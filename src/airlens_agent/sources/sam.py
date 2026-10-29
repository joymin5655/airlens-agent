"""SAM 3.1 (Meta Model API) text-prompted segmentation of satellite images."""

import base64
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
)
from PIL import Image, ImageDraw

from .. import config, http, quota

USD_PER_CALL = 0.0025  # $2.50 per 1,000 images
OP = "meta.sam"
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_COLORS = ((255, 64, 64), (64, 160, 255), (255, 200, 40), (80, 220, 120))
# An object id followed by its tokens, and a box token's fields (the parser's own record grammar).
_RECORD = re.compile(r"([0-9]+)((?:<\|[^|]*\|>)+)")
_BOX_TOKEN = re.compile(r"<\|box;([^|]*)\|>")


@dataclass
class Region:
    prompt: str
    object_id: str
    box_px: tuple[int, int, int, int]  # x1, y1, x2, y2 in sidecar-frame pixels (x2/y2 exclusive)
    box_lonlat: tuple[float, float, float, float]  # west, south, east, north
    area_frac: float  # mask pixels / frame pixels (box area / frame area if the mask is missing)
    confidence: float | None
    mask: np.ndarray | None  # bool, frame size (height, width)


def sidecar_path(image_path: Path) -> Path:
    return image_path.with_name(image_path.name + ".json")


def _text_of(payload: dict) -> str:
    """The model's text from a Responses payload (`output_text` or output[].content[])."""
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts = [
        c.get("text") or ""
        for item in payload.get("output") or []
        for c in item.get("content") or []
        if c.get("type") == "output_text"
    ]
    return "\n".join(parts)


def _source_sizes(text: str) -> dict[str, tuple[int, int]]:
    """(w, h) of the image each object's box coordinates refer to, from the box tokens.

    The parser checks `w`/`h` but its box record does not expose them. They differ from the
    sidecar frame when the API resizes the image before segmenting it.
    """
    sizes = {}
    for oid, tokens in _RECORD.findall(text):
        box = _BOX_TOKEN.search(tokens)
        if box is None:
            continue
        fields = {}
        for part in box.group(1).split(";"):
            key, _, value = part.partition("=")
            fields[key.strip()] = value.strip()
        try:
            size = int(fields["w"]), int(fields["h"])
        except (KeyError, ValueError):
            continue
        if min(size) > 0:
            sizes[oid] = size
    return sizes


def _scaled(
    box: tuple[int, int, int, int], source: tuple[int, int], frame: tuple[int, int]
) -> tuple[int, int, int, int]:
    """A box (x1, y1, x2, y2; x2/y2 exclusive) of a `source` (w, h) image on the `frame` pixel grid.

    Edges are rounded outward in integer arithmetic, so a box never shrinks to nothing and a box
    inside the source stays inside the frame. Clipping to the frame is left to the caller.
    """
    (x1, y1, x2, y2), (sw, sh), (fw, fh) = box, source, frame
    return x1 * fw // sw, y1 * fh // sh, -(-x2 * fw // sw), -(-y2 * fh // sh)


def _frame_mask(
    record: SegmentationMaskRecord, source: tuple[int, int], frame: tuple[int, int]
) -> np.ndarray:
    """Decode the box-local mask, scale it (nearest) to its box on the frame and place it in a
    frame array (height, width). The part of the box outside the frame is cropped."""
    m = record.mask
    raster = np.frombuffer(decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width)
    b = record.bounds
    left, top, right, bottom = _scaled(
        (int(b.left), int(b.top), int(b.right), int(b.bottom)), source, frame
    )
    ys, xs = np.arange(top, bottom), np.arange(left, right)
    rows = np.minimum((ys - top) * m.height // (bottom - top), m.height - 1)
    cols = np.minimum((xs - left) * m.width // (right - left), m.width - 1)
    in_y, in_x = (ys >= 0) & (ys < frame[1]), (xs >= 0) & (xs < frame[0])
    full = np.zeros((frame[1], frame[0]), dtype=bool)
    full[np.ix_(ys[in_y], xs[in_x])] = raster[np.ix_(rows[in_y], cols[in_x])].astype(bool)
    return full


def parse_regions(
    text: str, prompt: str, bbox: tuple[float, float, float, float], width: int, height: int
) -> list[Region]:
    """Turn the SAM special-token text into Regions (empty text = no match = no regions).

    Boxes and masks are in the pixels of the image the API segmented (the `w`/`h` of the box
    token), which can differ from the sidecar frame (`width` x `height`): they are scaled to the
    frame and clipped to it. Non-empty text that has no box, or that the parser reports an error
    for, raises HttpError instead of passing for "no match".
    """
    if not text.strip():
        return []
    parser = image_segmentation_format().create_parser()
    parser.push(text, emit=False)
    result = parser.finish(CompletedOutcome()).result
    boxes = {r.object_id: r for r in result.records if isinstance(r, SegmentationBoxRecord)}
    masks = {r.object_id: r for r in result.records if isinstance(r, SegmentationMaskRecord)}
    errors = list(dict.fromkeys(d.code for d in result.diagnostics if d.severity == "error"))
    # Warnings only mean the parser ignored something and kept the data.
    # ponytail: all-or-nothing, so one bad record also drops the good ones next to it (and a box
    # whose mask failed to decode no longer degrades to a box-only region). Upgrade: return the
    # good regions and report the errors beside them.
    if errors or not boxes:
        raise http.HttpError(
            f"{OP}: unparseable output: {', '.join(errors) or 'no box'}: "
            f"{http.redact(text)[:200]}"
        )
    sizes = _source_sizes(text)
    frame = (width, height)
    west, south, east, north = bbox
    regions = []
    for oid, box in boxes.items():
        source = sizes.get(oid, frame)  # the parser requires w/h on every box; guard anyway
        x1, y1, x2, y2 = _scaled(
            (int(box.left), int(box.top), int(box.right), int(box.bottom)), source, frame
        )
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
        mrec = masks.get(oid)
        mask = _frame_mask(mrec, source, frame) if mrec else None
        area = float(mask.sum()) / (width * height) if mask is not None else (
            (x2 - x1) * (y2 - y1) / (width * height)
        )
        regions.append(
            Region(
                prompt=prompt,
                object_id=oid,
                box_px=(x1, y1, x2, y2),
                box_lonlat=(
                    west + x1 / width * (east - west),
                    north - y2 / height * (north - south),
                    west + x2 / width * (east - west),
                    north - y1 / height * (north - south),
                ),
                area_frac=area,
                confidence=box.confidence if box.confidence is not None else (
                    mrec.confidence if mrec else None
                ),
                mask=mask,
            )
        )
    return regions


def segment(
    image_path: str | Path,
    prompt: str,
    *,
    ledger: sqlite3.Connection,
    transport: httpx.BaseTransport | None = None,
) -> list[Region]:
    """Segment `prompt` (one short noun phrase) in a GIBS image; [] for blank images or no match.

    The `<image>.json` sidecar written by gibs.fetch_image supplies the bbox and frame size.
    A reply that cannot be read raises HttpError, after the call has been charged.
    """
    image_path = Path(image_path)
    side = json.loads(sidecar_path(image_path).read_text())
    if (side.get("blank_check") or {}).get("blank"):
        return []
    quota.ensure_budget(ledger, "meta")
    data = base64.b64encode(image_path.read_bytes()).decode()
    url = f"data:{_MIME.get(image_path.suffix.lower(), 'image/jpeg')};base64,{data}"
    payload = {
        "model": config.SAM_MODEL,
        "input": [{
            "type": "message", "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": url},
            ],
        }],
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
    }
    resp = http.post_json(
        f"{config.META_API_BASE}/responses",
        payload,
        headers={"Authorization": f"Bearer {config.model_api_key()}"},
        op=OP,
        conn=ledger,
        transport=transport,
    )
    quota.charge(ledger, "meta", USD_PER_CALL)
    try:
        text = _text_of(resp.json())
    except ValueError as exc:
        raise http.HttpError(f"{OP}: not JSON: {http.redact(resp.text)[:200]}") from exc
    return parse_regions(text, prompt, tuple(side["bbox"]), side["width"], side["height"])


def overlay(image_path: str | Path, regions: list[Region], out_path: str | Path) -> Path:
    """Draw translucent masks and boxes on the image and save a PNG."""
    img = Image.open(image_path).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for i, r in enumerate(regions):
        color = _COLORS[i % len(_COLORS)]
        if r.mask is not None:
            tint = Image.new("RGBA", img.size, (*color, 90))
            layer.paste(tint, mask=Image.fromarray(r.mask.astype(np.uint8) * 255))
        draw.rectangle(r.box_px, outline=(*color, 255), width=3)
        draw.text((r.box_px[0] + 4, r.box_px[1] + 4), f"{r.prompt} {i}", fill=(*color, 255))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(img, layer).save(out, format="PNG")
    return out
