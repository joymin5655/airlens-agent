import base64
import json

import httpx
import numpy as np
import pytest
from meta_sam_parser._mask_codec import _encode_raster
from PIL import Image

from airlens_agent import config, http, quota
from airlens_agent.sources import sam

W, H = 40, 20  # the sidecar frame
BBOX = (126.0, 34.0, 130.0, 36.0)  # 0.1 deg per pixel on both axes
META_KEY = "FAKE-META-KEY-9f8e7d"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", META_KEY)


def _image(tmp_path, blank=False):
    path = tmp_path / "2026-02-22_layer_bbox.jpg"
    Image.new("RGB", (W, H), (90, 90, 90)).save(path)
    side = {"bbox": list(BBOX), "width": W, "height": H, "blank_check": {"blank": blank}}
    sam.sidecar_path(path).write_text(json.dumps(side))
    return path


def _obj(oid, box=(10, 4, 19, 11), mask_hw=(4, 5), size=(W, H), mask_first=False, bits=None):
    """One SAM object record: a box (inclusive x2/y2, in a `size` source image) with a box-local
    mask of mask_hw rows x cols (`bits`, row by row; default fully set). The two tokens may come
    in either order."""
    h, w = mask_hw
    x1, y1, x2, y2 = box
    bits = [1] * (h * w) if bits is None else bits
    box_tok = f"<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={size[0]};h={size[1]};c=0.8|>"
    mask_tok = f"<|mask;x=0;y=0;data={h},{w},{_encode_raster(bits, w, h)}|>"
    return f"{oid}" + (mask_tok + box_tok if mask_first else box_tok + mask_tok)


def _canned(box=(10, 4, 19, 11), mask_hw=(4, 5), size=(W, H), bits=None):
    """One frame, one object."""
    return "<0f>" + _obj(0, box, mask_hw, size, bits=bits)


# a 4 x 5 mask shaped like an "L": the whole top row and the whole left column
L_SHAPE = [1 if i < 5 or i % 5 == 0 else 0 for i in range(4 * 5)]


def _transport(text, key="output_text"):
    seen = []

    def handler(request):
        seen.append({"url": str(request.url), "auth": request.headers["authorization"],
                     "body": json.loads(request.content)})
        if key == "output_text":
            return httpx.Response(200, json={"output_text": text})
        return httpx.Response(200, json={"output": [
            {"type": "message", "content": [{"type": "output_text", "text": text}]}]})

    return httpx.MockTransport(handler), seen


def test_request_shape_and_parsed_region(tmp_path, conn):
    path = _image(tmp_path)
    transport, seen = _transport(_canned())
    regions = sam.segment(path, "smoke", ledger=conn, transport=transport)
    req = seen[0]
    assert req["url"] == "https://api.meta.ai/v1/responses"
    assert req["auth"] == "Bearer FAKE-META-KEY-9f8e7d"
    body = req["body"]
    assert body["model"] == config.SAM_MODEL
    assert body["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit"}
    parts = body["input"][0]["content"]
    assert parts[0] == {"type": "input_text", "text": "smoke"}
    assert parts[1]["type"] == "input_image"
    assert parts[1]["image_url"] == "data:image/jpeg;base64," + base64.b64encode(
        path.read_bytes()).decode()
    assert len(regions) == 1
    r = regions[0]
    assert r.prompt == "smoke" and r.box_px == (10, 4, 20, 12)  # x2/y2 inclusive -> exclusive
    assert r.confidence == pytest.approx(0.8)
    assert r.mask is not None and r.mask.shape == (H, W)
    assert r.area_frac == pytest.approx(10 * 8 / (W * H))
    assert quota.usage(conn)["meta.sam"]["used"] == 1
    assert quota.spent(conn, "meta") == pytest.approx(0.0025)


def test_responses_output_items_are_read(tmp_path, conn):
    transport, _ = _transport(_canned(), key="output")
    assert len(sam.segment(_image(tmp_path), "haze", ledger=conn, transport=transport)) == 1


def test_pixel_to_lonlat_mapping(tmp_path, conn):
    transport, _ = _transport(_canned())
    r = sam.segment(_image(tmp_path), "smoke", ledger=conn, transport=transport)[0]
    west, south, east, north = r.box_lonlat
    assert (west, east) == pytest.approx((127.0, 128.0))
    assert (south, north) == pytest.approx((34.8, 35.6))


def test_blank_image_skips_api(tmp_path, conn):
    transport, seen = _transport(_canned())
    assert sam.segment(_image(tmp_path, blank=True), "smoke", ledger=conn,
                       transport=transport) == []
    assert seen == [] and quota.spent(conn, "meta") == 0


@pytest.mark.parametrize("text", ["", " \n "])
def test_no_match_returns_empty_and_still_charges(tmp_path, conn, text):
    transport, _ = _transport(text)
    assert sam.segment(_image(tmp_path), "smoke", ledger=conn, transport=transport) == []
    assert quota.spent(conn, "meta") == pytest.approx(0.0025)


# --- non-empty output the parser cannot use must not pass for "no match" ----------------------


@pytest.mark.parametrize(("text", "why"), [
    ("I could not find any smoke in this image.", "no box"),  # prose, no records
    ("<0f>", "malformed_record"),  # a frame header without objects
    ("<0f>" + _obj(0, box=(10, 4, 45, 11)), "invalid_box"),  # x2 outside the 40 px source
    ("<0f>" + _obj(0).replace("data=4,5,", "data=4,6,"), "invalid_mask_payload"),
    # one bad record next to a good one: do not return the good one as if the answer were whole
    ("<0f>" + _obj(0) + "," + _obj(1, box=(10, 4, 45, 11)), "invalid_box"),
])
def test_unparseable_output_raises_but_keeps_the_charge(tmp_path, conn, text, why):
    transport, _ = _transport(text)
    with pytest.raises(http.HttpError, match=f"meta.sam: unparseable output: {why}"):
        sam.segment(_image(tmp_path), "smoke", ledger=conn, transport=transport)
    assert quota.spent(conn, "meta") == pytest.approx(0.0025)  # the call was billed
    assert quota.usage(conn)["meta.sam"]["used"] == 1


def test_parser_warnings_do_not_fail_the_call(tmp_path, conn):
    # an unknown box field and an unknown token are ignored with warnings; the data is kept
    text = "<0f>" + _obj(0).replace("c=0.8|>", "c=0.8;zzz=1|>") + "<|extra|>"
    transport, _ = _transport(text)
    assert len(sam.segment(_image(tmp_path), "smoke", ledger=conn, transport=transport)) == 1


def test_unparseable_snippet_redacts_a_key_cut_by_the_limit(tmp_path, conn):
    transport, _ = _transport("z" * 193 + META_KEY + " tail")  # prose; key 7 chars before the cut
    with pytest.raises(http.HttpError, match="unparseable output") as err:
        sam.segment(_image(tmp_path), "smoke", ledger=conn, transport=transport)
    assert META_KEY[:6] not in str(err.value) and "***" in str(err.value)


def test_not_json_snippet_redacts_a_key_cut_by_the_limit(tmp_path, conn):
    body = "q" * 193 + META_KEY + " tail"  # the key starts 7 characters before the 200-char cut
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    with pytest.raises(http.HttpError, match="not JSON") as err:
        sam.segment(_image(tmp_path), "smoke", ledger=conn, transport=transport)
    # cutting first would leave a key prefix that redact() can no longer recognise
    assert META_KEY[:6] not in str(err.value) and "***" in str(err.value)
    assert quota.spent(conn, "meta") == pytest.approx(0.0025)


# --- the box token's w/h is the image the coordinates refer to; the sidecar frame is W x H -----


def _one_region(tmp_path, conn, text):
    transport, _ = _transport(text)
    (region,) = sam.segment(_image(tmp_path), "smoke", ledger=conn, transport=transport)
    return region


@pytest.mark.parametrize(("size", "box"), [
    ((80, 40), (20, 8, 39, 23)),  # the source is twice the frame
    ((20, 10), (5, 2, 9, 5)),  # the source is half the frame
])
def test_box_and_mask_are_scaled_to_the_sidecar_frame(tmp_path, conn, size, box):
    r = _one_region(tmp_path, conn, _canned(box=box, size=size, bits=L_SHAPE))
    assert r.box_px == (10, 4, 20, 12)  # the same 10 x 8 pixel box as in the unscaled case
    west, south, east, north = r.box_lonlat
    assert (west, east) == pytest.approx((127.0, 128.0))
    assert (south, north) == pytest.approx((34.8, 35.6))
    # the L covers the top 2 rows of the box in full and the left 2 columns of the other 6
    assert r.mask.shape == (H, W) and r.mask[4:6, 10:20].all() and r.mask[6:12, 10:12].all()
    assert r.mask.sum() == 2 * 10 + 6 * 2 and not r.mask[6:12, 12:20].any()
    assert r.area_frac == pytest.approx(32 / (W * H))


def test_a_larger_source_no_longer_overflows_the_frame(tmp_path, conn):
    # bottom-right corner of an 80x40 source: unscaled it lies entirely outside the 40x20 frame
    r = _one_region(tmp_path, conn, _canned(box=(60, 20, 79, 39), size=(80, 40)))
    assert r.box_px == (30, 10, 40, 20) and r.mask[10:20, 30:40].all() and r.mask.sum() == 100


def test_scaling_rounds_the_box_outward_never_to_nothing(tmp_path, conn):
    # one pixel of a source 10x larger than the frame covers a part of one frame pixel: keep it
    r = _one_region(tmp_path, conn, _canned(box=(101, 101, 101, 101), size=(400, 200),
                                            mask_hw=(1, 1)))
    assert r.box_px == (10, 10, 11, 11) and r.mask.sum() == 1


def test_each_object_uses_its_own_source_size_whatever_the_token_order(tmp_path, conn):
    text = ("<0f>" + _obj(0, (20, 8, 39, 23), size=(80, 40), mask_first=True)
            + "," + _obj(1, (0, 0, 9, 9)))
    transport, _ = _transport(text)
    first, second = sam.segment(_image(tmp_path), "smoke", ledger=conn, transport=transport)
    assert first.box_px == (10, 4, 20, 12) and second.box_px == (0, 0, 10, 10)


def test_unknown_source_size_falls_back_to_the_frame_and_clips(tmp_path, conn, monkeypatch):
    monkeypatch.setattr(sam, "_source_sizes", lambda text: {})
    # the first two of the mask's five columns are set, on a 20 px wide box whose right half
    # lies outside the frame: they cover x = 30..37 of the box
    bits = [1 if i % 5 < 2 else 0 for i in range(4 * 5)]
    r = _one_region(tmp_path, conn, _canned(box=(30, 10, 49, 29), size=(80, 40), bits=bits))
    assert r.box_px == (30, 10, 40, 20)  # clipped to the 40x20 frame
    # cropped, not squeezed (squeezing the mask into the visible 10 px would set only x = 30..33)
    assert r.mask.shape == (H, W) and r.mask[10:20, 30:38].all() and not r.mask[:, 38:].any()
    assert r.mask.sum() == 8 * 10 and r.area_frac == pytest.approx(80 / (W * H))


def test_budget_checked_before_request(tmp_path, conn, monkeypatch):
    monkeypatch.setitem(config.SPEND_CAPS, "meta", 0.001)
    quota.charge(conn, "meta", 0.01)
    transport, seen = _transport(_canned())
    with pytest.raises(quota.BudgetExceeded):
        sam.segment(_image(tmp_path), "smoke", ledger=conn, transport=transport)
    assert seen == []


def test_overlay_writes_png(tmp_path, conn):
    path = _image(tmp_path)
    transport, _ = _transport(_canned())
    regions = sam.segment(path, "smoke", ledger=conn, transport=transport)
    out = sam.overlay(path, regions, tmp_path / "out" / "o.png")
    img = Image.open(out)
    assert img.format == "PNG" and img.size == (W, H)
    assert np.asarray(img.convert("RGB")).std() > 0
