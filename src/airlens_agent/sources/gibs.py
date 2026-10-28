"""NASA GIBS true-colour and aerosol satellite images (owner: implementer B)."""

import io
import json
import sqlite3
from datetime import date
from pathlib import Path

from PIL import Image, ImageStat

from .. import http

WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
MAX_PIXELS = 20000

TRUE_COLOR_LAYERS = (
    "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
    "VIIRS_NOAA21_CorrectedReflectance_TrueColor",
    "MODIS_Terra_CorrectedReflectance_TrueColor",
    "MODIS_Aqua_CorrectedReflectance_TrueColor",
)
AEROSOL_LAYERS = (
    "OMPS_NOAA20_NadirMapper_AerosolIndex_380",
    "MODIS_Combined_Value_Added_AOD",
)
# Aerosol layers are mostly transparent; JPEG has no alpha, so they use PNG.
LAYER_FORMATS = {layer: "image/png" for layer in AEROSOL_LAYERS}
DEFAULT_FORMAT = "image/jpeg"
_EXT = {"image/jpeg": "jpg", "image/png": "png"}

UNIFORM_STDDEV = 2.0  # mean per-band stddev (0-255) below this = near-uniform
MIN_OPAQUE_FRACTION = 0.01  # RGBA images with less opaque area than this are empty


def image_size(bbox: tuple[float, float, float, float], width: int) -> tuple[int, int]:
    """(width, height) in pixels; height follows the bbox aspect in degrees (EPSG:4326)."""
    west, south, east, north = bbox
    if east <= west or north <= south:
        raise ValueError(f"invalid bbox {bbox}")
    height = round(width * (north - south) / (east - west))
    if not (1 <= width <= MAX_PIXELS and 1 <= height <= MAX_PIXELS):
        raise ValueError(f"image size {width}x{height} outside 1..{MAX_PIXELS}")
    return width, height


def wms_bbox(bbox: tuple[float, float, float, float]) -> str:
    """WMS 1.3.0 + EPSG:4326 wants lat-first axis order; lon-first returns an all-black image."""
    west, south, east, north = bbox
    return f"{south},{west},{north},{east}"


def bbox_slug(bbox: tuple[float, float, float, float]) -> str:
    return "_".join(f"{v:g}" for v in bbox)


def blank_check(img: Image.Image) -> dict:
    """Pixel statistics and a blank/near-uniform verdict."""
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    opaque_fraction = sum(alpha.histogram()[1:]) / (rgba.width * rgba.height)
    stat = ImageStat.Stat(rgba.convert("RGB"))
    mean = sum(stat.mean) / 3
    stddev = sum(stat.stddev) / 3
    reason = None
    if opaque_fraction < MIN_OPAQUE_FRACTION:
        reason = "fully transparent"
    elif stddev < UNIFORM_STDDEV:
        reason = "near-uniform"
    return {
        "blank": reason is not None,
        "reason": reason,
        "mean": round(mean, 2),
        "stddev": round(stddev, 2),
        "opaque_fraction": round(opaque_fraction, 4),
    }


def fetch_image(
    bbox: tuple[float, float, float, float],
    day: date,
    layer: str,
    *,
    width: int = 1024,
    out_dir: Path,
    conn: sqlite3.Connection | None = None,
) -> Path:
    """Download a WMS image of `layer` for bbox (west, south, east, north) on `day` into out_dir
    and return the file path. A sidecar `<image>.json` records the request and the blank check.

    GIBS snaps TIME to the nearest available date, so the sidecar holds the *requested* date.
    """
    w, h = image_size(bbox, width)
    fmt = LAYER_FORMATS.get(layer, DEFAULT_FORMAT)
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": fmt,
        "CRS": "EPSG:4326",
        "BBOX": wms_bbox(bbox),
        "WIDTH": w,
        "HEIGHT": h,
        "TIME": day.isoformat(),
    }
    resp = http.get(WMS_URL, params, op="gibs.wms", conn=conn)
    try:
        img = Image.open(io.BytesIO(resp.content))
        img.load()
    except Exception as exc:  # WMS reports errors as XML with HTTP 200
        snippet = resp.content[:120].decode("utf-8", "replace")
        raise http.HttpError(f"gibs.wms: not an image: {snippet!r}") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day.isoformat()}_{layer}_{bbox_slug(bbox)}.{_EXT[fmt]}"
    path.write_bytes(resp.content)
    sidecar = {
        "layer": layer,
        "bbox": list(bbox),
        "requested_date": day.isoformat(),
        "format": fmt,
        "width": img.width,
        "height": img.height,
        "bytes": len(resp.content),
        "blank_check": blank_check(img),
    }
    path.with_name(path.name + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return path
