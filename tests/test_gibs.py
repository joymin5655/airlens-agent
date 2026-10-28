import json
from datetime import date
from io import BytesIO

import httpx
import pytest
from PIL import Image

from airlens_agent import http
from airlens_agent.sources import gibs

BBOX = (127.5, 35.0, 130.5, 37.5)


def _png(color=(10, 20, 30, 255), size=(8, 8), noise=False) -> bytes:
    img = Image.new("RGBA", size, color)
    if noise:
        for x in range(size[0]):
            for y in range(size[1]):
                img.putpixel((x, y), ((x * 30) % 256, (y * 30) % 256, (x * y) % 256, 255))
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _patch_get(monkeypatch, content, calls):
    def fake(url, params=None, *, op, conn=None, **_kw):
        calls.append((url, params, op))
        return httpx.Response(200, content=content)

    monkeypatch.setattr(http, "get", fake)


def test_wms_bbox_is_lat_first():
    assert gibs.wms_bbox(BBOX) == "35.0,127.5,37.5,130.5"


def test_image_size_follows_aspect():
    assert gibs.image_size(BBOX, 1024) == (1024, 853)  # 2.5 / 3.0 * 1024
    assert gibs.image_size((115.0, 30.0, 132.0, 42.0), 1024) == (1024, 723)
    with pytest.raises(ValueError):
        gibs.image_size((0, 0, 1, 100), 1024)  # height 102400 > 20000
    with pytest.raises(ValueError):
        gibs.image_size((1, 0, 0, 1), 1024)


def test_blank_check_flags_uniform_and_transparent():
    assert gibs.blank_check(Image.new("RGB", (8, 8), (0, 0, 0)))["reason"] == "near-uniform"
    clear = gibs.blank_check(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    assert clear["reason"] == "fully transparent"
    ok = gibs.blank_check(Image.open(BytesIO(_png(noise=True))))
    assert ok["blank"] is False and ok["stddev"] > gibs.UNIFORM_STDDEV


def test_fetch_image_writes_file_and_sidecar(tmp_path, monkeypatch):
    calls = []
    _patch_get(monkeypatch, _png(noise=True), calls)
    layer = "OMPS_NOAA20_NadirMapper_AerosolIndex_380"
    path = gibs.fetch_image(BBOX, date(2025, 3, 22), layer, width=300, out_dir=tmp_path)

    assert path.name == f"2025-03-22_{layer}_127.5_35_130.5_37.5.png"
    _, params, op = calls[0]
    assert op == "gibs.wms" and params["BBOX"] == "35.0,127.5,37.5,130.5"
    assert (params["WIDTH"], params["HEIGHT"]) == (300, 250)
    assert params["FORMAT"] == "image/png" and params["TIME"] == "2025-03-22"
    side = json.loads(path.with_name(path.name + ".json").read_text())
    assert side["requested_date"] == "2025-03-22" and side["layer"] == layer
    assert side["bbox"] == list(BBOX) and side["blank_check"]["blank"] is False


def test_true_colour_uses_jpeg(tmp_path, monkeypatch):
    calls = []
    _patch_get(monkeypatch, _png(), calls)
    layer = gibs.TRUE_COLOR_LAYERS[0]
    path = gibs.fetch_image(BBOX, date(2025, 3, 22), layer, out_dir=tmp_path)
    assert calls[0][1]["FORMAT"] == "image/jpeg" and path.suffix == ".jpg"
    assert json.loads(path.with_name(path.name + ".json").read_text())["blank_check"]["blank"]


def test_xml_error_with_200_raises(tmp_path, monkeypatch):
    _patch_get(monkeypatch, b"<ServiceExceptionReport>bad TIME</ServiceExceptionReport>", [])
    with pytest.raises(http.HttpError, match="not an image"):
        gibs.fetch_image(BBOX, date(2025, 3, 22), gibs.TRUE_COLOR_LAYERS[0], out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
