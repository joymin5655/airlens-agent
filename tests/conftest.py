import pytest

from airlens_agent import store


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "t.sqlite")
    yield c
    c.close()


# --- a small replay-event fixture shared by the tool, agent and CLI tests -------------------------

import json  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

from meta_sam_parser._mask_codec import _encode_raster  # noqa: E402
from PIL import Image  # noqa: E402

from airlens_agent.config import KST  # noqa: E402
from airlens_agent.models import Candidate, GridSample, Hotspot, Observation, Station  # noqa: E402

EVENT_TS = datetime(2026, 2, 22, 10, tzinfo=KST)
IMG_BBOX = (126.0, 36.0, 128.0, 38.0)  # 0.05 deg per pixel on a 40x40 image
IMG_SIZE = 40
NOAA20 = "VIIRS_NOAA20_CorrectedReflectance_TrueColor"  # afternoon pass, about 14:00 KST
TERRA = "MODIS_Terra_CorrectedReflectance_TrueColor"  # morning pass, about 11:00 KST


def sam_text(box=(10, 10, 19, 19), mask_hw=(4, 5)) -> str:
    """Canned SAM answer: one object whose box-local mask is fully set."""
    h, w = mask_hw
    payload = _encode_raster([1] * (h * w), w, h)
    x1, y1, x2, y2 = box
    return (f"<0f>0<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={IMG_SIZE};h={IMG_SIZE};c=0.9|>"
            f"<|mask;x=0;y=0;data={h},{w},{payload}|>")


def make_candidate(event_id="evt-1", ts=EVENT_TS, severity=2.0, kind="regional") -> Candidate:
    return Candidate(
        event_id, ts, kind, "coarse", ["서울|A", "서울|B"], 37.0, 127.0, (126.9, 36.9, 127.1, 37.1),
        300.0, 90.0, 0.3, severity,
        {"co_z_mean": 0.2, "no2_z_mean": 0.1, "rise_3h_pm10": 150.0, "rise_3h_pm25": 40.0,
         "extent_km": 12.0, "provinces": ["서울"], "hotspots_50km_24h": 0},
    )


@pytest.fixture
def event_db(tmp_path):
    """Event DB with a candidate at EVENT_TS, hourly data around it, and some 'future' rows."""
    c = store.connect(tmp_path / "event.sqlite")
    stations = [Station(f"서울|{n}", n, "서울", 37.0 + i * 0.01, 127.0, None, "t", "도시대기")
                for i, n in enumerate("ABC")]
    store.upsert_stations(c, stations)
    obs = []
    for h in range(0, 24):
        ts = datetime(2026, 2, 22, h, tzinfo=KST)
        peak = max(0, 10 - abs(h - 10))  # rises to the event hour, falls after
        for key, scale in (("서울|A", 30), ("서울|B", 25), ("서울|C", 3)):
            obs.append(Observation(key, ts, pm10=20 + peak * scale, pm25=10 + peak * scale / 3,
                                   co=0.4 + peak * 0.05, no2=0.02))
    store.upsert_observations(c, obs)
    store.insert_hotspots(c, [
        Hotspot(37.2, 127.5, datetime(2026, 2, 21, 20, tzinfo=UTC), "N20", "t", frp=12.0),
        Hotspot(37.0, 127.2, datetime(2026, 2, 22, 4, tzinfo=UTC),
                "N20", "t", frp=99.0),  # 13:00 KST: after the event hour
        Hotspot(30.0, 120.0, datetime(2026, 2, 21, 20, tzinfo=UTC),
                "N20", "t", frp=5.0),  # far away
    ])
    grid = []
    for h in range(0, 24):
        ts = datetime(2026, 2, 22, h, tzinfo=KST)
        grid.append(GridSample(37.0, 127.0, ts, "openmeteo.cams", {
            "pm10": 50.0 + h, "pm2_5": 20.0, "dust": 100.0 + 10 * h, "aerosol_optical_depth": 0.8,
            "carbon_monoxide": 300.0, "nitrogen_dioxide": 5.0}))
        grid.append(GridSample(37.0, 127.0, ts, "openmeteo.era5", {
            "wind_speed_10m": 4.0, "wind_direction_10m": 270.0}))
        grid.append(GridSample(37.0, 127.0, ts, "openmeteo.hist_fc", {
            "wind_speed_850hPa": 12.0, "wind_direction_850hPa": 300.0}))
        grid.append(GridSample(45.0, 140.0, ts, "openmeteo.cams", {"pm10": 1.0}))  # far point
    store.upsert_grid_samples(c, grid)
    store.save_candidates(c, [make_candidate()])
    yield c
    c.close()


def write_image(directory, day, layer=NOAA20, *, blank=False):
    """A cached GIBS true-colour image with its sidecar; an existing one is replaced."""
    path = directory / f"{day}_{layer}_126_36_128_38.jpg"
    Image.new("RGB", (IMG_SIZE, IMG_SIZE), (90, 90, 90)).save(path)
    path.with_name(path.name + ".json").write_text(json.dumps({
        "bbox": list(IMG_BBOX), "width": IMG_SIZE, "height": IMG_SIZE,
        "blank_check": {"blank": blank}}))
    return path


@pytest.fixture
def image_dir(tmp_path):
    """One non-blank and one blank true-colour image for the event day, with GIBS sidecars."""
    d = tmp_path / "images"
    d.mkdir()
    for day, blank in (("2026-02-22", False), ("2026-02-21", True)):
        write_image(d, day, blank=blank)
    return d
