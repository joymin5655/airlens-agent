import json
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest

from airlens_agent import http
from airlens_agent.config import KST
from airlens_agent.sources import openmeteo

FIX = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIX / name).read_text())


def _patch_get(monkeypatch, responder):
    """Replace http.get; `responder(url, params)` returns JSON data. Returns the call list."""
    calls = []

    def fake(url, params=None, *, op, conn=None, **_kw):
        calls.append((url, params, op))
        return httpx.Response(200, json=responder(url, params))

    monkeypatch.setattr(http, "get", fake)
    return calls


def test_grid_points():
    pts = openmeteo.grid_points((126.0, 36.0, 127.0, 37.0))
    assert len(pts) == 9
    assert pts[0] == (36.0, 126.0) and pts[-1] == (37.0, 127.0)
    row = openmeteo.grid_points((0.0, 0.0, 1.0, 0.3), step=0.5)
    assert row == [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0)]
    assert len(openmeteo.grid_points((124.0, 33.0, 132.0, 39.0))) == 17 * 13


def test_air_quality_list_response(conn, monkeypatch):
    calls = _patch_get(monkeypatch, lambda url, params: _load("openmeteo_cams_multi.json"))
    pts = [(37.57, 126.98), (36.57, 128.73)]
    start = datetime(2026, 2, 22, tzinfo=KST)
    samples = openmeteo.fetch_air_quality(conn, pts, start, start + timedelta(days=1))

    assert len(calls) == 1  # both locations in one batched request
    url, params, op = calls[0]
    assert op == "openmeteo.air_quality" and url == openmeteo.AIR_QUALITY_URL
    assert params["latitude"] == "37.57,36.57" and params["longitude"] == "126.98,128.73"
    assert params["timezone"] == "Asia/Seoul" and params["domains"] == "cams_global"
    assert params["start_date"] == "2026-02-22" and params["end_date"] == "2026-02-23"
    assert set(params["hourly"].split(",")) == set(openmeteo.CAMS_VARS)

    # Seoul's third hour is all-null and dropped: 2 + 3 samples
    assert len(samples) == 5
    first = samples[0]
    assert (first.lat, first.lon) == (37.57, 126.98) and first.source == "openmeteo.cams"
    assert first.ts == datetime(2026, 2, 22, 0, 0, tzinfo=KST)
    assert first.variables["pm10"] == 85.2
    assert conn.execute("SELECT COUNT(*) FROM grid_samples").fetchone()[0] == 5


def test_air_quality_single_object_response(conn, monkeypatch):
    _patch_get(monkeypatch, lambda url, params: _load("openmeteo_cams_single.json"))
    start = datetime(2026, 2, 22, tzinfo=KST)
    samples = openmeteo.fetch_air_quality(conn, [(37.57, 126.98)], start, start)
    assert [s.variables["pm10"] for s in samples] == [85.2, 88.0]


def test_location_count_mismatch_raises(conn, monkeypatch):
    _patch_get(monkeypatch, lambda url, params: _load("openmeteo_cams_single.json"))
    start = datetime(2026, 2, 22, tzinfo=KST)
    with pytest.raises(http.HttpError, match="expected 2"):
        openmeteo.fetch_air_quality(conn, [(37.0, 126.0), (36.0, 128.0)], start, start)


def test_use_archive_boundary():
    now = datetime(2026, 10, 28, 12, 0, tzinfo=KST)
    assert openmeteo.use_archive(datetime(2026, 10, 22, tzinfo=KST), now)  # 6 days old
    assert not openmeteo.use_archive(datetime(2026, 10, 23, tzinfo=KST), now)  # 5 days old
    assert not openmeteo.use_archive(now, now)


def test_wind_old_range_uses_archive_and_historical_forecast(conn, monkeypatch):
    def responder(url, params):
        return _load(
            "openmeteo_wind_archive.json" if url == openmeteo.ARCHIVE_URL
            else "openmeteo_wind_850.json"
        )

    calls = _patch_get(monkeypatch, responder)
    start = datetime(2025, 3, 22, tzinfo=KST)
    samples = openmeteo.fetch_wind(conn, [(36.57, 128.73)], start, start + timedelta(days=3))

    assert [(c[0], c[2]) for c in calls] == [
        (openmeteo.ARCHIVE_URL, "openmeteo.archive"),
        (openmeteo.HIST_FORECAST_URL, "openmeteo.historical_forecast"),
    ]
    for _, params, _ in calls:
        assert params["wind_speed_unit"] == "ms" and params["timezone"] == "Asia/Seoul"
        assert params["start_date"] == "2025-03-22" and params["end_date"] == "2025-03-25"
    archive_params, hist_params = calls[0][1], calls[1][1]
    assert archive_params["models"] == "era5" and "models" not in hist_params
    assert {s.source for s in samples} == {"openmeteo.era5", "openmeteo.hist_fc"}
    era5 = next(s for s in samples if s.source == "openmeteo.era5")
    assert era5.variables["wind_speed_10m"] == 3.2
    hist = next(s for s in samples if s.source == "openmeteo.hist_fc")
    assert hist.variables["wind_speed_850hPa"] == 12.1


def test_wind_recent_range_uses_forecast_api(conn, monkeypatch):
    now = datetime.now(KST)
    stamp = now.strftime("%Y-%m-%dT00:00")
    old = (now - timedelta(days=2)).strftime("%Y-%m-%dT00:00")

    def responder(url, params):
        names = openmeteo.WIND_VARS + openmeteo.WIND_850_VARS
        return {
            "hourly": {"time": [old, stamp], **{n: [1.0, 2.0] for n in names}},
        }

    calls = _patch_get(monkeypatch, responder)
    samples = openmeteo.fetch_wind(conn, [(37.5, 127.0)], now - timedelta(days=1), now)

    assert len(calls) == 1
    url, params, op = calls[0]
    assert url == openmeteo.FORECAST_URL and op == "openmeteo.forecast"
    assert params["past_days"] == 1 and params["forecast_days"] == 1
    assert "wind_speed_850hPa" in params["hourly"]
    # the sample two days back is outside start..end and dropped
    assert [s.ts.date() for s in samples] == [now.date()]
    assert samples[0].source == "openmeteo.forecast"


def test_wind_forecast_window_checked(conn, monkeypatch):
    _patch_get(monkeypatch, lambda url, params: {})
    now = datetime.now(KST)
    with pytest.raises(ValueError):
        openmeteo.fetch_wind(
            conn, [(37.5, 127.0)], now - timedelta(days=4), now + timedelta(days=30)
        )
