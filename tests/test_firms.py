from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from airlens_agent import http
from airlens_agent.sources import firms

FIX = Path(__file__).parent / "fixtures"
FAKE_KEY = "FAKE-MAP-KEY-0123456789"
HEADER = "latitude,longitude,acq_date,acq_time,satellite,confidence,frp,daynight\n"
ROW = "36.5,128.7,2025-03-22,0512,N20,n,12.4,D\n"
BBOX = (124.0, 33.0, 132.0, 39.0)


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("NASA_FIRMS_MAP_KEY", FAKE_KEY)


def _patch_get(monkeypatch, responder):
    """Replace http.get; `responder(url)` returns CSV text. Returns the list of URLs requested."""
    urls = []

    def fake(url, params=None, *, op, conn=None, **_kw):
        assert op == "firms.area"
        urls.append(url)
        return httpx.Response(200, text=responder(url))

    monkeypatch.setattr(http, "get", fake)
    return urls


def test_parse_viirs():
    rows = firms.parse_csv((FIX / "firms_viirs_sample.csv").read_text(), "VIIRS_NOAA20_NRT")
    assert len(rows) == 2
    first = rows[0]
    assert (first.lat, first.lon) == (36.43012, 128.71234)
    assert first.ts_utc == datetime(2025, 3, 22, 5, 12, tzinfo=UTC)  # acq_time 512 -> 05:12
    assert first.satellite == "N20" and first.confidence == "n" and first.daynight == "D"
    assert first.frp == 12.4 and first.source == "VIIRS_NOAA20_NRT"
    assert rows[1].ts_utc.hour == 16 and rows[1].confidence == "h"


def test_parse_modis_numeric_confidence():
    (row,) = firms.parse_csv((FIX / "firms_modis_sample.csv").read_text(), "MODIS_NRT")
    assert row.confidence == "82" and row.satellite == "Terra"
    assert row.ts_utc == datetime(2025, 3, 22, 1, 59, tzinfo=UTC)


def test_parse_header_only_and_empty():
    assert firms.parse_csv(HEADER, "MODIS_NRT") == []
    assert firms.parse_csv("", "MODIS_NRT") == []


def test_non_csv_body_is_redacted():
    body = f"Invalid MAP_KEY. key={FAKE_KEY} {FAKE_KEY}"
    with pytest.raises(http.HttpError) as exc:
        firms.parse_csv(body, "MODIS_NRT")
    assert FAKE_KEY not in str(exc.value)
    assert "Invalid MAP_KEY" in str(exc.value)


def test_url_shape():
    url = firms._area_url("K", "MODIS_NRT", (1.0, 2.0, 3.0, 4.0), 3, date(2025, 3, 22))
    assert url == f"{firms.BASE_URL}/K/MODIS_NRT/1.0,2.0,3.0,4.0/3/2025-03-22"
    assert firms._area_url("K", "MODIS_NRT", (1.0, 2.0, 3.0, 4.0), 1, None).endswith("/1")


def test_fetch_hotspots_one_call_per_source_and_stores(conn, monkeypatch):
    urls = _patch_get(monkeypatch, lambda url: HEADER + ROW)
    rows = firms.fetch_hotspots(conn, BBOX)
    assert len(urls) == len(firms.LIVE_SOURCES)
    assert {r.source for r in rows} == set(firms.LIVE_SOURCES)
    # same position/time/satellite from three sources dedupes in the store
    assert conn.execute("SELECT COUNT(*) FROM hotspots").fetchone()[0] == 1


def test_day_range_validated(conn, monkeypatch):
    _patch_get(monkeypatch, lambda url: HEADER)
    with pytest.raises(ValueError):
        firms.fetch_hotspots(conn, BBOX, day_range=6)


def test_chunks_at_most_five_days():
    assert firms._chunks(date(2026, 2, 20), date(2026, 2, 20)) == [(date(2026, 2, 20), 1)]
    assert firms._chunks(date(2026, 2, 20), date(2026, 2, 24)) == [(date(2026, 2, 20), 5)]
    assert firms._chunks(date(2026, 2, 20), date(2026, 3, 2)) == [
        (date(2026, 2, 20), 5),
        (date(2026, 2, 25), 5),
        (date(2026, 3, 2), 1),
    ]


def test_range_splits_into_chunks_per_source(conn, monkeypatch):
    urls = _patch_get(monkeypatch, lambda url: HEADER + ROW)
    firms.fetch_hotspots_range(
        conn, BBOX, date(2026, 2, 20), date(2026, 2, 26), sources=("VIIRS_NOAA20_SP",)
    )
    assert [u.split("/")[-2:] for u in urls] == [["5", "2026-02-20"], ["2", "2026-02-25"]]


def test_sp_falls_back_to_nrt_when_empty(conn, monkeypatch):
    def responder(url):
        return HEADER if "_SP/" in url else HEADER + ROW

    urls = _patch_get(monkeypatch, responder)
    fallbacks = []
    rows = firms.fetch_hotspots_range(
        conn, BBOX, date(2026, 2, 20), date(2026, 2, 24),
        sources=("VIIRS_NOAA20_SP",), fallbacks=fallbacks,
    )
    assert [u.split("/")[-4] for u in urls] == ["VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"]
    assert fallbacks == [("VIIRS_NOAA20_SP", date(2026, 2, 20))]
    assert [r.source for r in rows] == ["VIIRS_NOAA20_NRT"]


def test_sp_with_rows_does_not_fall_back(conn, monkeypatch):
    urls = _patch_get(monkeypatch, lambda url: HEADER + ROW)
    fallbacks = []
    rows = firms.fetch_hotspots_range(
        conn, BBOX, date(2025, 3, 22), date(2025, 3, 22),
        sources=("MODIS_SP",), fallbacks=fallbacks,
    )
    assert len(urls) == 1 and fallbacks == [] and rows[0].source == "MODIS_SP"
