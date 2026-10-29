from datetime import UTC, datetime, timedelta

from airlens_agent import store
from airlens_agent.config import KST
from airlens_agent.models import Hotspot
from conftest import EVENT_TS, make_candidate


def test_candidate_roundtrip(event_db):
    got = store.candidate(event_db, "evt-1")
    want = make_candidate()
    assert got == want and got.ts.tzinfo is not None
    assert store.candidate(event_db, "missing") is None


def test_hotspots_between_filters_by_time(event_db):
    start = datetime(2026, 2, 21, 19, tzinfo=UTC)
    got = store.hotspots_between(event_db, start, start + timedelta(hours=2))
    assert sorted(got["frp"]) == [5.0, 12.0]
    assert store.hotspots_between(event_db, start + timedelta(hours=1), start).empty


def test_hotspots_between_handles_other_utc_offsets(conn):
    # a row stamped with a +09:00 offset still matches by instant, not by string order
    at = datetime(2026, 2, 22, 9, tzinfo=KST)  # 00:00 UTC
    store.insert_hotspots(conn, [Hotspot(37.0, 127.0, at, "N20", "t", frp=1.0)])
    got = store.hotspots_between(conn, at - timedelta(minutes=1), at + timedelta(minutes=1))
    assert len(got) == 1
    assert store.hotspots_between(conn, at + timedelta(minutes=1), at + timedelta(hours=1)).empty


def test_hotspots_in_filters_bbox_and_time(event_db):
    start = datetime(2026, 2, 21, 0, tzinfo=UTC)
    end = datetime(2026, 2, 23, 0, tzinfo=UTC)
    got = store.hotspots_in(event_db, (126.0, 36.0, 128.0, 38.0), start, end)
    assert sorted(got["frp"]) == [12.0, 99.0]
    narrow = store.hotspots_in(event_db, (127.4, 37.1, 127.6, 37.3), start, end)
    assert list(narrow["frp"]) == [12.0]


def test_grid_samples_near_picks_nearest_point(event_db):
    start, end = EVENT_TS - timedelta(hours=2), EVENT_TS
    got = store.grid_samples_near(event_db, 37.2, 127.1, start, end, source="openmeteo.cams")
    assert [(g.lat, g.lon) for g in got] == [(37.0, 127.0)] * 3
    assert [g.ts for g in got] == sorted(g.ts for g in got)
    assert got[0].variables["pm10"] == 58.0
    mixed = store.grid_samples_near(event_db, 37.0, 127.0, start, end)
    assert {g.source for g in mixed} == {"openmeteo.cams", "openmeteo.era5", "openmeteo.hist_fc"}


def test_grid_samples_near_respects_max_km_and_window(event_db):
    assert store.grid_samples_near(event_db, 33.0, 127.0, EVENT_TS, EVENT_TS) == []
    assert store.grid_samples_near(event_db, 37.0, 127.0, EVENT_TS + timedelta(days=5),
                                   EVENT_TS + timedelta(days=6)) == []
