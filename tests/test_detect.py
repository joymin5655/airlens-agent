"""Detector tests on synthetic observations (no network, no real AirKorea data)."""

from datetime import datetime, timedelta

import pytest

from airlens_agent import detect, store
from airlens_agent.config import KST
from airlens_agent.models import Candidate, Observation, Station

AT = datetime(2026, 2, 22, 12, tzinfo=KST)


def add_stations(conn, names, lat0=37.5, lon0=127.0, network="도시대기", step=0.05):
    sts = [
        Station(f"서울|{n}", n, "서울", lat0 + i * step, lon0, None, "t", network)
        for i, n in enumerate(names)
    ]
    store.upsert_stations(conn, sts)
    return [s.key for s in sts]


def add_obs(conn, key, spike=None, pm25_spike=None, base10=30.0, base25=15.0, flags=None, at=AT):
    """7 days of flat baseline (small wobble) then an optional spike at `at`."""
    rows = []
    for h in range(7 * 24, 0, -1):
        wobble = (h % 5) - 2
        rows.append(Observation(key, at - timedelta(hours=h), pm10=base10 + wobble,
                                pm25=base25 + wobble / 2, co=0.5, no2=0.02))
    rows.append(Observation(key, at, pm10=spike if spike is not None else base10,
                            pm25=pm25_spike if pm25_spike is not None else base25,
                            co=0.5, no2=0.02, flags=flags or {}))
    store.upsert_observations(conn, rows)


def test_dust_like_cluster_is_coarse_regional(conn):
    for key in add_stations(conn, "abcd"):
        add_obs(conn, key, spike=320, pm25_spike=60)
    [c] = detect.detect(conn, AT)
    assert c.kind == "regional" and c.hint == "coarse"
    assert c.peak_pm10 == 320 and len(c.station_keys) == 4
    assert c.ratio_pm25_pm10 == pytest.approx(60 / 320)


def test_smoke_like_cluster_is_fine_regional(conn):
    for key in add_stations(conn, "abc"):
        add_obs(conn, key, spike=200, pm25_spike=160)
    [c] = detect.detect(conn, AT)
    assert (c.kind, c.hint) == ("regional", "fine")


def test_single_station_spike_is_local(conn):
    keys = add_stations(conn, "abcd")
    add_obs(conn, keys[0], spike=300, pm25_spike=50)
    for key in keys[1:]:
        add_obs(conn, key)
    [c] = detect.detect(conn, AT)
    assert c.kind == "local" and c.station_keys == [keys[0]]


def test_none_and_flagged_values_ignored(conn):
    keys = add_stations(conn, "abc")
    add_obs(conn, keys[0], spike=None)
    add_obs(conn, keys[1], flags={"pm10": "통신장애"})
    add_obs(conn, keys[2])
    # flagged values arrive as None, and a flag on a stored value masks it too
    store.upsert_observations(conn, [
        Observation(keys[0], AT, pm10=None, pm25=None, flags={"pm10": "점검"}),
        Observation(keys[1], AT, pm10=400, pm25=200,
                    flags={"pm10": "통신장애", "pm25": "통신장애"}),
    ])
    assert detect.detect(conn, AT) == []


def test_roadside_skipped_by_default(conn):
    keys = add_stations(conn, "abc", network="도로변대기")
    for key in keys:
        add_obs(conn, key, spike=400, pm25_spike=80)
    assert detect.detect(conn, AT) == []
    assert len(detect.detect(conn, AT, include_roadside=True, save=False)) == 1


def test_short_baseline_uses_absolute_thresholds_only(conn):
    [key] = add_stations(conn, "a")
    store.upsert_observations(conn, [
        Observation(key, AT - timedelta(hours=h), pm10=10, pm25=5) for h in range(1, 10)
    ] + [Observation(key, AT, pm10=120, pm25=20)])
    assert detect.detect(conn, AT) == []  # z would be huge but 120 < very-bad 151
    store.upsert_observations(conn, [Observation(key, AT, pm10=160, pm25=20)])
    assert len(detect.detect(conn, AT)) == 1


def test_z_requires_bad_grade_as_well(conn):
    [key] = add_stations(conn, "a")
    add_obs(conn, key, spike=60, pm25_spike=20)  # huge z against a flat baseline but < 81
    assert detect.detect(conn, AT) == []


def test_stations_without_coordinates_counted_not_clustered(conn):
    store.upsert_stations(
        conn, [Station("서울|nc", "nc", "서울", None, None, None, "t", "도시대기")])
    add_obs(conn, "서울|nc", spike=300, pm25_spike=50)
    keys = add_stations(conn, "ab")
    for key in keys:
        add_obs(conn, key, spike=300, pm25_spike=50)
    [c] = detect.detect(conn, AT)
    assert len(c.station_keys) == 2 and c.features["elevated_without_coords"] == 1


def test_event_linking_across_hours(conn):
    keys = add_stations(conn, "abcd")
    for key in keys:
        add_obs(conn, key, spike=300, pm25_spike=50)
        store.upsert_observations(conn, [
            Observation(key, AT + timedelta(hours=1), pm10=310, pm25=55)])
    [first] = detect.detect(conn, AT)
    [second] = detect.detect(conn, AT + timedelta(hours=1))
    assert second.event_id == first.event_id
    assert first.event_id.startswith("evt-2026022212-")
    # far away and no shared stations -> a new id
    far = [Station(f"부산|{n}", n, "부산", 35.1 + i * 0.05, 129.0, None, "t", "도시대기")
           for i, n in enumerate("xyz")]
    store.upsert_stations(conn, far)
    for st in far:
        add_obs(conn, st.key, spike=300, pm25_spike=50, at=AT + timedelta(hours=2))
    ids = {c.event_id for c in detect.detect(conn, AT + timedelta(hours=2))}
    assert ids and first.event_id not in ids


def test_hotspots_counted_within_50km(conn):
    from airlens_agent.models import Hotspot
    for key in add_stations(conn, "abc"):
        add_obs(conn, key, spike=300, pm25_spike=200)
    near = Hotspot(37.6, 127.1, AT.astimezone(KST) - timedelta(hours=3), "N20", "t")
    far = Hotspot(35.0, 129.0, AT - timedelta(hours=3), "N20", "t")
    store.insert_hotspots(conn, [near, far])
    [c] = detect.detect(conn, AT)
    assert c.features["hotspots_50km_24h"] == 1


def cand(eid, hour, severity, day=22):
    return Candidate(eid, datetime(2026, 2, day, hour, tzinfo=KST), "regional", "coarse", ["k"],
                     37.0, 127.0, (0, 0, 0, 0), 300, 50, 0.2, severity)


def test_daily_selection_cap_counts_each_event_once():
    cands = [cand(f"e{i}", 10, float(i)) for i in range(10)]
    cands += [cand("e9", 11, 3.0), cand("e1", 9, 50.0, day=23)]
    chosen = detect.select_for_investigation(cands, per_day=6)
    day22 = [c for c in chosen if c.ts.day == 22]
    assert len(day22) == 6 and len({c.event_id for c in chosen}) == len(chosen)
    assert [c.event_id for c in day22][:2] == ["e9", "e8"]
    assert [c.event_id for c in chosen if c.ts.day == 23] == ["e1"]


def test_hint_and_severity_formulas():
    assert detect.hint_for(0.2, 120) == "coarse"
    assert detect.hint_for(0.2, 60) == "mixed"  # low ratio but not elevated PM10
    assert detect.hint_for(0.7, 100) == "fine"
    assert detect.hint_for(None, 300) == "mixed"
    assert detect.severity(151, None, 1) == pytest.approx(1.0)
    assert detect.severity(None, 152, 4) == pytest.approx(2 * (1 + 1.3862943611), rel=1e-6)


def test_rerun_same_hour_keeps_ids(conn):
    for key in add_stations(conn, "abcd"):
        add_obs(conn, key, spike=300, pm25_spike=50)
    first = detect.detect(conn, AT)
    second = detect.detect(conn, AT)
    assert first and [c.event_id for c in second] == [c.event_id for c in first]
    assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == len(first)


def test_rerun_latest_hour_after_link_keeps_id(conn):
    for key in add_stations(conn, "abcd"):
        add_obs(conn, key, spike=300, pm25_spike=50)
        store.upsert_observations(conn, [
            Observation(key, AT + timedelta(hours=1), pm10=310, pm25=55)])
    [first] = detect.detect(conn, AT)
    detect.detect(conn, AT + timedelta(hours=1))
    [again] = detect.detect(conn, AT + timedelta(hours=1))
    assert again.event_id == first.event_id
    assert conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE event_id=?", (first.event_id,)
    ).fetchone()[0] == 1


def test_detect_on_empty_db_returns_nothing(conn):
    assert detect.detect(conn, AT) == []
