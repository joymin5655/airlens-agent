"""Replay on a synthetic event DB (no network)."""

from datetime import datetime, timedelta

from airlens_agent import config, events, replay, store
from airlens_agent.config import KST
from airlens_agent.models import Observation, Station


def test_months_span():
    assert replay._months(datetime(2025, 12, 20, tzinfo=KST), datetime(2026, 2, 3, tzinfo=KST)) == [
        (2025, 12), (2026, 1), (2026, 2)]


def test_events_definitions():
    fire, dust = events.EVENTS["fire-2025-03"], events.EVENTS["dust-2026-02"]
    assert fire.replay_start == datetime(2025, 3, 22, tzinfo=KST)
    assert dust.expected_hint == "coarse" and dust.focus_bbox == (125.5, 36.5, 128.0, 38.3)


def test_run_replay_summarises_days(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path))
    ev = events.Event("syn", datetime(2026, 1, 1, tzinfo=KST), datetime(2026, 1, 9, 23, tzinfo=KST),
                      (126, 36, 128, 38), datetime(2026, 1, 9, tzinfo=KST), "coarse")
    monkeypatch.setitem(events.EVENTS, "syn", ev)
    conn = store.connect(config.event_db_path("syn"))
    sts = [Station(f"서울|{n}", n, "서울", 37.5 + i * 0.05, 127.0, None, "t", "도시대기")
           for i, n in enumerate("abcd")]
    store.upsert_stations(conn, sts)
    rows = []
    for s in sts:
        t = ev.start
        while t <= ev.end:
            dust = t >= datetime(2026, 1, 9, 6, tzinfo=KST)
            rows.append(Observation(s.key, t, pm10=300 if dust else 30 + t.hour % 3,
                                    pm25=45 if dust else 15 + t.hour % 2))
            t += timedelta(hours=1)
    store.upsert_observations(conn, rows)
    out = replay.run_replay("syn")
    assert list(out["days"]) == ["2026-01-08", "2026-01-09"]
    assert out["days"]["2026-01-08"]["regional_events"] == 0
    top = out["days"]["2026-01-09"]["top_event"]
    assert out["days"]["2026-01-09"]["regional_events"] == 1
    assert top["hint"] == "coarse" and top["provinces"] == ["서울"] and top["peak_pm10"] == 300
    assert top["provinces_seen"] == ["서울"] and top["max_stations"] == 4
    assert top["first_ts"] <= top["ts"] <= top["last_ts"]
