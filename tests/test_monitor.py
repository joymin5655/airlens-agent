import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from airlens_agent import config, detect, http, monitor, quota, store
from airlens_agent.sources import airkorea, firms, gibs, openmeteo
from conftest import make_candidate


class FakeReport:
    claims = ["a", "b"]
    unverified_claims: list = []
    models = ["muse-spark"]
    cost_usd = 0.0
    cause = "dust"
    confidence = 0.8
    fallback = None

    def model_dump(self, mode="json"):
        return {"cause": self.cause, "cost_usd": self.cost_usd}


@pytest.fixture
def live(tmp_path, monkeypatch, conn):
    """Everything networked is replaced; calls are recorded in `calls`."""
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path / "data"))
    calls = {"investigate": [], "image": [], "wind": 0}
    cands = [make_candidate("evt-a", severity=3.0), make_candidate("evt-b", severity=2.5)]
    monkeypatch.setattr(airkorea, "fetch_realtime_nation", lambda c: [])
    monkeypatch.setattr(airkorea, "fetch_stations", lambda c: [])
    monkeypatch.setattr(firms, "fetch_hotspots", lambda c, bbox: [])
    monkeypatch.setattr(detect, "detect_latest", lambda c: cands)

    def wind(c, points, start, end):
        calls["wind"] += 1
        return []

    monkeypatch.setattr(openmeteo, "fetch_wind", wind)
    monkeypatch.setattr(openmeteo, "fetch_air_quality", lambda c, p, s, e: [])

    def image(bbox, day, layer, *, out_dir, conn=None, **kw):
        calls["image"].append((bbox, day, layer))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{day}_{layer}_{gibs.bbox_slug(bbox)}.jpg"
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr(gibs, "fetch_image", image)
    monkeypatch.setattr(monitor.agent, "triage", lambda cand, ledger: SimpleNamespace(
        model_used="m"))

    def investigate(c, cand, tri, **kw):
        calls["investigate"].append(cand.event_id)
        return FakeReport()

    monkeypatch.setattr(monitor.agent, "investigate", investigate)
    calls["cands"] = cands
    return calls


def test_investigates_the_strongest_and_writes_status(live, conn, tmp_path):
    status = monitor.run_once(conn)
    assert live["investigate"] == ["evt-a"]
    data = tmp_path / "data" / "live"
    assert json.loads((data / "reports" / "evt-a.json").read_text())["cause"] == "dust"
    on_disk = json.loads((data / "status.json").read_text())
    assert on_disk["investigations"][0]["event_id"] == "evt-a"
    assert on_disk["investigations"][0]["report"] == "evt-a.json"
    rows = {c["event_id"]: c for c in on_disk["candidates"]}
    assert rows["evt-a"]["report"] == "evt-a.json" and rows["evt-b"]["report"] is None
    assert {"ran_at", "last_observation", "stations_reporting", "skips", "budget"} <= set(status)
    assert on_disk["budget"]["meta"]["daily_cap"] == config.DAILY_SPEND_CAPS["meta"]
    assert quota.usage(conn)["monitor.investigate"]["used"] == 1


def test_inputs_are_gathered_with_a_padded_bbox(live, conn):
    monitor.run_once(conn)
    assert live["wind"] == 1
    assert live["image"]
    w, s, e, n = live["image"][0][0]
    assert e - w >= 2.5 and n - s >= 2.5


def test_already_investigated_is_skipped(live, conn, tmp_path):
    report = tmp_path / "data" / "live" / "reports" / "evt-a.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    monitor.run_once(conn)
    assert live["investigate"] == ["evt-b"]  # evt-a already has its report


def test_nothing_new_records_no_candidate(live, conn, tmp_path):
    reports = tmp_path / "data" / "live" / "reports"
    reports.mkdir(parents=True)
    for eid in ("evt-a", "evt-b"):
        (reports / f"{eid}.json").write_text("{}")
    status = monitor.run_once(conn)
    assert live["investigate"] == []
    assert [s["reason"] for s in status["skips"]] == ["no_candidate"]


def test_weak_candidates_are_not_investigated(live, conn, monkeypatch):
    weak = [replace(c, severity=1.2) for c in live["cands"]]
    monkeypatch.setattr(detect, "detect_latest", lambda c: weak)
    status = monitor.run_once(conn)
    assert live["investigate"] == []
    assert "weak_candidates" in [s["reason"] for s in status["skips"]]


def test_local_candidates_are_not_investigated(live, conn, monkeypatch):
    local = [replace(c, kind="local") for c in live["cands"]]
    monkeypatch.setattr(detect, "detect_latest", lambda c: local)
    monitor.run_once(conn)
    assert live["investigate"] == []


def test_daily_investigation_cap(live, conn, monkeypatch):
    monkeypatch.setitem(config.QUOTA_CAPS, "monitor.investigate", 1)
    first = monitor.run_once(conn)
    assert first["investigations"]
    second = monitor.run_once(conn, max_investigations=2)
    assert live["investigate"] == ["evt-a"]  # the cap held: evt-b was not run
    assert second["skips"][-1]["reason"] == "cap_reached"


def test_max_investigations_runs_several(live, conn):
    monitor.run_once(conn, max_investigations=2)
    assert live["investigate"] == ["evt-a", "evt-b"]


def test_no_investigate_flag(live, conn):
    status = monitor.run_once(conn, investigate=False)
    assert live["investigate"] == [] and status["investigations"] == []
    assert len(status["candidates"]) == 2


def test_daily_budget_exceeded_skips_investigation(live, conn, monkeypatch):
    quota.charge(conn, "meta", config.DAILY_SPEND_CAPS["meta"])  # today's cap is used up
    status = monitor.run_once(conn)
    assert live["investigate"] == []
    assert status["skips"][-1]["reason"] == "budget_exceeded"
    assert status["budget"]["meta"]["spent_today"] >= config.DAILY_SPEND_CAPS["meta"]


def test_budget_exceeded_during_investigation_is_recorded(live, conn, monkeypatch):
    def broke(*a, **kw):
        raise quota.BudgetExceeded("meta: budget reached")

    monkeypatch.setattr(monitor.agent, "investigate", broke)
    status = monitor.run_once(conn)
    assert status["investigations"] == []
    assert status["skips"][-1]["reason"] == "budget_exceeded"


def test_a_crashing_investigation_does_not_stop_the_run(live, conn, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(monitor.agent, "investigate", boom)
    status = monitor.run_once(conn)
    assert status["skips"][-1]["reason"] == "investigation_failed"
    assert (monitor.live_dir() / "status.json").exists()


def test_failing_sources_do_not_stop_the_run(live, conn, monkeypatch):
    def down(*a, **kw):
        raise http.HttpError("airkorea down")

    monkeypatch.setattr(airkorea, "fetch_realtime_nation", down)
    monkeypatch.setattr(firms, "fetch_hotspots", down)
    monkeypatch.setattr(openmeteo, "fetch_wind", down)
    status = monitor.run_once(conn)
    assert status["sources"]["airkorea.realtime"].startswith("failed")
    assert status["sources"]["firms"].startswith("failed")
    assert sum(s["reason"] == "source_failed" for s in status["skips"]) >= 2
    assert live["investigate"] == ["evt-a"]  # detection and the investigation still ran
    assert status["investigations"][0]["inputs"]["wind"].startswith("failed")


def test_missing_key_is_a_failed_source_not_a_crash(live, conn, monkeypatch):
    def nokey(c):
        raise config.MissingKeyError("DATA_GO_KR_SERVICE_KEY is not set")

    monkeypatch.setattr(airkorea, "fetch_realtime_nation", nokey)
    status = monitor.run_once(conn)
    assert status["sources"]["airkorea.realtime"].startswith("failed")


def test_status_counts_stations_of_the_latest_hour(live, conn):
    from airlens_agent.models import Observation

    ts = make_candidate().ts
    store.upsert_observations(conn, [Observation("s1", ts, pm10=10.0), Observation("s2", ts)])
    status = monitor.run_once(conn)
    assert status["stations_reporting"] == 2
    assert status["last_observation"] == ts.isoformat()
