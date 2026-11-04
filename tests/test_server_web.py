"""Web server: events API, media safety, ask endpoint limits, status. No network, no keys."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from airlens_agent import ask, server, store
from airlens_agent.config import KST
from airlens_agent.models import Candidate, Hotspot, Observation, Station

NAME, REPORT_ID = "fire-2025-03", "evt-2025032421-0"
TS = datetime(2025, 3, 25, 10, tzinfo=KST)
BBOX = (127.5, 35.0, 130.0, 37.5)
IMG = "2025-03-23_VIIRS_NOAA20_CorrectedReflectance_TrueColor_127.5_35_130_37.5.png"


def _report(**extra) -> dict:
    rep = {
        "headline": "Smoke", "cause": "wildfire_smoke", "confidence": 0.8, "summary": "s",
        "claims": [], "alternatives": [], "limitations": [], "unverified_claims": [],
        "models": ["m"], "cost_usd": 0.01,
        "evidence": [{
            "evidence_id": "E2", "tool": "satellite_segments", "source": "SAM",
            "data": {"overlay_file": "ov.png"},
        }],
    }
    return rep | extra


def _png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(path)


def _past_db(path):
    conn = store.connect(path)
    store.upsert_stations(conn, [
        Station("a", "A", "x", 36.0, 128.5, None, "t"),  # in the event
        Station("b", "B", "x", 36.5, 129.0, None, "t"),
        Station("far", "F", "x", 10.0, 10.0, None, "t"),  # outside the image box
    ])
    store.upsert_observations(conn, [
        Observation("a", TS, pm10=300.0, pm25=200.0),
        Observation("b", TS, pm10=20.0, pm25=10.0),
        Observation("far", TS, pm10=20.0, pm25=10.0),
        Observation("b", TS - timedelta(hours=1), pm10=999.0),  # an older hour is not used
    ])
    store.insert_hotspots(conn, [
        Hotspot(36.1, 128.6, (TS - timedelta(hours=3)).astimezone(UTC), "N20", "VIIRS", 5.0),
        Hotspot(36.2, 128.7, (TS - timedelta(hours=30)).astimezone(UTC), "N20", "VIIRS", 99.0),
        Hotspot(36.3, 128.8, (TS + timedelta(hours=2)).astimezone(UTC), "N20", "VIIRS", 50.0),
    ])
    store.save_candidates(conn, [Candidate(
        REPORT_ID, TS, "regional", "fine", ["a"], 36.0, 128.5, (128.0, 35.5, 129.5, 36.5),
        368.0, 253.0, 0.6, 3.0, {},
    )])
    conn.close()


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path))
    monkeypatch.delenv("AIRLENS_LEDGER_FILE", raising=False)
    server._hits.clear()
    (tmp_path / "events").mkdir()
    _past_db(tmp_path / "events" / f"{NAME}.sqlite")
    base = tmp_path / "events" / NAME
    (base / "reports").mkdir(parents=True)
    (base / "reports" / f"{REPORT_ID}.json").write_text(json.dumps(_report()))
    _png(base / "images" / IMG)
    _png(base / "reports" / "ov.png")
    (base / "reports" / "secret.json").write_text("{}")
    return TestClient(server.app)


def test_events_lists_the_past_case_and_skips_missing_ones(site):
    body = site.get("/api/events").json()
    assert [e["id"] for e in body["past"]] == [NAME]  # dust-2026-02 has no files here
    assert body["past"][0]["cause"] == "wildfire_smoke" and body["live"] == []


def test_past_detail_has_scene_data(site):
    d = site.get(f"/api/events/past/{NAME}").json()
    assert d["observed_at"] == TS.isoformat()
    by_lat = {s["lat"]: s for s in d["stations"]}
    assert set(by_lat) == {36.0, 36.5}  # the far station is outside the image box
    assert by_lat[36.0]["in_event"] and not by_lat[36.5]["in_event"]
    assert by_lat[36.5]["pm10"] == 20.0  # the event hour, not the older one
    assert [h["frp"] for h in d["hotspots"]] == [5.0]  # only the 24 h before the event
    (img,) = d["images"]
    assert img["bbox"] == list(BBOX) and img["date"] == "2025-03-23"
    assert d["overlay"].endswith("/reports/ov.png")
    assert site.get(img["file"]).status_code == 200
    assert site.get(d["overlay"]).headers["content-type"] == "image/png"


def test_unknown_case_is_404(site):
    assert site.get("/api/events/past/nope").status_code == 404
    assert site.get("/api/events/other/x").status_code == 404
    assert site.get("/api/events/live/..%2Fx").status_code == 404


@pytest.mark.parametrize("path", [
    "../live.sqlite",
    f"events/{NAME}/images/../../{NAME}.sqlite",
    f"events/{NAME}/reports/secret.json",  # not an image
    f"events/{NAME}/sam/x.png",  # a folder that is not served
    "%2e%2e/%2e%2e/etc/passwd",
    f"events/{NAME}/images",
])
def test_media_refuses_everything_but_images_in_the_allowed_folders(site, path):
    assert site.get(f"/media/{path}").status_code == 404


def test_live_candidate_without_report_says_why(site, tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    row = {
        "event_id": "evt-live-1", "ts": TS.isoformat(), "kind": "regional", "hint": "fine",
        "centroid": [36.0, 128.5], "bbox": [128.0, 35.5, 129.5, 36.5], "peak_pm10": 200.0,
        "peak_pm25": 120.0, "severity": 2.4, "n_stations": 5, "report": None,
    }
    (live / "status.json").write_text(json.dumps({
        "ran_at": TS.isoformat(), "stations_reporting": 600, "candidates": [row],
        "skips": [{"reason": "cap_reached", "detail": "x", "event_id": "evt-live-1"}],
    }))
    (item,) = site.get("/api/events").json()["live"]
    assert not item["has_report"] and "limit" in item["why_no_report"]
    d = site.get("/api/events/live/evt-live-1").json()
    assert d["report"] is None and d["candidate"]["centroid_lat"] == 36.0 and d["box"]
    (live / "reports").mkdir()
    (live / "reports" / "evt-live-1.json").write_text(json.dumps(_report(headline="Live smoke")))
    (item,) = site.get("/api/events").json()["live"]
    assert item["has_report"] and item["title"] == "Live smoke"
    st = site.get("/api/status").json()
    assert st["live"]["stations_reporting"] == 600 and "meta" in st["budget"]


def test_status_before_the_first_run_and_eval(site, tmp_path):
    assert site.get("/api/status").json()["live"]["ran_at"] is None
    assert site.get("/api/eval").status_code == 404
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "latest.json").write_text('{"runs": 2, "summary": {}}')
    assert site.get("/api/eval").json()["runs"] == 2


# --- ask ----------------------------------------------------------------------------------------


@pytest.fixture
def fake_answer(monkeypatch):
    calls = []

    def fake(question, *, ledger, live_conn, lat=None, lon=None, **kw):
        files = [r[2] for r in ledger.execute("PRAGMA database_list")]
        calls.append({"q": question, "ledger": files[0], "lat": lat})
        return ask.Answer(answer="ok", risk_level="good", confidence=0.5)

    monkeypatch.setattr(server.ask, "answer", fake)
    return calls


def test_ask_answers_and_charges_the_web_ledger(site, tmp_path, monkeypatch, fake_answer):
    ledger = tmp_path / "web-ledger.sqlite"
    monkeypatch.setenv("AIRLENS_LEDGER_FILE", str(ledger))
    r = site.post("/api/ask", json={"question": "  Is Seoul   OK? ", "lat": 37.5, "lon": 127.0})
    assert r.status_code == 200 and r.json()["answer"] == "ok"
    assert fake_answer[0]["q"] == "Is Seoul OK?" and fake_answer[0]["lat"] == 37.5
    assert fake_answer[0]["ledger"].endswith("web-ledger.sqlite")


@pytest.mark.parametrize("question", ["", "   ", "x" * 301])
def test_ask_validates_before_calling_the_model(site, fake_answer, question):
    r = site.post("/api/ask", json={"question": question})
    assert r.status_code == 422 and "error" in r.json() and not fake_answer


def test_ask_per_client_limit(site, fake_answer):
    for _ in range(server.PER_MINUTE):
        assert site.post("/api/ask", json={"question": "air?"}).status_code == 200
    r = site.post("/api/ask", json={"question": "air?"})
    assert r.status_code == 429 and "error" in r.json()
    # another client is not affected
    other = site.post("/api/ask", json={"question": "air?"}, headers={"x-forwarded-for": "9.9.9.9"})
    assert other.status_code == 200


def test_ask_failure_hides_the_traceback(site, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret detail /Users/x/file.py")

    monkeypatch.setattr(server.ask, "answer", boom)
    r = site.post("/api/ask", json={"question": "air?"})
    assert r.status_code == 502 and "secret" not in r.text and "Traceback" not in r.text


def test_site_is_served_at_the_root(site):
    r = site.get("/")
    assert r.status_code == 200 and "AirLens Agent" in r.text
    assert site.get("/health").status_code == 200
