import dataclasses
import json
from datetime import date, datetime

import httpx
import pytest

from airlens_agent import agent, cli, config, http, store
from airlens_agent.config import KST
from airlens_agent.events import get_event
from airlens_agent.sources import airkorea, firms, gibs, sam
from conftest import NOAA20, TERRA, make_candidate, sam_text, write_image

EVENT = "dust-2026-02"


@pytest.fixture(autouse=True)
def gibs_calls(monkeypatch):
    """Keep every test here offline: GIBS downloads are recorded instead of made, and any request
    that would leave the machine fails the test. Returns the recorded downloads."""
    calls = []

    def fetch_image(bbox, day, layer, *, out_dir, **_kw):
        calls.append({"bbox": bbox, "day": day, "layer": layer, "out_dir": out_dir})

    def no_network(self, request):
        raise AssertionError(f"real network access: {request.url}")

    monkeypatch.setattr(gibs, "fetch_image", fetch_image)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)  # mock transports pass
    return calls


def test_the_offline_guard_blocks_real_requests():
    with pytest.raises(AssertionError, match="real network access"):
        http.get("https://gibs.invalid/wms", op="t.op", retries=0)


def test_fetch_now_continues_when_firms_is_down(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(airkorea, "fetch_realtime_nation", lambda conn: [])

    def down(conn, bbox):
        raise http.HttpError("firms.area: HTTP 503")

    monkeypatch.setattr(firms, "fetch_hotspots", down)
    assert cli.cmd_fetch_now(None) == {
        "observations": 0, "latest_ts": None, "hotspots": None, "candidates": 0}
    assert "[firms] skipped" in capsys.readouterr().err


# --- investigate / sam-trial -----------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path))
    return tmp_path


def _event_db(name, candidates):
    conn = store.connect(config.event_db_path(name))
    store.save_candidates(conn, candidates)
    conn.close()


def _fake_agent(monkeypatch, seen, *, model="open-fake", fallback=None):
    tri = agent.Triage(category="dust", hypotheses=[], checks=[])
    tri._model, tri._fallback = model, fallback

    def fake_triage(cand, *, ledger, **kw):
        seen["triage"] = cand.event_id
        return tri

    def fake_investigate(conn, cand, triage, *, ledger, **kw):
        seen["investigate"] = (cand.event_id, kw)
        return agent.Report(
            headline="h", cause="dust", confidence=0.7, summary="s",
            claims=[agent.Claim(text="a", evidence_ids=["E1"]),
                    agent.Claim(text="b", evidence_ids=["E1"])],
            alternatives=[], limitations=[], models=["muse-fake"], unverified_claims=[1],
            triage=triage, triage_fallback=triage.fallback,
        )

    monkeypatch.setattr(agent, "triage", fake_triage)
    monkeypatch.setattr(agent, "investigate", fake_investigate)


def test_investigate_picks_top_severity_regional_and_writes_report(data_dir, monkeypatch, capsys):
    key_day = datetime(2026, 2, 22, 9, tzinfo=KST)  # the event's key day gets no preference
    other = datetime(2026, 2, 21, 9, tzinfo=KST)
    _event_db(EVENT, [
        make_candidate("evt-loud-other-day", other, severity=9.0),
        make_candidate("evt-key-low", key_day, severity=1.0),
        make_candidate("evt-key-mid", key_day, severity=3.0),
        make_candidate("evt-key-local", key_day, severity=50.0, kind="local"),  # not regional
    ])
    seen = {}
    _fake_agent(monkeypatch, seen)
    assert cli.main(["investigate", EVENT, "--max-steps", "4"]) == 0
    assert seen["triage"] == "evt-loud-other-day"
    assert seen["investigate"][0] == "evt-loud-other-day"
    assert seen["investigate"][1]["max_steps"] == 4
    assert seen["investigate"][1]["investigator"] == "meta"
    summary = json.loads(capsys.readouterr().out)
    assert summary["cause"] == "dust" and summary["claims_verified"] == "1/2"
    assert summary["models"] == ["open-fake", "muse-fake"] and summary["cost_usd"] == 0
    assert summary["fallback"] is None and summary["triage_fallback"] is None
    path = data_dir / "events" / EVENT / "reports" / "evt-loud-other-day.json"
    assert json.loads(path.read_text())["headline"] == "h"
    assert str(data_dir) not in path.read_text()


def test_investigate_breaks_severity_ties_by_time_then_event_id(data_dir, monkeypatch):
    early, late = datetime(2026, 2, 20, 9, tzinfo=KST), datetime(2026, 2, 21, 9, tzinfo=KST)
    _event_db(EVENT, [  # stored in the opposite order of the one they must be picked in
        make_candidate("evt-z", early, severity=5.0),
        make_candidate("evt-a", late, severity=5.0),
        make_candidate("evt-b", early, severity=5.0),
    ])
    seen = {}
    _fake_agent(monkeypatch, seen)
    cli.main(["investigate", EVENT])
    assert seen["triage"] == "evt-b"  # the earliest time, then the lowest id


def _far(event_id, severity):
    """A candidate in the southeast, outside the dust event's focus bbox (and its model grid)."""
    return dataclasses.replace(
        make_candidate(event_id, severity=severity), centroid_lat=35.54, centroid_lon=129.19)


def test_investigate_prefers_the_top_candidate_inside_the_focus_bbox(data_dir, monkeypatch, capsys):
    _event_db(EVENT, [_far("evt-far", 9.0), make_candidate("evt-inside", severity=3.0)])
    seen = {}
    _fake_agent(monkeypatch, seen)
    cli.main(["investigate", EVENT])
    assert seen["triage"] == "evt-inside"  # the grid and the images exist there
    assert "warning" not in capsys.readouterr().err


def test_investigate_falls_back_to_a_candidate_outside_the_bbox_with_a_warning(
    data_dir, monkeypatch, capsys
):
    _event_db(EVENT, [_far("evt-far", 9.0), _far("evt-far-low", 2.0)])
    seen = {}
    _fake_agent(monkeypatch, seen)
    cli.main(["investigate", EVENT])
    assert seen["triage"] == "evt-far"
    assert "evt-far lies outside" in capsys.readouterr().err


def test_the_summary_and_the_report_file_show_a_triage_fallback(data_dir, monkeypatch, capsys):
    _event_db(EVENT, [make_candidate("evt-1")])
    note = "baseten -> meta: baseten: budget $40.00 reached"
    _fake_agent(monkeypatch, {}, fallback=note)
    cli.main(["investigate", EVENT])
    summary = json.loads(capsys.readouterr().out)
    assert summary["triage_fallback"] == note
    saved = json.loads((data_dir / "events" / EVENT / "reports" / "evt-1.json").read_text())
    assert saved["triage_fallback"] == note


def test_a_neutral_triage_adds_no_blank_model_name(data_dir, monkeypatch, capsys):
    _event_db(EVENT, [make_candidate("evt-1")])
    note = "triage answer invalid after a retry, neutral triage used (ValidationError: ...)"
    _fake_agent(monkeypatch, {}, model="", fallback=note)  # no model gave a usable answer
    cli.main(["investigate", EVENT])
    summary = json.loads(capsys.readouterr().out)
    assert summary["models"] == ["muse-fake"] and summary["triage_fallback"] == note


@pytest.mark.parametrize(("given", "used"), [
    ("0", 1), ("-5", 1), ("1", 1), ("8", 8), ("12", 12), ("13", 12), ("99", 12),
])
def test_max_steps_is_kept_between_1_and_12(data_dir, monkeypatch, capsys, given, used):
    _event_db(EVENT, [make_candidate("evt-1")])
    seen = {}
    _fake_agent(monkeypatch, seen)
    cli.main(["investigate", EVENT, "--max-steps", given])
    assert seen["investigate"][1]["max_steps"] == used
    err = capsys.readouterr().err
    assert (f"--max-steps {given} limited to {used}" in err) == (int(given) != used)
    assert cli.MAX_STEPS == 12


def test_max_steps_defaults_to_8(data_dir, monkeypatch):
    _event_db(EVENT, [make_candidate("evt-1")])
    seen = {}
    _fake_agent(monkeypatch, seen)
    cli.main(["investigate", EVENT])
    assert seen["investigate"][1]["max_steps"] == 8


def test_investigate_event_id_overrides_the_default_pick(data_dir, monkeypatch, capsys):
    _event_db(EVENT, [
        make_candidate("evt-a", datetime(2026, 2, 20, 9, tzinfo=KST), severity=1.0),
        make_candidate("evt-b", datetime(2026, 2, 20, 9, tzinfo=KST), severity=2.0),
        make_candidate("evt-local", datetime(2026, 2, 20, 9, tzinfo=KST), severity=9.0,
                       kind="local"),
    ])
    seen = {}
    _fake_agent(monkeypatch, seen)
    cli.main(["investigate", EVENT])
    assert seen["triage"] == "evt-b"
    cli.main(["investigate", EVENT, "--event-id", "evt-a", "--investigator", "baseten"])
    assert seen["triage"] == "evt-a" and seen["investigate"][1]["investigator"] == "baseten"
    cli.main(["investigate", EVENT, "--event-id", "evt-local"])  # explicit ids may be non-regional
    assert seen["triage"] == "evt-local"
    with pytest.raises(SystemExit, match="no candidate"):
        cli.main(["investigate", EVENT, "--event-id", "nope"])


def test_investigate_needs_a_regional_candidate(data_dir, monkeypatch, gibs_calls):
    _event_db(EVENT, [make_candidate("evt-local", severity=9.0, kind="local")])
    seen = {}
    _fake_agent(monkeypatch, seen)
    with pytest.raises(SystemExit, match="no regional candidates"):
        cli.main(["investigate", EVENT])
    assert seen == {} and gibs_calls == []  # nothing was fetched or asked for


def test_investigate_needs_event_data(data_dir):
    with pytest.raises(SystemExit, match="fetch-event"):
        cli.main(["investigate", EVENT])


def test_investigate_fetches_the_images_the_satellite_tool_needs_before_triage(
    data_dir, monkeypatch, gibs_calls
):
    _event_db(EVENT, [make_candidate("evt-1", datetime(2026, 2, 22, 15, tzinfo=KST))])
    seen = {}
    _fake_agent(monkeypatch, seen)
    fake_triage = agent.triage

    def triage(cand, *, ledger, **kw):
        seen["fetched_before_triage"] = list(gibs_calls)
        return fake_triage(cand, ledger=ledger, **kw)

    monkeypatch.setattr(agent, "triage", triage)
    cli.main(["investigate", EVENT])
    images = data_dir / "events" / EVENT / "images"
    bbox = get_event(EVENT).focus_bbox
    # a 15:00 event sees its own day's two passes, newest first (tools.image_passes)
    assert seen["fetched_before_triage"] == [
        {"bbox": bbox, "day": date(2026, 2, 22), "layer": NOAA20, "out_dir": images},
        {"bbox": bbox, "day": date(2026, 2, 22), "layer": TERRA, "out_dir": images},
    ]
    assert seen["investigate"][1]["image_dir"] == images


# --- _ensure_images ---------------------------------------------------------------------------

# a 10:00 event can see the 22nd's Terra pass, then the 21st's NOAA-20 and Terra passes
MORNING = datetime(2026, 2, 22, 10, tzinfo=KST)


def _cached(img_dir, day, layer=NOAA20):
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{day}_{layer}_125.5_36.5_128_38.3.jpg").touch()


def test_ensure_images_fetches_the_two_newest_missing_passes(tmp_path, gibs_calls):
    images = tmp_path / "images"  # does not exist yet
    cli._ensure_images(EVENT, make_candidate(ts=MORNING), images)
    bbox = get_event(EVENT).focus_bbox
    assert gibs_calls == [  # not the 21st's Terra pass, the third newest
        {"bbox": bbox, "day": date(2026, 2, 22), "layer": TERRA, "out_dir": images},
        {"bbox": bbox, "day": date(2026, 2, 21), "layer": NOAA20, "out_dir": images},
    ]


def test_ensure_images_skips_a_pass_that_is_already_cached(tmp_path, gibs_calls):
    _cached(tmp_path, "2026-02-22", layer=TERRA)
    # another layer's file, or another day's, is not the wanted image
    _cached(tmp_path, "2026-02-21", layer=TERRA)
    _cached(tmp_path, "2026-02-22")
    cli._ensure_images(EVENT, make_candidate(ts=MORNING), tmp_path)
    assert [(c["day"], c["layer"]) for c in gibs_calls] == [(date(2026, 2, 21), NOAA20)]
    _cached(tmp_path, "2026-02-21")
    gibs_calls.clear()
    cli._ensure_images(EVENT, make_candidate(ts=MORNING), tmp_path)
    assert gibs_calls == []


def test_ensure_images_warns_and_goes_on_when_gibs_fails(tmp_path, monkeypatch, capsys):
    def down(bbox, day, layer, *, out_dir, **_kw):
        raise http.HttpError("gibs.wms: failed after 4 attempts (HTTP 503)")

    monkeypatch.setattr(gibs, "fetch_image", down)
    cli._ensure_images(EVENT, make_candidate(ts=MORNING), tmp_path)  # does not raise
    err = capsys.readouterr().err
    assert err.count("warning: gibs.wms: failed after 4 attempts (HTTP 503)") == 2  # per pass


def test_ensure_images_only_swallows_http_errors(tmp_path, monkeypatch):
    def broken(bbox, day, layer, *, out_dir, **_kw):
        raise ValueError("bad bbox")

    monkeypatch.setattr(gibs, "fetch_image", broken)
    with pytest.raises(ValueError, match="bad bbox"):
        cli._ensure_images(EVENT, make_candidate(ts=MORNING), tmp_path)


def test_sam_trial_writes_overlays_and_summary(data_dir, image_dir, monkeypatch, capsys):
    monkeypatch.setenv("MODEL_API_KEY", "FAKE-META-KEY-9f8e7d")
    target = data_dir / "events" / "dust-2026-02" / "images"
    target.parent.mkdir(parents=True)
    image_dir.rename(target)
    calls = []
    real_post = http.post_json

    def post(*a, **kw):
        calls.append(a[1]["input"][0]["content"][0]["text"])
        kw["transport"] = httpx.MockTransport(
            lambda r: httpx.Response(200, json={"output_text": sam_text()}))
        return real_post(*a, **kw)

    monkeypatch.setattr(http, "post_json", post)
    assert cli.main(["sam-trial", "dust-2026-02", "--prompt", "smoke", "--prompt", "haze"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert calls == ["smoke", "haze"]  # only the non-blank image (02-22) is sent
    assert summary["images_tried"] == ["2026-02-22"] and summary["cost_usd"] == pytest.approx(0.005)
    assert [r["prompt"] for r in summary["results"]] == ["smoke", "haze"]
    out = data_dir / "events" / "dust-2026-02" / "sam"
    assert (out / "summary.json").exists() and (out / f"2026-02-22_{NOAA20}_smoke.png").exists()
    assert sam.OP == "meta.sam"


def test_sam_trial_names_overlays_by_layer_when_a_day_has_two_images(
    data_dir, image_dir, monkeypatch, capsys
):
    monkeypatch.setenv("MODEL_API_KEY", "FAKE-META-KEY-9f8e7d")
    write_image(image_dir, "2026-02-22", TERRA)  # the morning image next to the afternoon one
    target = data_dir / "events" / "dust-2026-02" / "images"
    target.parent.mkdir(parents=True)
    image_dir.rename(target)
    real_post = http.post_json

    def post(*a, **kw):
        kw["transport"] = httpx.MockTransport(
            lambda r: httpx.Response(200, json={"output_text": sam_text()}))
        return real_post(*a, **kw)

    monkeypatch.setattr(http, "post_json", post)
    assert cli.main(["sam-trial", "dust-2026-02", "--prompt", "smoke"]) == 0
    results = json.loads(capsys.readouterr().out)["results"]
    assert sorted(r["layer"] for r in results) == sorted([NOAA20, TERRA])
    assert len({r["overlay"] for r in results}) == 2  # neither overwrites the other
    out = data_dir / "events" / "dust-2026-02" / "sam"
    assert all((out / r["overlay"]).exists() for r in results)
