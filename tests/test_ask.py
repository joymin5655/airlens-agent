import json
from datetime import date, datetime, timedelta

import httpx
import pytest

from airlens_agent import ask, cli, config, http, llm, quota, store
from airlens_agent.config import KST
from airlens_agent.models import Forecast, Observation, Station
from airlens_agent.sources import airkorea

NOW = datetime(2026, 11, 4, 10, 30, tzinfo=KST)
LIVE_PLACE = {"name": "Seoul", "latitude": 37.566, "longitude": 126.978, "country": "South Korea",
              "country_code": "KR", "admin1": "Seoul", "timezone": "Asia/Seoul"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(http, "_sleep", lambda _s: None)


# --- a scripted model and a mocked network ------------------------------------------------------


def tc(cid, name, args):
    text = json.dumps(args)
    return {"id": cid, "name": name, "arguments": args, "raw_arguments": text,
            "_raw": {"id": cid, "type": "function", "function": {"name": name, "arguments": text}}}


class FakeModel:
    """Scripted llm.chat / llm.json_chat; every call is charged to the ledger like the real ones."""

    def __init__(self, monkeypatch, steps, draft=None):
        self.steps, self.draft = list(steps), draft
        self.chats: list[list[dict]] = []
        monkeypatch.setattr(llm, "chat", self.chat)
        monkeypatch.setattr(llm, "json_chat", self.json_chat)

    def result(self, ledger, content="", calls=(), model="muse-fake"):
        quota.charge(ledger, "meta", 0.001)
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = [c["_raw"] for c in calls]
        public = [{k: v for k, v in c.items() if k != "_raw"} for c in calls]
        return llm.ChatResult(content, "", public, message, {}, 0.001, model, "stop", "meta")

    def chat(self, provider, messages, *, ledger, **kw):
        assert provider == "meta"
        self.chats.append([dict(m) for m in messages])
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return self.result(ledger, *step)

    def json_chat(self, provider, messages, model_cls, *, ledger, **kw):
        self.chats.append([dict(m) for m in messages])
        assert model_cls is ask.AnswerDraft
        return self.draft, self.result(ledger, self.draft.model_dump_json())


def draft(answer, *, ids=("E3",), risk="moderate", in_scope=True, confidence=0.8, limits=()):
    return ask.AnswerDraft(
        in_scope=in_scope, answer=answer, risk_level=risk, time_window="Nov 5, 06:00-12:00 KST",
        confidence=confidence, evidence_ids=list(ids), limitations=list(limits),
    )


def cams_json():
    times, pm10, pm25, ozone = [], [], [], []
    start = datetime(2026, 11, 4)
    for h in range(7 * 24):
        t = start + timedelta(hours=h)
        times.append(t.strftime("%Y-%m-%dT%H:%M"))
        morning = t.date() == date(2026, 11, 5) and 6 <= t.hour < 12
        pm10.append(40.0 if morning else 100.0)
        pm25.append(20.0 if morning else 60.0)
        ozone.append(55.0)
    return {"utc_offset_seconds": 32400, "hourly": {
        "time": times, "pm10": pm10, "pm2_5": pm25, "ozone": ozone,
        "dust": [10.0] * len(times), "aerosol_optical_depth": [0.2] * len(times)}}


def network(seen=None, places=None):
    places = {"Seoul": [LIVE_PLACE]} if places is None else places

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request.url.host)
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": places.get(request.url.params["name"], [])})
        if request.url.host == "air-quality-api.open-meteo.com":
            return httpx.Response(200, json=cams_json())
        raise AssertionError(f"unexpected request to {request.url}")

    return httpx.MockTransport(handler)


@pytest.fixture
def live(conn):
    """Live DB with two Seoul stations and a Busan one, read at 10:00 KST."""
    store.upsert_stations(conn, [
        Station("서울|중구", "중구", "서울", 37.564, 126.975, None, "t"),
        Station("서울|종로구", "종로구", "서울", 37.572, 126.980, None, "t"),
        Station("부산|광복동", "광복동", "부산", 35.098, 129.033, None, "t"),
    ])
    ts = datetime(2026, 11, 4, 10, tzinfo=KST)
    store.upsert_observations(conn, [
        Observation("서울|중구", ts, pm10=60.0, pm25=30.0, o3=0.031, no2=0.02),
        Observation("서울|종로구", ts, pm10=70.0, pm25=34.0, o3=0.030, no2=0.021),
        Observation("부산|광복동", ts, pm10=20.0, pm25=9.0),
    ])
    return conn


# --- tool loop, citations, risk ----------------------------------------------------------------


def test_tool_loop_builds_a_verified_answer(live, monkeypatch):
    model = FakeModel(monkeypatch, [
        ("", [tc("a", "geocode", {"place": "Seoul"})]),
        ("", [tc("b", "cams_forecast", {"lat": 37.57, "lon": 126.98, "hours_ahead": 20})]),
        ("ready",),
    ], draft(
        "Tomorrow morning in Seoul the CAMS model forecasts PM10 near 40 µg/m³ and PM2.5 "
        "near 20 µg/m³. That is the moderate grade.",
        ids=["E3"],
    ))
    out = ask.answer("Is the air OK for a run in Seoul tomorrow morning?", ledger=live,
                     live_conn=live, now=NOW, transport=network())
    assert out.risk_level == "moderate"
    assert out.cached is False and out.models == ["muse-fake"]
    assert out.cost_usd == pytest.approx(0.004)  # 3 chats + the final answer
    assert "40" in out.answer and "20" in out.answer
    pm10 = next(c for c in out.citations if "PM10" in c.detail)
    assert pm10.value == 40.0 and pm10.unit == "µg/m³"
    assert "2026-11-05 06:00" in pm10.time
    assert ask.SAFETY_NOTE in out.limitations
    # the tool results went back to the model, in the window the tool computed
    sent = json.dumps(model.chats[2])
    assert "Seoul" in sent and "2026-11-05 06:00 to 2026-11-05 12:00" in sent
    # the first message carries the safety scope and the current time
    assert "never health or medical advice" in model.chats[0][0]["content"]
    assert "2026-11-04 10:30 KST" in model.chats[0][1]["content"]


def test_loop_stops_after_five_steps(live, monkeypatch):
    endless = [("", [tc(f"c{i}", "active_events", {})]) for i in range(10)]
    model = FakeModel(monkeypatch, endless, draft("No live alert is published.", ids=[]))
    ask.answer("Any alert?", ledger=live, live_conn=live, now=NOW)
    assert len(model.chats) == ask.MAX_STEPS + 1  # five tool steps and the final answer
    assert "Step limit reached" in model.chats[-1][-2]["content"]


def test_unverified_sentence_is_dropped_and_flagged(live, monkeypatch):
    FakeModel(monkeypatch, [
        ("", [tc("b", "cams_forecast", {"lat": 37.57, "lon": 126.98, "hours_ahead": 20})]),
        ("ready",),
    ], draft(
        "PM10 is forecast near 40 µg/m³. PM10 will reach 400 µg/m³ by noon.",
        ids=["E2"],
    ))
    out = ask.answer("Seoul air tomorrow?", ledger=live, live_conn=live, now=NOW,
                     transport=network())
    assert "400" not in out.answer and "40" in out.answer
    assert any("removed" in x for x in out.limitations)
    assert out.confidence <= 0.5


def test_nothing_verified_gives_a_plain_notice(live, monkeypatch):
    FakeModel(monkeypatch, [("ready",)], draft("PM10 is 999 µg/m³.", ids=[]))
    out = ask.answer("Seoul air?", ledger=live, live_conn=live, now=NOW)
    assert out.answer == ask.NOTHING_VERIFIED
    assert out.risk_level == "unknown"


def test_medical_question_stays_informational(live, monkeypatch):
    FakeModel(monkeypatch, [
        ("", [tc("l", "live_readings", {"lat": 37.57, "lon": 126.98})]),
        ("ready",),
    ], draft(
        "You should skip your run today. The nearest stations read about 65 µg/m³ PM10, "
        "which is the moderate grade (30 to 80). Please follow the official guidance of local "
        "health authorities.",
        ids=["E1", "E2"],
    ))
    out = ask.answer("Should I take my asthma medicine before a run in Seoul?", ledger=live,
                     live_conn=live, now=NOW)
    assert "should" not in out.answer.lower()
    assert "moderate grade" in out.answer and "official guidance" in out.answer
    assert out.risk_level == "moderate"
    assert out.limitations[-1] == ask.SAFETY_NOTE
    assert any(c.source == ask.SOURCE_BANDS for c in out.citations)  # the grade definitions


def test_unknown_place_answers_with_unknown_risk(live, monkeypatch):
    model = FakeModel(monkeypatch, [
        ("", [tc("g", "geocode", {"place": "Qwertyville"})]),
        ("ready",),
    ], draft("I could not find a place called Qwertyville.", ids=["E2"], risk="good",
             confidence=0.9))
    out = ask.answer("Air in Qwertyville?", ledger=live, live_conn=live, now=NOW,
                     transport=network())
    assert out.risk_level == "unknown" and out.confidence <= 0.3
    assert out.citations == []
    assert "no place with this name" in json.dumps(model.chats[2])


def test_off_topic_is_refused_briefly(live, monkeypatch):
    FakeModel(monkeypatch, [("I only cover air quality.",)],
              draft("Here is a pancake recipe.", ids=[], in_scope=False))
    out = ask.answer("How do I make pancakes?", ledger=live, live_conn=live, now=NOW)
    assert out.answer == ask.REFUSAL and out.citations == [] and out.risk_level == "unknown"


def test_bad_tool_arguments_go_back_as_errors(live):
    ctx = ask.AskContext(live, NOW)
    assert ask.call_tool(ctx, "cams_forecast", {"lat": 999, "lon": 0}).startswith("error:")
    assert ask.call_tool(ctx, "nope", {}).startswith("error: unknown tool")
    assert ask.call_tool(ctx, "geocode", None, "{bad").startswith("error: arguments")
    assert ctx.evidence.items == []


# --- caps and failures --------------------------------------------------------------------------


def test_question_length_and_empty(live, monkeypatch):
    FakeModel(monkeypatch, [])  # any model call would fail on the empty script
    long = ask.answer("a" * 301, ledger=live, live_conn=live, now=NOW)
    assert "300" in long.answer and not long.cached
    assert ask.answer("   ", ledger=live, live_conn=live, now=NOW).cost_usd == 0
    assert quota.usage(live)[ask.OP]["used"] == 0


def test_cap_reached_returns_cached_example(live, monkeypatch, tmp_path):
    monkeypatch.setitem(config.QUOTA_CAPS, ask.OP, 1)
    saved = ask.Answer(answer="Seoul was moderate in the example.", risk_level="moderate",
                       confidence=0.7, limitations=["model forecast"])
    path = ask.examples_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"examples": [
        {"place": "Seoul", "question": ask.EXAMPLE_QUESTIONS[0], "answer": saved.model_dump()},
        {"place": "Busan", "question": ask.EXAMPLE_QUESTIONS[1],
         "answer": ask.Answer(answer="Busan example.").model_dump()},
    ]}))
    FakeModel(monkeypatch, [("ready",)], draft("PM10 is unknown.", ids=[]))
    first = ask.answer("How is Seoul air?", ledger=live, live_conn=live, now=NOW)
    assert not first.cached
    second = ask.answer("Is the air OK for a run in Seoul tomorrow?", ledger=live,
                        live_conn=live, now=NOW)
    assert second.cached and second.answer == saved.answer and second.cost_usd == 0
    assert "daily limit" in second.limitations[-1]
    other = ask.answer("Air in Lima?", ledger=live, live_conn=live, now=NOW)
    assert other.cached and "limit" in other.answer and other.risk_level == "unknown"


def test_model_failure_and_budget_fall_back_to_cached(live, monkeypatch):
    FakeModel(monkeypatch, [http.HttpError("meta.chat: HTTP 500")])
    out = ask.answer("Air in Seoul?", ledger=live, live_conn=live, now=NOW)
    assert out.cached and "assistant unavailable" in out.limitations[0]
    FakeModel(monkeypatch, [quota.BudgetExceeded("meta: budget $40.00 reached")])
    out = ask.answer("Air in Seoul?", ledger=live, live_conn=live, now=NOW)
    assert out.cached and "budget" in out.limitations[0]


# --- tools ----------------------------------------------------------------------------------------


def test_live_readings_medians_and_staleness(live):
    ctx = ask.AskContext(live, NOW)
    _, data, cites = ask.tool_live_readings(ctx, {"lat": 37.57, "lon": 126.98, "radius_km": 10})
    assert [s["station"] for s in data["stations"]] == ["종로구", "중구"]  # nearest first
    assert data["median_pm10"] == 65.0 and data["median_pm25"] == 32.0
    assert data["latest_time"] == "2026-11-04 10:00" and "note" not in data
    assert {c["detail"].split(",")[0] for c in cites} == {"PM10", "PM2.5"}
    late = ask.AskContext(live, NOW + timedelta(hours=8))
    assert "hours old" in ask.tool_live_readings(late, {"lat": 37.57, "lon": 126.98})[1]["note"]
    _, far, none = ask.tool_live_readings(ctx, {"lat": 35.68, "lon": 139.7})  # Tokyo
    assert far["stations"] == [] and none == [] and "Korea only" in far["note"]


def test_official_forecast_is_cached_per_day(live, monkeypatch):
    calls = []

    def fake(conn, day):
        calls.append(day)
        issued = datetime(2026, 11, 4, 5, tzinfo=KST)
        return [
            Forecast(issued_at=issued, target_date=date(2026, 11, 5), pollutant="PM10",
                     overall="Normal nationwide", cause="stagnant air",
                     grades={"서울": "보통", "제주": "좋음"}),
            Forecast(issued_at=issued, target_date=date(2026, 11, 5), pollutant="PM25",
                     overall="Bad in the west", cause="", grades={"서울": "나쁨"}),
        ]

    monkeypatch.setattr(airkorea, "fetch_forecast", fake)
    ctx = ask.AskContext(live, NOW)
    _, data, cites = ask.tool_official_forecast(ctx, {"day": "tomorrow"})
    assert data["day"] == "2026-11-05" and data["pm10"]["grades"]["서울"] == "보통"
    assert len(cites) == 2 and cites[0]["time"] == "2026-11-04 05:00"
    _, today, _ = ask.tool_official_forecast(ctx, {"day": "today"})
    assert "note" in today  # the bulletin has nothing for today in this fixture
    assert calls == [date(2026, 11, 4)]  # one fetch served both questions


def test_tools_write_only_to_the_cache_connection(live, monkeypatch, tmp_path):
    """On Modal the web container reads a read-only copy of the live DB; caches go to the ledger."""
    import sqlite3

    path = live.execute("PRAGMA database_list").fetchone()[2]
    read_only = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    ledger = store.connect(tmp_path / "ledger.sqlite")
    used = []

    def fake(conn, day):
        used.append(conn)
        issued = datetime(2026, 11, 4, 5, tzinfo=KST)
        return [Forecast(issued_at=issued, target_date=date(2026, 11, 5), pollutant="PM10",
                         overall="Normal", cause="", grades={"서울": "보통"})]

    monkeypatch.setattr(airkorea, "fetch_forecast", fake)
    ctx = ask.AskContext(read_only, NOW, cache=ledger)
    _, data, _ = ask.tool_official_forecast(ctx, {"day": "tomorrow"})
    _, readings, _ = ask.tool_live_readings(ctx, {"lat": 37.566, "lon": 126.978, "radius_km": 10})
    assert data["pm10"]["overall"] == "Normal" and used == [ledger]
    assert readings  # read through the read-only connection
    tables = {r[0] for r in ledger.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "ask_forecast_cache" in tables


def test_active_events_tolerates_absence(live, tmp_path):
    ctx = ask.AskContext(live, NOW)
    assert ask.tool_active_events(ctx, {})[1]["available"] is False
    path = config.data_dir() / "live" / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"updated_at": "2026-11-04T10:00", "candidates": list(range(9))}))
    _, data, cites = ask.tool_active_events(ctx, {})
    assert data["available"] and data["candidates"] == [0, 1, 2, 3, 4] and len(cites) == 1
    path.write_text("not json")
    assert ask.tool_active_events(ctx, {})[1]["available"] is False


def test_cams_window_outside_the_forecast_is_an_error(live, monkeypatch):
    monkeypatch.setattr(ask.openmeteo, "fetch_cams_window",
                        lambda *a, **k: {"utc_offset_seconds": 0, "hours": {}})
    ctx = ask.AskContext(live, NOW)
    assert ask.call_tool(ctx, "cams_forecast", {"lat": 1, "lon": 1}).startswith("error:")
    assert ctx.evidence.items == []


def test_grade_bands():
    assert [ask.grade("pm10", v) for v in (30, 81, 151)] == [0, 2, 3]
    assert [ask.grade("pm25", v) for v in (15, 76, None)] == [0, 3, None]
    assert ask._worst(20, 40) == "unhealthy"  # PM2.5 40 is above 35


# --- examples and cli ---------------------------------------------------------------------------


def test_five_example_questions():
    assert len(ask.EXAMPLE_QUESTIONS) == 5
    joined = " ".join(ask.EXAMPLE_QUESTIONS)
    for word in ("run in Seoul tomorrow morning", "Busan today", "Tokyo", "Los Angeles"):
        assert word in joined


def test_build_examples_saves_only_live_answers(live, monkeypatch):
    def fake_answer(question, **_):
        if "Tokyo" in question:
            return ask.Answer(answer="limit", cached=True)
        return ask.Answer(answer=f"A: {question}", risk_level="good")

    monkeypatch.setattr(ask, "answer", fake_answer)
    summary = ask.build_examples(ledger=live, live_conn=live)
    assert summary["saved"] == 4 and len(summary["skipped"]) == 1
    saved = json.loads(ask.examples_path().read_text())["examples"]
    assert {e["place"] for e in saved} == {"Seoul", "Busan", "Los Angeles", "Korea"}
    assert ask._cached("Air in Busan?", "x").answer.startswith("A: How is the air in Busan")


def test_cli_ask(monkeypatch, capsys, tmp_path):
    seen = {}

    def fake_answer(question, *, ledger, live_conn, lat, lon, **_):
        seen.update(question=question, lat=lat, lon=lon)
        return ask.Answer(answer="ok")

    monkeypatch.setattr(ask, "answer", fake_answer)
    monkeypatch.setattr(cli, "_live", lambda: store.connect(tmp_path / "live.sqlite"))
    assert cli.main(["ask", "Air in Seoul?", "--lat", "37.5", "--lon", "127"]) == 0
    assert seen == {"question": "Air in Seoul?", "lat": 37.5, "lon": 127.0}
    assert json.loads(capsys.readouterr().out)["answer"] == "ok"
