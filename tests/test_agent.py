import dataclasses
import json
from typing import get_args

import httpx
import pytest

from airlens_agent import agent, config, http, llm, quota, tools
from conftest import make_candidate

CALL = {"type": "function"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "FAKE-META-KEY-9f8e7d")
    monkeypatch.setenv("BASETEN_API_KEY", "FAKE-BASETEN-KEY-1a2b3c")
    monkeypatch.setattr(http, "_sleep", lambda _s: None)


def reply(content="", tool_calls=None, model="muse-fake"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "model": model,
        "choices": [{"message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
    }


def tool_call(cid, name, arguments):
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": text}}


class Fake:
    """Scripted model API: one reply queue per host; records every request body."""

    def __init__(self, meta=(), baseten=()):
        self.queues = {"api.meta.ai": list(meta), "inference.baseten.co": list(baseten)}
        self.requests: list[dict] = []
        self.transport = httpx.MockTransport(self.handler)

    def handler(self, request):
        self.requests.append({"host": request.url.host, "body": json.loads(request.content)})
        queue = self.queues[request.url.host]
        assert queue, f"unexpected extra request to {request.url.host}"
        item = queue.pop(0)
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)

    def on(self, host):
        return [r["body"] for r in self.requests if r["host"] == host]


TRIAGE = agent.Triage(
    category="dust",
    hypotheses=[agent.Hypothesis(cause="dust", likelihood=0.7, rationale="coarse")],
    checks=["station_readings"],
)


def report_json(claims, **over):
    body = {
        "headline": "Dust event", "cause": "dust", "confidence": 0.8, "summary": "s",
        "claims": claims, "alternatives": [{"cause": "traffic", "why_less_likely": "regional"}],
        "limitations": ["no lidar"],
    }
    return json.dumps({**body, **over})


# E1 is the detector summary of every run; the first tool result is E2
GOOD = {"text": "Peak PM10 was 320 ug/m3 at station A", "evidence_ids": ["E2"]}


def scripted_meta(final_claims, extra=()):
    calls = [
        tool_call("c1", "station_readings", {}),
        tool_call("c2", "bogus_tool", {}),
        tool_call("c3", "wind", {"level": "500hPa"}),
        tool_call("c4", "fire_hotspots", "{not json"),
    ]
    return [
        reply(tool_calls=calls),
        reply("I have enough."),
        reply(report_json(final_claims)),
        *extra,
    ]


def run(event_db, conn, fake, **kw):
    return agent.investigate(
        event_db, make_candidate(), TRIAGE, ledger=conn, transport=fake.transport, **kw)


def test_loop_tool_results_and_final_report(event_db, conn):
    fake = Fake(meta=scripted_meta([GOOD]))
    report = run(event_db, conn, fake)
    meta = fake.on("api.meta.ai")
    assert len(meta) == 3 and "tool_choice" not in meta[0] and meta[0]["tools"]
    assert "response_format" in meta[2] and "tools" not in meta[2]
    # step 2 resends the assistant tool calls, then one tool message per call, in order
    msgs = meta[1]["messages"]
    assistant = next(m for m in msgs if m.get("tool_calls"))
    assert [c["id"] for c in assistant["tool_calls"]] == ["c1", "c2", "c3", "c4"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2", "c3", "c4"]
    assert json.loads(tool_msgs[0]["content"])["evidence_id"] == "E2"  # E1 is the detector
    assert "unknown tool 'bogus_tool'" in tool_msgs[1]["content"]
    assert tool_msgs[2]["content"].startswith("error: level must be")
    assert "not a valid JSON object" in tool_msgs[3]["content"]
    # code attaches the extras
    assert report.cause == "dust" and report.claims == [agent.Claim(**GOOD)]
    assert [e["evidence_id"] for e in report.evidence] == ["E1", "E2"]
    assert report.evidence[0]["source"] == "AirLens anomaly detector"
    assert report.evidence[1]["tool"] == "station_readings" and report.evidence[1]["args"] == {}
    assert report.models == ["muse-fake"] and report.triage == TRIAGE
    assert report.provider_used == "meta" and report.fallback is None
    assert report.cost_usd == pytest.approx(3 * (1000 * 0.10 + 200 * 0.20) / 1e6)
    assert report.unverified_claims == [] and report.dropped_claims == []


def test_step_cap_forces_final_report(event_db, conn):
    loop = reply(tool_calls=[tool_call("c1", "wind", {})])
    fake = Fake(meta=[loop, loop, reply(report_json([]))])
    report = run(event_db, conn, fake, max_steps=2)
    assert len(fake.on("api.meta.ai")) == 3 and report.claims == []
    assert "Step limit reached" in json.dumps(fake.on("api.meta.ai")[2]["messages"])


def test_repair_round_fixes_a_failed_claim(event_db, conn):
    bad = {"text": "PM10 reached 999 ug/m3", "evidence_ids": ["E2"]}
    fixed = {"claims": [{"index": 1, "text": "PM10 reached 320 ug/m3", "evidence_ids": ["E2"]}]}
    fake = Fake(meta=scripted_meta([GOOD, bad], extra=[reply(json.dumps(fixed))]))
    report = run(event_db, conn, fake)
    assert [c.text for c in report.claims] == [GOOD["text"], "PM10 reached 320 ug/m3"]
    assert report.unverified_claims == [] and report.dropped_claims == []
    repair_prompt = fake.on("api.meta.ai")[3]["messages"][-1]["content"]
    assert '1: "PM10 reached 999' in repair_prompt and '\n0: ' not in repair_prompt


def test_still_failing_claims_are_flagged_not_removed(event_db, conn):
    bad = {"text": "PM10 reached 999 ug/m3", "evidence_ids": ["E2"]}
    nowhere = {"text": "Smoke was seen", "evidence_ids": ["E42"]}
    still = {"claims": [{"index": 1, "text": "PM10 reached 998 ug/m3", "evidence_ids": ["E2"]}]}
    fake = Fake(meta=scripted_meta([GOOD, bad, nowhere], extra=[reply(json.dumps(still))]))
    report = run(event_db, conn, fake)
    assert [c.text for c in report.claims] == [GOOD["text"], "PM10 reached 998 ug/m3"]
    assert report.unverified_claims == [1]
    assert report.dropped_claims == ["Smoke was seen"]  # the model dropped it; recorded, not silent


REPAIR_BAD = {"text": "PM10 reached 999 ug/m3", "evidence_ids": ["E2"]}
CALL_COST = (1000 * 0.10 + 200 * 0.20) / 1e6  # one scripted meta reply


@pytest.mark.parametrize(("extra", "paid", "why"), [
    ([reply("not json"), reply("still not json")], 5, "ValidationError"),  # json_chat's retry too
    ([httpx.Response(500, text="upstream exploded")], 3, "HTTP 500"),  # a failed call is not billed
])
def test_a_failed_repair_round_keeps_the_draft_and_flags_its_claims(
    event_db, conn, extra, paid, why
):
    fake = Fake(meta=scripted_meta([GOOD, REPAIR_BAD], extra=extra))
    report = run(event_db, conn, fake)  # does not raise
    assert [c.text for c in report.claims] == [GOOD["text"], REPAIR_BAD["text"]]  # draft kept
    assert report.unverified_claims == [1] and report.dropped_claims == []
    assert report.limitations[0] == "no lidar"  # the model's own limitations stay first
    assert len(report.limitations) == 2 and "claim repair round failed" in report.limitations[1]
    assert why in report.limitations[1] and len(report.limitations[1]) < 400
    assert report.cost_usd == pytest.approx(paid * CALL_COST)  # what was paid for is still counted
    assert report.models == ["muse-fake"] and report.provider_used == "meta"


def test_a_repair_round_that_reaches_no_provider_keeps_the_draft(event_db, conn, monkeypatch):
    monkeypatch.setitem(config.SPEND_CAPS, "meta", 0.0004)  # spent after the third call
    monkeypatch.delenv("BASETEN_API_KEY")
    fake = Fake(meta=scripted_meta([GOOD, REPAIR_BAD]))
    report = run(event_db, conn, fake)
    assert report.unverified_claims == [1] and len(report.claims) == 2
    assert "MissingKeyError" in report.limitations[-1]
    assert "BASETEN_API_KEY" in report.limitations[-1]  # the variable name, never a value
    assert "meta -> baseten" in report.fallback and "budget" in report.fallback
    assert report.provider_used == "meta"  # baseten never answered


@pytest.mark.parametrize(("failure", "why"), [
    ([reply("{not json"), reply("{still not json")], "ValidationError"),
    ([httpx.Response(500, text="bad key FAKE-META-KEY-9f8e7d")], "HTTP 500"),
])
def test_a_failed_final_report_returns_a_minimal_report_with_the_evidence(
    event_db, conn, failure, why
):
    loop = [reply(tool_calls=[tool_call("c1", "station_readings", {})]), reply("done")]
    fake = Fake(meta=[*loop, *failure])
    report = run(event_db, conn, fake)  # does not raise
    assert (report.cause, report.confidence, report.claims) == ("unknown", 0.0, [])
    assert "could not be generated" in report.headline
    assert why in report.summary and "evidence collected" in report.summary
    assert len(report.limitations) == 1 and why in report.limitations[0]
    assert [e["evidence_id"] for e in report.evidence] == ["E1", "E2"]  # collected work is kept
    assert report.triage == TRIAGE and report.models == ["muse-fake"]
    assert report.provider_used == "meta" and report.fallback is None
    assert report.cost_usd > 0 and report.unverified_claims == [] and report.dropped_claims == []
    assert "FAKE-META-KEY-9f8e7d" not in json.dumps(report.model_dump(mode="json"))  # CLI writes it


def test_a_failed_model_call_in_the_tool_loop_keeps_the_evidence_and_still_reports(event_db, conn):
    fake = Fake(meta=[
        reply(tool_calls=[tool_call("c1", "station_readings", {})]),
        httpx.Response(500, text="upstream exploded"),  # step 2 fails
        reply(report_json([GOOD])),
    ])
    report = run(event_db, conn, fake)  # does not raise
    assert [e["evidence_id"] for e in report.evidence] == ["E1", "E2"]  # step 1's work is kept
    assert report.claims == [agent.Claim(**GOOD)] and report.unverified_claims == []
    assert report.limitations[0] == "no lidar"  # the model's own limitations stay first
    assert len(report.limitations) == 2
    assert report.limitations[1].startswith("The tool loop stopped early: HttpError: ")
    assert "HTTP 500" in report.limitations[1]
    meta = fake.on("api.meta.ai")
    assert len(meta) == 3 and "response_format" in meta[2]  # the third request is the report
    final = meta[2]["messages"]
    assert [m["role"] for m in final].count("tool") == 1  # the model is shown step 1's result
    assert "Step limit reached" not in json.dumps(final)  # the cap was not what ended the loop
    assert report.cost_usd == pytest.approx(2 * CALL_COST)  # the failed call was not billed
    assert report.models == ["muse-fake"] and report.provider_used == "meta"


def test_a_failed_tool_loop_and_a_failed_final_report_leave_a_minimal_report_with_both_reasons(
    event_db, conn
):
    fake = Fake(meta=[
        reply(tool_calls=[tool_call("c1", "station_readings", {})]),
        httpx.Response(500, text="loop exploded"),
        httpx.Response(500, text="final exploded"),
    ])
    report = run(event_db, conn, fake)  # does not raise
    assert (report.cause, report.confidence, report.claims) == ("unknown", 0.0, [])
    assert "could not be generated" in report.headline
    assert [e["evidence_id"] for e in report.evidence] == ["E1", "E2"]  # collected work is kept
    assert len(report.limitations) == 2
    assert report.limitations[0].startswith("The final report failed: HttpError: ")
    assert "final exploded" in report.limitations[0]
    assert report.limitations[1].startswith("The tool loop stopped early: HttpError: ")
    assert "loop exploded" in report.limitations[1]
    assert report.models == ["muse-fake"] and report.cost_usd == pytest.approx(CALL_COST)


def test_the_detector_summary_is_evidence_e1_and_claims_can_cite_it(event_db, conn):
    claim = {"text": "Peak PM10 was 300 ug/m3 with a PM2.5/PM10 ratio of 0.30 over 12 km, and PM10 "
             "rose 150 ug/m3 in 3 h", "evidence_ids": ["E1"]}
    fake = Fake(meta=[reply("done"), reply(report_json([claim]))])
    report = run(event_db, conn, fake)
    e1 = report.evidence[0]
    assert (e1["evidence_id"], e1["tool"]) == ("E1", "detector")
    assert e1["source"] == agent.DETECTOR_SOURCE == "AirLens anomaly detector"
    assert e1["data"]["peak_pm10"] == 300.0 and e1["data"]["extent_km"] == 12.0
    assert report.unverified_claims == [] and len(fake.on("api.meta.ai")) == 2  # no repair needed
    prompt = fake.on("api.meta.ai")[0]["messages"][1]["content"]
    assert "evidence E1" in prompt and "E2, E3" in prompt
    for request in fake.on("api.meta.ai"):  # the system message and the final request agree
        assert "E1" in request["messages"][0]["content"]
    assert "E1" in fake.on("api.meta.ai")[1]["messages"][-1]["content"]


def test_every_number_the_model_is_shown_for_the_candidate_is_citable_as_e1():
    cand = make_candidate()
    ev = tools.Evidence()
    ev.add("detector", {}, agent.DETECTOR_SOURCE, agent.detector_summary(cand))
    first, *rest = agent.describe(cand).splitlines()  # the first line is the event id and time
    assert "evt-1" in first
    draft = agent.ReportDraft(
        headline="h", cause="dust", confidence=0.5, summary="s", alternatives=[], limitations=[],
        claims=[agent.Claim(text=" ".join(rest), evidence_ids=["E1"])])
    check = agent.verify(draft, ev).checks[0]
    assert check.ok, check.problems


def test_the_detector_summary_copes_with_missing_features():
    bare = dataclasses.replace(
        make_candidate(), features={}, peak_pm10=None, peak_pm25=None, ratio_pm25_pm10=None)
    data = agent.detector_summary(bare)
    assert data["peak_pm10"] is None and data["ratio_pm25_pm10"] is None
    assert data["extent_km"] is None and data["hotspots_50km_24h"] is None
    assert data["provinces"] == [] and data["n_stations"] == 2


def test_missing_tool_call_ids_get_the_same_synthesized_id_in_both_messages(
    event_db, conn, monkeypatch
):
    nameless = [
        {"type": "function", "function": {"name": "station_readings", "arguments": "{}"}},  # no id
        {"id": "", "type": "function", "function": {"name": "wind", "arguments": "{}"}},
    ]
    fake = Fake(meta=[reply(tool_calls=nameless), reply("done"), reply(report_json([]))])
    answers = []
    real_chat = agent.llm.chat

    def spy(provider, messages, **kw):
        answers.append(real_chat(provider, messages, **kw))
        return answers[-1]

    monkeypatch.setattr(agent.llm, "chat", spy)
    run(event_db, conn, fake)
    msgs = fake.on("api.meta.ai")[1]["messages"]
    assistant = next(m for m in msgs if m.get("tool_calls"))
    assert [c["id"] for c in assistant["tool_calls"]] == ["call_0", "call_1"]
    assert [m["tool_call_id"] for m in msgs if m["role"] == "tool"] == ["call_0", "call_1"]
    assert assistant["tool_calls"][0]["function"]["name"] == "station_readings"  # the rest intact
    raw = answers[0].message["tool_calls"]  # what the model sent was copied, not edited in place
    assert "id" not in raw[0] and raw[1]["id"] == ""


@pytest.mark.parametrize("empty", ["", "  \n"])
def test_an_empty_reply_is_not_resent_as_an_empty_assistant_message(event_db, conn, empty):
    fake = Fake(meta=[reply(empty), reply(report_json([]))])  # e.g. reasoning used every token
    report = run(event_db, conn, fake)
    final = fake.on("api.meta.ai")[1]["messages"]
    assert not any(m["role"] == "assistant" for m in final)
    assert final[-1] == {"role": "user", "content": agent.FINAL_REQUEST}
    assert report.cause == "dust"
    # a reply with text is still resent
    fake = Fake(meta=[reply("I have enough."), reply(report_json([]))])
    run(event_db, conn, fake)
    resent = fake.on("api.meta.ai")[1]["messages"]
    assert {"role": "assistant", "content": "I have enough."} in resent


def test_fallback_to_baseten_when_meta_key_missing(event_db, conn, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY")
    fake = Fake(baseten=[
        reply(tool_calls=[tool_call("c1", "station_readings", {})], model="open-fake"),
        reply("done", model="open-fake"),
        reply(report_json([GOOD]), model="open-fake"),
    ])
    report = run(event_db, conn, fake)
    assert fake.on("api.meta.ai") == [] and len(fake.on("inference.baseten.co")) == 3
    assert report.provider_used == "baseten" and report.investigator == "meta"
    assert "meta -> baseten" in report.fallback and "MODEL_API_KEY" in report.fallback
    assert report.models == ["open-fake"]
    assert quota.spent(conn, "meta") == 0 and quota.spent(conn, "baseten") > 0


def test_fallback_to_baseten_when_meta_budget_exhausted(event_db, conn, monkeypatch):
    monkeypatch.setitem(config.SPEND_CAPS, "meta", 1.0)
    quota.charge(conn, "meta", 1.0)
    fake = Fake(baseten=[reply("done"), reply(report_json([]))])
    report = run(event_db, conn, fake)
    assert report.provider_used == "baseten" and "budget" in report.fallback
    assert fake.on("api.meta.ai") == []


def test_no_fallback_when_both_are_unavailable(event_db, conn, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY")
    monkeypatch.delenv("BASETEN_API_KEY")
    fake = Fake()
    report = run(event_db, conn, fake)  # no traceback: a minimal report that says why
    assert fake.requests == [] and report.cost_usd == 0 and report.models == []
    assert report.cause == "unknown" and [e["evidence_id"] for e in report.evidence] == ["E1"]
    assert report.provider_used == "" and "meta -> baseten" in report.fallback
    assert len(report.limitations) == 2
    assert "The final report failed: MissingKeyError" in report.limitations[0]
    assert "The tool loop stopped early: MissingKeyError" in report.limitations[1]
    assert "BASETEN_API_KEY" in report.limitations[1]  # the variable name, never a value


def test_triage_uses_candidate_features_and_validates(conn):
    body = {"category": "dust", "hypotheses": [
        {"cause": "dust", "likelihood": 0.8, "rationale": "r"}], "checks": ["wind"]}
    fake = Fake(baseten=[reply(json.dumps(body), model="open-fake")])
    out = agent.triage(make_candidate(), ledger=conn, transport=fake.transport)
    assert out.category == "dust" and out.checks == ["wind"] and out.model_used == "open-fake"
    assert out.fallback is None
    req = fake.on("inference.baseten.co")[0]
    assert req["reasoning_effort"] == "medium" and req["max_tokens"] == 8192
    prompt = req["messages"][1]["content"]
    assert "0.30" in prompt and "co" in prompt.lower() and "Seoul" not in prompt
    assert "서울" in prompt and "300" in prompt
    schema = req["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["checks"]["items"]["enum"] == list(tools.TOOL_NAMES)
    assert "$ref" not in json.dumps(schema)


def triage_json(*causes):
    return json.dumps({"category": "dust", "checks": ["wind"], "hypotheses": [
        {"cause": cause, "likelihood": 0.5, "rationale": "r"} for cause in causes]})


def test_triage_retries_once_when_two_hypotheses_share_a_cause(conn):
    dup = triage_json("traffic", "traffic")  # Muse Spark Open once did this for a dust event
    fake = Fake(baseten=[reply(dup, model="open-fake"),
                         reply(triage_json("dust", "traffic"), model="open-fake")])
    out = agent.triage(make_candidate(), ledger=conn, transport=fake.transport)
    assert [h.cause for h in out.hypotheses] == ["dust", "traffic"]
    assert out.model_used == "open-fake"
    requests = fake.on("inference.baseten.co")
    assert len(requests) == 2
    # the retry shows the model its own answer and why it was rejected
    assert requests[1]["messages"][-2] == {"role": "assistant", "content": dup}
    assert requests[1]["messages"][-1]["role"] == "user"
    assert "each hypothesis must name a different cause" in requests[1]["messages"][-1]["content"]
    assert quota.spent(conn, "baseten") == pytest.approx(2 * (1000 * 1.25 + 200 * 4.25) / 1e6)


def test_triage_is_neutral_when_the_retry_also_repeats_a_cause(conn):
    dup = triage_json("traffic", "traffic")
    fake = Fake(baseten=[reply(dup), reply(dup)])
    out = agent.triage(make_candidate(), ledger=conn, transport=fake.transport)
    assert len(fake.on("inference.baseten.co")) == 2  # one retry, never a third try
    assert (out.category, out.hypotheses) == ("unknown", [])
    assert out.checks == list(tools.TOOL_NAMES)  # no hypothesis to test: every tool is worth a look
    assert out.model_used == ""  # no model gave a usable answer
    assert "triage answer invalid" in out.fallback and "different cause" in out.fallback
    assert quota.spent(conn, "baseten") > 0  # both attempts were paid for


def test_triage_is_neutral_when_the_provider_call_fails(conn):
    fake = Fake(baseten=[httpx.Response(500, text="upstream exploded")])
    out = agent.triage(make_candidate(), ledger=conn, transport=fake.transport)  # does not raise
    assert (out.category, out.hypotheses, out.model_used) == ("unknown", [], "")
    assert out.checks == list(tools.TOOL_NAMES)
    assert "triage call failed" in out.fallback and "HTTP 500" in out.fallback
    assert "answer invalid" not in out.fallback


def test_triage_is_neutral_when_neither_provider_has_a_key(conn, monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY")
    monkeypatch.delenv("MODEL_API_KEY")
    out = agent.triage(make_candidate(), ledger=conn, transport=Fake().transport)
    assert (out.category, out.hypotheses) == ("unknown", [])
    assert out.checks == list(tools.TOOL_NAMES)
    # the switch that was tried comes first, then why the neutral triage was used
    assert out.fallback.startswith("baseten -> meta: ") and "triage call failed" in out.fallback
    assert "MODEL_API_KEY" in out.fallback  # the variable name, never a value


def test_a_plain_value_error_from_json_chat_is_an_invalid_answer_not_a_crash(
    event_db, conn, monkeypatch
):
    # json_chat re-raises whatever ValueError its second answer gave, not only ValidationError
    def invalid(*_a, **_kw):
        raise ValueError("no JSON object in the answer")

    monkeypatch.setattr(llm, "json_chat", invalid)
    out = agent.triage(make_candidate(), ledger=conn, transport=Fake().transport)
    assert (out.category, out.hypotheses) == ("unknown", [])
    assert "triage answer invalid" in out.fallback and "no JSON object" in out.fallback
    report = run(event_db, conn, Fake(meta=[reply("done")]))  # the final report fails the same way
    assert (report.cause, report.claims) == ("unknown", [])
    assert "could not be generated" in report.headline
    assert "ValueError: no JSON object" in report.limitations[0]
    assert [e["evidence_id"] for e in report.evidence] == ["E1"]


def test_a_neutral_triage_still_lets_the_investigation_run(event_db, conn):
    dup = triage_json("traffic", "traffic")
    triage_fake = Fake(baseten=[reply(dup), reply(dup)])
    neutral = agent.triage(make_candidate(), ledger=conn, transport=triage_fake.transport)
    fake = Fake(meta=[reply("done"), reply(report_json([]))])
    report = agent.investigate(
        event_db, make_candidate(), neutral, ledger=conn, transport=fake.transport)
    assert report.cause == "dust" and report.triage == neutral
    assert report.triage_fallback == neutral.fallback and "triage answer invalid" in (
        report.triage_fallback)
    prompt = fake.on("api.meta.ai")[0]["messages"][1]["content"]
    assert '"category":"unknown"' in prompt and '"hypotheses":[]' in prompt


def test_triage_records_a_provider_switch(conn, monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY")
    fake = Fake(meta=[reply(triage_json("dust"), model="meta-fake")])
    out = agent.triage(make_candidate(), ledger=conn, transport=fake.transport)
    assert out.model_used == "meta-fake" and fake.on("inference.baseten.co") == []
    assert "baseten -> meta" in out.fallback and "BASETEN_API_KEY" in out.fallback


def test_a_report_carries_the_triage_fallback_apart_from_its_own(event_db, conn):
    tri = agent.Triage(category="dust", hypotheses=[], checks=[])
    tri._fallback = "baseten -> meta: baseten: budget $40.00 reached"
    fake = Fake(meta=[reply("done"), reply(report_json([]))])
    report = agent.investigate(
        event_db, make_candidate(), tri, ledger=conn, transport=fake.transport)
    assert report.triage_fallback == "baseten -> meta: baseten: budget $40.00 reached"
    assert report.fallback is None  # the investigator's own route did not switch
    assert report.model_dump(mode="json")["triage_fallback"] == report.triage_fallback


def test_category_guide_defines_every_category_and_reaches_both_system_prompts(event_db, conn):
    for category in get_args(agent.Category):
        assert f"{category} = " in agent.CATEGORY_GUIDE
    triage_fake = Fake(baseten=[reply(triage_json("dust"))])
    agent.triage(make_candidate(), ledger=conn, transport=triage_fake.transport)
    triage_system = triage_fake.on("inference.baseten.co")[0]["messages"][0]
    assert triage_system["role"] == "system" and agent.CATEGORY_GUIDE in triage_system["content"]
    fake = Fake(meta=[reply("done"), reply(report_json([]))])
    run(event_db, conn, fake)
    for request in fake.on("api.meta.ai"):  # the tool loop and the final report share the prompt
        assert agent.CATEGORY_GUIDE in request["messages"][0]["content"]


def test_report_schema_is_flat():
    schema = json.dumps(agent.llm.flat_schema(agent.ReportDraft))
    assert "$ref" not in schema and "$defs" not in schema and "minimum" not in schema


# --- verify() ---------------------------------------------------------------------------------


def evidence():
    ev = tools.Evidence()
    ev.add("station_readings", {}, "AirKorea", {"peak_pm10": 300.0, "ratio": 0.31, "n": 2,
                                                "time": "02-22 10:00", "level": "850hPa"})
    ev.add("wind", {}, "Open-Meteo", {"mean_speed_ms": 12.0})
    ev.add("fire_hotspots", {}, "NASA FIRMS", {"total_frp_mw": 1950.0, "low": 5.0, "high": 8.0,
                                               "gust": 2.5})
    ev.add("cams_forecast", {}, "CAMS", {"extent": 2024.3})
    return ev


def verdict(text, ids=("E1",)):
    draft = agent.ReportDraft(headline="h", cause="dust", confidence=0.5, summary="s",
                              claims=[agent.Claim(text=text, evidence_ids=list(ids))],
                              alternatives=[], limitations=[])
    return agent.verify(draft, evidence()).checks[0]


@pytest.mark.parametrize("text", [
    "PM10 peaked at 300 ug/m3",
    "PM10 peaked near 310 ug/m3",  # within 5 %
    "On 2026-02-22 at 10:00 PM10 peaked at 300 ug/m3",  # dates and times are ignored
    "In 2026 PM10 peaked at 300",
    "The PM2.5/PM10 ratio was 31%",
    "The ratio was 0.3",  # rounding
    "Two stations (E1) rose",  # E1 is an id, not a number
    "Wind at the 850hPa level",
    "Peak 3,00 is not here".replace("3,00", "300"),
    "No numbers at all",
    # every spelling of a date or time is ignored
    "On Feb 22 PM10 peaked at 300 ug/m3",
    "On February 22, 2026 PM10 peaked at 300 ug/m3",
    "On 22 Feb PM10 peaked at 300 ug/m3",
    "On the 22nd of February PM10 peaked at 300 ug/m3",
    "On Sept. 5 and 5 September PM10 peaked at 300 ug/m3",
    "On May 30 PM10 peaked at 300 ug/m3",
    "At 2026-02-22T10:00:00+09:00 PM10 peaked at 300 ug/m3",
    "On 2/22 PM10 peaked at 300 ug/m3",
    "At 9:30 PM10 peaked at 300 ug/m3",
    "At 02-22 10:00 PM10 peaked at 300 ug/m3",  # the tools' own time format
])
def test_verify_passes(text):
    assert verdict(text).ok, verdict(text).problems


@pytest.mark.parametrize(("text", "ids"), [
    ("Wind blew at 5-8 m/s", ("E3",)),  # a range is checked number by number
    ("Total FRP was 1950 MW", ("E3",)),  # 1950 before a unit is a value, not a year
    ("CO reached 1950 ug/m3", ("E3",)),
    ("The extent was 2024.3 km", ("E4",)),  # nor is the integer part of a decimal
    ("In 2024, FRP was 1950 MW", ("E3",)),  # a plain year still is ignored
])
def test_verify_passes_ranges_and_year_like_values_that_are_in_the_evidence(text, ids):
    assert verdict(text, ids).ok, verdict(text, ids).problems


@pytest.mark.parametrize("unit", [
    "MW", "km", "ug/m3", "µg/m3", "μg/m3", "hPa", "ppm", "ppb",  # micro and Greek mu
])
def test_a_four_digit_number_before_a_unit_is_a_value_not_a_year(unit):
    assert verdict(f"It reached 1950 {unit}", ("E3",)).ok  # the evidence says 1950
    assert verdict(f"It reached 2090 {unit}", ("E3",)).problems == [
        "number 2090 not in cited evidence"]


def test_a_four_digit_percentage_is_a_value_too():
    assert verdict("It reached 2090%", ("E3",)).problems == ["number 2090% not in cited evidence"]


@pytest.mark.parametrize(("text", "ids", "flagged"), [
    # each of these used to pass because a date pattern swallowed the number
    ("PM10 decreased 25 ug/m3", ("E1",), ["25"]),  # "dec" + "reased" read as a month
    ("The ratio marked 40", ("E1",), ["40"]),  # "mar" + "ked"
    ("Gusts may 30 m/s", ("E2",), ["30"]),  # lower-case "may" is the verb, not the month
    ("Wind was 5-9 m/s", ("E3",), ["9"]),  # 5-8 is in the evidence, 9 is not
    ("The plume spanned 30-50 km", ("E1",), ["30", "50"]),  # a range, not "30-50" the date
    ("Total FRP was 2090 MW", ("E3",), ["2090"]),  # a value before a unit, not a year
    ("FRP reached 2090.5 MW", ("E3",), ["2090.5"]),  # not the year 2090 plus a stray ".5"
    ("The ratios were 0.31/0.4", ("E1",), ["0.4"]),  # decimals are not "31/0"
    ("The 30/50 split", ("E1",), ["30", "50"]),  # there is no month 30
])
def test_verify_checks_the_numbers_the_date_filter_used_to_swallow(text, ids, flagged):
    assert verdict(text, ids).problems == [f"number {n} not in cited evidence" for n in flagged]


def test_a_number_matches_both_roundings_of_an_evidence_value_halfway_between():
    assert verdict("Gusts of 3 m/s", ("E3",)).ok  # the evidence says 2.5
    assert verdict("Gusts of 2 m/s", ("E3",)).ok  # round() alone only accepts the even one
    assert not verdict("Gusts of 4 m/s", ("E3",)).ok
    assert agent._matches(3.0, 0, False, [2.5]) and agent._matches(2.0, 0, False, [2.5])
    assert not agent._matches(3.0, 0, False, [2.4]) and agent._matches(2.0, 0, False, [2.4])
    assert agent._matches(0.3, 1, False, [0.25]) and agent._matches(0.2, 1, False, [0.25])
    # a percentage quotes a fraction: half a percent either side of 0.045
    assert agent._matches(5.0, 0, True, [0.045]) and not agent._matches(6.0, 0, True, [0.045])


def test_dates_and_times_in_the_evidence_are_not_numbers_a_claim_can_match():
    ev = tools.Evidence()
    ev.add("station_readings", {}, "AirKorea", {
        "peak_pm10": 300.0, "peak_time_kst": "02-22 13:00", "image_date": "2026-03-25",
        "taken": "2026-03-25T13:40:00+09:00",
    })

    def problems(text):
        draft = agent.ReportDraft(
            headline="h", cause="dust", confidence=0.5, summary="s", alternatives=[],
            limitations=[], claims=[agent.Claim(text=text, evidence_ids=["E1"])])
        return agent.verify(draft, ev).checks[0].problems

    assert problems("PM10 peaked at 300 ug/m3") == []  # a real number still passes
    for n in ("22", "13", "25", "40", "9"):  # parts of the date, time and ISO timestamp above
        assert problems(f"The count was {n}") == [f"number {n} not in cited evidence"]
    # a string that also holds a value keeps it: only the date and time are dropped
    ev.add("wind", {}, "Open-Meteo", {"note": "gust of 12 m/s at 02-22 13:00"})
    draft = agent.ReportDraft(
        headline="h", cause="dust", confidence=0.5, summary="s", alternatives=[], limitations=[],
        claims=[agent.Claim(text="A gust of 12 m/s", evidence_ids=["E2"])])
    assert agent.verify(draft, ev).checks[0].ok


@pytest.mark.parametrize(("text", "ids", "why"), [
    ("PM10 peaked at 400 ug/m3", ("E1",), "400"),
    ("The ratio was 0.4", ("E1",), "0.4"),
    ("Wind blew at 300", ("E2",), "300"),  # right number, wrong evidence
    ("PM10 peaked at 300", (), "no evidence"),
    ("PM10 peaked at 300", ("E9",), "unknown evidence id E9"),
    ("PM10 peaked at 300", ("E1", "E9"), "E9"),
])
def test_verify_fails(text, ids, why):
    check = verdict(text, ids)
    assert not check.ok and any(why in p for p in check.problems)


def test_verification_counts():
    draft = agent.ReportDraft(
        headline="h", cause="dust", confidence=0.5, summary="s", alternatives=[], limitations=[],
        claims=[agent.Claim(text="PM10 300", evidence_ids=["E1"]),
                agent.Claim(text="PM10 999", evidence_ids=["E1"])])
    v = agent.verify(draft, evidence())
    assert (v.verified, v.total, [c.index for c in v.failed]) == (1, 2, [1])
