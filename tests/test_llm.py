import json

import httpx
import pytest
from pydantic import BaseModel

from airlens_agent import config, http, llm, quota

META_KEY = "FAKE-META-KEY-9f8e7d"
BASETEN_KEY = "FAKE-BASETEN-KEY-1a2b3c"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", META_KEY)
    monkeypatch.setenv("BASETEN_API_KEY", BASETEN_KEY)
    monkeypatch.setattr(http, "_sleep", lambda _s: None)


_USAGE = {"prompt_tokens": 1000, "completion_tokens": 500}


def _reply(content="hi", tool_calls=None, usage=_USAGE, finish="stop", model="m"):
    """A chat completion body; `usage=None` leaves the usage key out."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    body = {"model": model, "choices": [{"message": msg, "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return body


def _transport(*replies):
    """Answers 200 with each reply in turn (a str reply is sent as a raw, non-JSON body)."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "headers": request.headers,
                     "body": json.loads(request.content)})
        reply = replies[min(len(seen) - 1, len(replies) - 1)]
        if isinstance(reply, str):
            return httpx.Response(200, text=reply)
        return httpx.Response(200, json=reply)

    return httpx.MockTransport(handler), seen


def test_meta_request_shape(conn):
    transport, seen = _transport(_reply())
    llm.chat("meta", [{"role": "user", "content": "x"}], tools=[{"type": "function"}],
             reasoning_effort="low", max_tokens=123, ledger=conn, transport=transport)
    req = seen[0]
    assert req["url"] == "https://api.meta.ai/v1/chat/completions"
    assert req["headers"]["authorization"] == f"Bearer {META_KEY}"
    body = req["body"]
    assert body["model"] == config.MUSE_SPARK_MODEL
    assert body["max_completion_tokens"] == 123 and "max_tokens" not in body
    assert body["reasoning_effort"] == "low" and "tool_choice" not in body


def test_baseten_request_shape(conn):
    transport, seen = _transport(_reply())
    llm.chat("baseten", [{"role": "user", "content": "x"}], max_tokens=77, ledger=conn,
             transport=transport)
    req = seen[0]
    assert req["url"] == "https://inference.baseten.co/v1/chat/completions"
    assert req["headers"]["authorization"] == f"Bearer {BASETEN_KEY}"
    assert req["body"]["model"] == config.MUSE_SPARK_OPEN_MODEL
    assert req["body"]["max_tokens"] == 77 and "max_completion_tokens" not in req["body"]
    assert "tools" not in req["body"] and req["body"]["reasoning_effort"] == "medium"  # default


def test_tool_calls_parsed_with_invalid_arguments(conn):
    calls = [
        {"id": "c1", "type": "function", "function": {"name": "wind", "arguments": '{"a": 1}'}},
        {"id": "c2", "type": "function", "function": {"name": "wind", "arguments": "{oops"}},
        {"id": "c3", "type": "function", "function": {"name": "wind", "arguments": "[1]"}},
    ]
    transport, _ = _transport(_reply(content=None, tool_calls=calls, finish="tool_calls"))
    res = llm.chat("meta", [], ledger=conn, transport=transport)
    assert res.finish_reason == "tool_calls"
    assert [c["arguments"] for c in res.tool_calls] == [{"a": 1}, None, None]
    assert res.tool_calls[1]["raw_arguments"] == "{oops"
    assert res.message["tool_calls"] == calls and res.message["role"] == "assistant"


def test_tool_call_arguments_that_arrive_already_parsed_are_accepted(conn):
    def call(cid, arguments):
        return {"id": cid, "type": "function", "function": {"name": "wind", "arguments": arguments}}

    calls = [call("c1", {"level": "10m"}), call("c2", {}), call("c3", '{"level": "10m"}')]
    transport, _ = _transport(_reply(content=None, tool_calls=calls, finish="tool_calls"))
    res = llm.chat("meta", [], ledger=conn, transport=transport)  # a billed reply: no exception
    assert [c["arguments"] for c in res.tool_calls] == [{"level": "10m"}, {}, {"level": "10m"}]
    # the raw text of a parsed object is its JSON dump; a string is kept as the server sent it
    assert [c["raw_arguments"] for c in res.tool_calls] == [
        json.dumps({"level": "10m"}), "{}", '{"level": "10m"}']
    assert quota.spent(conn, "meta") == pytest.approx((1000 * 0.10 + 500 * 0.20) / 1e6)


def test_cost_and_ledger_charge(conn):
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
             "prompt_tokens_details": {"cached_tokens": 400_000}}
    transport, _ = _transport(_reply(usage=usage))
    res = llm.chat("meta", [], ledger=conn, transport=transport)
    expected = 0.6 * 0.10 + 0.4 * 0.002 + 1.0 * 0.20
    assert res.cost_usd == pytest.approx(expected)
    assert quota.spent(conn, "meta") == pytest.approx(expected)
    assert quota.spent(conn, "baseten") == 0
    assert quota.spent(conn, "meta", day="2000-01-01") == 0
    transport, _ = _transport(_reply(usage={"prompt_tokens": 1_000_000, "completion_tokens": 0}))
    res = llm.chat("baseten", [], ledger=conn, transport=transport)
    assert res.cost_usd == pytest.approx(1.25)


def test_budget_exceeded_before_any_request(conn, monkeypatch):
    monkeypatch.setitem(config.SPEND_CAPS, "meta", 1.0)
    quota.charge(conn, "meta", 1.0)
    transport, seen = _transport(_reply())
    with pytest.raises(quota.BudgetExceeded):
        llm.chat("meta", [], ledger=conn, transport=transport)
    assert seen == []


@pytest.mark.parametrize("broken", [
    {"choices": []},
    {},
    {"choices": [{"finish_reason": "stop"}]},
    {"choices": [{"message": "oops"}]},
    {"choices": [{"message": {"tool_calls": ["not-a-dict"]}}]},
])
def test_unusable_reply_is_still_charged_before_it_raises(conn, broken):
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
    transport, _ = _transport({**broken, "usage": usage})
    with pytest.raises(http.HttpError, match="unexpected response"):
        llm.chat("baseten", [], ledger=conn, transport=transport)
    assert quota.spent(conn, "baseten") == pytest.approx(1.25)  # the provider billed the call


def _estimate(body: dict, max_tokens: int, usd_in: float, usd_out: float) -> float:
    """The request at ~4 characters per token as uncached input plus all max_tokens as output."""
    return (len(json.dumps(body)) / 4 * usd_in + max_tokens * usd_out) / 1e6


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": None, "completion_tokens": None}])
def test_missing_usage_is_charged_a_conservative_estimate(conn, usage):
    transport, seen = _transport(_reply(usage=usage))
    res = llm.chat("meta", [{"role": "user", "content": "x"}], max_tokens=1000, ledger=conn,
                   transport=transport)
    expected = _estimate(seen[0]["body"], 1000, 0.10, 0.20)
    assert expected > 1000 * 0.20 / 1e6  # more than the output cap alone: the input counts too
    assert res.cost_usd == pytest.approx(expected) and res.usage == {}
    assert quota.spent(conn, "meta") == pytest.approx(expected)


def test_reported_zero_usage_is_trusted_not_estimated(conn):
    transport, _ = _transport(_reply(usage={"prompt_tokens": 0, "completion_tokens": 0}))
    assert llm.chat("meta", [], ledger=conn, transport=transport).cost_usd == 0
    assert quota.spent(conn, "meta") == 0


def test_non_json_reply_is_charged_an_estimate_before_it_raises(conn):
    transport, seen = _transport("<html>bad gateway</html>")
    with pytest.raises(http.HttpError, match="unexpected response"):
        llm.chat("meta", [{"role": "user", "content": "x"}], max_tokens=1000, ledger=conn,
                 transport=transport)
    assert quota.spent(conn, "meta") == pytest.approx(_estimate(seen[0]["body"], 1000, 0.10, 0.20))


@pytest.mark.parametrize(("provider", "key"), [("meta", META_KEY), ("baseten", BASETEN_KEY)])
@pytest.mark.parametrize("shape", ["json", "text"])
def test_unexpected_reply_snippet_redacts_a_key_cut_by_the_limit(conn, provider, key, shape):
    pad = "z" * 183  # the key starts 7 characters before the 200-character cut
    text = json.dumps({"error": pad + key + " tail"}, separators=(",", ":"))
    transport, _ = _transport(text if shape == "json" else "z" * 193 + key + " tail")
    with pytest.raises(http.HttpError, match="unexpected response") as err:
        llm.chat(provider, [], ledger=conn, transport=transport)
    # cutting first would leave a key prefix that redact() can no longer recognise
    assert key[:6] not in str(err.value) and "***" in str(err.value)


@pytest.mark.parametrize(("provider", "effort"), [("meta", None), ("baseten", "medium")])
def test_provider_default_reasoning_effort(conn, provider, effort):
    transport, seen = _transport(_reply())
    llm.chat(provider, [], ledger=conn, transport=transport)
    assert seen[0]["body"].get("reasoning_effort") == effort  # meta: left to the API default
    llm.chat(provider, [], reasoning_effort="high", ledger=conn, transport=transport)
    assert seen[1]["body"]["reasoning_effort"] == "high"  # the caller's choice wins


class Answer(BaseModel):
    label: str
    score: float


def test_json_chat_retries_once_after_invalid_answer(conn):
    fenced = '```json\n{"label":"a","score":1}\n```'
    transport, seen = _transport(_reply("not json"), _reply(fenced))
    obj, res = llm.json_chat("baseten", [{"role": "user", "content": "q"}], Answer,
                             schema_name="answer", ledger=conn, transport=transport)
    assert obj == Answer(label="a", score=1.0)
    assert len(seen) == 2 and seen[1]["body"]["messages"][-1]["role"] == "user"
    assert "invalid" in seen[1]["body"]["messages"][-1]["content"]
    rf = seen[0]["body"]["response_format"]["json_schema"]
    assert rf["strict"] is True and rf["name"] == "answer"
    assert res.cost_usd == pytest.approx(2 * (1000 * 1.25 + 500 * 4.25) / 1e6)


@pytest.mark.parametrize(("first", "resent"), [
    ("", False), (" \n ", False),  # nothing to show the model: only the correction request
    ("not json", True),
])
def test_json_chat_retry_does_not_resend_an_empty_reply(conn, first, resent):
    transport, seen = _transport(_reply(first), _reply('{"label": "a", "score": 1}'))
    obj, _ = llm.json_chat("meta", [{"role": "user", "content": "q"}], Answer,
                           schema_name="answer", ledger=conn, transport=transport)
    assert obj == Answer(label="a", score=1.0)
    retry = seen[1]["body"]["messages"]
    expected = ["user", "assistant", "user"] if resent else ["user", "user"]
    assert [m["role"] for m in retry] == expected
    assert retry[0] == {"role": "user", "content": "q"}
    assert retry[-1]["content"].startswith("That answer was invalid (")
    assert all(m["content"] for m in retry)  # no empty message goes back to the API


def test_json_chat_raises_after_second_failure(conn):
    transport, seen = _transport(_reply("nope"))
    with pytest.raises(ValueError):
        llm.json_chat("meta", [], Answer, schema_name="answer", ledger=conn, transport=transport)
    assert len(seen) == 2


@pytest.mark.parametrize(("provider", "effort"), [("meta", None), ("baseten", "medium")])
def test_json_chat_applies_the_provider_default_effort(conn, provider, effort):
    transport, seen = _transport(_reply('{"label": "a", "score": 1}'))
    llm.json_chat(provider, [], Answer, schema_name="answer", ledger=conn, transport=transport)
    assert seen[0]["body"].get("reasoning_effort") == effort
    llm.json_chat(provider, [], Answer, schema_name="answer", reasoning_effort="low", ledger=conn,
                  transport=transport)
    assert seen[1]["body"]["reasoning_effort"] == "low"


def test_flat_schema_has_no_refs():
    class Inner(BaseModel):
        x: int

    class Outer(BaseModel):
        items: list[Inner]
        note: str = "d"

    schema = llm.flat_schema(Outer)
    assert "$ref" not in json.dumps(schema) and "$defs" not in schema
    assert schema["additionalProperties"] is False and schema["required"] == ["items", "note"]
    assert schema["properties"]["items"]["items"]["required"] == ["x"]


def test_flat_schema_leaves_map_objects_open():
    class Inner(BaseModel):
        x: int

    class Outer(BaseModel):
        scores: dict[str, float]
        by_name: dict[str, Inner]
        inner: Inner

    schema = llm.flat_schema(Outer)
    props = schema["properties"]
    # a dict field stays a map; forcing "no extra keys" would turn it into an empty object
    assert props["scores"] == {"type": "object", "additionalProperties": {"type": "number"}}
    named = props["by_name"]
    assert "properties" not in named and "required" not in named
    # refs inside a map are still inlined, and the objects with declared fields stay closed
    assert named["additionalProperties"]["required"] == ["x"]
    assert named["additionalProperties"]["additionalProperties"] is False
    assert props["inner"]["additionalProperties"] is False and props["inner"]["required"] == ["x"]
    assert schema["required"] == ["scores", "by_name", "inner"]
    assert schema["additionalProperties"] is False and "$ref" not in json.dumps(schema)
