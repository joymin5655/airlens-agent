"""OpenAI-format chat clients for Muse Spark (Meta Model API) and Muse Spark Open (Baseten).

Every call checks the USD budget first and records its computed cost in the spend ledger.
"""

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import httpx
from pydantic import BaseModel, ValidationError

from . import config, http, quota


@dataclass(frozen=True)
class Provider:
    name: str  # ledger key and quota prefix
    base_url: str
    key: Callable[[], str]
    model: str
    usd_per_m_input: float
    usd_per_m_cached: float
    usd_per_m_output: float
    token_param: str  # name of the output-token limit parameter
    op: str  # quota op
    default_effort: str | None = None  # reasoning_effort when the caller passes none


PROVIDERS: dict[str, Provider] = {
    "meta": Provider(
        "meta", config.META_API_BASE, config.model_api_key, config.MUSE_SPARK_MODEL,
        0.10, 0.002, 0.20, "max_completion_tokens", "meta.chat",
    ),
    "baseten": Provider(
        "baseten", config.BASETEN_API_BASE, config.baseten_api_key, config.MUSE_SPARK_OPEN_MODEL,
        1.25, 0.15, 4.25, "max_tokens", "baseten.chat",
        # its own default is xhigh, which can spend the whole max_tokens on reasoning
        default_effort="medium",
    ),
}


@dataclass
class ChatResult:
    content: str
    reasoning: str
    tool_calls: list[dict]  # {id, name, arguments: dict | None, raw_arguments: str}
    message: dict  # assistant message to resend before the tool results
    usage: dict
    cost_usd: float
    model: str
    finish_reason: str | None
    provider: str = ""
    extra: dict = field(default_factory=dict)


def cost_usd(provider: Provider, usage: dict) -> float:
    """Cost of one call: uncached input, cached input, and output (reasoning included)."""
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cached = min(cached, prompt)
    output = usage.get("completion_tokens") or 0
    return (
        (prompt - cached) * provider.usd_per_m_input
        + cached * provider.usd_per_m_cached
        + output * provider.usd_per_m_output
    ) / 1_000_000


def _has_usage(usage) -> bool:
    """True when the reply reports at least one token count."""
    return isinstance(usage, dict) and any(
        isinstance(usage.get(k), int | float) for k in ("prompt_tokens", "completion_tokens")
    )


def estimate_cost_usd(provider: Provider, payload: dict) -> float:
    """Cost to book for a call whose reply carries no usage.

    ponytail: assumes the request is ~4 characters per token, all uncached input, and that the
    model used the whole output limit. It over-counts on purpose, so a billed call that reports
    nothing cannot vanish from the ledger.
    """
    prompt_tokens = len(json.dumps(payload)) / 4
    output_tokens = payload.get(provider.token_param) or 0
    return (
        prompt_tokens * provider.usd_per_m_input + output_tokens * provider.usd_per_m_output
    ) / 1_000_000


def _parse_tool_calls(raw: list[dict]) -> list[dict]:
    calls = []
    for call in raw:
        fn = call.get("function") or {}
        text = fn.get("arguments") or ""
        try:
            args = json.loads(text) if text else {}
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            args = None  # the agent returns an error to the model instead of guessing
        calls.append(
            {"id": call.get("id"), "name": fn.get("name"), "arguments": args, "raw_arguments": text}
        )
    return calls


def chat(
    provider: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int = 8192,
    ledger: sqlite3.Connection,
    transport: httpx.BaseTransport | None = None,
) -> ChatResult:
    """One chat completion. `tool_choice` is never sent (only "auto" works on the Meta API).

    `reasoning_effort=None` uses the provider's `default_effort` (None = the API's own default).
    The call is charged as soon as the reply arrives, before its choices are read: a reply that
    turns out to be unusable was still billed.
    """
    prov = PROVIDERS[provider]
    quota.ensure_budget(ledger, prov.name)
    payload: dict = {"model": prov.model, "messages": messages, prov.token_param: max_tokens}
    if tools:
        payload["tools"] = tools
    if response_format:
        payload["response_format"] = response_format
    if effort := reasoning_effort or prov.default_effort:
        payload["reasoning_effort"] = effort
    resp = http.post_json(
        f"{prov.base_url}/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {prov.key()}"},
        op=prov.op,
        conn=ledger,
        transport=transport,
    )
    try:
        data = resp.json()
    except ValueError:
        data = None  # not JSON: charged as if usage were missing, then rejected below
    usage = data.get("usage") if isinstance(data, dict) else None
    known = _has_usage(usage)
    cost = cost_usd(prov, usage) if known else estimate_cost_usd(prov, payload)
    quota.charge(ledger, prov.name, cost)
    try:
        choice = data["choices"][0]
        msg = choice["message"]
        raw_calls = msg.get("tool_calls") or []
        tool_calls = _parse_tool_calls(raw_calls)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise http.HttpError(
            f"{prov.op}: unexpected response: {http.redact(resp.text)[:200]}"
        ) from exc
    message: dict = {"role": "assistant", "content": msg.get("content") or ""}
    if raw_calls:
        message["tool_calls"] = raw_calls
    return ChatResult(
        content=msg.get("content") or "",
        reasoning=msg.get("reasoning_content") or "",
        tool_calls=tool_calls,
        message=message,
        usage=usage if known else {},
        cost_usd=cost,
        model=data.get("model") or prov.model,
        finish_reason=choice.get("finish_reason"),
        provider=prov.name,
    )


# Title/default and range constraints are left to pydantic; strict mode only needs the shape.
_DROP_KEYS = frozenset({
    "title", "default", "$defs", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "minItems", "maxItems",
})


def flat_schema(model_cls: type[BaseModel]) -> dict:
    """JSON schema for strict structured output: refs inlined, every key required, no extras.

    Only objects with declared `properties` are closed this way; a map (`dict[str, X]`, which
    has an `additionalProperties` schema instead) keeps its shape, or it would become an empty
    object.

    ponytail: no recursive models and no Optional fields (the Meta API rejects both); keep
    response models flat and nested at most a few levels.
    """
    raw = model_cls.model_json_schema()
    defs = raw.get("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
        out = {k: walk(v) for k, v in node.items() if k not in _DROP_KEYS}
        if out.get("type") == "object" and "properties" in node:
            out["properties"] = {k: walk(v) for k, v in node["properties"].items()}
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    return walk(raw)


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def _validate(model_cls: type[BaseModel], text: str) -> BaseModel:
    text = text.strip()
    fenced = _FENCE.match(text)
    return model_cls.model_validate_json(fenced.group(1) if fenced else text)


def json_chat(
    provider: str,
    messages: list[dict],
    model_cls: type[BaseModel],
    *,
    schema_name: str,
    schema: dict | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int = 8192,
    ledger: sqlite3.Connection,
    transport: httpx.BaseTransport | None = None,
) -> tuple[BaseModel, ChatResult]:
    """Chat with a json_schema response format and validate the answer with `model_cls`.

    Baseten does not enforce the schema, so on a validation failure the call is retried once
    with the error appended as a user message. The returned cost covers both attempts.
    `reasoning_effort=None` falls back to the provider's `default_effort` (see `chat`).
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name, "schema": schema or flat_schema(model_cls), "strict": True,
        },
    }
    msgs = list(messages)
    spent = 0.0
    for attempt in range(2):
        res = chat(
            provider, msgs, response_format=response_format, reasoning_effort=reasoning_effort,
            max_tokens=max_tokens, ledger=ledger, transport=transport,
        )
        spent += res.cost_usd
        try:
            return _validate(model_cls, res.content), replace(res, cost_usd=spent)
        except (ValidationError, ValueError) as exc:
            if attempt == 1:
                raise
            msgs += [
                {"role": "assistant", "content": res.content},
                {
                    "role": "user",
                    "content": f"That answer was invalid ({str(exc)[:600]}). "
                    "Reply again with only the corrected JSON object.",
                },
            ]
    raise AssertionError("unreachable")
