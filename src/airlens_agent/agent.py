"""Triage, tool-calling investigation and claim verification for one candidate event.

Muse Spark Open (Baseten) triages, Muse Spark 1.3 (Meta) investigates with tools and writes the
report. Code, not the model, attaches the evidence, the models used, the cost and the triage.
"""

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from . import config, http, llm, quota, tools
from .models import Candidate

Category = Literal["dust", "wildfire_smoke", "industrial", "traffic", "mixed", "unknown"]
ToolName = Literal[
    "station_readings", "fire_hotspots", "wind", "cams_forecast", "satellite_segments"
]
EFFORT = "medium"
MAX_TOKENS = 8192
NUMBER_TOLERANCE = 0.05
DETECTOR_SOURCE = "AirLens anomaly detector"
# A paid step that fails this way must not throw away the work already paid for. ValueError is
# an invalid answer: pydantic's ValidationError or anything else json_chat re-raises; the other
# errors here are RuntimeErrors (a failed call).
_STEP_ERRORS = (
    ValueError, http.HttpError, quota.QuotaExceeded, quota.BudgetExceeded,
    config.MissingKeyError,
)


class Hypothesis(BaseModel):
    cause: Category
    likelihood: float = Field(ge=0, le=1, description="0 to 1")
    rationale: str


class Triage(BaseModel):
    category: Category
    hypotheses: list[Hypothesis]
    checks: list[ToolName] = Field(description="tools worth running to test the hypotheses")
    _model: str = PrivateAttr(default="")  # set by triage(); not part of the schema
    _fallback: str | None = PrivateAttr(default=None)  # provider switch or invalid answer

    @field_validator("hypotheses")
    @classmethod
    def _distinct_causes(cls, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        # Muse Spark Open once answered traffic three times for a coarse-dust event.
        causes = [h.cause for h in hypotheses]
        if len(set(causes)) != len(causes):
            raise ValueError("each hypothesis must name a different cause")
        return hypotheses

    @property
    def model_used(self) -> str:
        return self._model

    @property
    def fallback(self) -> str | None:
        return self._fallback


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(description="ids such as E1 from the tool results")


class Alternative(BaseModel):
    cause: Category
    why_less_likely: str


class ReportDraft(BaseModel):
    headline: str
    cause: Category
    confidence: float = Field(ge=0, le=1, description="0 to 1")
    summary: str
    claims: list[Claim]
    alternatives: list[Alternative]
    limitations: list[str]


class Report(ReportDraft):
    """The model's report plus what code attaches to it."""

    evidence: list[dict] = []
    models: list[str] = []
    investigator: str = ""  # provider that was asked for
    provider_used: str = ""  # provider that finished the report
    fallback: str | None = None
    cost_usd: float = 0.0
    triage: Triage | None = None
    triage_fallback: str | None = None  # the triage step's provider switch or invalid answer
    unverified_claims: list[int] = []  # indexes into `claims` that still fail verification
    dropped_claims: list[str] = []  # claims the model removed in the repair round


class RepairedClaim(BaseModel):
    index: int
    text: str
    evidence_ids: list[str]


class ClaimRepair(BaseModel):
    claims: list[RepairedClaim]


@dataclass
class ClaimCheck:
    index: int
    ok: bool
    problems: list[str] = field(default_factory=list)


@dataclass
class Verification:
    checks: list[ClaimCheck]

    @property
    def failed(self) -> list[ClaimCheck]:
        return [c for c in self.checks if not c.ok]

    @property
    def verified(self) -> int:
        return len(self.checks) - len(self.failed)

    @property
    def total(self) -> int:
        return len(self.checks)


# --- provider routing -------------------------------------------------------------------------

_OTHER = {"meta": "baseten", "baseten": "meta"}


class Route:
    """The provider in use; a missing key or an exhausted budget switches to the other once."""

    def __init__(self, provider: str):
        self.requested = self.provider = provider
        self.answered = ""  # provider of the last call that succeeded
        self.fallback: str | None = None

    def call(self, fn: Callable[[str], object]):
        try:
            out = fn(self.provider)
        except (config.MissingKeyError, quota.BudgetExceeded) as exc:
            if self.fallback is not None:
                raise
            self.fallback = f"{self.provider} -> {_OTHER[self.provider]}: {exc}"
            self.provider = _OTHER[self.provider]
            out = fn(self.provider)
        self.answered = self.provider
        return out


def _why(exc: Exception) -> str:
    """One line for a failure report: type and message, key values redacted, cut short."""
    return f"{type(exc).__name__}: {' '.join(http.redact(str(exc)).split())[:200]}"


# --- prompts ----------------------------------------------------------------------------------


def describe(c: Candidate) -> str:
    f = c.features

    def num(v, digits=1):
        return "n/a" if v is None else f"{v:.{digits}f}"

    return "\n".join([
        f"Event {c.event_id} at {c.ts.isoformat()} (KST), {c.kind} anomaly in "
        f"{', '.join(f.get('provinces') or []) or 'unknown provinces'}.",
        f"Centroid {c.centroid_lat:.2f}N {c.centroid_lon:.2f}E, {len(c.station_keys)} stations, "
        f"extent {num(f.get('extent_km'), 0)} km.",
        f"Peak PM10 {num(c.peak_pm10, 0)}, peak PM2.5 {num(c.peak_pm25, 0)} ug/m3, "
        f"PM2.5/PM10 ratio {num(c.ratio_pm25_pm10, 2)} (detector hint: {c.hint}).",
        f"CO z-score {num(f.get('co_z_mean'))}, NO2 z-score {num(f.get('no2_z_mean'))}.",
        f"3 h rise: PM10 {num(f.get('rise_3h_pm10'), 0)}, PM2.5 {num(f.get('rise_3h_pm25'), 0)}.",
        f"Fire hotspots within 50 km in the last 24 h: {f.get('hotspots_50km_24h', 'n/a')}.",
    ])


def detector_summary(c: Candidate) -> dict:
    """The numbers `describe()` shows, rounded the same way, so claims can cite them (as E1).

    No timestamp or id strings: their digits would end up in the pool claims are checked against.
    ponytail: the 3 h, 50 km and 24 h windows are copied from describe() and detect.py, not shared.
    """
    f = c.features

    def r(v, digits=1):
        return None if v is None else round(v, digits)

    return {
        "kind": c.kind, "hint": c.hint, "provinces": f.get("provinces") or [],
        "n_stations": len(c.station_keys),
        "centroid_lat": round(c.centroid_lat, 2), "centroid_lon": round(c.centroid_lon, 2),
        "extent_km": r(f.get("extent_km"), 0),
        "peak_pm10": r(c.peak_pm10, 0), "peak_pm25": r(c.peak_pm25, 0),
        "ratio_pm25_pm10": r(c.ratio_pm25_pm10, 2),
        "co_z_mean": r(f.get("co_z_mean")), "no2_z_mean": r(f.get("no2_z_mean")),
        "rise_window_h": 3,
        "rise_3h_pm10": r(f.get("rise_3h_pm10"), 0), "rise_3h_pm25": r(f.get("rise_3h_pm25"), 0),
        "hotspot_radius_km": 50, "hotspot_window_h": 24,
        "hotspots_50km_24h": f.get("hotspots_50km_24h"),
    }


CATEGORY_GUIDE = (
    "Cause categories: dust = wind-blown mineral dust, coarse particles dominate (low PM2.5/PM10 "
    "ratio); wildfire_smoke = biomass-burning smoke, fine particles dominate, often with CO and "
    "fires upwind; industrial = emissions from industrial sources; traffic = local vehicle "
    "emissions, NO2 and CO rise in cities; mixed = two or more causes together; unknown = the "
    "data do not point to a cause."
)
TRIAGE_SYSTEM = (
    "You triage air-pollution anomalies in South Korea from detector features. Give the most "
    "likely cause category, two to four hypotheses with different causes ranked by likelihood, "
    f"and which tools to run to test them (tools: {', '.join(tools.TOOL_NAMES)}). "
    f"{CATEGORY_GUIDE} Answer with JSON only. Do not give health or medical advice."
)
INVESTIGATE_SYSTEM = (
    "You are an air-pollution event investigator for South Korea. A statistical detector flagged "
    "an anomaly and a first-pass triage proposed hypotheses. Verify or refute them with the "
    "tools; call only what you need (satellite_segments costs money). Every claim you make must "
    "cite evidence ids (E1, E2, ...): E1 holds the detector's own numbers, the tool results are "
    "numbered after it. Every number in a claim must be copied from the cited evidence. "
    f"{CATEGORY_GUIDE} Report uncertainty and what you could not check. Write in English. "
    "Do not give health or medical advice."
)
FINAL_REQUEST = (
    "Write the final report now as JSON. Claims must cite evidence ids (E1 = the detector's "
    "numbers, then the tool results); copy numbers exactly from the cited evidence."
)


# --- steps ------------------------------------------------------------------------------------


def triage(
    candidate: Candidate, *, ledger: sqlite3.Connection, provider: str = "baseten", transport=None
) -> Triage:
    """First-pass category and hypotheses from the candidate's detector features.

    A provider switch is kept on the result (`fallback`). When the answer is invalid even after
    json_chat's one retry, or the call fails (an outage, no key for either provider), a neutral
    triage (unknown, no hypotheses, every tool) is returned so the investigation still runs.
    """
    messages = [
        {"role": "system", "content": TRIAGE_SYSTEM},
        {"role": "user", "content": describe(candidate)},
    ]
    route = Route(provider)
    try:
        obj, res = route.call(
            lambda p: llm.json_chat(
                p, messages, Triage, schema_name="triage", reasoning_effort=EFFORT,
                max_tokens=MAX_TOKENS, ledger=ledger, transport=transport,
            )
        )
    except _STEP_ERRORS as exc:
        obj = Triage(category="unknown", hypotheses=[], checks=list(tools.TOOL_NAMES))
        what = "answer invalid after a retry" if isinstance(exc, ValueError) else "call failed"
        note = f"triage {what}, neutral triage used ({_why(exc)})"
        obj._fallback = "; ".join(filter(None, [route.fallback, note]))
        return obj
    obj._model = res.model
    obj._fallback = route.fallback
    return obj


# What counts as a date or time, and so is not checked as a number. Narrow on purpose: whatever
# it strips is never verified, so "decreased 25", "5-8 m/s" or "1950 MW" must stay numbers.
_MONTH = (  # real month words only; "may" needs a capital (lower case is the verb)
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|(?-i:May|MAY)|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_MM, _DD = r"(?:0?[1-9]|1[0-2])", r"(?:0?[1-9]|[12]\d|3[01])"
# a 4-digit number right before one of these is a value, not a year (u00b5 = micro, u03bc = mu)
# ponytail: a heuristic, any other "1999 hotspots" is still read as a year and skipped
_UNIT = r"(?:mw|km|ug|[\u00b5\u03bc]g|hpa|ppm|ppb)(?![a-z])|%"
_DATE_OR_TIME = re.compile(
    rf"(?<![\d.-])\d\d\d\d-{_MM}-{_DD}(?!\d)"  # 2026-02-22 (also before a T10:00 time)
    rf"|(?<![\d.-]){_MM}-{_DD}(?=\s+\d\d?:\d\d(?!\d))"  # 02-22 13:00, as the tools print it
    r"|(?<![\d:.])\d\d?:\d\d(?::\d\d)?(?!\d)"  # 9:30, 10:00:00
    rf"|(?<![\d.]){_MM}/{_DD}(?!\d|\.\d)"  # 2/22
    rf"|\b{_MONTH}\.? {_DD}(?:st|nd|rd|th)?\b"  # Feb 22
    rf"|\b{_DD}(?:st|nd|rd|th)?(?: of)? {_MONTH}\b"  # 22 Feb
    rf"|(?<![\w.])(?:19|20)\d\d(?!\w|\.\d|\s*(?:{_UNIT}))",  # 2026, but not 1950 MW or 2024.3
    re.I,
)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(%?)")


def _numbers(text: str, *, skip_dates: bool) -> list[tuple[float, int, bool]]:
    """(value, decimals, is_percent) per number; IDs like E3 and names like PM2.5 are skipped."""
    if skip_dates:
        text = _DATE_OR_TIME.sub(" ", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0).rstrip("%")
        out.append((float(raw), len(raw.split(".")[1]) if "." in raw else 0, bool(m.group(1))))
    return out


def _evidence_numbers(data) -> list[float]:
    if isinstance(data, bool):
        return []
    if isinstance(data, int | float):
        return [float(data)]
    if isinstance(data, str):  # "02-22 13:00" or an ISO timestamp must not make a 22 or 13 pass
        return [v for v, _, _ in _numbers(data, skip_dates=True)]
    if isinstance(data, dict):
        data = list(data.values())
    if isinstance(data, list | tuple):
        return [n for item in data for n in _evidence_numbers(item)]
    return []


def _matches(value: float, decimals: int, pct: bool, pool: list[float]) -> bool:
    """Within 5 % of an evidence number, or within half a unit of the claim's last digit (so
    "3" matches 2.5 and "2" does too; round() would only accept the even one)."""
    for scale in (1, 100) if pct else (1,):  # "31%" may quote the fraction 0.31
        c = value / scale
        half_digit = 0.5 * 10**-decimals / scale + 1e-9  # + epsilon for float error
        if any(abs(c - e) <= max(NUMBER_TOLERANCE * abs(e), half_digit) for e in pool):
            return True
    return False


def verify(report: ReportDraft, evidence: tools.Evidence) -> Verification:
    """Check each claim: at least one existing evidence id, and every number in the claim matches
    a number in the cited evidence within 5 % or rounding. Years, dates and times are ignored on
    both sides."""
    checks = []
    for i, claim in enumerate(report.claims):
        problems = []
        cited = [evidence.get(eid) for eid in claim.evidence_ids]
        if not claim.evidence_ids:
            problems.append("cites no evidence")
        for eid, item in zip(claim.evidence_ids, cited, strict=True):
            if item is None:
                problems.append(f"unknown evidence id {eid}")
        pool = [n for item in cited if item for n in _evidence_numbers(item["data"])]
        if any(cited):
            for value, decimals, pct in _numbers(claim.text, skip_dates=True):
                if not _matches(value, decimals, pct, pool):
                    problems.append(f"number {value:g}{'%' if pct else ''} not in cited evidence")
        checks.append(ClaimCheck(i, not problems, problems))
    return Verification(checks)


def _repair(
    report: ReportDraft, check: Verification, evidence: tools.Evidence, route: Route,
    ledger: sqlite3.Connection, transport,
) -> tuple[ClaimRepair, str]:
    lines = []
    for c in check.failed:
        claim = report.claims[c.index]
        problems = "; ".join(c.problems)
        lines.append(f'{c.index}: "{claim.text}" cites {claim.evidence_ids}; problems: {problems}')
    evidence_json = "\n".join(
        f'{i["evidence_id"]}: {i["data"]}' for i in evidence.items
    )
    messages = [
        {"role": "system", "content": INVESTIGATE_SYSTEM},
        {"role": "user", "content": (
            "Some claims in your report failed verification.\n" + "\n".join(lines)
            + "\n\nEvidence:\n" + evidence_json
            + "\n\nReturn JSON {\"claims\": [...]} with a corrected version (same index) of each "
            "claim you can fix using only this evidence. Omit a claim to drop it."
        )},
    ]
    obj, res = route.call(
        lambda p: llm.json_chat(
            p, messages, ClaimRepair, schema_name="claim_repair", reasoning_effort=EFFORT,
            max_tokens=MAX_TOKENS, ledger=ledger, transport=transport,
        )
    )
    return obj, res.model


def investigate(
    conn: sqlite3.Connection,
    candidate: Candidate,
    triage: Triage,
    *,
    ledger: sqlite3.Connection,
    investigator: str = "meta",
    max_steps: int = 8,
    image_dir: Path | None = None,
    out_dir: Path | None = None,
    transport=None,
) -> Report:
    """Tool-calling loop over the event DB, then a structured report whose claims are verified.

    The detector's numbers are evidence E1, the tool results follow. A failed model call, repair
    round or final report does not throw the paid work away: the loop stops and reports on the
    evidence so far, the claims are kept and flagged, or a minimal report carries the evidence.
    """
    ctx = tools.Context(conn, candidate, ledger=ledger, image_dir=image_dir, out_dir=out_dir,
                        transport=transport)
    ctx.evidence.add("detector", {}, DETECTOR_SOURCE, detector_summary(candidate))  # E1
    route = Route(investigator)
    spent_before = quota.spent_total(ledger)
    models: list[str] = []
    messages: list[dict] = [
        {"role": "system", "content": INVESTIGATE_SYSTEM},
        {"role": "user", "content": (
            f"{describe(candidate)}\n\nThese detector numbers are evidence E1; tool results get "
            f"the next ids (E2, E3, ...). Cite E1 for them.\n\nTriage: {triage.model_dump_json()}"
            "\n\nInvestigate which cause explains this event."
        )},
    ]
    specs = tools.specs()

    def attach(draft: ReportDraft, **extra) -> Report:
        """The draft plus what code attaches: evidence, models, route, cost and triage."""
        return Report(
            **draft.model_dump(),
            evidence=ctx.evidence.items,
            models=list(dict.fromkeys(models)),
            investigator=investigator,
            provider_used=route.answered,
            fallback=route.fallback,
            cost_usd=round(quota.spent_total(ledger) - spent_before, 6),
            triage=triage,
            triage_fallback=triage.fallback,
            **extra,
        )

    early: list[str] = []  # limitations known before the final report is written
    for _ in range(max_steps):
        try:
            res = route.call(
                lambda p: llm.chat(
                    p, messages, tools=specs, reasoning_effort=EFFORT, max_tokens=MAX_TOKENS,
                    ledger=ledger, transport=transport,
                )
            )
        except _STEP_ERRORS as exc:  # the evidence so far still goes into the report
            early.append(f"The tool loop stopped early: {_why(exc)}")
            break
        models.append(res.model)
        ids = [tc["id"] or f"call_{n}" for n, tc in enumerate(res.tool_calls)]
        if res.tool_calls or res.content.strip():  # an empty reply would be an invalid message
            message = res.message
            if res.tool_calls:  # resend the ids the tool messages answer (a copy, not an edit)
                message = {**message, "tool_calls": [
                    {**raw, "id": cid} for raw, cid in zip(message["tool_calls"], ids, strict=True)
                ]}
            messages.append(message)
        if not res.tool_calls:
            break
        for tc, cid in zip(res.tool_calls, ids, strict=True):
            messages.append({
                "role": "tool", "tool_call_id": cid,
                "content": tools.call(ctx, tc["name"], tc["arguments"], tc["raw_arguments"]),
            })
    else:
        messages.append({"role": "user", "content": "Step limit reached; stop calling tools."})
    messages.append({"role": "user", "content": FINAL_REQUEST})
    try:
        draft, res = route.call(
            lambda p: llm.json_chat(
                p, messages, ReportDraft, schema_name="report", reasoning_effort=EFFORT,
                max_tokens=MAX_TOKENS, ledger=ledger, transport=transport,
            )
        )
    except _STEP_ERRORS as exc:
        why = _why(exc)
        return attach(ReportDraft(
            headline="The report could not be generated", cause="unknown", confidence=0.0,
            summary=f"The final report failed ({why}). The evidence collected so far is attached.",
            claims=[], alternatives=[], limitations=[f"The final report failed: {why}", *early],
        ))
    models.append(res.model)

    claims, dropped, limitations = list(draft.claims), [], [*draft.limitations, *early]
    check = verify(draft, ctx.evidence)
    if check.failed:
        try:
            fix, repair_model = _repair(draft, check, ctx.evidence, route, ledger, transport)
        except _STEP_ERRORS as exc:  # keep the draft; the failing claims stay flagged below
            limitations.append(
                f"The claim repair round failed ({_why(exc)}); claims that fail verification "
                "are kept and flagged."
            )
        else:
            models.append(repair_model)
            fixed = {c.index: c for c in fix.claims}
            claims = []
            for i, claim in enumerate(draft.claims):
                if i not in {c.index for c in check.failed}:
                    claims.append(claim)
                elif i in fixed:
                    claims.append(Claim(text=fixed[i].text, evidence_ids=fixed[i].evidence_ids))
                else:
                    dropped.append(claim.text)
    final = draft.model_copy(update={"claims": claims, "limitations": limitations})
    check = verify(final, ctx.evidence)
    return attach(
        final, unverified_claims=[c.index for c in check.failed], dropped_claims=dropped
    )
