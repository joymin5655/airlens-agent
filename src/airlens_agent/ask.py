"""Citizen question answering: air-quality risk information with sources, never health advice.

Muse Spark 1.3 (Meta) calls tools in a short loop (geocode, live readings, official forecast, CAMS
forecast, active events) and then writes a structured answer. Code, not the model, attaches the
citations, the risk level, the models and the cost, and drops every sentence whose numbers are
not in the collected evidence (the same check as `agent.verify`).
"""

import json
import re
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from . import agent, config, http, llm, quota, tools
from .config import KST
from .detect import haversine_km
from .sources import airkorea, geocode, openmeteo

RiskLevel = Literal["good", "moderate", "unhealthy", "very_unhealthy", "unknown"]

MAX_QUESTION_CHARS = 300
MAX_STEPS = 5
MAX_CALLS_PER_STEP = 6  # a model that fans out dozens of calls in one message is cut off here
MAX_SENTENCES = 4
OP = "site.ask"
SOURCE_LIVE = "AirKorea real-time stations (data.go.kr)"
SOURCE_FORECAST = "AirKorea official forecast (data.go.kr)"
SOURCE_CAMS = "CAMS global forecast via Open-Meteo (CC BY 4.0)"
SOURCE_MONITOR = "AirLens Agent anomaly monitor"
SOURCE_BANDS = "AirKorea air-quality grade bands"
UNIT = "µg/m³"
STALE_HOURS = 6

# AirKorea grade bands (daily-mean ug/m3, upper edge of each grade): good, moderate ("normal"),
# unhealthy ("bad"), above that very_unhealthy ("very bad"). Applied to AirKorea values in Korea
# and, unchanged, to CAMS values elsewhere: CAMS is a model, so overseas grades are indicative.
BANDS = {"pm10": (30, 80, 150), "pm25": (15, 35, 75)}
LEVELS = ("good", "moderate", "unhealthy", "very_unhealthy")
GRADE_KO = {"좋음": "good", "보통": "moderate", "나쁨": "unhealthy", "매우나쁨": "very_unhealthy"}

SAFETY_NOTE = (
    "Air-quality risk information only, not health or medical advice; for personal decisions "
    "follow the official guidance of your local health authority (in Korea, the Ministry of "
    "Environment and AirKorea)."
)
REFUSAL = (
    "I can only answer questions about air quality and its forecast, for example the "
    "particulate level in a city today or tomorrow."
)
NOTHING_VERIFIED = (
    "I could not confirm enough of the figures behind an answer, so I am not giving one."
)

# What the answer text may not contain: a personal recommendation. Such a sentence is dropped.
# ponytail: a phrase list, not a classifier; the system message is the main guard (upgrade:
# a second model pass).
_ADVICE = re.compile(
    r"\b(you\s+(should|shouldn'?t|must|need\s+to|can\s+safely|could\s+safely)"
    r"|i\s+(recommend|advise|suggest)"
    r"|(it\s+is|it'?s)\s+(safe|fine|okay|ok|unsafe)\s+(for\s+you|to\s+\w+)"
    r"|(wear|put\s+on)\s+(a\s+|an\s+)?(mask|n95|kf94)"
    r"|(take|use)\s+(your\s+)?(medication|medicine|inhaler)"
    r"|(avoid|skip|cancel)\s+(your\s+)?(run|jog|jogging|exercise|workout|outdoor)"
    r"|consult\s+(a|your)\s+(doctor|physician))\b",
    re.I,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


# --- answer model -----------------------------------------------------------------------------


class Citation(BaseModel):
    source: str
    detail: str
    value: float | str | None = None
    unit: str = ""
    time: str = ""


class Answer(BaseModel):
    answer: str
    risk_level: RiskLevel = "unknown"
    time_window: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    citations: list[Citation] = []
    limitations: list[str] = []
    cached: bool = False
    models: list[str] = []
    cost_usd: float = 0.0


class AnswerDraft(BaseModel):
    """What the model writes; the rest of `Answer` is attached by code."""

    in_scope: bool = Field(description="false when the question is not about air quality")
    answer: str = Field(description="2 to 4 plain English sentences")
    risk_level: RiskLevel = Field(description="from the AirKorea grade bands in E1")
    time_window: str = Field(description="the time the answer covers, with its time zone")
    confidence: float = Field(description="0 to 1")
    evidence_ids: list[str] = Field(description="ids such as E2 of the evidence the answer uses")
    limitations: list[str]


# --- grading ----------------------------------------------------------------------------------


def grade(pollutant: str, value: float | None) -> int | None:
    """Index into LEVELS for a PM10 / PM2.5 value, None when there is no value."""
    if value is None:
        return None
    for i, edge in enumerate(BANDS[pollutant]):
        if value <= edge:
            return i
    return len(BANDS[pollutant])


def _worst(pm10: float | None, pm25: float | None) -> RiskLevel | None:
    grades = [g for g in (grade("pm10", pm10), grade("pm25", pm25)) if g is not None]
    return LEVELS[max(grades)] if grades else None


def derive_risk(items: list[dict]) -> RiskLevel | None:
    """The level the evidence itself gives: the worse of the PM10 and PM2.5 grades.

    The asked window's CAMS mean first, else the nearest stations' median now. The official
    forecast has grades per region, which code cannot match to the asked place; the model's
    reading of it is used instead (see `answer`).
    """
    for tool in ("cams_forecast", "live_readings"):
        for item in items:
            if item.get("tool") != tool:
                continue
            d = item["data"]
            if tool == "cams_forecast":
                pm10, pm25 = d.get("pm10") or {}, d.get("pm2_5") or {}
                level = _worst(pm10.get("mean"), pm25.get("mean"))
            else:
                level = _worst(d.get("median_pm10"), d.get("median_pm25"))
            if level:
                return level
    return None


# --- tools ------------------------------------------------------------------------------------


@dataclass
class AskContext:
    conn: sqlite3.Connection  # live DB, only read: stations, observations
    now: datetime
    transport: object = None
    evidence: tools.Evidence = field(default_factory=tools.Evidence)
    cites: dict[str, list[dict]] = field(default_factory=dict)  # evidence id -> citations
    # Caches and API call counters. On Modal the web container's live DB is a read-only copy
    # that a refresh replaces, so answer() passes the ledger, the one file it may write.
    cache: sqlite3.Connection | None = None

    @property
    def writable(self) -> sqlite3.Connection:
        return self.cache if self.cache is not None else self.conn


def _num(args: dict, name: str, lo: float, hi: float, default: float | None = None) -> float:
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or not lo <= value <= hi:
        raise tools.ToolError(f"{name} must be a number between {lo:g} and {hi:g}")
    return float(value)


def _fmt(ts: datetime | str) -> str:
    return (datetime.fromisoformat(ts) if isinstance(ts, str) else ts).strftime("%Y-%m-%d %H:%M")


def _r(v, digits=1):
    return None if v is None else round(float(v), digits)


def tool_geocode(ctx: AskContext, args: dict) -> tuple[str, dict, list[dict]]:
    place = args.get("place")
    if not isinstance(place, str) or not place.strip():
        raise tools.ToolError("place must be a non-empty string")
    results = geocode.search(ctx.writable, place[:100], transport=ctx.transport)
    data: dict = {"results": results}
    if not results:
        data["note"] = "no place with this name was found"
    return "Open-Meteo geocoding (CC BY 4.0)", data, []


def tool_live_readings(ctx: AskContext, args: dict) -> tuple[str, dict, list[dict]]:
    lat, lon = _num(args, "lat", -90, 90), _num(args, "lon", -180, 180)
    radius = _num(args, "radius_km", 1, 100, 30)
    rows = ctx.conn.execute(
        "SELECT key, name, sido, lat, lon FROM stations WHERE lat IS NOT NULL AND lon IS NOT NULL"
    ).fetchall()
    near = sorted(
        (float(haversine_km(lat, lon, r[3], r[4])), r) for r in rows
    )
    stations, latest = [], None
    for dist, (key, name, sido, _la, _lo) in near:
        if dist > radius or len(stations) >= 5:
            break
        obs = ctx.conn.execute(
            "SELECT ts, pm10, pm25, o3, no2 FROM observations WHERE station_key = ? "
            "AND (pm10 IS NOT NULL OR pm25 IS NOT NULL) ORDER BY ts DESC LIMIT 1", (key,)
        ).fetchone()
        if obs is None:
            continue
        ts = datetime.fromisoformat(obs[0])
        latest = ts if latest is None or ts > latest else latest
        stations.append({
            "station": name, "province": sido, "distance_km": _r(dist), "time": _fmt(ts),
            "pm10": obs[1], "pm25": obs[2], "o3_ppm": obs[3], "no2_ppm": obs[4],
        })
    data: dict = {"radius_km": radius, "stations": stations}
    if not stations:
        data["note"] = "no AirKorea station with a reading within the radius (Korea only)"
        return SOURCE_LIVE, data, []

    def median(key):
        vals = [s[key] for s in stations if s[key] is not None]
        return _r(statistics.median(vals), 4) if vals else None

    data |= {
        "median_pm10": median("pm10"), "median_pm25": median("pm25"),
        "median_o3_ppm": median("o3_ppm"), "median_no2_ppm": median("no2_ppm"),
        "latest_time": _fmt(latest),
    }
    if ctx.now - latest > timedelta(hours=STALE_HOURS):
        data["note"] = f"the latest reading is more than {STALE_HOURS} hours old"
    detail = f"median of the {len(stations)} nearest stations (KST)"
    cites = [
        {"source": SOURCE_LIVE, "detail": f"PM10, {detail}", "value": data["median_pm10"],
         "unit": UNIT, "time": data["latest_time"]},
        {"source": SOURCE_LIVE, "detail": f"PM2.5, {detail}", "value": data["median_pm25"],
         "unit": UNIT, "time": data["latest_time"]},
    ]
    return SOURCE_LIVE, data, [c for c in cites if c["value"] is not None]


def _forecast_items(conn: sqlite3.Connection, issue_day: date) -> list[dict]:
    """The AirKorea bulletins of `issue_day`, fetched once per day (the call quota is 6/day)."""
    conn.execute("CREATE TABLE IF NOT EXISTS ask_forecast_cache (day TEXT PRIMARY KEY, items TEXT)")
    row = conn.execute(
        "SELECT items FROM ask_forecast_cache WHERE day = ?", (issue_day.isoformat(),)
    ).fetchone()
    if row:
        return json.loads(row[0])
    items = [
        {"target": f.target_date.isoformat(), "pollutant": f.pollutant, "overall": f.overall,
         "cause": f.cause, "grades": f.grades, "issued": f.issued_at.isoformat()}
        for f in airkorea.fetch_forecast(conn, issue_day)
    ]
    if items:  # an empty answer is not cached, so a later question tries again
        conn.execute(
            "INSERT OR REPLACE INTO ask_forecast_cache VALUES (?, ?)",
            (issue_day.isoformat(), json.dumps(items, ensure_ascii=False)),
        )
        conn.commit()
    return items


def tool_official_forecast(ctx: AskContext, args: dict) -> tuple[str, dict, list[dict]]:
    day = args.get("day", "today")
    if day not in ("today", "tomorrow"):
        raise tools.ToolError("day must be 'today' or 'tomorrow'")
    issue = ctx.now.astimezone(KST).date()
    target = (issue + timedelta(days=1 if day == "tomorrow" else 0)).isoformat()
    items = _forecast_items(ctx.writable, issue)
    data: dict = {"day": target}
    cites = []
    for code, key in (("PM10", "pm10"), ("PM25", "pm25")):
        # several bulletins a day: the newest one for this target
        mine = [i for i in items if i["target"] == target and i["pollutant"] == code]
        if not mine:
            continue
        top = max(mine, key=lambda i: i["issued"])
        data[key] = {k: top[k] for k in ("overall", "cause", "grades")}
        data["issued"] = _fmt(top["issued"])
        cites.append({
            "source": SOURCE_FORECAST, "detail": f"{code} forecast for {target}: "
            f"{top['overall'][:160]}", "value": None, "unit": "", "time": _fmt(top["issued"]),
        })
    if len(data) == 1:
        data["note"] = "the bulletins carry no forecast for this day (Korea only)"
    return SOURCE_FORECAST, data, cites


def tool_cams_forecast(ctx: AskContext, args: dict) -> tuple[str, dict, list[dict]]:
    lat, lon = _num(args, "lat", -90, 90), _num(args, "lon", -180, 180)
    ahead = int(_num(args, "hours_ahead", 0, 96, 0))
    length = int(_num(args, "duration_hours", 1, 24, 6))
    raw = openmeteo.fetch_cams_window(ctx.writable, lat, lon, transport=ctx.transport)
    # the API answers in the point's local time; "now" there, to the hour
    local_now = (ctx.now.astimezone(UTC) + timedelta(seconds=raw["utc_offset_seconds"])).replace(
        minute=0, second=0, microsecond=0, tzinfo=None
    )
    start = local_now + timedelta(hours=ahead)
    end = start + timedelta(hours=length)
    rows = [
        v for t, v in raw["hours"].items() if start <= datetime.fromisoformat(t) < end
    ]
    if not rows:
        raise tools.ToolError("the forecast does not cover this window (about 5 days ahead)")
    window = f"{_fmt(start)} to {_fmt(end)} local time"
    data: dict = {"lat": _r(lat, 2), "lon": _r(lon, 2), "window": window, "hours": len(rows)}
    cites = []
    for var, label, unit in (
        ("pm10", "PM10", UNIT), ("pm2_5", "PM2.5", UNIT), ("ozone", "ozone", UNIT),
        ("dust", "dust", UNIT), ("aerosol_optical_depth", "aerosol optical depth", ""),
    ):
        vals = [r[var] for r in rows if r.get(var) is not None]
        if not vals:
            continue
        data[var] = {"mean": _r(statistics.fmean(vals)), "max": _r(max(vals))}
        cites.append({
            "source": SOURCE_CAMS, "detail": f"{label}, mean over {window}",
            "value": data[var]["mean"], "unit": unit, "time": window,
        })
    return SOURCE_CAMS, data, cites


def tool_active_events(ctx: AskContext, args: dict) -> tuple[str, dict, list[dict]]:
    path = config.data_dir() / "live" / "status.json"
    try:
        status = json.loads(path.read_text())
    except (OSError, ValueError):
        return SOURCE_MONITOR, {"available": False, "note": "no monitor status is published"}, []
    if not isinstance(status, dict):
        return SOURCE_MONITOR, {"available": False, "note": "no monitor status is published"}, []
    # ponytail: the first 5 of each list, whatever fields they carry (the monitor lane owns the
    # file's shape); upgrade: pick named fields
    data: dict = {"available": True}
    for key in ("updated_at", "candidates", "reports"):
        if key in status:
            value = status[key]
            data[key] = value[:5] if isinstance(value, list) else value
    cites = [{"source": SOURCE_MONITOR, "detail": "today's anomaly candidates and reports",
              "value": None, "unit": "", "time": str(data.get("updated_at") or "")}]
    return SOURCE_MONITOR, data, cites


def _spec(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return tools._spec(name, description, properties, required)


_LAT = {"type": "number", "description": "latitude in degrees"}
_LON = {"type": "number", "description": "longitude in degrees"}

ASK_TOOLS = {
    "geocode": (tool_geocode, _spec(
        "geocode",
        "Find a place by name: up to 3 matches with lat, lon, country and admin names.",
        {"place": {"type": "string", "description": "a city or place name, e.g. 'Busan'"}},
        ["place"])),
    "live_readings": (tool_live_readings, _spec(
        "live_readings",
        "Latest PM10, PM2.5, O3 and NO2 of the nearest AirKorea stations with their times, and "
        "the median of the nearest stations. South Korea only.",
        {"lat": _LAT, "lon": _LON,
         "radius_km": {"type": "integer", "description": "search radius, 1-100 km (default 30)"}},
        ["lat", "lon"])),
    "official_forecast": (tool_official_forecast, _spec(
        "official_forecast",
        "The AirKorea official PM10 and PM2.5 forecast: overall text, cause and the grade per "
        "region (좋음 good, 보통 normal, 나쁨 bad, 매우나쁨 very bad). South Korea only.",
        {"day": {"type": "string", "enum": ["today", "tomorrow"]}})),
    "cams_forecast": (tool_cams_forecast, _spec(
        "cams_forecast",
        "CAMS model forecast for a window at a point, worldwide: mean and max of PM10, PM2.5, "
        "ozone, dust and aerosol optical depth. The window starts hours_ahead hours from now "
        "and is returned in the place's local time; check it against what was asked.",
        {"lat": _LAT, "lon": _LON,
         "hours_ahead": {"type": "integer", "description": "window start, 0-96 hours from now"},
         "duration_hours": {"type": "integer", "description": "window length, 1-24 (default 6)"}},
        ["lat", "lon"])),
    "active_events": (tool_active_events, _spec(
        "active_events",
        "Today's air-pollution anomaly candidates and investigation reports from the AirLens "
        "Agent monitor (South Korea), if it has published any.", {})),
}


def call_tool(ctx: AskContext, name: str, args: dict | None, raw: str = "") -> str:
    """Run one tool call and return the tool-message text; failures come back as 'error: ...'."""
    if name not in ASK_TOOLS:
        return f"error: unknown tool {name!r}; available: {', '.join(ASK_TOOLS)}"
    if args is None:
        return f"error: arguments are not a valid JSON object: {raw[:200]!r}"
    try:
        source, data, cites = ASK_TOOLS[name][0](ctx, args)
    except (tools.ToolError, http.HttpError, quota.QuotaExceeded, quota.BudgetExceeded,
            config.MissingKeyError) as exc:
        return f"error: {exc}"
    except Exception as exc:  # a tool bug must not abort the paid run
        return f"error: {name} failed: {type(exc).__name__}: {http.redact(str(exc))[:200]}"
    shown = {k: (v[:60] if isinstance(v, str) else v) for k, v in args.items()}
    item = ctx.evidence.add(name, shown, source, data)
    ctx.cites[item["evidence_id"]] = cites
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str)


# --- prompts ----------------------------------------------------------------------------------

SYSTEM = (
    "You answer citizens' questions about air quality using tools. You give air-quality risk "
    "information only, never health or medical advice. Use only the tools' results; call only "
    "the tools you need (geocode first for a named place; live_readings and official_forecast "
    "cover South Korea only, cams_forecast works worldwide). Evidence ids: E1 holds the AirKorea "
    "grade bands, tool results are numbered after it. Questions such as 'should I run, "
    "exercise, wear a mask or take medicine' are personal health decisions: do not recommend or "
    "advise anything; report the forecast values and grades, state what the official grade "
    "definitions in E1 mean, and say that personal decisions should follow official guidance "
    "from local health authorities. If the question is not about air quality, call no tool and "
    "decline briefly. Write in English, plain sentences, no markdown. Every number you write "
    "must be copied from the evidence. Say what you could not check."
)
FINAL_REQUEST = (
    "Write the final answer now as JSON: 2 to 4 plain English sentences with the asked time "
    "window, the risk level from the grade bands in E1, a confidence from 0 to 1 (lower when "
    "data are missing, stale or only a model forecast), the evidence ids you used, and "
    "limitations. Copy numbers exactly from the evidence. No personal recommendation."
)


def _bands_item(ctx: AskContext) -> None:
    """E1: the grade definitions, so answers can quote them and the number check accepts them."""
    data = {
        "pm10_ug_m3_upper_edges": dict(zip(LEVELS, BANDS["pm10"], strict=False)),
        "pm25_ug_m3_upper_edges": dict(zip(LEVELS, BANDS["pm25"], strict=False)),
        "meaning": "good (좋음), moderate (보통, normal), unhealthy (나쁨, bad), "
        "very_unhealthy (매우나쁨, very bad); the highest grade has no upper edge",
    }
    item = ctx.evidence.add("grade_bands", {}, SOURCE_BANDS, data)
    ctx.cites[item["evidence_id"]] = [{
        "source": SOURCE_BANDS,
        "detail": "PM10 (ug/m3) good up to 30, moderate up to 80, unhealthy up to 150; "
        "PM2.5 good up to 15, moderate up to 35, unhealthy up to 75; above that very unhealthy",
        "value": None, "unit": "", "time": "",
    }]


# --- steps ------------------------------------------------------------------------------------

_STEP_ERRORS = agent._STEP_ERRORS


def _why(exc: Exception) -> str:
    return agent._why(exc)


def _checked_sentences(text: str, evidence: tools.Evidence) -> tuple[list[str], list[str]]:
    """(kept, dropped) sentences: dropped = a personal recommendation, or a number that is not in
    the evidence (within 5 % or rounding, dates and times ignored, as in `agent.verify`)."""
    pool = [n for item in evidence.items for n in agent._evidence_numbers(item["data"])]
    kept, dropped = [], []
    for sentence in (s.strip() for s in _SENTENCE.split(text.strip())):
        if not sentence:
            continue
        numbers = agent._numbers(sentence, skip_dates=True)
        bad_number = any(not agent._matches(v, d, pct, pool) for v, d, pct in numbers)
        (dropped if bad_number or _ADVICE.search(sentence) else kept).append(sentence)
    return kept, dropped


def _build(draft: AnswerDraft, ctx: AskContext, models: list[str], cost: float) -> Answer:
    base = {"models": list(dict.fromkeys(models)), "cost_usd": round(cost, 6)}
    if not draft.in_scope:
        return Answer(answer=REFUSAL, limitations=[SAFETY_NOTE], **base)
    items = ctx.evidence.items
    cited = [i for i in dict.fromkeys(draft.evidence_ids) if ctx.evidence.get(i)]
    used = [i for i in items if i["evidence_id"] in cited] or items
    kept, dropped = _checked_sentences(draft.answer, ctx.evidence)
    limitations = [s.strip() for s in draft.limitations if s.strip()]
    if dropped:
        limitations.append(
            f"{len(dropped)} sentence(s) were removed: a number could not be matched to the "
            "evidence, or the sentence was a personal recommendation."
        )
    answer = " ".join(kept[:MAX_SENTENCES]) or NOTHING_VERIFIED
    risk = derive_risk(used) or derive_risk(items)
    if risk is None and any(i["tool"] == "official_forecast" for i in used):
        risk = draft.risk_level  # grades per region: the model's reading of them
    confidence = min(max(draft.confidence, 0.0), 1.0)
    if risk is None:
        risk, confidence = "unknown", min(confidence, 0.3)
    if dropped or not kept:
        confidence = min(confidence, 0.5)
    cites = [
        Citation(**c)
        for i in used if i["tool"] != "grade_bands" or i["evidence_id"] in cited
        for c in ctx.cites.get(i["evidence_id"], [])
    ]
    return Answer(
        answer=answer, risk_level=risk, time_window=draft.time_window[:120],
        confidence=round(confidence, 2), citations=cites,
        limitations=[*limitations, SAFETY_NOTE], **base,
    )


def _run(question: str, *, ledger, ctx: AskContext, lat, lon) -> Answer:
    before = quota.spent_total(ledger)
    _bands_item(ctx)
    where = f"\nThe user's location: lat {lat}, lon {lon}." if lat is not None else ""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"Current time: {ctx.now.astimezone(KST).strftime('%Y-%m-%d %H:%M')} KST "
            f"({ctx.now.astimezone(UTC).strftime('%Y-%m-%d %H:%M')} UTC).{where}\n"
            f"Question: {question}"
        )},
    ]
    specs = [spec for _, spec in ASK_TOOLS.values()]
    models: list[str] = []
    for _ in range(MAX_STEPS):
        res = llm.chat(
            "meta", messages, tools=specs, reasoning_effort=agent.EFFORT,
            max_tokens=agent.MAX_TOKENS, ledger=ledger, transport=ctx.transport,
        )
        models.append(res.model)
        ids = [tc["id"] or f"call_{n}" for n, tc in enumerate(res.tool_calls)]
        if res.tool_calls or res.content.strip():  # an empty reply would be an invalid message
            message = res.message
            if res.tool_calls:
                message = {**message, "tool_calls": [
                    {**raw, "id": cid} for raw, cid in zip(message["tool_calls"], ids, strict=True)
                ]}
            messages.append(message)
        if not res.tool_calls:
            break
        for n, (tc, cid) in enumerate(zip(res.tool_calls, ids, strict=True)):
            content = (
                call_tool(ctx, tc["name"], tc["arguments"], tc["raw_arguments"])
                if n < MAX_CALLS_PER_STEP else "error: too many tool calls in one step"
            )
            messages.append({"role": "tool", "tool_call_id": cid, "content": content})
    else:
        messages.append({"role": "user", "content": "Step limit reached; stop calling tools."})
    messages.append({"role": "user", "content": FINAL_REQUEST})
    draft, res = llm.json_chat(
        "meta", messages, AnswerDraft, schema_name="answer", reasoning_effort=agent.EFFORT,
        max_tokens=agent.MAX_TOKENS, ledger=ledger, transport=ctx.transport,
    )
    models.append(res.model)
    return _build(draft, ctx, models, quota.spent_total(ledger) - before)


# --- examples and fallbacks -------------------------------------------------------------------

EXAMPLES = (
    ("Seoul", "Is the air OK for a run in Seoul tomorrow morning?"),
    ("Busan", "How is the air in Busan today?"),
    ("Tokyo", "What will the air quality be like in Tokyo tomorrow?"),
    ("Los Angeles", "What is the air quality forecast for Los Angeles this evening?"),
    ("Korea", "Is there any dust or smoke alert in Korea today?"),
)
EXAMPLE_QUESTIONS = [q for _, q in EXAMPLES]


def examples_path():
    return config.data_dir() / "ask" / "examples.json"


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _cached(question: str, reason: str) -> Answer:
    """A saved example answer for the question's place (else a generic notice), cached=True."""
    best, score = None, 0
    try:
        saved = json.loads(examples_path().read_text()).get("examples", [])
    except (OSError, ValueError, AttributeError):
        saved = []
    for ex in saved:
        place = str(ex.get("place", "")).lower()
        if not place or place not in question.lower():
            continue
        s = 1 + len(_words(question) & _words(str(ex.get("question", ""))))
        if s > score:
            try:
                best, score = Answer.model_validate(ex["answer"]), s
            except (KeyError, ValueError):
                continue
    if best is not None:
        note = f"A saved example answer, not a live reading ({reason})."
        return best.model_copy(update={
            "cached": True, "cost_usd": 0.0, "limitations": [*best.limitations, note],
        })
    return Answer(
        answer="The daily question limit has been reached or the assistant is unavailable "
        "right now. Please try again later, or look at the example answers on this page.",
        limitations=[reason, SAFETY_NOTE], cached=True,
    )


def build_examples(*, ledger, live_conn, now=None, transport=None) -> dict:
    """Run the example questions for real (paid) and save them as examples.json.

    A question whose answer fell back to a cached one (a limit, a failure) is not saved.
    """
    saved, skipped = [], []
    for place, question in EXAMPLES:
        result = answer(question, ledger=ledger, live_conn=live_conn, now=now, transport=transport)
        if result.cached:
            skipped.append(question)
            continue
        saved.append({"place": place, "question": question, "answer": result.model_dump()})
    if saved:
        path = examples_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"generated": datetime.now(KST).isoformat(timespec="minutes"), "examples": saved},
            ensure_ascii=False, indent=2,
        ) + "\n")
    return {"saved": len(saved), "skipped": skipped, "path": examples_path().name}


# --- entry point ------------------------------------------------------------------------------


def answer(
    question: str,
    *,
    ledger: sqlite3.Connection,
    live_conn: sqlite3.Connection,
    now: datetime | None = None,
    lat: float | None = None,
    lon: float | None = None,
    transport=None,
) -> Answer:
    """Answer one citizen question; never raises for a limit or a model failure.

    A question over 300 characters or empty is refused without using the daily cap. After the
    cap, a spent budget or a failed model call, a saved example answer (cached=True) is returned.
    """
    question = " ".join(question.split())
    if not question:
        return Answer(answer="Please ask a question about air quality.")
    if len(question) > MAX_QUESTION_CHARS:
        return Answer(
            answer=f"Please keep the question under {MAX_QUESTION_CHARS} characters."
        )
    try:
        quota.reserve(ledger, OP)
    except quota.QuotaExceeded as exc:
        return _cached(question, f"daily limit: {exc}")
    ctx = AskContext(live_conn, now or datetime.now(KST), transport, cache=ledger)
    try:
        return _run(question, ledger=ledger, ctx=ctx, lat=lat, lon=lon)
    except _STEP_ERRORS as exc:
        return _cached(question, f"assistant unavailable: {_why(exc)}")
