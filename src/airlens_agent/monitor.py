"""Hourly live monitor: fetch today's data, detect anomalies, investigate the worst new event.

Everything is capped: AirKorea calls (quota.QUOTA_CAPS), model money per day and all-time
(quota.ensure_budget), and investigations per KST day (config.MONITOR_INVESTIGATIONS_PER_DAY).
The outcome, including why nothing was investigated, goes to data_dir()/live/status.json.
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

from . import agent, config, detect, http, quota, store, tools
from .config import KST
from .models import Candidate
from .sources import airkorea, firms, gibs, openmeteo

# Detect's severity is max(pm10/151, pm25/76) * (1 + ln n): 1.0 is "very bad" at one station.
# A regional candidate (>= 3 stations) at "very bad" scores 1 * (1 + ln 3) = 2.1, so 2.0 keeps
# roughly "very bad grade at three or more stations" and skips the z-score-only "bad" events
# (about 1.1) that a paid investigation would not be worth.
MIN_SEVERITY = 2.0
FOCUS_MIN_DEG = 2.5  # like the replay focus boxes: the model grid and images cover this much
GRID_STEP_DEG = 0.5
HISTORY_HOURS = 12  # wind and CAMS samples before the candidate's hour
SOURCE_ERRORS = (http.HttpError, config.MissingKeyError, quota.QuotaExceeded)


def _say(message: str) -> None:
    print(message, file=sys.stderr)


def live_dir() -> Path:
    return config.data_dir() / "live"


def _write_json(path: Path, obj) -> None:
    """Write via a temp file and rename, so the web container never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def focus_bbox(cand: Candidate) -> tuple[float, float, float, float]:
    """The candidate bbox padded to at least FOCUS_MIN_DEG on each side-to-side span."""
    w, s, e, n = cand.bbox
    cx, cy = (w + e) / 2, (s + n) / 2
    half_x = max(FOCUS_MIN_DEG / 2, (e - w) / 2 + 0.25)
    half_y = max(FOCUS_MIN_DEG / 2, (n - s) / 2 + 0.25)
    return (cx - half_x, cy - half_y, cx + half_x, cy + half_y)


def _report_path(event_id: str) -> Path:
    return live_dir() / "reports" / f"{event_id}.json"


def _skip(skips: list, reason: str, detail: str, event_id: str | None = None) -> None:
    _say(f"[monitor] skip {reason}: {detail}")
    skips.append({"reason": reason, "detail": detail, "event_id": event_id})


def _step(name: str, fn, sources: dict, skips: list):
    """Run one source step; a failed source is recorded and skipped, never fatal."""
    try:
        out = fn()
    except SOURCE_ERRORS as exc:
        sources[name] = f"failed: {exc}"
        _skip(skips, "source_failed", f"{name}: {exc}")
        return None
    sources[name] = "ok"
    return out


def _fetch(conn: sqlite3.Connection, sources: dict, skips: list) -> None:
    """AirKorea nation-wide realtime (plus the station list once) and FIRMS hotspots."""
    if conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0] == 0:
        _step("airkorea.stations", lambda: airkorea.fetch_stations(conn), sources, skips)
    obs = _step("airkorea.realtime", lambda: airkorea.fetch_realtime_nation(conn), sources, skips)
    if obs is not None:
        store.upsert_observations(conn, obs)
    _step("firms", lambda: firms.fetch_hotspots(conn, firms.WATCH_BBOX), sources, skips)


def _gather_inputs(conn: sqlite3.Connection, cand: Candidate, img_dir: Path) -> dict:
    """Wind and CAMS samples around the candidate and its true-colour images, into the DB and
    image folder the agent tools read. A failed source is noted; the tools then say what is
    missing."""
    bbox = focus_bbox(cand)
    points = openmeteo.grid_points(bbox, GRID_STEP_DEG)
    start = cand.ts - timedelta(hours=HISTORY_HOURS)
    end = cand.ts + timedelta(hours=tools.CLAMP_HOURS)
    sources: dict = {}
    skips: list = []
    for label, fn in (("wind", openmeteo.fetch_wind), ("cams", openmeteo.fetch_air_quality)):
        _step(label, lambda fn=fn: fn(conn, points, start, end), sources, skips)
    # the bbox slug is part of the cache key: two events on one day have different boxes
    slug = gibs.bbox_slug(bbox)
    # ponytail: only the two newest passes, as cli._ensure_images; upgrade: walk further back
    # while both come back blank
    for day, layer in tools.image_passes(cand)[:2]:
        if any(img_dir.glob(f"{day.isoformat()}_{layer}_{slug}.*")):
            continue
        _step(
            f"gibs {day} {layer}",
            lambda day=day, layer=layer: gibs.fetch_image(
                bbox, day, layer, out_dir=img_dir, conn=conn
            ),
            sources, skips,
        )
    return sources


def _investigate(conn: sqlite3.Connection, cand: Candidate) -> dict:
    """Triage and investigate `cand` as cli.cmd_investigate does; write its report JSON."""
    root = live_dir()
    img_dir = root / "images"
    inputs = _gather_inputs(conn, cand, img_dir)
    before = quota.spent_total(conn)
    tri = agent.triage(cand, ledger=conn)
    report = agent.investigate(
        conn, cand, tri, ledger=conn, investigator="meta", image_dir=img_dir,
        out_dir=root / "reports",
    )
    report.models = list(dict.fromkeys(m for m in (tri.model_used, *report.models) if m))
    report.cost_usd = round(quota.spent_total(conn) - before, 6)  # triage included
    out = _report_path(cand.event_id)
    _write_json(out, report.model_dump(mode="json"))
    total = len(report.claims)
    return {
        "event_id": cand.event_id, "report": out.name, "cause": report.cause,
        "confidence": report.confidence, "cost_usd": report.cost_usd,
        "claims_verified": f"{total - len(report.unverified_claims)}/{total}",
        "fallback": report.fallback, "inputs": inputs,
    }


def _pick(candidates: list[Candidate], limit: int, skips: list) -> list[Candidate]:
    """Strongest regional candidates with no report yet, at most `limit`."""
    regional = [c for c in detect.select_for_investigation(candidates) if c.kind == "regional"]
    strong = [c for c in regional if c.severity >= MIN_SEVERITY]
    if regional and not strong:
        _skip(skips, "weak_candidates",
              f"{len(regional)} regional candidate(s) below severity {MIN_SEVERITY}")
    fresh = [c for c in strong if not _report_path(c.event_id).exists()]
    fresh.sort(key=lambda c: -c.severity)
    if not fresh:
        _skip(skips, "no_candidate", "no new regional candidate to investigate")
    return fresh[:limit]


def _candidate_row(c: Candidate) -> dict:
    return {
        "event_id": c.event_id, "ts": c.ts.isoformat(), "kind": c.kind, "hint": c.hint,
        "centroid": [round(c.centroid_lat, 3), round(c.centroid_lon, 3)],
        "bbox": [round(v, 3) for v in c.bbox],
        "peak_pm10": c.peak_pm10, "peak_pm25": c.peak_pm25, "severity": round(c.severity, 2),
        "n_stations": len(c.station_keys),
        "report": _report_path(c.event_id).name if _report_path(c.event_id).exists() else None,
    }


def _stations_reporting(conn: sqlite3.Connection, last: datetime | None) -> int:
    if last is None:
        return 0
    row = conn.execute(
        "SELECT COUNT(DISTINCT station_key) FROM observations WHERE ts = ?", (last.isoformat(),)
    ).fetchone()
    return int(row[0])


def run_once(
    conn: sqlite3.Connection | None = None,
    *,
    now: datetime | None = None,
    investigate: bool = True,
    max_investigations: int = 1,
) -> dict:
    """One monitor pass; returns (and writes to live/status.json) what happened.

    ponytail: only the latest observation hour is detected; an hour the job missed is not
    back-filled (detection is stateless per hour, the next run sees the new latest hour).
    """
    own = conn is None
    conn = conn or store.connect(config.live_db_path())
    now = (now or datetime.now(KST)).astimezone(KST)
    sources: dict = {}
    skips: list = []
    done: list = []
    try:
        _fetch(conn, sources, skips)
        candidates = detect.detect_latest(conn)
        if investigate:
            for cand in _pick(candidates, max_investigations, skips):
                try:
                    quota.ensure_budget(conn, "meta")
                    quota.reserve(conn, "monitor.investigate")
                except quota.BudgetExceeded as exc:
                    _skip(skips, "budget_exceeded", str(exc), cand.event_id)
                    break
                except quota.QuotaExceeded as exc:
                    _skip(skips, "cap_reached", str(exc), cand.event_id)
                    break
                try:
                    done.append(_investigate(conn, cand))
                except quota.BudgetExceeded as exc:
                    _skip(skips, "budget_exceeded", str(exc), cand.event_id)
                    break
                except quota.QuotaExceeded as exc:
                    _skip(skips, "cap_reached", str(exc), cand.event_id)
                    break
                except Exception as exc:  # one failed event must not stop the job
                    _skip(skips, "investigation_failed", f"{type(exc).__name__}: {exc}",
                          cand.event_id)
        last = store.last_observation_ts(conn)
        status = {
            "ran_at": now.isoformat(),
            "last_observation": last.isoformat() if last else None,
            "stations_reporting": _stations_reporting(conn, last),
            "candidates": [_candidate_row(c) for c in candidates],
            "investigations": done,
            "skips": skips,
            "sources": sources,
            "budget": quota.budget_status(conn),
            "quota": quota.usage(conn),
        }
        _write_json(live_dir() / "status.json", status)
        return status
    finally:
        if own:
            conn.close()
