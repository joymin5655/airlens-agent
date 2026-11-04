"""Public web server: JSON API over the saved investigations, the question endpoint, the site.

Modal contract (deploy/README.md): the hourly job is the only writer of live.sqlite and live/;
this process opens them read-only and charges model spend to config.ledger_db_path().
"""

import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ask, config, monitor, quota, store

log = logging.getLogger(__name__)
app = FastAPI(title="AirLens Agent")

# The two curated past cases (replayed with real data); name -> report id and display text.
PAST_CASES: dict[str, dict] = {
    "fire-2025-03": {
        "report": "evt-2025032421-0",
        "title": "Gyeongbuk wildfire smoke",
        "region": "Uiseong to Yeongdeok, North Gyeongsang",
    },
    "dust-2026-02": {
        "report": "evt-2026022210-1",
        "title": "Seoul metropolitan dust",
        "region": "Seoul, Gyeonggi and the west coast",
    },
}
_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
IMAGE_SUFFIXES = {".jpg": "image/jpeg", ".png": "image/png"}
MEDIA_DIRS = ("events/*/images", "events/*/reports", "live/images", "live/reports")
QUESTION_MAX = 300
PER_MINUTE, PER_DAY = 5, 30


# --- small readers ------------------------------------------------------------------------------


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _ro(path: Path) -> sqlite3.Connection | None:
    """A read-only connection, or None when the file is missing or not a database."""
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        return conn
    except sqlite3.Error:
        return None


def _live_dir() -> Path:
    return config.data_dir() / "live"


def _past_dir(name: str) -> Path:
    return config.data_dir() / "events" / name


def _media_url(path: Path) -> str:
    return "/media/" + path.relative_to(config.data_dir()).as_posix()


def _report_at(path: Path) -> dict | None:
    rep = _read_json(path)
    return rep if isinstance(rep, dict) and "evidence" in rep else None


# --- events -------------------------------------------------------------------------------------


def _candidate_from_db(conn: sqlite3.Connection | None, event_id: str) -> dict | None:
    if conn is None:
        return None
    try:
        row = conn.execute("SELECT * FROM candidates WHERE event_id = ?", (event_id,)).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    cand = dict(row)
    cand["bbox"] = json.loads(cand["bbox"])
    cand["station_keys"] = json.loads(cand.get("station_keys") or "[]")
    cand.pop("features", None)
    return cand


def _candidate_from_status(row: dict) -> dict:
    lat, lon = row["centroid"]
    return {
        "event_id": row["event_id"], "ts": row["ts"], "kind": row["kind"], "hint": row.get("hint"),
        "centroid_lat": lat, "centroid_lon": lon, "bbox": row["bbox"],
        "peak_pm10": row.get("peak_pm10"), "peak_pm25": row.get("peak_pm25"),
        "severity": row.get("severity"), "station_keys": [],
    }


def _status() -> dict | None:
    st = _read_json(_live_dir() / "status.json")
    return st if isinstance(st, dict) else None


def _why_no_report(status: dict, row: dict) -> str:
    for skip in status.get("skips", []):
        if skip.get("event_id") == row["event_id"]:
            reason = skip.get("reason")
            if reason == "cap_reached":
                return "The daily limit on investigations was reached."
            if reason == "budget_exceeded":
                return "The daily model budget was reached."
            if reason == "investigation_failed":
                return "The investigation failed and will be retried."
    if row["kind"] != "regional":
        return "Only regional events (three or more stations) are investigated."
    if (row.get("severity") or 0) < monitor.MIN_SEVERITY:
        return "Too mild to investigate: below the severity threshold."
    return "Queued for the next hourly run."


def _past_item(name: str) -> dict | None:
    meta = PAST_CASES[name]
    rep = _report_at(_past_dir(name) / "reports" / f"{meta['report']}.json")
    conn = _ro(config.event_db_path(name))
    try:
        cand = _candidate_from_db(conn, meta["report"])
    finally:
        if conn:
            conn.close()
    if rep is None or cand is None:
        return None
    return _item("past", name, meta["title"], meta["region"], cand, rep)


def _item(kind: str, id_: str, title: str, region: str, cand: dict, rep: dict | None) -> dict:
    return {
        "kind": kind, "id": id_, "title": title, "region": region, "ts": cand["ts"],
        "cause": rep["cause"] if rep else None,
        "confidence": rep["confidence"] if rep else None,
        "headline": rep["headline"] if rep else None,
        "has_report": rep is not None,
        "candidate": {
            k: cand.get(k) for k in ("kind", "hint", "peak_pm10", "peak_pm25", "severity")
        } | {"n_stations": len(cand["station_keys"]) or cand.get("n_stations")},
    }


def _live_items() -> list[dict]:
    status = _status() or {}
    rows = {r["event_id"]: r for r in status.get("candidates", [])}
    conn = _ro(config.live_db_path())
    items = []
    try:
        ids = list(rows)
        for p in sorted((_live_dir() / "reports").glob("*.json")):
            if p.stem not in rows and _ID.match(p.stem):
                ids.append(p.stem)
        for event_id in ids:
            row = rows.get(event_id)
            cand = _candidate_from_db(conn, event_id) or (
                _candidate_from_status(row) if row else None
            )
            if cand is None:
                continue
            rep = _report_at(_live_dir() / "reports" / f"{event_id}.json")
            lat, lon = cand["centroid_lat"], cand["centroid_lon"]
            item = _item("live", event_id, rep["headline"] if rep else _hint_title(cand),
                         f"Around {lat:.2f}N {lon:.2f}E", cand, rep)
            if rep is None and row is not None:
                item["why_no_report"] = _why_no_report(status, row)
            items.append(item)
    finally:
        if conn:
            conn.close()
    items.sort(key=lambda i: i["ts"], reverse=True)
    return items


def _hint_title(cand: dict) -> str:
    return {"coarse": "Coarse dust episode", "fine": "Fine particle episode"}.get(
        cand.get("hint") or "", "Pollution event"
    )


@app.get("/api/events")
def events() -> dict:
    past = [i for i in (_past_item(n) for n in PAST_CASES) if i]
    return {"live": _live_items(), "past": past}


# --- event detail -------------------------------------------------------------------------------


def _bbox_from_name(name: str) -> list[float] | None:
    try:
        return [float(v) for v in Path(name).stem.split("_")[-4:]]
    except ValueError:
        return None


def _images(img_dir: Path, want: tuple | None) -> list[dict]:
    """Satellite images in a folder; `want` (a bbox) keeps only images of that box."""
    out = []
    for p in sorted(img_dir.glob("*")):
        if p.suffix not in IMAGE_SUFFIXES:
            continue
        bbox = _bbox_from_name(p.name)
        if bbox is None or len(bbox) != 4:
            continue
        if want and any(abs(a - b) > 0.01 for a, b in zip(bbox, want, strict=True)):
            continue
        out.append({
            "file": _media_url(p), "date": p.name[:10],
            "layer": "_".join(p.stem.split("_")[1:-4]), "bbox": bbox,
        })
    return out


def _scene(db_path: Path, img_dir: Path, rep_dir: Path, cand: dict, rep: dict,
           focus: tuple | None) -> dict:
    """Stations, hotspots, images and overlay the scene draws for one candidate."""
    images = _images(img_dir, focus)
    box = images[0]["bbox"] if images else list(focus or cand["bbox"])
    w, s, e, n = box
    ts = datetime.fromisoformat(cand["ts"])
    conn = _ro(db_path)
    stations, hotspots, at = [], [], None
    if conn is not None:
        try:
            at = conn.execute("SELECT MAX(ts) FROM observations WHERE ts <= ?",
                              (cand["ts"],)).fetchone()[0]
            if at is None:
                at = conn.execute("SELECT MIN(ts) FROM observations").fetchone()[0]
            keys = set(cand.get("station_keys") or [])
            cw, cs, ce, cn = cand["bbox"]
            for r in conn.execute(
                "SELECT s.key, s.lat, s.lon, o.pm10, o.pm25 FROM observations o "
                "JOIN stations s ON s.key = o.station_key WHERE o.ts = ? AND s.lat BETWEEN ? AND ? "
                "AND s.lon BETWEEN ? AND ?", (at, s, n, w, e)):
                st = {k: r[k] for k in ("lat", "lon", "pm10", "pm25")}
                st["in_event"] = r["key"] in keys if keys else (cw <= r["lon"] <= ce
                                                                and cs <= r["lat"] <= cn)
                stations.append(st)
            lo = (ts - timedelta(hours=24)).astimezone(UTC).isoformat()
            hi = ts.astimezone(UTC).isoformat()
            hotspots = [dict(r) for r in conn.execute(
                "SELECT lat, lon, frp, ts_utc FROM hotspots WHERE lat BETWEEN ? AND ? "
                "AND lon BETWEEN ? AND ? AND ts_utc >= ? AND ts_utc <= ? "
                "ORDER BY frp DESC LIMIT 3000", (s, n, w, e, lo, hi))]
        except sqlite3.Error:
            pass  # a DB without these tables shows the scene without the layers
        finally:
            conn.close()
    overlay = None
    for ev in rep.get("evidence", []):
        data = ev.get("data") or {}
        name = data.get("overlay_file") if ev.get("tool") == "satellite_segments" else None
        if name and (rep_dir / Path(name).name).is_file():
            overlay = _media_url(rep_dir / Path(name).name)
    return {"observed_at": at, "box": box, "stations": stations, "hotspots": hotspots,
            "images": images, "overlay": overlay}


@app.get("/api/events/{kind}/{event_id}")
def event_detail(kind: str, event_id: str) -> dict:
    if kind not in ("past", "live") or not _ID.match(event_id):
        raise HTTPException(404, "No such case.")
    if kind == "past":
        meta = PAST_CASES.get(event_id)
        item = _past_item(event_id) if meta else None
        if item is None:
            raise HTTPException(404, "No such case.")
        base = _past_dir(event_id)
        rep = _report_at(base / "reports" / f"{meta['report']}.json")
        conn = _ro(config.event_db_path(event_id))
        try:
            cand = _candidate_from_db(conn, meta["report"])
        finally:
            conn.close()
        scene = _scene(config.event_db_path(event_id), base / "images", base / "reports",
                       cand, rep, None)
    else:
        item = next((i for i in _live_items() if i["id"] == event_id), None)
        if item is None:
            raise HTTPException(404, "No such case.")
        conn = _ro(config.live_db_path())
        try:
            cand = _candidate_from_db(conn, event_id)
        finally:
            if conn:
                conn.close()
        row = next((r for r in (_status() or {}).get("candidates", [])
                    if r["event_id"] == event_id), None)
        cand = cand or _candidate_from_status(row)
        rep = _report_at(_live_dir() / "reports" / f"{event_id}.json")
        scene = _scene(config.live_db_path(), _live_dir() / "images", _live_dir() / "reports",
                       cand, rep or {}, tuple(monitor.focus_bbox(_as_candidate(cand))))
    return {**item, "report": rep, "candidate": cand, **scene}


def _as_candidate(cand: dict):
    """Just enough of a Candidate for monitor.focus_bbox (it reads only .bbox)."""
    class _Box:
        bbox = tuple(cand["bbox"])
    return _Box()


# --- media --------------------------------------------------------------------------------------


@app.get("/media/{path:path}")
def media(path: str) -> FileResponse:
    root = config.data_dir().resolve()
    target = (root / path).resolve()
    allowed = [d for pat in MEDIA_DIRS for d in root.glob(pat)]
    if (
        target.suffix.lower() not in IMAGE_SUFFIXES
        or not target.is_file()
        or not any(target.parent == d.resolve() for d in allowed)
    ):
        raise HTTPException(404, "Not found.")
    return FileResponse(
        target, media_type=IMAGE_SUFFIXES[target.suffix.lower()],
        headers={"Cache-Control": "public, max-age=3600"},
    )


# --- status, eval -------------------------------------------------------------------------------


def _ledger() -> sqlite3.Connection:
    return store.connect(config.ledger_db_path())


@app.get("/api/status")
def status() -> dict:
    ledger = _ledger()
    try:
        live = _status() or {
            "ran_at": None, "candidates": [], "note": "The monitor has not run yet.",
        }
        return {
            "live": live,
            "budget": quota.budget_status(ledger),
            "ask": quota.usage(ledger).get(ask.OP, {}),
        }
    finally:
        ledger.close()


@app.get("/api/eval")
def eval_latest() -> dict:
    data = _read_json(config.data_dir() / "eval" / "latest.json")
    if data is None:
        raise HTTPException(404, "No comparison has been run yet.")
    return data


# --- ask ----------------------------------------------------------------------------------------

# ponytail: in-memory per-process limits keyed by client IP, reset on restart and not shared
# between containers (Modal runs one web container). Upgrade: a counter table in the ledger DB.
_hits: dict[str, deque] = defaultdict(deque)


def _client(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _rate_limited(client: str, now: float) -> str | None:
    hits = _hits[client]
    while hits and now - hits[0] > 86400:
        hits.popleft()
    if len(hits) >= PER_DAY:
        return "You have reached the daily limit of questions. Please try again tomorrow."
    if sum(1 for t in hits if now - t <= 60) >= PER_MINUTE:
        return "Too many questions in a minute. Please wait a little and try again."
    hits.append(now)
    return None


class AskIn(BaseModel):
    question: str
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)


def _error(code: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=code)


def _live_copy() -> sqlite3.Connection:
    """The live DB copied into memory: the ask tools write caches (geocode, forecast, call
    counts) to their connection, which must never reach the read-only file.

    ponytail: copies the whole file per question (about 1.3 MB, 1.5 MB more per day); upgrade:
    an attached in-memory cache schema.
    """
    mem = sqlite3.connect(":memory:", check_same_thread=False)
    mem.executescript(store.SCHEMA)
    src = _ro(config.live_db_path())
    if src is not None:
        try:
            src.backup(mem)
        except sqlite3.Error:
            pass
        finally:
            src.close()
    return mem


@app.post("/api/ask")
def ask_endpoint(body: AskIn, request: Request):
    question = " ".join(body.question.split())
    if not question:
        return _error(422, "Please type a question.")
    if len(question) > QUESTION_MAX:
        return _error(422, f"Please keep the question under {QUESTION_MAX} characters.")
    if (limited := _rate_limited(_client(request), time.time())) is not None:
        return _error(429, limited)
    try:
        ledger, live = _ledger(), _live_copy()
        try:
            result = ask.answer(question, ledger=ledger, live_conn=live, lat=body.lat, lon=body.lon)
        finally:
            ledger.close()
            live.close()
    except Exception:  # never show a stack trace; the log has it
        log.exception("ask failed")
        return _error(502, "The agent could not answer right now. Please try again in a minute.")
    return result.model_dump()


# --- health, legacy -----------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    conn = store.connect(config.live_db_path())
    try:
        last = store.last_observation_ts(conn)
        return {
            "status": "ok",
            "last_observation": last.isoformat() if last else None,
            "quota": quota.usage(conn),
        }
    finally:
        conn.close()


@app.get("/anomalies")
def anomalies(limit: int = Query(20, ge=1, le=100)) -> list[dict]:
    conn = store.connect(config.live_db_path())
    try:
        return store.latest_candidates(conn, limit)
    finally:
        conn.close()


# --- static site (last, so the API routes win) --------------------------------------------------


def web_dir() -> Path | None:
    """web/ next to the checkout, else /root/web (the Modal image), else AIRLENS_WEB_DIR."""
    for p in (os.environ.get("AIRLENS_WEB_DIR"), Path(__file__).resolve().parents[2] / "web",
              "/root/web"):
        if p and Path(p, "index.html").is_file():
            return Path(p)
    return None


if (_web := web_dir()) is not None:
    app.mount("/", StaticFiles(directory=_web, html=True), name="site")
