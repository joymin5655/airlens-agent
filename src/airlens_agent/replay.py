"""Past-event data download and replay."""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

from . import config, detect, store
from .config import KST
from .events import Event, get_event
from .models import Candidate
from .sources import airkorea_archive, firms, gibs, openmeteo

FIRMS_MARGIN_DEG = 2.0
GRID_STEP_DEG = 0.5
IMAGE_LAYERS = (gibs.TRUE_COLOR_LAYERS[0], gibs.AEROSOL_LAYERS[0])


def _say(message: str) -> None:
    print(message, file=sys.stderr)


def _months(start: datetime, end: datetime) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _expand(
    bbox: tuple[float, float, float, float], deg: float
) -> tuple[float, float, float, float]:
    w, s, e, n = bbox
    return (w - deg, s - deg, e + deg, n + deg)


def _step(label: str, fn) -> object:
    """Run a connector step; a not-yet-built one is skipped so it can be re-run later."""
    try:
        return fn()
    except NotImplementedError:
        _say(f"[{label}] connector not built yet, skipped (re-run fetch-event later)")
        return None


def _load_archive(ev: Event, conn: sqlite3.Connection) -> dict:
    """Steps 2: download, extract and store the needed AirKorea months inside the window."""
    raw_dir = config.data_dir() / "raw" / "airkorea"
    seen, n_obs = {}, 0
    for year, month in _months(ev.start, ev.end):
        zip_path = airkorea_archive.download_year(year, raw_dir, conn)
        xlsx = airkorea_archive.extract_month(zip_path, year, month, raw_dir)
        obs, stations = airkorea_archive.parse_frame(airkorea_archive.read_frame(xlsx))
        obs = [o for o in obs if ev.start <= o.ts <= ev.end]
        n_obs += store.upsert_observations(conn, obs)
        seen.update({s.key: s for s in stations})
    report = airkorea_archive.merge_archive_stations(conn, list(seen.values()))
    report["observations"] = n_obs
    return report


def fetch_event(name: str) -> dict:
    """Download and store all data for event `name` into config.event_db_path(name)."""
    ev = get_event(name)
    live = store.connect(config.live_db_path())
    if live.execute("SELECT COUNT(*) FROM stations").fetchone()[0] == 0:
        raise SystemExit("live DB has no stations: run `airlens-agent stations` first")
    conn = store.connect(config.event_db_path(name))
    airkorea_archive.copy_live_stations(live, conn)
    report = {"event": name, "archive": _load_archive(ev, conn)}
    _say(f"[archive] {report['archive']}")

    bbox = _expand(ev.focus_bbox, FIRMS_MARGIN_DEG)
    hs = _step(
        "firms",
        lambda: firms.fetch_hotspots_range(conn, bbox, ev.start.date(), ev.end.date()),
    )
    report["hotspots"] = None if hs is None else len(hs)

    points = openmeteo.grid_points(ev.focus_bbox, GRID_STEP_DEG)
    for label, fn in (("wind", openmeteo.fetch_wind), ("cams", openmeteo.fetch_air_quality)):
        got = _step(label, lambda fn=fn: fn(conn, points, ev.start, ev.end))
        report[label] = None if got is None else len(got)

    img_dir = config.event_db_path(name).with_suffix("") / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for offset in (-1, 0, 1):
        day = (ev.key_day + timedelta(days=offset)).date()
        for layer in IMAGE_LAYERS:
            path = _step(
                f"gibs {layer} {day}",
                lambda day=day, layer=layer: gibs.fetch_image(
                    ev.focus_bbox, day, layer, out_dir=img_dir, conn=conn
                ),
            )
            if path is not None:
                images.append(Path(path).name)
    report["images"] = images
    return report


def _summarise(cand: Candidate, seen: dict) -> dict:
    f = cand.features
    return {
        "event_id": cand.event_id,
        "ts": cand.ts.isoformat(),
        "n_stations": len(cand.station_keys),
        "centroid": [round(cand.centroid_lat, 3), round(cand.centroid_lon, 3)],
        "provinces": f.get("provinces"),
        "peak_pm10": cand.peak_pm10,
        "peak_pm25": cand.peak_pm25,
        "ratio": None if cand.ratio_pm25_pm10 is None else round(cand.ratio_pm25_pm10, 2),
        "hint": cand.hint,
        "hotspots_50km_24h": f.get("hotspots_50km_24h"),
        "severity": round(cand.severity, 2),
        # where the event went during the day; the snapshot above is only its most severe hour
        "provinces_seen": sorted(seen["provinces"]),
        "max_stations": seen["max_stations"],
        "first_ts": seen["first_ts"].isoformat(),
        "last_ts": seen["last_ts"].isoformat(),
    }


def run_replay(name: str) -> dict:
    """Run detection hour by hour over the event window and return a summary dict.

    Hours run from window start + 7 days (baseline) to the window end. Per KST day: the
    number of distinct regional events and each one at its most severe hour (top first), plus
    the provinces it covered, its largest station count and its first/last hour that day.
    """
    ev = get_event(name)
    conn = store.connect(config.event_db_path(name))
    conn.execute("DELETE FROM candidates")
    per_day: dict[str, dict] = {}
    at = ev.replay_start
    while at <= ev.end:
        # an hour stamped 00:00 closes the previous KST day's interval; count it by its own date
        day = at.astimezone(KST).date().isoformat()
        slot = per_day.setdefault(day, {"best": {}, "seen": {}})
        for cand in detect.detect(conn, at):
            if cand.kind != "regional":
                continue
            best = slot["best"].get(cand.event_id)
            if best is None or cand.severity > best.severity:
                slot["best"][cand.event_id] = cand
            seen = slot["seen"].setdefault(
                cand.event_id,
                {"provinces": set(), "max_stations": 0, "first_ts": cand.ts, "last_ts": cand.ts},
            )
            seen["provinces"].update(cand.features.get("provinces") or [])
            seen["max_stations"] = max(seen["max_stations"], len(cand.station_keys))
            seen["last_ts"] = cand.ts
        at += timedelta(hours=1)
    days = {}
    for day, slot in per_day.items():
        ranked = sorted(slot["best"].values(), key=lambda c: -c.severity)
        # every regional event of the day at its most severe hour
        summaries = [_summarise(c, slot["seen"][c.event_id]) for c in ranked]
        days[day] = {
            "regional_events": len(ranked),
            "top_event": summaries[0] if summaries else None,
            "events": summaries,
        }
    return {
        "event": name,
        "expected_hint": ev.expected_hint,
        "key_day": ev.key_day.date().isoformat(),
        "days": days,
    }
