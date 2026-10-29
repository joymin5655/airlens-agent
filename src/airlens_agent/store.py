"""SQLite persistence (WAL). One DB per live run and one per replay event."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import quota
from .config import KST
from .models import Candidate, GridSample, Hotspot, Observation, Station

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    key TEXT PRIMARY KEY, name TEXT, sido TEXT, lat REAL, lon REAL, code TEXT, source TEXT,
    network TEXT
);
CREATE TABLE IF NOT EXISTS observations (
    station_key TEXT NOT NULL, ts TEXT NOT NULL,
    pm10 REAL, pm25 REAL, o3 REAL, no2 REAL, co REAL, so2 REAL,
    flags TEXT, source TEXT,
    PRIMARY KEY (station_key, ts)
);
CREATE TABLE IF NOT EXISTS hotspots (
    lat REAL NOT NULL, lon REAL NOT NULL, ts_utc TEXT NOT NULL, satellite TEXT NOT NULL,
    source TEXT, frp REAL, confidence TEXT, daynight TEXT,
    PRIMARY KEY (lat, lon, ts_utc, satellite)
);
CREATE TABLE IF NOT EXISTS grid_samples (
    lat REAL NOT NULL, lon REAL NOT NULL, ts TEXT NOT NULL, source TEXT NOT NULL,
    variables TEXT,
    PRIMARY KEY (lat, lon, ts, source)
);
CREATE TABLE IF NOT EXISTS candidates (
    event_id TEXT PRIMARY KEY, ts TEXT, kind TEXT, hint TEXT, station_keys TEXT,
    centroid_lat REAL, centroid_lon REAL, bbox TEXT, peak_pm10 REAL, peak_pm25 REAL,
    ratio_pm25_pm10 REAL, severity REAL, features TEXT
);
"""

_OBS_COLS = ("pm10", "pm25", "o3", "no2", "co", "so2")


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    if "network" not in {r[1] for r in conn.execute("PRAGMA table_info(stations)")}:
        conn.execute("ALTER TABLE stations ADD COLUMN network TEXT")
    conn.execute(quota.SCHEMA)
    conn.commit()
    return conn


def _iso(ts: datetime) -> str:
    return ts.isoformat()


def upsert_stations(conn: sqlite3.Connection, stations: list[Station]) -> int:
    conn.executemany(
        "INSERT OR REPLACE INTO stations VALUES (?,?,?,?,?,?,?,?)",
        [(s.key, s.name, s.sido, s.lat, s.lon, s.code, s.source, s.network) for s in stations],
    )
    conn.commit()
    return len(stations)


def upsert_observations(conn: sqlite3.Connection, obs: list[Observation]) -> int:
    conn.executemany(
        "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (o.station_key, _iso(o.ts), o.pm10, o.pm25, o.o3, o.no2, o.co, o.so2,
             json.dumps(o.flags, ensure_ascii=False), o.source)
            for o in obs
        ],
    )
    conn.commit()
    return len(obs)


def insert_hotspots(conn: sqlite3.Connection, hotspots: list[Hotspot]) -> int:
    """Insert new hotspots only (dedupe on position, time and satellite); returns new rows."""
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO hotspots VALUES (?,?,?,?,?,?,?,?)",
        [
            (h.lat, h.lon, _iso(h.ts_utc), h.satellite, h.source, h.frp, h.confidence, h.daynight)
            for h in hotspots
        ],
    )
    conn.commit()
    return conn.total_changes - before


def upsert_grid_samples(conn: sqlite3.Connection, samples: list[GridSample]) -> int:
    conn.executemany(
        "INSERT OR REPLACE INTO grid_samples VALUES (?,?,?,?,?)",
        [(g.lat, g.lon, _iso(g.ts), g.source, json.dumps(g.variables)) for g in samples],
    )
    conn.commit()
    return len(samples)


def save_candidates(conn: sqlite3.Connection, candidates: list[Candidate]) -> int:
    conn.executemany(
        "INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (c.event_id, _iso(c.ts), c.kind, c.hint, json.dumps(c.station_keys),
             c.centroid_lat, c.centroid_lon, json.dumps(c.bbox), c.peak_pm10, c.peak_pm25,
             c.ratio_pm25_pm10, c.severity, json.dumps(c.features, default=str))
            for c in candidates
        ],
    )
    conn.commit()
    return len(candidates)


def stations_df(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM stations", conn)


def observations_between(conn: sqlite3.Connection, start: datetime, end: datetime) -> pd.DataFrame:
    """Observations with start <= ts <= end. `ts` is returned as tz-aware KST."""
    df = pd.read_sql_query(
        "SELECT * FROM observations WHERE ts >= ? AND ts <= ? ORDER BY ts",
        conn,
        params=(_iso(start.astimezone(KST)), _iso(end.astimezone(KST))),
    )
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True).dt.tz_convert(KST)
    return df


_TS_PAD = timedelta(days=1)  # covers any UTC offset when ts_utc strings carry another one


def _hotspots(conn, where: str, args: list, start: datetime, end: datetime) -> pd.DataFrame:
    """Filter on a padded time range in SQL (strings), then exactly in pandas."""
    cur = conn.execute(
        f"SELECT * FROM hotspots WHERE {where} ts_utc >= ? AND ts_utc <= ?",
        [*args, _utc_iso(start - _TS_PAD), _utc_iso(end + _TS_PAD)],
    )
    df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], format="ISO8601", utc=True)
    return df[(df["ts_utc"] >= start) & (df["ts_utc"] <= end)].reset_index(drop=True)


def _utc_iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat()


def hotspots_between(conn: sqlite3.Connection, start: datetime, end: datetime) -> pd.DataFrame:
    """Hotspots with start <= ts_utc <= end (timestamps stored as UTC ISO strings)."""
    return _hotspots(conn, "", [], start, end)


def hotspots_in(
    conn: sqlite3.Connection,
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Hotspots inside bbox (west, south, east, north) with start <= ts_utc <= end."""
    west, south, east, north = bbox
    where = "lon >= ? AND lon <= ? AND lat >= ? AND lat <= ? AND"
    return _hotspots(conn, where, [west, east, south, north], start, end)


def grid_samples_near(
    conn: sqlite3.Connection,
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    *,
    source: str | None = None,
    max_km: float = 60.0,
) -> list[GridSample]:
    """Samples at the grid point nearest to (lat, lon) with start <= ts <= end, oldest first.

    Empty when no point lies within max_km. `source` limits the rows (e.g. "openmeteo.cams").
    """
    from .detect import haversine_km  # detect imports store, so import it lazily

    where, args = "ts >= ? AND ts <= ?", [_iso(start.astimezone(KST)), _iso(end.astimezone(KST))]
    if source is not None:
        where, args = where + " AND source = ?", [*args, source]
    points = conn.execute(f"SELECT DISTINCT lat, lon FROM grid_samples WHERE {where}", args)
    best = min(
        points.fetchall(), key=lambda p: haversine_km(lat, lon, p[0], p[1]), default=None
    )
    if best is None or haversine_km(lat, lon, best[0], best[1]) > max_km:
        return []
    rows = conn.execute(
        f"SELECT lat, lon, ts, source, variables FROM grid_samples "
        f"WHERE lat = ? AND lon = ? AND {where} ORDER BY ts",
        [best[0], best[1], *args],
    ).fetchall()
    return [
        GridSample(la, lo, datetime.fromisoformat(ts), src, json.loads(variables))
        for la, lo, ts, src, variables in rows
    ]


def candidate(conn: sqlite3.Connection, event_id: str) -> Candidate | None:
    """The stored candidate `event_id`, or None."""
    cur = conn.execute("SELECT * FROM candidates WHERE event_id = ?", (event_id,))
    row = cur.fetchone()
    if row is None:
        return None
    d = dict(zip([c[0] for c in cur.description], row, strict=True))
    return Candidate(
        event_id=d["event_id"],
        ts=datetime.fromisoformat(d["ts"]).astimezone(KST),
        kind=d["kind"],
        hint=d["hint"],
        station_keys=json.loads(d["station_keys"] or "[]"),
        centroid_lat=d["centroid_lat"],
        centroid_lon=d["centroid_lon"],
        bbox=tuple(json.loads(d["bbox"])),
        peak_pm10=d["peak_pm10"],
        peak_pm25=d["peak_pm25"],
        ratio_pm25_pm10=d["ratio_pm25_pm10"],
        severity=d["severity"],
        features=json.loads(d["features"] or "{}"),
    )


def latest_candidates(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    cur = conn.execute("SELECT * FROM candidates ORDER BY ts DESC LIMIT ?", (limit,))
    cols = [c[0] for c in cur.description]
    rows = []
    for row in cur.fetchall():
        d = dict(zip(cols, row, strict=True))
        for k in ("station_keys", "bbox", "features"):
            d[k] = json.loads(d[k]) if d[k] else None
        rows.append(d)
    return rows


def last_observation_ts(conn: sqlite3.Connection) -> datetime | None:
    row = conn.execute("SELECT MAX(ts) FROM observations").fetchone()
    return datetime.fromisoformat(row[0]) if row and row[0] else None

