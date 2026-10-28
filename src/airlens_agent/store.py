"""SQLite persistence (WAL). One DB per live run and one per replay event."""

import json
import sqlite3
from datetime import datetime
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
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601").dt.tz_convert(KST)
    return df


def hotspots_between(conn: sqlite3.Connection, start: datetime, end: datetime) -> pd.DataFrame:
    """Hotspots with start <= ts_utc <= end (timestamps stored as UTC ISO strings)."""
    df = pd.read_sql_query("SELECT * FROM hotspots", conn)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], format="ISO8601", utc=True)
    return df[(df["ts_utc"] >= start) & (df["ts_utc"] <= end)].reset_index(drop=True)


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

