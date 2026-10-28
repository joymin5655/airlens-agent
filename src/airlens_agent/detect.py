"""Hourly anomaly detection on observations, no model calls.

Per station and pollutant, the current hour is compared with the previous 7 days
(robust z-score on median/MAD). Elevated stations are clustered by distance and each
cluster becomes a candidate event with a coarse/fine/mixed hint.
"""

import json
import math
import sqlite3
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import store
from .config import KST
from .models import Candidate

# Korean Ministry of Environment grades (ug/m3, 24h scale): "bad" and "very bad" lower bounds.
BAD = {"pm10": 81.0, "pm25": 36.0}
VERY_BAD = {"pm10": 151.0, "pm25": 76.0}
Z_MIN = 3.5
MIN_BASELINE_HOURS = 48
BASELINE_DAYS = 7
EPS = 1.0  # ug/m3 floor on the scale: archive values are integers, so MAD is often 0
CLUSTER_KM = 40.0
REGIONAL_MIN_STATIONS = 3
HOTSPOT_KM = 50.0
LINK_HOURS = 3
LINK_JACCARD = 0.3
LINK_KM = 50.0
GAS_Z = 2.0
COARSE_RATIO = 0.35
FINE_RATIO = 0.6
ROADSIDE = "도로변"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; accepts scalars or numpy arrays (broadcasting)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def _matrices(obs: pd.DataFrame, hours: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Pollutant -> hours x stations matrix with NaN for missing or flagged values."""
    flagged: dict[tuple[str, str], list[pd.Timestamp]] = defaultdict(list)
    for row in obs.loc[obs["flags"].notna() & (obs["flags"] != "{}")].itertuples():
        for pol in json.loads(row.flags):
            flagged[(pol, row.station_key)].append(row.ts)
    out = {}
    for pol in ("pm10", "pm25", "co", "no2"):
        mat = obs.pivot(index="ts", columns="station_key", values=pol).reindex(hours)
        for (fpol, key), stamps in flagged.items():
            if fpol == pol and key in mat.columns:
                mat.loc[mat.index.intersection(stamps), key] = np.nan
        out[pol] = mat.astype(float)
    return out


def robust_z(mat: pd.DataFrame, at: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """(current value, robust z) per station. z is NaN with fewer than 48 baseline hours."""
    base = mat.loc[mat.index < at]
    cur = mat.loc[at] if at in mat.index else pd.Series(np.nan, index=mat.columns)
    arr = base.to_numpy()
    n = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN stations
        med = np.nanmedian(arr, axis=0)
        mad = np.nanmedian(np.abs(arr - med), axis=0)
    z = (cur.to_numpy() - med) / (1.4826 * mad + EPS)
    z = np.where(n >= MIN_BASELINE_HOURS, z, np.nan)
    return cur, pd.Series(z, index=mat.columns)


def _elevated(pol: str, x: float, z: float) -> bool:
    if np.isnan(x):
        return False
    if x >= VERY_BAD[pol]:
        return True
    return bool(not np.isnan(z) and z >= Z_MIN and x >= BAD[pol])


def _clusters(lat: np.ndarray, lon: np.ndarray) -> list[list[int]]:
    """Single linkage within CLUSTER_KM (O(n^2) is fine for ~700 stations)."""
    n = len(lat)
    if n == 0:
        return []
    near = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) <= CLUSTER_KM
    seen, groups = np.zeros(n, dtype=bool), []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack, members = [start], []
        while stack:
            i = stack.pop()
            members.append(i)
            for j in np.flatnonzero(near[i] & ~seen):
                seen[j] = True
                stack.append(j)
        groups.append(sorted(members))
    return groups


def _mean(values) -> float | None:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    return float(arr.mean()) if arr.size else None


def _max(values) -> float | None:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    return float(arr.max()) if arr.size else None


def hint_for(ratio: float | None, peak_pm10: float | None) -> str:
    """A clue, not a verdict: low PM2.5/PM10 with high PM10 -> coarse (dust);
    high ratio -> fine (smoke, combustion); otherwise mixed."""
    if ratio is None:
        return "mixed"
    if ratio <= COARSE_RATIO and peak_pm10 is not None and peak_pm10 >= BAD["pm10"]:
        return "coarse"
    if ratio >= FINE_RATIO:
        return "fine"
    return "mixed"


def severity(peak_pm10: float | None, peak_pm25: float | None, n_stations: int) -> float:
    """max(peak_pm10 / 151, peak_pm25 / 76) * (1 + ln(n_stations)): how far above the
    'very bad' grade, scaled up (log) by how many stations see it."""
    grade = max((peak_pm10 or 0) / VERY_BAD["pm10"], (peak_pm25 or 0) / VERY_BAD["pm25"])
    return grade * (1 + math.log(max(n_stations, 1)))


def _hotspots_near(
    conn: sqlite3.Connection, at: datetime, lat: np.ndarray, lon: np.ndarray
) -> int:
    hs = store.hotspots_between(conn, at - timedelta(hours=24), at)
    if hs.empty:
        return 0
    d = haversine_km(
        hs["lat"].to_numpy()[:, None], hs["lon"].to_numpy()[:, None],
        lat[None, :], lon[None, :],
    )
    return int(np.sum(d.min(axis=1) <= HOTSPOT_KM))


def _link_ids(
    conn: sqlite3.Connection, at: datetime, drafts: list[dict]
) -> list[str]:
    """Reuse the event_id of a candidate from the previous 3 hours (station Jaccard >= 0.3 or
    centroids <= 50 km apart, best match first, each old id used once)."""
    rows = conn.execute(
        "SELECT event_id, station_keys, centroid_lat, centroid_lon FROM candidates "
        "WHERE ts >= ? AND ts < ?",
        ((at - timedelta(hours=LINK_HOURS)).isoformat(), at.isoformat()),
    ).fetchall()
    prev = [(eid, set(json.loads(keys)), lat, lon) for eid, keys, lat, lon in rows]
    pairs = []
    for i, d in enumerate(drafts):
        for eid, keys, lat, lon in prev:
            jac = len(keys & d["keys"]) / len(keys | d["keys"]) if keys | d["keys"] else 0.0
            km = float(haversine_km(lat, lon, d["lat"], d["lon"]))
            if jac >= LINK_JACCARD or km <= LINK_KM:
                pairs.append((-jac, km, i, eid))
    ids: list[str | None] = [None] * len(drafts)
    used: set[str] = set()
    for _, _, i, eid in sorted(pairs):
        if ids[i] is None and eid not in used:
            ids[i], _ = eid, used.add(eid)
    stamp = at.strftime("%Y%m%d%H")
    existing = {r[0] for r in conn.execute("SELECT event_id FROM candidates")}
    counter = 0
    for i in range(len(ids)):
        if ids[i] is None:
            while f"evt-{stamp}-{counter}" in existing | used:
                counter += 1
            ids[i] = f"evt-{stamp}-{counter}"
            used.add(ids[i])
    return ids  # type: ignore[return-value]


def detect(
    conn: sqlite3.Connection, at: datetime, *, include_roadside: bool = False, save: bool = True
) -> list[Candidate]:
    """Find anomaly candidates in the observations up to `at` (tz-aware KST)."""
    at = at.astimezone(KST)
    start = at - timedelta(days=BASELINE_DAYS)
    obs = store.observations_between(conn, start, at)
    if obs.empty:
        return []
    hours = pd.date_range(start, at, freq="h", tz=KST)
    ts_at = pd.Timestamp(at)
    mats = _matrices(obs, hours)
    pm = {pol: robust_z(mats[pol], ts_at) for pol in ("pm10", "pm25")}
    gas = {pol: robust_z(mats[pol], ts_at)[1] for pol in ("co", "no2")}

    stations = store.stations_df(conn).set_index("key")
    elevated: dict[str, list[str]] = {}
    for pol, (cur, z) in pm.items():
        for key in cur.index:
            if _elevated(pol, cur[key], z[key]):
                elevated.setdefault(key, []).append(pol)
    if not include_roadside:
        elevated = {
            k: v for k, v in elevated.items()
            if k not in stations.index or ROADSIDE not in str(stations.at[k, "network"] or "")
        }
    located = [
        k for k in elevated
        if k in stations.index
        and pd.notna(stations.at[k, "lat"])
        and pd.notna(stations.at[k, "lon"])
    ]
    no_coord = len(elevated) - len(located)
    if not located:
        return []
    lat = stations.loc[located, "lat"].to_numpy(dtype=float)
    lon = stations.loc[located, "lon"].to_numpy(dtype=float)

    ts_prev3 = pd.Timestamp(at - timedelta(hours=3))
    drafts = []
    for members in _clusters(lat, lon):
        keys = [located[i] for i in members]
        mlat, mlon = lat[members], lon[members]
        pm10 = pm["pm10"][0].reindex(keys)
        pm25 = pm["pm25"][0].reindex(keys)
        both = pm10.notna() & pm25.notna()
        ratio = (
            float(pm25[both].mean() / pm10[both].mean())
            if both.any() and pm10[both].mean() > 0 else None
        )
        peak10, peak25 = _max(pm10), _max(pm25)
        rise = {}
        for pol in ("pm10", "pm25"):
            prev = mats[pol].loc[ts_prev3].reindex(keys)
            cur = pm[pol][0].reindex(keys)
            rise[pol] = _mean((cur - prev).to_numpy())
        extent = float(
            haversine_km(mlat[:, None], mlon[:, None], mlat[None, :], mlon[None, :]).max()
        )
        co_z, no2_z = _mean(gas["co"].reindex(keys)), _mean(gas["no2"].reindex(keys))
        drafts.append({
            "keys": set(keys), "key_list": keys, "lat": float(mlat.mean()),
            "lon": float(mlon.mean()),
            "bbox": (float(mlon.min()), float(mlat.min()), float(mlon.max()), float(mlat.max())),
            "peak10": peak10, "peak25": peak25, "ratio": ratio,
            "features": {
                "n_stations": len(keys),
                "mean_pm10": _mean(pm10), "mean_pm25": _mean(pm25),
                "co_z_mean": co_z, "no2_z_mean": no2_z,
                "co_elevated": bool(co_z is not None and co_z >= GAS_Z),
                "no2_elevated": bool(no2_z is not None and no2_z >= GAS_Z),
                "rise_3h_pm10": rise["pm10"], "rise_3h_pm25": rise["pm25"],
                "extent_km": extent,
                "provinces": sorted({str(stations.at[k, "sido"]) for k in keys}),
                "hotspots_50km_24h": _hotspots_near(conn, at, mlat, mlon),
                "elevated_without_coords": no_coord,
            },
        })

    ids = _link_ids(conn, at, drafts)
    candidates = []
    for d, eid in zip(drafts, ids, strict=True):
        n = len(d["key_list"])
        candidates.append(
            Candidate(
                event_id=eid, ts=at,
                kind="regional" if n >= REGIONAL_MIN_STATIONS else "local",
                hint=hint_for(d["ratio"], d["peak10"]),
                station_keys=d["key_list"], centroid_lat=d["lat"], centroid_lon=d["lon"],
                bbox=d["bbox"], peak_pm10=d["peak10"], peak_pm25=d["peak25"],
                ratio_pm25_pm10=d["ratio"], severity=severity(d["peak10"], d["peak25"], n),
                features=d["features"],
            )
        )
    candidates.sort(key=lambda c: -c.severity)
    if save:
        store.save_candidates(conn, candidates)
    return candidates


def detect_latest(conn: sqlite3.Connection) -> list[Candidate]:
    """detect() at the latest observation hour in the DB; saves via store.save_candidates."""
    last = store.last_observation_ts(conn)
    return [] if last is None else detect(conn, last)


def select_for_investigation(candidates: list[Candidate], per_day: int = 6) -> list[Candidate]:
    """Top events per KST day by severity; each event counted once (its most severe hour)."""
    best: dict[str, Candidate] = {}
    for c in candidates:
        if c.event_id not in best or c.severity > best[c.event_id].severity:
            best[c.event_id] = c
    by_day: dict = defaultdict(list)
    for c in best.values():
        by_day[c.ts.astimezone(KST).date()].append(c)
    out: list[Candidate] = []
    for day in sorted(by_day):
        out += sorted(by_day[day], key=lambda c: -c.severity)[:per_day]
    return out
