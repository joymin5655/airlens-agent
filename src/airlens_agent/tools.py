"""Evidence tools the investigator model can call (OpenAI function-calling format).

Each tool returns a compact {"evidence_id", "source", "data"} dict that is also kept in the
run's Evidence store. In replay, tools only return data stamped at or before the candidate's
hour + CLAMP_HOURS, satellite imagery included (see image_passes).
"""

import copy
import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from . import http, quota, store
from .config import KST, MissingKeyError
from .detect import haversine_km
from .models import Candidate
from .sources import gibs, sam

CLAMP_HOURS = 3
# KST hour by which a layer's daytime pass over Korea (127E) has been imaged, with an hour of
# margin: Terra crosses at about 10:30 local solar time (about 11:00 KST), NOAA-20 at about 13:30
# (about 14:00 KST); the edges of a swath are imaged later. GIBS can serve an all-black (blank)
# Terra image for a recent date (seen for 2026-02-22): satellite_segments takes the first
# non-blank cached image of the walk over these passes, so a blank one falls back to the next.
OVERPASS_HOURS = {
    "MODIS_Terra_CorrectedReflectance_TrueColor": 12,
    "VIIRS_NOAA20_CorrectedReflectance_TrueColor": 15,
}
MAX_STATIONS = 8
MAX_HOTSPOTS = 5
MAX_REGIONS = 5
# ponytail: paid satellite_segments calls per run; across runs only the daily call cap and the USD
# budget (quota) limit SAM
MAX_SAM_CALLS = 4
SAM_SOURCE = "NASA GIBS + SAM 3.1"
MAX_ARG_CHARS = 60  # string arguments are cut to this before they are kept in the evidence
_COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


class ToolError(ValueError):
    """Bad tool arguments or missing data; the text goes back to the model."""


class Evidence:
    """Evidence collected during one agent run, ids E1, E2, ..."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, tool: str, args: dict, source: str, data: dict) -> dict:
        item = {"evidence_id": f"E{len(self.items) + 1}", "source": source, "data": data}
        self.items.append({**item, "tool": tool, "args": args})
        return item

    def get(self, evidence_id: str) -> dict | None:
        return next((i for i in self.items if i["evidence_id"] == evidence_id), None)


@dataclass
class Context:
    conn: sqlite3.Connection  # event (or live) DB
    candidate: Candidate
    evidence: Evidence = field(default_factory=Evidence)
    ledger: sqlite3.Connection | None = None  # spend ledger, needed by satellite_segments
    image_dir: Path | None = None  # GIBS images with sidecars
    out_dir: Path | None = None  # where overlays go
    transport: object = None  # httpx transport for tests
    sam_calls: int = 0  # paid SAM calls started in this run (a failed one may still be billed)
    sam_results: dict[str, dict] = field(default_factory=dict)  # prompt -> data, to reuse

    @property
    def limit(self) -> datetime:
        return self.candidate.ts + timedelta(hours=CLAMP_HOURS)


def image_passes(candidate: Candidate) -> list[tuple[date, str]]:
    """The (day, layer) true-colour passes already imaged by the candidate's hour + CLAMP_HOURS:
    those of that KST day whose OVERPASS_HOURS has come, and all of the day before; newest first.
    A 10:00 event gets that day's Terra pass, then the day before's, not an image taken later."""
    limit = (candidate.ts + timedelta(hours=CLAMP_HOURS)).astimezone(KST)
    passes = [
        (limit.date() - timedelta(days=back), layer)
        for back in (0, 1)
        for layer, hour in OVERPASS_HOURS.items()
        if back or hour <= limit.hour
    ]
    return sorted(passes, key=lambda p: (p[0], OVERPASS_HOURS[p[1]]), reverse=True)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def compass(deg: float) -> str:
    return _COMPASS[int((deg + 22.5) % 360 // 45)]


def _r(value, digits: int = 1):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(
        float(value), digits
    )


def _int_arg(args: dict, name: str, default: int, lo: int, hi: int) -> int:
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or not lo <= value <= hi:
        raise ToolError(f"{name} must be a number between {lo} and {hi}")
    return int(value)


def _window(ctx: Context, hours_before: int, hours_after: int = 0) -> tuple[datetime, datetime]:
    end = min(ctx.candidate.ts + timedelta(hours=hours_after), ctx.limit)
    return ctx.candidate.ts - timedelta(hours=hours_before), end


def station_readings(ctx: Context, args: dict) -> tuple[str, dict]:
    hours = _int_arg(args, "hours_before", 24, 1, 72)
    start, end = _window(ctx, hours)
    cand = ctx.candidate
    obs = store.observations_between(ctx.conn, start, end)
    mine = obs[obs["station_key"].isin(cand.station_keys)]
    stations = []
    for key, g in mine.groupby("station_key"):
        peak = g.loc[g["pm10"].idxmax()] if g["pm10"].notna().any() else g.iloc[-1]
        p10, p25 = g["pm10"].max(), g["pm25"].max()
        stations.append({
            "station": key, "peak_pm10": _r(p10), "peak_pm25": _r(p25),
            # the two peaks can be hours apart, so this is not the ratio at any one hour
            "ratio_of_peaks": _r(p25 / p10, 2) if p10 and p10 == p10 else None,
            "peak_time_kst": peak["ts"].strftime("%m-%d %H:00"),
            "max_co": _r(g["co"].max(), 2), "max_no2": _r(g["no2"].max(), 3),
        })
    stations.sort(key=lambda s: -(s["peak_pm10"] or 0))
    at_hour = obs[obs["ts"] == cand.ts]
    data = {
        "hours_before": hours, "n_stations": len(stations), "top_stations": stations[:MAX_STATIONS],
        "network_mean_pm10_at_event_hour": _r(at_hour["pm10"].mean()),
        "network_mean_pm25_at_event_hour": _r(at_hour["pm25"].mean()),
    }
    return "AirKorea", data


def fire_hotspots(ctx: Context, args: dict) -> tuple[str, dict]:
    radius = _int_arg(args, "radius_km", 300, 10, 1000)
    hours = _int_arg(args, "hours_before", 72, 1, 168)
    start, end = _window(ctx, hours)
    c = ctx.candidate
    dlat = radius / 111.0
    dlon = radius / (111.0 * max(math.cos(math.radians(c.centroid_lat)), 0.2))
    lat, lon = c.centroid_lat, c.centroid_lon
    box = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    # the same pixel can be stored once per satellite/product; count it once
    hs = store.hotspots_in(ctx.conn, box, start, end).drop_duplicates(["lat", "lon", "ts_utc"])
    if not hs.empty:
        hs = hs.assign(km=haversine_km(c.centroid_lat, c.centroid_lon, hs["lat"], hs["lon"]))
        hs = hs[hs["km"] <= radius].sort_values("km")
    data: dict = {"radius_km": radius, "hours_before": hours, "count": int(len(hs))}
    if hs.empty:
        return "NASA FIRMS", {**data, "total_frp_mw": 0.0, "nearest_km": None, "top": []}
    near = hs.iloc[0]
    bearing = bearing_deg(c.centroid_lat, c.centroid_lon, near["lat"], near["lon"])
    top = []
    for _, h in hs.sort_values("frp", ascending=False).head(MAX_HOTSPOTS).iterrows():
        b = bearing_deg(c.centroid_lat, c.centroid_lon, h["lat"], h["lon"])
        top.append({
            "lat": _r(h["lat"], 2), "lon": _r(h["lon"], 2), "frp_mw": _r(h["frp"]),
            "km": _r(h["km"], 0), "bearing_deg": _r(b, 0),
            "time_utc": h["ts_utc"].strftime("%m-%d %H:%M"),
        })
    data.update(
        total_frp_mw=_r(hs["frp"].sum()), nearest_km=_r(near["km"], 0),
        nearest_bearing_deg=_r(bearing, 0), nearest_bearing=compass(bearing), top=top,
    )
    return "NASA FIRMS", data


def _nearest(ctx: Context, start: datetime, end: datetime, source: str | None = None):
    c = ctx.candidate
    samples = store.grid_samples_near(
        ctx.conn, c.centroid_lat, c.centroid_lon, start, end, source=source
    )
    if not samples:
        raise ToolError("no model grid data near the candidate for this time window")
    return samples


def wind(ctx: Context, args: dict) -> tuple[str, dict]:
    level = args.get("level", "850hPa")
    if level not in ("850hPa", "10m"):
        raise ToolError('level must be "850hPa" or "10m"')
    hours = _int_arg(args, "hours_before", 24, 1, 72)
    start, end = _window(ctx, hours)
    speed_key, dir_key = f"wind_speed_{level}", f"wind_direction_{level}"
    rows = [s for s in _nearest(ctx, start, end) if s.variables.get(speed_key) is not None]
    if not rows:
        raise ToolError(f"no {level} wind data for this time window")
    speeds = np.array([s.variables[speed_key] for s in rows], dtype=float)
    dirs = np.radians([s.variables[dir_key] for s in rows])
    # direction is where the wind comes from, so the mean vector points downwind
    u, v = np.mean(-speeds * np.sin(dirs)), np.mean(-speeds * np.cos(dirs))
    from_deg = float(np.degrees(np.arctan2(-u, -v)) % 360)
    last = rows[-1]
    data = {
        "level": level, "hours_before": hours, "grid_lat": rows[0].lat, "grid_lon": rows[0].lon,
        "n_hours": len(rows), "mean_speed_ms": _r(speeds.mean()),
        "mean_from_deg": _r(from_deg, 0), "upwind_bearing_deg": _r(from_deg, 0),
        "upwind_direction": compass(from_deg),
        "last_hour": {
            "time_kst": last.ts.strftime("%m-%d %H:00"), "speed_ms": _r(last.variables[speed_key]),
            "from_deg": _r(last.variables[dir_key], 0),
        },
    }
    return "Open-Meteo", data


_CAMS = {
    "dust": "dust", "pm10": "pm10", "pm2_5": "pm2_5", "aod": "aerosol_optical_depth",
    "co": "carbon_monoxide", "no2": "nitrogen_dioxide",
}


def cams_forecast(ctx: Context, args: dict) -> tuple[str, dict]:
    before = _int_arg(args, "hours_before", 24, 1, 72)
    after = _int_arg(args, "hours_after", 6, 0, 24)
    start, end = _window(ctx, before, after)
    rows = _nearest(ctx, start, end, source="openmeteo.cams")
    at = next((s for s in rows if s.ts == ctx.candidate.ts), None)  # no such row: report None
    out = {}
    for name, key in _CAMS.items():
        vals = [(s.variables.get(key), s.ts) for s in rows if s.variables.get(key) is not None]
        if not vals:
            continue
        peak, peak_ts = max(vals, key=lambda p: p[0])
        out[name] = {
            "at_event": None if at is None else _r(at.variables.get(key), 2), "max": _r(peak, 2),
            "max_time_kst": peak_ts.strftime("%m-%d %H:00"),
        }
    data = {
        "hours_before": before, "hours_after": min(after, CLAMP_HOURS),
        "grid_lat": rows[0].lat, "grid_lon": rows[0].lon, "variables": out,
    }
    return "CAMS via Open-Meteo", data


def true_color_images(image_dir: Path, day, layer: str | None = None) -> list[Path]:
    """Non-blank true-colour GIBS images requested for `day`, from the sidecars; with `layer`,
    only that exact layer's (files are named {day}_{layer}_{bbox}.jpg)."""
    stem = f"{day.isoformat()}_" + (f"{layer}_" if layer else "")
    found = []
    for path in sorted(Path(image_dir).glob(f"{stem}*")):
        if path.suffix != ".jpg" or not any(name in path.name for name in gibs.TRUE_COLOR_LAYERS):
            continue
        side = sam.sidecar_path(path)
        if side.exists() and not json.loads(side.read_text()).get("blank_check", {}).get("blank"):
            found.append(path)
    return found


def _overlaps(bbox, region) -> bool:
    w, s, e, n = bbox
    rw, rs, re_, rn = region
    return not (re_ < w or rw > e or rn < s or rs > n)


def satellite_segments(ctx: Context, args: dict) -> tuple[str, dict]:
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not 0 < len(prompt.strip()) <= 40:
        raise ToolError("prompt must be a short noun phrase such as 'smoke' or 'haze'")
    prompt = prompt.strip()
    if ctx.image_dir is None or ctx.ledger is None:
        raise ToolError("no satellite imagery is available for this run")
    if prompt.lower() in ctx.sam_results:  # asked again: the earlier answer, no second charge
        return SAM_SOURCE, copy.deepcopy(ctx.sam_results[prompt.lower()])
    if ctx.sam_calls >= MAX_SAM_CALLS:
        raise ToolError(
            f"satellite_segments is limited to {MAX_SAM_CALLS} paid calls per run; "
            "work from the results you already have"
        )
    for day, layer in image_passes(ctx.candidate):  # newest pass first, blank images skipped
        if images := true_color_images(ctx.image_dir, day, layer):
            break
    else:
        raise ToolError("no usable (non-blank) true-colour image cached for this date")
    path = images[0]
    ctx.sam_calls += 1  # counted before the call: one that fails may still have been billed
    regions = sam.segment(path, prompt, ledger=ctx.ledger, transport=ctx.transport)
    side = json.loads(sam.sidecar_path(path).read_text())
    c = ctx.candidate
    start, end = _window(ctx, 24)
    hs = store.hotspots_in(ctx.conn, tuple(side["bbox"]), start, end)
    ranked = sorted(regions, key=lambda r: -r.area_frac)[:MAX_REGIONS]
    out = []
    for r in ranked:
        w, s, e, n = r.box_lonlat
        near = None
        if not hs.empty:
            near = _r(haversine_km((s + n) / 2, (w + e) / 2, hs["lat"], hs["lon"]).min(), 0)
        out.append({
            "area_frac": _r(r.area_frac, 3), "confidence": _r(r.confidence, 2),
            "box_lonlat": [_r(v, 2) for v in r.box_lonlat],
            "overlaps_candidate_bbox": _overlaps(c.bbox, r.box_lonlat),
            "nearest_hotspot_km": near,
        })
    data = {
        "prompt": prompt, "image_date": path.name[:10], "image_layer": layer,
        "n_regions": len(regions), "regions": out,
    }
    if regions and ctx.out_dir is not None:
        slug = "".join(ch if ch.isalnum() else "_" for ch in prompt)
        overlay = sam.overlay(
            path, regions, Path(ctx.out_dir) / f"{c.event_id}_{slug}_{path.name[:10]}.png"
        )
        data["overlay_file"] = overlay.name
    ctx.sam_results[prompt.lower()] = copy.deepcopy(data)
    return SAM_SOURCE, data


def _spec(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    params = {"type": "object", "properties": properties, "required": required or []}
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": params}}


_HOURS = {"type": "integer", "description": "hours before the event hour to look back"}

TOOLS = {
    "station_readings": (station_readings, _spec(
        "station_readings",
        "Per-station PM10/PM2.5 peaks, ratio_of_peaks (the PM2.5 peak divided by the PM10 peak; "
        "the two peaks can be hours apart), PM10 peak time and CO/NO2 for the stations in the "
        "event, plus the network mean at the event hour (AirKorea).",
        {"hours_before": _HOURS})),
    "fire_hotspots": (fire_hotspots, _spec(
        "fire_hotspots",
        "Satellite fire hotspots near the event centroid: count, total FRP, nearest distance "
        "and bearing, top 5 (NASA FIRMS).",
        {"radius_km": {"type": "integer", "description": "search radius, 10-1000 km"},
         "hours_before": _HOURS})),
    "wind": (wind, _spec(
        "wind",
        "Mean wind speed and direction at the nearest model grid point, and the upwind bearing "
        "(Open-Meteo).",
        {"level": {"type": "string", "enum": ["850hPa", "10m"]}, "hours_before": _HOURS})),
    "cams_forecast": (cams_forecast, _spec(
        "cams_forecast",
        "CAMS model dust, PM10, PM2.5, aerosol optical depth, CO and NO2 at the event centroid "
        "(at event hour and peak).",
        {"hours_before": _HOURS,
         "hours_after": {"type": "integer", "description": "hours after the event hour (max 3)"}})),
    "satellite_segments": (satellite_segments, _spec(
        "satellite_segments",
        "Segment a short noun phrase (e.g. 'smoke', 'haze') in the most recent true-colour "
        f"satellite image already taken by the event hour + {CLAMP_HOURS} h (MODIS Terra morning "
        "or NOAA-20 afternoon pass) with SAM 3.1; returns the image date and layer and regions "
        "with area, box, overlap with the event and distance to the nearest fire hotspot. "
        f"Costs money: at most {MAX_SAM_CALLS} calls per run; asking the same phrase again "
        "returns the earlier result.",
        {"prompt": {"type": "string", "description": "one short noun phrase"}}, ["prompt"])),
}
TOOL_NAMES = tuple(TOOLS)


def specs() -> list[dict]:
    return [spec for _, spec in TOOLS.values()]


def _kept_args(name: str, args: dict) -> dict:
    """What the evidence keeps of a call's arguments: the parameters the tool's spec declares,
    strings cut short. The model's other keys and long text are dropped."""
    declared = TOOLS[name][1]["function"]["parameters"]["properties"]
    return {
        k: v[:MAX_ARG_CHARS] if isinstance(v, str) else v for k, v in args.items() if k in declared
    }


def call(ctx: Context, name: str, args: dict | None, raw: str = "") -> str:
    """Run one tool call and return the tool-message text; failures come back as 'error: ...'."""
    if name not in TOOLS:
        return f"error: unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}"
    if args is None:
        return f"error: arguments are not a valid JSON object: {raw[:200]!r}"
    try:
        source, data = TOOLS[name][0](ctx, args)
    except (ToolError, http.HttpError, quota.QuotaExceeded, quota.BudgetExceeded,
            MissingKeyError) as exc:
        return f"error: {exc}"
    except Exception as exc:  # a tool bug or an unreadable file must not abort the paid run
        return f"error: {name} failed: {type(exc).__name__}: {http.redact(str(exc))[:200]}"
    item = ctx.evidence.add(name, _kept_args(name, args), source, data)
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str)
