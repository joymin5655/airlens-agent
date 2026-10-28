"""Open-Meteo wind and CAMS air-quality samples.

Weather data by Open-Meteo.com (CC BY 4.0) and CAMS. Free tier: non-commercial, <10,000 calls/day.
"""

import sqlite3
from datetime import datetime, timedelta

from .. import http, store
from ..config import KST
from ..models import GridSample

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CAMS_VARS = (
    "pm10", "pm2_5", "dust", "aerosol_optical_depth", "carbon_monoxide", "nitrogen_dioxide"
)
WIND_VARS = ("wind_speed_10m", "wind_direction_10m", "wind_speed_100m", "wind_direction_100m")
WIND_850_VARS = ("wind_speed_850hPa", "wind_direction_850hPa")

ARCHIVE_LAG_DAYS = 5  # ERA5 archive trails real time by about 5 days
MAX_PAST_DAYS = 92  # forecast API past_days limit
MAX_FORECAST_DAYS = 16
POINTS_PER_REQUEST = 50  # keeps the URL short; each location counts as one call for budgeting


def grid_points(
    bbox: tuple[float, float, float, float], step: float = 0.5
) -> list[tuple[float, float]]:
    """(lat, lon) points covering bbox (west, south, east, north), edges included."""
    west, south, east, north = bbox
    n_lat = int((north - south) / step + 1e-9) + 1
    n_lon = int((east - west) / step + 1e-9) + 1
    return [
        (round(south + i * step, 4), round(west + j * step, 4))
        for i in range(n_lat)
        for j in range(n_lon)
    ]


def _fetch_hourly(
    conn: sqlite3.Connection,
    url: str,
    op: str,
    points: list[tuple[float, float]],
    hourly: tuple[str, ...],
    source: str,
    extra: dict,
) -> list[GridSample]:
    """Batch-request hourly variables for points; one GridSample per point and hour."""
    samples: list[GridSample] = []
    for i in range(0, len(points), POINTS_PER_REQUEST):
        batch = points[i : i + POINTS_PER_REQUEST]
        params = {
            "latitude": ",".join(str(p[0]) for p in batch),
            "longitude": ",".join(str(p[1]) for p in batch),
            "hourly": ",".join(hourly),
            "timezone": "Asia/Seoul",
            **extra,
        }
        data = http.get(url, params, op=op, conn=conn).json()
        results = data if isinstance(data, list) else [data]  # single location = bare object
        if len(results) != len(batch):
            raise http.HttpError(f"{op}: expected {len(batch)} locations, got {len(results)}")
        for (lat, lon), res in zip(batch, results, strict=True):
            block = res.get("hourly", {})
            for k, t in enumerate(block.get("time", [])):
                values = {name: block[name][k] for name in hourly if name in block}
                if all(v is None for v in values.values()):
                    continue
                ts = datetime.fromisoformat(t).replace(tzinfo=KST)
                samples.append(GridSample(lat=lat, lon=lon, ts=ts, source=source, variables=values))
    return samples


def fetch_air_quality(
    conn: sqlite3.Connection, points: list[tuple[float, float]], start: datetime, end: datetime
) -> list[GridSample]:
    """Hourly CAMS air-quality samples at (lat, lon) points for the KST days start..end;
    stored via upsert_grid_samples."""
    extra = {
        "start_date": start.astimezone(KST).date().isoformat(),
        "end_date": end.astimezone(KST).date().isoformat(),
        "domains": "cams_global",
    }
    samples = _fetch_hourly(
        conn, AIR_QUALITY_URL, "openmeteo.air_quality", points, CAMS_VARS, "openmeteo.cams", extra
    )
    store.upsert_grid_samples(conn, samples)
    return samples


def use_archive(end: datetime, now: datetime | None = None) -> bool:
    """ERA5 archive is used only when `end` is older than its ~5-day lag."""
    now = now or datetime.now(KST)
    cutoff = now.astimezone(KST).date() - timedelta(days=ARCHIVE_LAG_DAYS)
    return end.astimezone(KST).date() < cutoff


def fetch_wind(
    conn: sqlite3.Connection, points: list[tuple[float, float]], start: datetime, end: datetime
) -> list[GridSample]:
    """Hourly wind samples at (lat, lon) points for the KST days start..end; stored via
    store.upsert_grid_samples.

    Old ranges come from ERA5 (`openmeteo.era5`, 10/100 m) plus the historical forecast
    (`openmeteo.hist_fc`, 850 hPa); recent ones from the forecast API (`openmeteo.forecast`,
    all levels in one request). Wind speed is m/s.
    """
    now = datetime.now(KST)
    start_day = start.astimezone(KST).date()
    end_day = end.astimezone(KST).date()
    common = {"wind_speed_unit": "ms"}
    if use_archive(end, now):
        dates = {"start_date": start_day.isoformat(), "end_date": end_day.isoformat()}
        samples = _fetch_hourly(
            conn, ARCHIVE_URL, "openmeteo.archive", points, WIND_VARS, "openmeteo.era5",
            {**common, **dates, "models": "era5"},
        )
        samples += _fetch_hourly(
            conn, HIST_FORECAST_URL, "openmeteo.historical_forecast", points, WIND_850_VARS,
            "openmeteo.hist_fc", {**common, **dates},
        )
    else:
        today = now.date()
        past_days = max(0, (today - start_day).days)
        forecast_days = max(1, (end_day - today).days + 1)
        if past_days > MAX_PAST_DAYS or forecast_days > MAX_FORECAST_DAYS:
            raise ValueError(f"{start_day}..{end_day} is outside the forecast API window")
        samples = _fetch_hourly(
            conn, FORECAST_URL, "openmeteo.forecast", points, WIND_VARS + WIND_850_VARS,
            "openmeteo.forecast",
            {**common, "past_days": past_days, "forecast_days": forecast_days},
        )
        samples = [s for s in samples if start_day <= s.ts.date() <= end_day]
    store.upsert_grid_samples(conn, samples)
    return samples
