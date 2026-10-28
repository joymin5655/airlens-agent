"""NASA FIRMS fire hotspots.

The MAP_KEY sits in the URL path, so URLs are never logged and errors go through http.redact.
"""

import csv
import io
import sqlite3
from datetime import UTC, date, datetime, timedelta

from .. import config, http, store
from ..models import Hotspot

LIVE_SOURCES = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT")
HISTORY_SOURCES = ("VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP", "VIIRS_SNPP_SP", "MODIS_SP")
WATCH_BBOX = (110.0, 30.0, 132.0, 45.0)  # west, south, east, north

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_DAY_RANGE = 5  # FIRMS area API: DAY_RANGE is 1..5


def _area_url(
    key: str, source: str, bbox: tuple[float, float, float, float], day_range: int, day: date | None
) -> str:
    west, south, east, north = bbox
    url = f"{BASE_URL}/{key}/{source}/{west},{south},{east},{north}/{day_range}"
    return f"{url}/{day.isoformat()}" if day else url


def _num(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def parse_csv(text: str, source: str) -> list[Hotspot]:
    """Parse a FIRMS area CSV (VIIRS or MODIS) by header name.

    Non-CSV bodies (FIRMS answers an invalid key or quota problem with plain text) raise
    http.HttpError with the body redacted. A header-only body means no hotspots.
    """
    body = text.strip()
    if not body:
        return []
    header = body.splitlines()[0].lower()
    if "latitude" not in header or "longitude" not in header:
        snippet = http.redact(body)[:120]
        raise http.HttpError(f"firms.area: non-CSV response: {snippet!r}")
    rows = []
    for row in csv.DictReader(io.StringIO(body)):
        acq_time = (row.get("acq_time") or "0").zfill(4)
        ts = datetime.strptime(f"{row['acq_date']} {acq_time}", "%Y-%m-%d %H%M").replace(tzinfo=UTC)
        rows.append(
            Hotspot(
                lat=float(row["latitude"]),
                lon=float(row["longitude"]),
                ts_utc=ts,
                satellite=row.get("satellite") or "",
                source=source,
                frp=_num(row.get("frp")),
                confidence=row.get("confidence") or None,
                daynight=row.get("daynight") or None,
            )
        )
    return rows


def _fetch_one(
    conn: sqlite3.Connection,
    bbox: tuple[float, float, float, float],
    source: str,
    day_range: int,
    day: date | None,
) -> list[Hotspot]:
    if not 1 <= day_range <= MAX_DAY_RANGE:
        raise ValueError(f"day_range must be 1..{MAX_DAY_RANGE}, got {day_range}")
    url = _area_url(config.firms_map_key(), source, bbox, day_range, day)
    resp = http.get(url, op="firms.area", conn=conn)
    return parse_csv(resp.text, source)


def fetch_hotspots(
    conn: sqlite3.Connection,
    bbox: tuple[float, float, float, float],
    *,
    day_range: int = 2,
    day: date | None = None,
    sources: tuple[str, ...] = LIVE_SOURCES,
) -> list[Hotspot]:
    """Fetch hotspots inside bbox for `day_range` days starting at `day` (default: latest data),
    one call per source; store them with store.insert_hotspots and return them.

    Without `day` FIRMS counts back from now in UTC, so day_range=1 is empty early in the UTC
    day (live check: 0 rows at 02:00 UTC, 765 with day_range=2). Hence the default of 2; the
    store dedupes the overlap."""
    hotspots: list[Hotspot] = []
    for source in sources:
        hotspots.extend(_fetch_one(conn, bbox, source, day_range, day))
    store.insert_hotspots(conn, hotspots)
    return hotspots


def _chunks(start: date, end: date) -> list[tuple[date, int]]:
    """Split start..end (inclusive) into (first_day, day_range) chunks of at most 5 days."""
    out = []
    cur = start
    while cur <= end:
        n = min(MAX_DAY_RANGE, (end - cur).days + 1)
        out.append((cur, n))
        cur += timedelta(days=n)
    return out


def fetch_hotspots_range(
    conn: sqlite3.Connection,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    *,
    sources: tuple[str, ...] = HISTORY_SOURCES,
    fallbacks: list[tuple[str, date]] | None = None,
) -> list[Hotspot]:
    """Fetch start..end (inclusive) in <=5-day chunks per source and store the hotspots.

    If an `_SP` request returns no rows for a chunk, that chunk is retried with the matching
    `_NRT` source (SP may not exist yet for recent months). `Hotspot.source` records the source
    actually used; `fallbacks`, if given, collects (sp_source, chunk_start) for each retry.
    """
    hotspots: list[Hotspot] = []
    for source in sources:
        for first, n in _chunks(start, end):
            rows = _fetch_one(conn, bbox, source, n, first)
            if not rows and source.endswith("_SP"):
                if fallbacks is not None:
                    fallbacks.append((source, first))
                rows = _fetch_one(conn, bbox, source.removesuffix("_SP") + "_NRT", n, first)
            hotspots.extend(rows)
    store.insert_hotspots(conn, hotspots)
    return hotspots
