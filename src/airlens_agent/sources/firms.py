"""NASA FIRMS fire hotspots (owner: implementer B)."""

import sqlite3
from datetime import date

from ..models import Hotspot

LIVE_SOURCES = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT")
WATCH_BBOX = (110.0, 30.0, 132.0, 45.0)  # west, south, east, north


def fetch_hotspots(
    conn: sqlite3.Connection,
    bbox: tuple[float, float, float, float],
    *,
    day_range: int = 1,
    day: date | None = None,
    sources: tuple[str, ...] = LIVE_SOURCES,
) -> list[Hotspot]:
    """Fetch hotspots inside bbox for `day_range` days ending at `day` (default: latest),
    store them with store.insert_hotspots and return them."""
    raise NotImplementedError
