"""Open-Meteo wind and CAMS air-quality samples (owner: implementer B)."""

import sqlite3
from datetime import datetime

from ..models import GridSample


def fetch_wind(
    conn: sqlite3.Connection, points: list[tuple[float, float]], start: datetime, end: datetime
) -> list[GridSample]:
    """Hourly wind samples at (lat, lon) points; stored via store.upsert_grid_samples."""
    raise NotImplementedError


def fetch_air_quality(
    conn: sqlite3.Connection, points: list[tuple[float, float]], start: datetime, end: datetime
) -> list[GridSample]:
    """Hourly CAMS air-quality samples at (lat, lon) points; stored via upsert_grid_samples."""
    raise NotImplementedError
