"""Hourly anomaly detection on observations, no model calls (owner: implementer C)."""

import sqlite3
from datetime import datetime

from .models import Candidate


def detect(conn: sqlite3.Connection, at: datetime) -> list[Candidate]:
    """Find anomaly candidates in the observations up to `at` (tz-aware KST)."""
    raise NotImplementedError


def detect_latest(conn: sqlite3.Connection) -> list[Candidate]:
    """detect() at the latest observation hour in the DB; saves via store.save_candidates."""
    raise NotImplementedError
