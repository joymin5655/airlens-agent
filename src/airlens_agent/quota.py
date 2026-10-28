"""Per-operation daily API call ledger (KST day)."""

import sqlite3
from datetime import datetime

from .config import KST, QUOTA_CAPS

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_calls (
    op TEXT NOT NULL, day TEXT NOT NULL, n INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (op, day)
)"""


class QuotaExceeded(RuntimeError):
    pass


def today() -> str:
    return datetime.now(KST).date().isoformat()


def reserve(conn: sqlite3.Connection, op: str) -> None:
    """Count one call; raise QuotaExceeded if the op has a cap that is already reached."""
    conn.execute(SCHEMA)
    day = today()
    row = conn.execute("SELECT n FROM api_calls WHERE op=? AND day=?", (op, day)).fetchone()
    n = row[0] if row else 0
    cap = QUOTA_CAPS.get(op)
    if cap is not None and n >= cap:
        raise QuotaExceeded(f"{op}: daily cap {cap} reached ({day})")
    conn.execute(
        "INSERT INTO api_calls(op, day, n) VALUES (?, ?, 1) "
        "ON CONFLICT(op, day) DO UPDATE SET n = n + 1",
        (op, day),
    )
    conn.commit()


def usage(conn: sqlite3.Connection, day: str | None = None) -> dict[str, dict]:
    conn.execute(SCHEMA)
    rows = conn.execute("SELECT op, n FROM api_calls WHERE day=?", (day or today(),)).fetchall()
    used = dict(rows)
    ops = sorted(set(used) | set(QUOTA_CAPS))
    return {op: {"used": used.get(op, 0), "cap": QUOTA_CAPS.get(op)} for op in ops}
