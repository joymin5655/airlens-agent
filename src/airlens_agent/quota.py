"""Per-operation daily API call ledger (KST day)."""

import sqlite3
from datetime import datetime

from .config import KST, QUOTA_CAPS, SPEND_CAPS

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_calls (
    op TEXT NOT NULL, day TEXT NOT NULL, n INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (op, day)
)"""
SPEND_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    provider TEXT NOT NULL, day TEXT NOT NULL, usd REAL NOT NULL DEFAULT 0,
    calls INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, day)
)"""


class QuotaExceeded(RuntimeError):
    pass


def today() -> str:
    return datetime.now(KST).date().isoformat()


def reserve(conn: sqlite3.Connection, op: str) -> None:
    """Count one call; raise QuotaExceeded if the op has a cap that is already reached."""
    conn.execute(SCHEMA)
    day = today()
    cap = QUOTA_CAPS.get(op)
    cur = conn.execute(
        "INSERT INTO api_calls(op, day, n) VALUES (:op, :day, 1) "
        "ON CONFLICT(op, day) DO UPDATE SET n = n + 1 WHERE :cap IS NULL OR n < :cap",
        {"op": op, "day": day, "cap": cap},
    )
    conn.commit()
    if cur.rowcount == 0:
        raise QuotaExceeded(f"{op}: daily cap {cap} reached ({day})")


class BudgetExceeded(RuntimeError):
    pass


def charge(conn: sqlite3.Connection, provider: str, usd: float) -> None:
    """Add the cost of one paid call to today's spend for `provider`."""
    conn.execute(SPEND_SCHEMA)
    conn.execute(
        "INSERT INTO spend(provider, day, usd, calls) VALUES (:p, :day, :usd, 1) "
        "ON CONFLICT(provider, day) DO UPDATE SET usd = usd + :usd, calls = calls + 1",
        {"p": provider, "day": today(), "usd": usd},
    )
    conn.commit()


def spent(conn: sqlite3.Connection, provider: str, day: str | None = None) -> float:
    """USD spent on `provider` on `day`, or all-time when `day` is None."""
    conn.execute(SPEND_SCHEMA)
    sql, args = "SELECT COALESCE(SUM(usd), 0) FROM spend WHERE provider=?", [provider]
    if day is not None:
        sql, args = sql + " AND day=?", [*args, day]
    return float(conn.execute(sql, args).fetchone()[0])


def spent_total(conn: sqlite3.Connection) -> float:
    """All-time USD spent across providers."""
    return sum(spent(conn, p) for p in SPEND_CAPS)


def ensure_budget(conn: sqlite3.Connection, provider: str) -> None:
    """Raise BudgetExceeded when all-time spend has reached the provider's USD cap.

    ponytail: a soft cap on past spend only. One call can overshoot it by its own cost, and the
    check is not atomic with the charge; fine while this CLI makes its paid calls one after
    another. Upgrade: reserve the call's estimated cost here, settle it in `charge`.
    """
    cap = SPEND_CAPS[provider]
    total = spent(conn, provider)
    if total >= cap:
        raise BudgetExceeded(f"{provider}: budget ${cap:.2f} reached (spent ${total:.4f})")


def usage(conn: sqlite3.Connection, day: str | None = None) -> dict[str, dict]:
    conn.execute(SCHEMA)
    rows = conn.execute("SELECT op, n FROM api_calls WHERE day=?", (day or today(),)).fetchall()
    used = dict(rows)
    ops = sorted(set(used) | set(QUOTA_CAPS))
    return {op: {"used": used.get(op, 0), "cap": QUOTA_CAPS.get(op)} for op in ops}
