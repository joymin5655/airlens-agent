"""Per-operation daily API call ledger (KST day)."""

import sqlite3
from datetime import datetime
from pathlib import Path

from . import config
from .config import DAILY_SPEND_CAPS, KST, QUOTA_CAPS, SPEND_CAPS

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


def _own_file(conn: sqlite3.Connection) -> Path | None:
    row = conn.execute("PRAGMA database_list").fetchone()
    return Path(row[2]).resolve() if row and row[2] else None


def peer_spent(conn: sqlite3.Connection, provider: str, day: str | None = None) -> float:
    """USD that the other writers' ledger files (config.peer_ledger_paths) record for `provider`.

    A peer file is a complete snapshot (Modal: the other container uploads a checkpointed copy),
    so it is opened read-only and immutable: no -wal/-shm files are created next to it. A missing
    file or table counts as 0; a file that exists but cannot be read raises BudgetExceeded
    (fail closed: an unreadable ledger must not lift the cap).
    """
    own, total = _own_file(conn), 0.0
    for path in config.peer_ledger_paths():
        if not path.exists() or path.resolve() == own:
            continue
        sql, args = "SELECT COALESCE(SUM(usd), 0) FROM spend WHERE provider=?", [provider]
        if day is not None:
            sql, args = sql + " AND day=?", [*args, day]
        try:
            peer = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
            try:
                total += float(peer.execute(sql, args).fetchone()[0])
            finally:
                peer.close()
        except sqlite3.DatabaseError as exc:
            if "no such table" in str(exc):
                continue
            raise BudgetExceeded(f"{provider}: peer ledger {path.name} unreadable ({exc})") from exc
    return total


def ensure_budget(conn: sqlite3.Connection, provider: str) -> None:
    """Raise BudgetExceeded when the provider's all-time or today's (KST) USD cap is reached.

    Spend is this ledger plus the peer ledgers (other containers), see peer_spent.

    ponytail: a soft cap on past spend only. One call can overshoot it by its own cost, and the
    check is not atomic with the charge; fine while this CLI makes its paid calls one after
    another. A peer's spend is only as fresh as its last snapshot. Upgrade: reserve the call's
    estimated cost here, settle it in `charge`; one shared ledger service.
    """
    cap = SPEND_CAPS[provider]
    total = spent(conn, provider) + peer_spent(conn, provider)
    if total >= cap:
        raise BudgetExceeded(f"{provider}: budget ${cap:.2f} reached (spent ${total:.4f})")
    daily = DAILY_SPEND_CAPS.get(provider)
    if daily is not None:
        day = today()
        used = spent(conn, provider, day) + peer_spent(conn, provider, day)
        if used >= daily:
            raise BudgetExceeded(
                f"{provider}: daily budget ${daily:.2f} reached (spent ${used:.4f} on {day})"
            )


def budget_status(conn: sqlite3.Connection) -> dict[str, dict]:
    """Per provider: USD spent today and all-time (all writers) with their caps."""
    day = today()
    return {
        p: {
            "spent_today": round(spent(conn, p, day) + peer_spent(conn, p, day), 6),
            "daily_cap": DAILY_SPEND_CAPS.get(p),
            "spent_total": round(spent(conn, p) + peer_spent(conn, p), 6),
            "cap": cap,
        }
        for p, cap in SPEND_CAPS.items()
    }


def usage(conn: sqlite3.Connection, day: str | None = None) -> dict[str, dict]:
    conn.execute(SCHEMA)
    rows = conn.execute("SELECT op, n FROM api_calls WHERE day=?", (day or today(),)).fetchall()
    used = dict(rows)
    ops = sorted(set(used) | set(QUOTA_CAPS))
    return {op: {"used": used.get(op, 0), "cap": QUOTA_CAPS.get(op)} for op in ops}
