import pytest

from airlens_agent import config, quota, store
from airlens_agent.config import QUOTA_CAPS


def test_cap_enforced(conn):
    for _ in range(QUOTA_CAPS["airkorea.stations"]):
        quota.reserve(conn, "airkorea.stations")
    with pytest.raises(quota.QuotaExceeded):
        quota.reserve(conn, "airkorea.stations")
    assert quota.usage(conn)["airkorea.stations"] == {"used": 3, "cap": 3}


def test_uncapped_ops_only_count(conn):
    for _ in range(100):
        quota.reserve(conn, "firms.area")
    assert quota.usage(conn)["firms.area"] == {"used": 100, "cap": None}


def test_other_day_not_counted(conn):
    quota.reserve(conn, "airkorea.realtime")
    assert quota.usage(conn, day="2000-01-01")["airkorea.realtime"]["used"] == 0


def test_refused_reserve_leaves_usage_at_cap(conn):
    cap = QUOTA_CAPS["airkorea.forecast"]
    for _ in range(cap + 3):
        try:
            quota.reserve(conn, "airkorea.forecast")
        except quota.QuotaExceeded:
            pass
    assert quota.usage(conn)["airkorea.forecast"]["used"] == cap


# --- daily caps and peer ledgers ---------------------------------------------------------------


def test_daily_cap_blocks_even_under_the_all_time_cap(conn):
    quota.ensure_budget(conn, "meta")
    quota.charge(conn, "meta", config.DAILY_SPEND_CAPS["meta"])
    with pytest.raises(quota.BudgetExceeded, match="daily"):
        quota.ensure_budget(conn, "meta")
    quota.ensure_budget(conn, "baseten")  # other providers are counted separately


def test_yesterdays_spend_does_not_count_against_the_daily_cap(conn):
    conn.execute(quota.SPEND_SCHEMA)
    conn.execute("INSERT INTO spend VALUES ('meta', '2000-01-01', 4.99, 1)")
    quota.ensure_budget(conn, "meta")


def test_provider_without_a_daily_cap_only_has_the_all_time_cap(conn, monkeypatch):
    monkeypatch.delitem(config.DAILY_SPEND_CAPS, "baseten")
    quota.charge(conn, "baseten", 10.0)
    quota.ensure_budget(conn, "baseten")
    assert quota.budget_status(conn)["baseten"]["daily_cap"] is None


def test_budget_status(conn):
    quota.charge(conn, "meta", 1.5)
    s = quota.budget_status(conn)["meta"]
    assert s == {"spent_today": 1.5, "daily_cap": config.DAILY_SPEND_CAPS["meta"],
                 "spent_total": 1.5, "cap": config.SPEND_CAPS["meta"]}


def _peer_ledger(tmp_path, usd):
    peer = store.connect(tmp_path / "peer.sqlite")
    quota.charge(peer, "meta", usd)
    peer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    peer.close()
    return tmp_path / "peer.sqlite"


def test_peer_ledger_spend_counts_toward_the_caps(conn, tmp_path, monkeypatch):
    peer = _peer_ledger(tmp_path, config.DAILY_SPEND_CAPS["meta"])
    monkeypatch.setenv("AIRLENS_LEDGER_PEERS", str(peer))
    with pytest.raises(quota.BudgetExceeded, match="daily"):
        quota.ensure_budget(conn, "meta")
    assert quota.budget_status(conn)["meta"]["spent_today"] == config.DAILY_SPEND_CAPS["meta"]


def test_own_file_is_not_counted_twice_as_a_peer(conn, tmp_path, monkeypatch):
    quota.charge(conn, "meta", 1.0)
    own = conn.execute("PRAGMA database_list").fetchone()[2]
    monkeypatch.setenv("AIRLENS_LEDGER_PEERS", own)
    assert quota.budget_status(conn)["meta"]["spent_total"] == 1.0


def test_missing_peer_file_counts_zero_but_an_unreadable_one_fails_closed(
    conn, tmp_path, monkeypatch
):
    monkeypatch.setenv("AIRLENS_LEDGER_PEERS", str(tmp_path / "nope.sqlite"))
    quota.ensure_budget(conn, "meta")
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"this is not a database" * 50)
    monkeypatch.setenv("AIRLENS_LEDGER_PEERS", str(bad))
    with pytest.raises(quota.BudgetExceeded, match="unreadable"):
        quota.ensure_budget(conn, "meta")
