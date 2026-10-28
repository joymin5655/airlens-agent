import pytest

from airlens_agent import quota
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
