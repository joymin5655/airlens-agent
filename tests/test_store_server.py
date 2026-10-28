from datetime import UTC, datetime

from fastapi.testclient import TestClient

from airlens_agent import config, server, store
from airlens_agent.config import KST
from airlens_agent.models import Candidate, GridSample, Hotspot, Observation, Station

TS = datetime(2026, 1, 2, 10, tzinfo=KST)


def test_upserts_are_idempotent(conn):
    st = [Station("서울|중구", "중구", "서울", 37.5, 127.0, None, "t")]
    obs = [Observation("서울|중구", TS, pm10=40.0, flags={"pm25": "통신장애"})]
    for _ in range(2):
        store.upsert_stations(conn, st)
        store.upsert_observations(conn, obs)
    assert len(store.stations_df(conn)) == 1
    df = store.observations_between(conn, TS, TS)
    assert len(df) == 1 and df.iloc[0]["pm10"] == 40.0
    assert df.iloc[0]["ts"] == TS
    assert store.last_observation_ts(conn) == TS


def test_hotspot_dedupe_and_grid_upsert(conn):
    h = Hotspot(36.1, 128.5, datetime(2026, 1, 2, 4, tzinfo=UTC), "N20", "VIIRS", 3.2)
    assert store.insert_hotspots(conn, [h]) == 1
    assert store.insert_hotspots(conn, [h]) == 0
    df = store.hotspots_between(conn, datetime(2026, 1, 1, tzinfo=UTC),
                                datetime(2026, 1, 3, tzinfo=UTC))
    assert len(df) == 1
    g = GridSample(37.0, 127.0, TS, "openmeteo", {"wind_speed_10m": 3.0})
    store.upsert_grid_samples(conn, [g])
    store.upsert_grid_samples(conn, [g])
    assert conn.execute("SELECT COUNT(*) FROM grid_samples").fetchone()[0] == 1


def _cand(event_id: str) -> Candidate:
    return Candidate(event_id, TS, "regional", "coarse", ["서울|중구"], 37.5, 127.0,
                     (126.0, 36.0, 128.0, 38.0), 300.0, 100.0, 0.33, 2.5, {"n": 3})


def test_candidates_roundtrip(conn):
    store.save_candidates(conn, [_cand("e1")])
    store.save_candidates(conn, [_cand("e1")])
    (row,) = store.latest_candidates(conn, 10)
    assert row["event_id"] == "e1" and row["bbox"] == [126.0, 36.0, 128.0, 38.0]
    assert row["features"] == {"n": 3}


def test_server_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path))
    c = store.connect(config.live_db_path())
    store.upsert_observations(c, [Observation("서울|중구", TS, pm10=1.0)])
    store.save_candidates(c, [_cand("e1")])
    c.close()
    client = TestClient(server.app)
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["last_observation"].startswith("2026-01-02T10")
    assert health["quota"]["airkorea.realtime"]["cap"] == 30
    anomalies = client.get("/anomalies?limit=5").json()
    assert [a["event_id"] for a in anomalies] == ["e1"]


def test_observations_between_empty_is_tz_aware(conn):
    df = store.observations_between(conn, TS, TS)
    assert df.empty and df["ts"].dt.tz is not None


def test_anomalies_limit_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path))
    client = TestClient(server.app)
    for limit in (-1, 0, 101):
        assert client.get(f"/anomalies?limit={limit}").status_code == 422
