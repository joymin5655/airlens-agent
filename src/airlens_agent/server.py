"""Read-only HTTP API over the live DB."""

from fastapi import FastAPI

from . import config, quota, store

app = FastAPI(title="AirLens Agent")


@app.get("/health")
def health() -> dict:
    conn = store.connect(config.live_db_path())
    try:
        last = store.last_observation_ts(conn)
        return {
            "status": "ok",
            "last_observation": last.isoformat() if last else None,
            "quota": quota.usage(conn),
        }
    finally:
        conn.close()


@app.get("/anomalies")
def anomalies(limit: int = 20) -> list[dict]:
    conn = store.connect(config.live_db_path())
    try:
        return store.latest_candidates(conn, limit)
    finally:
        conn.close()
