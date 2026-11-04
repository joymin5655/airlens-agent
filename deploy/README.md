# Deploying AirLens Agent on Modal

`deploy/modal_app.py` defines two functions in one app (`airlens-agent`):

| Function | What | Resources |
|---|---|---|
| `monitor_hourly` | cron `10 * * * *` (UTC, every hour): fetch, detect, investigate at most one new event | 1 CPU, 1 GiB, timeout 15 min, `max_containers=1` |
| `web` | `airlens_agent.server:app` via `@modal.asgi_app()` | 0.5 CPU, 512 MiB, scales to zero after 120 s, `max_containers=1` |

## One-time setup

1. Secret `airlens-agent` from the local key file (values never go in git):
   `modal secret create airlens-agent DATA_GO_KR_SERVICE_KEY=... NASA_FIRMS_MAP_KEY=... MODEL_API_KEY=... BASETEN_API_KEY=...`
   (add `OPENAI_API_KEY`, `FIREWORKS_API_KEY` if used; empty values are fine). A convenient form:
2. Data: `deploy/upload_data.sh` (past events `fire-2025-03`, `dust-2026-02` and the live DB, whose
   spend table carries the model spend made so far into the cloud budget caps).
3. `modal deploy deploy/modal_app.py`
4. Check: `modal run deploy/modal_app.py::monitor_hourly` once by hand, then open the web URL
   and `/health`.

## How the two containers share data

SQLite needs one writer per file, and WAL files do not belong on a Volume. So:

- Volume `airlens-agent-data` is mounted at `/vol` and holds only complete snapshots.
- Both containers work on a local copy in `/srv/data` (`AIRLENS_AGENT_DATA`).
- **The hourly job is the only writer of `live.sqlite` and `live/`** (`status.json`, `reports/`,
  `images/`). It pulls them from the Volume, runs, writes a consistent snapshot back
  (SQLite backup API, journal mode DELETE) and calls `volume.commit()`.
- The web container never writes those. Every 60 s (checked on request) it calls
  `volume.reload()` and copies the new files to local disk (a failed reload, for example because
  a file is open, is skipped and retried). It serves from the local copy.
- The web container keeps its own spend ledger `/srv/data/web-ledger.sqlite`
  (`AIRLENS_LEDGER_FILE`). After a request that charged money it uploads a snapshot to
  `/web-ledger.sqlite` with `Volume.batch_upload` (lands without `commit()`).
- Each side reads the other's ledger read-only (`AIRLENS_LEDGER_PEERS`), so the budget checks in
  `quota.ensure_budget` count both. A peer's spend is only as fresh as its last snapshot
  (web to job: next `reload()` at the start of a run; job to web: next refresh), so a cap can be
  overshot by about one hour of the other side's spend.

**Contract for code that spends money in the web container:** charge a connection to
`config.ledger_db_path()` (not `live_db_path()`), and call `quota.ensure_budget` before the call.
In the web container `live.sqlite` is a read-only copy that is replaced on refresh, so spend
charged there would be lost.

## Hard caps

| Cap | Value | Where |
|---|---|---|
| Model spend per KST day, `meta` | $5.00 (`AIRLENS_DAILY_META_USD`) | `config.DAILY_SPEND_CAPS`, `quota.ensure_budget` |
| Model spend per KST day, `baseten` | $3.00 (`AIRLENS_DAILY_BASETEN_USD`) | same |
| Model spend all-time, per provider | $40 (`AIRLENS_BUDGET_<P>_USD`) | `config.SPEND_CAPS` |
| Investigations per KST day (hourly job) | 4 (`AIRLENS_MONITOR_PER_DAY`), at most 1 per hourly run | `config.MONITOR_INVESTIGATIONS_PER_DAY`, `monitor.run_once` |
| AirKorea realtime calls per KST day | 30 (`QUOTA_CAPS["airkorea.realtime"]`); 24 hourly runs use 24, leaving 6 for manual runs. A day with more than 30 attempts gets "source failed" in `status.json`, no crash | `quota.reserve` |
| AirKorea station list | 3 per day (fetched only when the station table is empty) | `QUOTA_CAPS` |
| Model calls per day (dev safety) | meta chat 400, meta SAM 300, baseten chat 400 | `QUOTA_CAPS` |
| SAM calls per investigation | 4 | `tools.MAX_SAM_CALLS` |
| Investigation severity floor | 2.0 (about "very bad" grade at 3 or more stations) | `monitor.MIN_SEVERITY` |
| Modal workspace | hard usage limit $27, spend limit $0 | Modal settings, not code |

Compute cost estimate: 24 runs of about 1 minute at 1 CPU per day plus an occasionally warm
0.5 CPU web container is a small fraction of a dollar per day; check the Modal Usage tab.

## Notes and limits

- The first days after a fresh start have a short baseline (detect needs 48 hours per station for
  z-scores; before that only "very bad" grades trigger candidates). Uploading `data/live.sqlite`
  does not help for the baseline, since realtime data is one hour per call.
- `live/images/` and `live.sqlite` grow without pruning (about 1.5 MB/day for observations).
  Fine for a hackathon; prune before running for months.
- Cron times are UTC on Modal: `10 * * * *` is minute 10 of every hour in any timezone.
- The `web/` folder, if present in the repo, is added to the image at `/root/web`.
