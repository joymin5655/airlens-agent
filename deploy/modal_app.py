"""Modal app: the public FastAPI site plus the hourly live monitor.

Deploy (after review): `modal deploy deploy/modal_app.py`. Design and caps: deploy/README.md.

One rule shapes everything: a SQLite file has exactly one writer, and WAL files do not belong on
a Volume. So both containers work on a local copy under /srv/data (AIRLENS_AGENT_DATA) and the
Volume (/vol) only holds complete snapshots:

- the hourly job is the only writer of /vol/live.sqlite and /vol/live/ (status, reports, images);
- the web container never writes them; it copies them to local disk and keeps its own spend
  ledger (AIRLENS_LEDGER_FILE) in a separate file, uploaded to /vol/web-ledger.sqlite;
- each side reads the other's ledger snapshot read-only (AIRLENS_LEDGER_PEERS), so the daily
  and all-time USD caps hold across both.
"""

import asyncio
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
VOL = "/vol"  # the Volume mount: snapshots only
LOCAL = "/srv/data"  # each container's working copy (AIRLENS_AGENT_DATA)
WEB_LEDGER = "web-ledger.sqlite"
REFRESH_SECONDS = 60  # how often the web container re-reads the Volume

app = modal.App("airlens-agent")
volume = modal.Volume.from_name("airlens-agent-data", create_if_missing=True)
# the user creates this from the local key file: see deploy/README.md
secret = modal.Secret.from_name("airlens-agent")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(REPO), frozen=True)  # pyproject.toml + uv.lock, locked; project itself is below
    .env({"AIRLENS_AGENT_DATA": LOCAL})
    .add_local_dir(
        str(REPO / "src" / "airlens_agent"), "/root/airlens_agent",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)
if (REPO / "web").is_dir():  # the static front end, added by the web lane
    image = image.add_local_dir(str(REPO / "web"), "/root/web")

# per-function environment: which ledger each container writes and which one it only reads
CRON_ENV = modal.Secret.from_dict({"AIRLENS_LEDGER_PEERS": f"{VOL}/{WEB_LEDGER}"})
WEB_ENV = modal.Secret.from_dict({
    "AIRLENS_LEDGER_FILE": f"{LOCAL}/{WEB_LEDGER}",
    "AIRLENS_LEDGER_PEERS": f"{LOCAL}/live.sqlite",
})


# --- Volume <-> local copies ------------------------------------------------------------------


def _copy_file(src: Path, dst: Path) -> None:
    """Copy via a temp name and rename; an open reader keeps the old file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy files that are new or changed; never SQLite -wal/-shm files."""
    if not src.is_dir():
        return
    for path in src.rglob("*"):
        if not path.is_file() or path.name.endswith(("-wal", "-shm", ".tmp")):
            continue
        out = dst / path.relative_to(src)
        if out.exists() and out.stat().st_size == path.stat().st_size and (
            out.stat().st_mtime >= path.stat().st_mtime
        ):
            continue
        _copy_file(path, out)


def _snapshot(src: Path, dst: Path) -> None:
    """A consistent standalone copy of a (possibly live, WAL-mode) SQLite file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    source, target = sqlite3.connect(src), sqlite3.connect(tmp)
    try:
        source.backup(target)
        target.execute("PRAGMA journal_mode=DELETE")  # no -wal/-shm needed to read the copy
    finally:
        target.close()
        source.close()
    os.replace(tmp, dst)


def pull_for_cron() -> None:
    """The job continues from the last committed live DB and live/ folder."""
    _copy_tree(Path(VOL) / "live", Path(LOCAL) / "live")
    if (Path(VOL) / "live.sqlite").exists():
        _copy_file(Path(VOL) / "live.sqlite", Path(LOCAL) / "live.sqlite")


def push_from_cron() -> None:
    local = Path(LOCAL)
    if (local / "live.sqlite").exists():
        _snapshot(local / "live.sqlite", Path(VOL) / "live.sqlite")
    _copy_tree(local / "live", Path(VOL) / "live")


def pull_for_web() -> None:
    """Local copies of everything the site reads: the live snapshot, the past events, the saved
    example answers (ask/) and the latest model comparison (eval/). Our own ledger is restored
    only when missing (the local file is newer than any upload)."""
    vol, local = Path(VOL), Path(LOCAL)
    for folder in ("live", "events", "ask", "eval"):
        _copy_tree(vol / folder, local / folder)
    for name in ("live.sqlite", WEB_LEDGER):
        if (vol / name).exists() and (name == "live.sqlite" or not (local / name).exists()):
            _copy_file(vol / name, local / name)


def _ledger_fingerprint(path: Path) -> tuple[float, int] | None:
    """(USD, calls) of the ledger: changes exactly when a paid call was charged."""
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(path)
        try:
            return tuple(conn.execute(
                "SELECT COALESCE(SUM(usd), 0), COALESCE(SUM(calls), 0) FROM spend").fetchone())
        finally:
            conn.close()
    except sqlite3.OperationalError:  # no spend table yet
        return None


class VolumeSync:
    """ASGI wrapper around the FastAPI app: refresh local copies from the Volume every
    REFRESH_SECONDS, and upload the web ledger after a request that charged money.

    ponytail: one container (max_containers=1) so one ledger writer; the cron job sees a web
    charge only after its next volume.reload(), so across the two a cap can be overshot by one
    hour of web spend. Upgrade: a shared ledger service.
    """

    def __init__(self, inner):
        self.inner = inner
        self.lock = threading.Lock()
        self.refreshed = time.monotonic()
        self.pushed = _ledger_fingerprint(Path(LOCAL) / WEB_LEDGER)

    def _refresh(self) -> None:
        with self.lock:
            if time.monotonic() - self.refreshed < REFRESH_SECONDS:
                return
            self.refreshed = time.monotonic()
            try:
                volume.reload()  # fails while a file on the Volume is open: try again later
            except Exception as exc:
                print(f"volume.reload skipped: {exc}")
                return
            pull_for_web()

    def _push_ledger(self) -> None:
        with self.lock:
            path = Path(LOCAL) / WEB_LEDGER
            now = _ledger_fingerprint(path)
            if now is None or now == self.pushed:
                return
            snap = Path("/tmp/web-ledger.snapshot.sqlite")
            _snapshot(path, snap)
            with volume.batch_upload(force=True) as batch:  # lands without volume.commit()
                batch.put_file(str(snap), f"/{WEB_LEDGER}")
            self.pushed = now

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            await asyncio.to_thread(self._refresh)
        await self.inner(scope, receive, send)
        if scope["type"] == "http":
            await asyncio.to_thread(self._push_ledger)


# --- functions ---------------------------------------------------------------------------------


@app.function(
    image=image, volumes={VOL: volume}, secrets=[secret, CRON_ENV],
    schedule=modal.Cron("10 * * * *"),  # minute 10 of every hour (UTC); AirKorea publishes ~:00
    timeout=15 * 60, max_containers=1, cpu=1.0, memory=1024,
)
def monitor_hourly() -> dict:
    """Fetch, detect, investigate at most one new event, then publish a snapshot."""
    volume.reload()  # see the web container's latest ledger upload before checking budgets
    pull_for_cron()
    from airlens_agent import monitor  # after pull: reads AIRLENS_AGENT_DATA at call time

    try:
        return monitor.run_once()
    finally:
        push_from_cron()  # also after a failure: keep the fetched observations
        volume.commit()


@app.function(
    image=image, volumes={VOL: volume}, secrets=[secret, WEB_ENV],
    max_containers=1, scaledown_window=120, cpu=0.5, memory=512,  # CPU only, scales to zero
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    volume.reload()
    pull_for_web()
    from airlens_agent.server import app as fastapi_app

    return VolumeSync(fastapi_app)
