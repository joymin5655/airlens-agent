"""Environment, paths and quota caps."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Checkout root from this file (src/airlens_agent/config.py), never from the cwd. A worktree at
# <repo>/.worktrees/<name>/ also reads the main checkout's .env and shares its data/ folder.
# ponytail: assumes a source checkout (uv run); a packaged deploy sets AIRLENS_AGENT_DATA and
# env vars.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = PROJECT_ROOT.parent.parent if PROJECT_ROOT.parent.name == ".worktrees" else PROJECT_ROOT


def env_files() -> list[Path]:
    """.env files read at import, nearest first; nothing above the checkout is read."""
    return list(dict.fromkeys([PROJECT_ROOT / ".env", MAIN_ROOT / ".env"]))


for _path in env_files():
    load_dotenv(_path)  # a missing file is skipped; set variables are never overridden

KST = ZoneInfo("Asia/Seoul")

# AirKorea allows 500/day per operation, shared with AirLens (up to 408/day).
QUOTA_CAPS: dict[str, int] = {
    "airkorea.realtime": 30,
    "airkorea.forecast": 6,
    "airkorea.stations": 3,
    # Daily dev safety caps for paid model calls; the USD ledger (SPEND_CAPS) is the real limit.
    "meta.chat": 400,
    "meta.sam": 300,
    "baseten.chat": 400,
}

META_API_BASE = "https://api.meta.ai/v1"
BASETEN_API_BASE = "https://inference.baseten.co/v1"
MUSE_SPARK_MODEL = "muse-spark-1.3-contributor"
MUSE_SPARK_OPEN_MODEL = "meta-models/muse-spark-open-1026"
SAM_MODEL = "sam-3.1"

# All-time USD budget per provider (dev and test share it); override with AIRLENS_BUDGET_<P>_USD.
SPEND_CAPS: dict[str, float] = {
    "meta": float(os.environ.get("AIRLENS_BUDGET_META_USD") or 40.0),
    "baseten": float(os.environ.get("AIRLENS_BUDGET_BASETEN_USD") or 40.0),
}


class MissingKeyError(RuntimeError):
    pass


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingKeyError(f"environment variable {name} is not set (see .env.example)")
    return value


def data_go_kr_key() -> str:
    return _require("DATA_GO_KR_SERVICE_KEY")


def firms_map_key() -> str:
    return _require("NASA_FIRMS_MAP_KEY")


def model_api_key() -> str:
    return _require("MODEL_API_KEY")


def baseten_api_key() -> str:
    return _require("BASETEN_API_KEY")


def data_dir() -> Path:
    return Path(os.environ.get("AIRLENS_AGENT_DATA") or MAIN_ROOT / "data")


def live_db_path() -> Path:
    return data_dir() / "live.sqlite"


def event_db_path(name: str) -> Path:
    return data_dir() / "events" / f"{name}.sqlite"
