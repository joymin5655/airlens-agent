"""Environment, paths and quota caps."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import find_dotenv, load_dotenv

# usecwd=True walks up from the cwd, so a repo-root .env is found from .worktrees/<name>/ too.
load_dotenv(find_dotenv(usecwd=True))

KST = ZoneInfo("Asia/Seoul")

# AirKorea allows 500/day per operation, shared with AirLens (up to 408/day).
QUOTA_CAPS: dict[str, int] = {
    "airkorea.realtime": 30,
    "airkorea.forecast": 6,
    "airkorea.stations": 3,
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


def data_dir() -> Path:
    return Path(os.environ.get("AIRLENS_AGENT_DATA") or Path.cwd() / "data")


def live_db_path() -> Path:
    return data_dir() / "live.sqlite"


def event_db_path(name: str) -> Path:
    return data_dir() / "events" / f"{name}.sqlite"
