"""Command line entry point."""

import argparse
import json
import sys
from datetime import datetime

from . import config, detect, http, quota, replay, store
from .sources import airkorea, firms


def _live():
    return store.connect(config.live_db_path())


def _step(name: str, fn):
    """Run one optional step; a failed source or missing key is reported and skipped."""
    try:
        return fn()
    except (http.HttpError, config.MissingKeyError) as exc:
        print(f"[{name}] skipped: {exc}", file=sys.stderr)
        return None


def cmd_fetch_now(_args) -> dict:
    conn = _live()
    obs = airkorea.fetch_realtime_nation(conn)
    store.upsert_observations(conn, obs)
    hotspots = _step("firms", lambda: firms.fetch_hotspots(conn, firms.WATCH_BBOX))
    candidates = detect.detect_latest(conn)
    return {
        "observations": len(obs),
        "latest_ts": max((o.ts for o in obs), default=None),
        "hotspots": None if hotspots is None else len(hotspots),
        "candidates": len(candidates),
    }


def cmd_forecast(_args) -> dict:
    conn = _live()
    day = datetime.now(config.KST).date()
    items = airkorea.fetch_forecast(conn, day)
    return {"count": len(items), "items": [vars(f) for f in items]}


def cmd_stations(_args) -> dict:
    conn = _live()
    stations = airkorea.fetch_stations(conn)
    return {
        "stations": len(stations),
        "with_coords": sum(1 for s in stations if s.lat is not None),
    }


def cmd_quota(_args) -> dict:
    return quota.usage(_live())


def cmd_fetch_event(args) -> dict:
    return replay.fetch_event(args.name)


def cmd_replay(args) -> dict:
    return replay.run_replay(args.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="airlens-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (
        ("fetch-now", cmd_fetch_now), ("forecast", cmd_forecast),
        ("stations", cmd_stations), ("quota", cmd_quota),
    ):
        sub.add_parser(name).set_defaults(fn=fn)
    for name, fn in (("fetch-event", cmd_fetch_event), ("replay", cmd_replay)):
        p = sub.add_parser(name)
        p.add_argument("name")
        p.set_defaults(fn=fn)
    args = parser.parse_args(argv)
    result = args.fn(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
