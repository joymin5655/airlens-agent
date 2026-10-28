"""Command line entry point."""

import argparse
import json
import sys
from datetime import datetime

from . import config, detect, quota, replay, store
from .sources import airkorea, firms


def _live():
    return store.connect(config.live_db_path())


def _step(name: str, fn):
    """Run one pipeline step; a not-yet-built step is reported and skipped."""
    try:
        return fn()
    except NotImplementedError:
        print(f"[{name}] not built yet, skipping", file=sys.stderr)
        return None


def cmd_fetch_now(_args) -> dict:
    conn = _live()
    obs = airkorea.fetch_realtime_nation(conn)
    store.upsert_observations(conn, obs)
    hotspots = _step("firms", lambda: firms.fetch_hotspots(conn, firms.WATCH_BBOX))
    candidates = _step("detect", lambda: detect.detect_latest(conn))
    return {
        "observations": len(obs),
        "latest_ts": max((o.ts for o in obs), default=None),
        "hotspots": None if hotspots is None else len(hotspots),
        "candidates": None if candidates is None else len(candidates),
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
    replay.fetch_event(args.name)
    return {"event": args.name, "status": "fetched"}


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
    try:
        result = args.fn(args)
    except NotImplementedError:
        print(f"{args.command}: not built yet", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
