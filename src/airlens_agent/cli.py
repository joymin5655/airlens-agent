"""Command line entry point."""

import argparse
import json
import sys
from datetime import datetime, timedelta

from . import agent, ask, config, detect, http, monitor, quota, replay, store, tools
from .events import get_event
from .sources import airkorea, firms, gibs, sam

SAM_PROMPTS = ("smoke", "smoke plume", "haze", "dust")
MAX_STEPS = 12  # upper bound for --max-steps: every step is a paid model call


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


def _event_dir(name: str):
    return config.event_db_path(name).with_suffix("")


def _pick_candidate(conn, name: str, event_id: str | None):
    """--event-id, else the top-severity regional candidate inside the event's focus bbox.

    Not the key day's: the key day picks satellite images, and the strongest impact can come
    days later (the 2025 Uiseong smoke peaked on 03-25, three days after ignition). Inside the
    focus bbox because the event's model grid and satellite images only cover that area; when
    no candidate lies there, the top one overall is used with a warning.
    """
    if event_id:
        found = store.candidate(conn, event_id)
        if found is None:
            raise SystemExit(f"no candidate {event_id!r} in event {name!r}")
        return found
    regional = "SELECT event_id FROM candidates WHERE kind = 'regional' "
    order = "ORDER BY severity DESC, ts, event_id LIMIT 1"  # ties go the same way every run
    w, s, e, n = get_event(name).focus_bbox
    row = conn.execute(
        regional + "AND centroid_lon BETWEEN ? AND ? AND centroid_lat BETWEEN ? AND ? " + order,
        (w, e, s, n),
    ).fetchone()
    if row is None:
        row = conn.execute(regional + order).fetchone()
        if row is not None:
            print(f"warning: no regional candidate inside the focus bbox of {name!r}; {row[0]} "
                  "lies outside it, so wind, CAMS and satellite evidence may be missing",
                  file=sys.stderr)
    if row is None:
        raise SystemExit(f"no regional candidates in event {name!r}; run `replay {name}` first")
    return store.candidate(conn, row[0])


def _ensure_images(name: str, cand, img_dir) -> None:
    """Fetch the two newest true-colour passes satellite_segments may use for `cand`, if not cached.

    ponytail: only two passes; if both come back blank the tool finds no image (upgrade: go on
    down tools.image_passes until a non-blank one)
    """
    for day, layer in tools.image_passes(cand)[:2]:
        if any(img_dir.glob(f"{day.isoformat()}_{layer}_*")):
            continue
        try:
            gibs.fetch_image(get_event(name).focus_bbox, day, layer, out_dir=img_dir)
        except http.HttpError as exc:  # the tool then reports that no image is cached
            print(f"warning: {exc}", file=sys.stderr)


def cmd_investigate(args) -> dict:
    if not config.event_db_path(args.name).exists():
        raise SystemExit(f"no data for event {args.name!r}; run `fetch-event {args.name}` first")
    conn = store.connect(config.event_db_path(args.name))
    ledger = _live()
    cand = _pick_candidate(conn, args.name, args.event_id)
    root = _event_dir(args.name)
    _ensure_images(args.name, cand, root / "images")
    max_steps = min(max(args.max_steps, 1), MAX_STEPS)
    if max_steps != args.max_steps:
        print(f"warning: --max-steps {args.max_steps} limited to {max_steps} "
              f"(allowed 1-{MAX_STEPS})", file=sys.stderr)
    before = quota.spent_total(ledger)
    tri = agent.triage(cand, ledger=ledger)
    report = agent.investigate(
        conn, cand, tri, ledger=ledger, investigator=args.investigator, max_steps=max_steps,
        image_dir=root / "images", out_dir=root / "reports",
    )
    # a neutral triage (invalid answer, failed call) has no model name
    report.models = list(dict.fromkeys(m for m in (tri.model_used, *report.models) if m))
    report.cost_usd = round(quota.spent_total(ledger) - before, 6)  # triage included
    out = root / "reports" / f"{cand.event_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n")
    total = len(report.claims)
    return {
        "event_id": cand.event_id,
        "cause": report.cause,
        "confidence": report.confidence,
        "claims_verified": f"{total - len(report.unverified_claims)}/{total}",
        "cost_usd": report.cost_usd,
        "models": report.models,
        "fallback": report.fallback,
        "triage_fallback": report.triage_fallback,
        "report": out.name,
    }


def cmd_sam_trial(args) -> dict:
    ev = get_event(args.name)
    ledger = _live()
    root = _event_dir(args.name)
    out_dir = root / "sam"
    before = quota.spent_total(ledger)
    results = []
    for offset in (-1, 0, 1):
        day = (ev.key_day + timedelta(days=offset)).date()
        for path in tools.true_color_images(root / "images", day):
            # a day can have a morning and an afternoon image: name overlays by layer too
            layer = next(name for name in gibs.TRUE_COLOR_LAYERS if name in path.name)
            for prompt in args.prompt or SAM_PROMPTS:
                regions = sam.segment(path, prompt, ledger=ledger)
                slug = prompt.replace(" ", "_")
                overlay = sam.overlay(
                    path, regions, out_dir / f"{day.isoformat()}_{layer}_{slug}.png"
                )
                results.append({
                    "date": day.isoformat(), "layer": layer, "prompt": prompt,
                    "n_regions": len(regions),
                    "overlay": overlay.name,
                    "regions": [
                        {"area_frac": round(r.area_frac, 3), "confidence": r.confidence,
                         "box_lonlat": [round(v, 2) for v in r.box_lonlat]}
                        for r in regions
                    ],
                })
    summary = {
        "event": args.name, "images_tried": sorted({r["date"] for r in results}),
        "cost_usd": round(quota.spent_total(ledger) - before, 6), "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def cmd_monitor(args) -> dict:
    return monitor.run_once(
        investigate=not args.no_investigate, max_investigations=args.max_investigations
    )


def cmd_ask(args) -> dict:
    conn = _live()
    if args.build_examples:
        return ask.build_examples(ledger=conn, live_conn=conn)
    if not args.question:
        raise SystemExit("give a question, or --build-examples")
    return ask.answer(
        args.question, ledger=conn, live_conn=conn, lat=args.lat, lon=args.lon
    ).model_dump()


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
    p = sub.add_parser("investigate")
    p.add_argument("name")
    p.add_argument("--event-id")
    p.add_argument("--investigator", choices=("meta", "baseten"), default="meta")
    p.add_argument("--max-steps", type=int, default=8)
    p.set_defaults(fn=cmd_investigate)
    p = sub.add_parser("sam-trial")
    p.add_argument("name")
    p.add_argument("--prompt", action="append")
    p.set_defaults(fn=cmd_sam_trial)
    p = sub.add_parser("monitor")
    p.add_argument("--no-investigate", action="store_true")
    p.add_argument("--max-investigations", type=int, default=1)
    p.set_defaults(fn=cmd_monitor)
    p = sub.add_parser("ask")
    p.add_argument("question", nargs="?")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--build-examples", action="store_true")
    p.set_defaults(fn=cmd_ask)
    args = parser.parse_args(argv)
    result = args.fn(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
