"""Compare the investigation pipeline across models on the past events.

Each (event, model, run) picks the same candidate as `investigate`, triages and investigates with
that model's own provider and no cross-provider fallback: a failure counts as a failure. Reports
go under `<data>/eval/reports/`, never over the demo reports in `<data>/events/*/reports/`.
"""

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from . import agent, cli, config, llm, quota, store

# Ground truth for the two past events, with the reason it is what it is.
GROUND_TRUTH: dict[str, tuple[str, str]] = {
    "fire-2025-03": ("wildfire_smoke", "Uiseong wildfire from 2025-03-22, smoke plumes in image"),
    "dust-2026-02": ("dust", "Seoul PM10 reached 311 ug/m3 on 2026-02-22, coarse particles"),
}

# ponytail: a flat guess of one investigation's calls and size, not a measurement. Upgrade: use
# the mean cost of a first measured run.
EST_CALLS, EST_INPUT_TOKENS, EST_OUTPUT_TOKENS = 12, 30_000, 8_192


class EvalError(RuntimeError):
    """The eval cannot start (no key, or the remaining budget cannot cover the runs)."""


def estimate_investigation_usd(provider: str) -> float:
    """Conservative cost of one investigation: every call at full input and output size."""
    p = llm.PROVIDERS[provider]
    per_call = (EST_INPUT_TOKENS * p.usd_per_m_input + EST_OUTPUT_TOKENS * p.usd_per_m_output) / 1e6
    return EST_CALLS * per_call


def preflight(events, models, runs, ledger) -> None:
    for name in events:
        if name not in GROUND_TRUTH:
            raise EvalError(f"no ground truth for event {name!r}; known: {sorted(GROUND_TRUTH)}")
        if not config.event_db_path(name).exists():
            raise EvalError(f"no data for event {name!r}; run `fetch-event {name}` first")
    for model in models:
        if model not in llm.PROVIDERS:
            raise EvalError(f"unknown model {model!r}; choose from {sorted(llm.PROVIDERS)}")
        try:
            llm.PROVIDERS[model].key()
        except config.MissingKeyError as exc:
            raise EvalError(str(exc)) from exc
        need = estimate_investigation_usd(model) * runs * len(events)
        left = config.SPEND_CAPS[model] - quota.spent(ledger, model)
        if need > left:
            raise EvalError(
                f"{model}: estimated ${need:.2f} for {runs * len(events)} investigations exceeds "
                f"the remaining budget ${left:.2f}"
            )


def _failed(report: agent.Report) -> bool:
    """A degraded run (provider switch, neutral triage, no final report) is a failure."""
    no_report = any(x.startswith("The final report failed") for x in report.limitations)
    return bool(report.fallback or report.triage_fallback or no_report)


def _one(model, event, run, ledger, out_root, transport) -> dict:
    truth = GROUND_TRUTH[event][0]
    row: dict = {"event": event, "model": model, "run": run, "truth": truth}
    conn = store.connect(config.event_db_path(event))
    before_model, before_all = quota.spent(ledger, model), quota.spent_total(ledger)
    start = time.monotonic()
    try:
        cand = cli._pick_candidate(conn, event, None)
        root = config.event_db_path(event).with_suffix("")
        cli._ensure_images(event, cand, root / "images")
        out_dir = out_root / "reports" / model / event
        tri = agent.triage(cand, ledger=ledger, provider=model, transport=transport,
                           allow_fallback=False)
        report = agent.investigate(
            conn, cand, tri, ledger=ledger, investigator=model, transport=transport,
            allow_fallback=False, image_dir=root / "images", out_dir=out_dir,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{cand.event_id}_run{run}.json"
        path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        )
        total = len(report.claims)
        row.update(
            candidate=cand.event_id, cause=report.cause, confidence=report.confidence,
            correct=report.cause == truth and not _failed(report),
            claims_total=total, claims_verified=total - len(report.unverified_claims),
            tool_calls=sum(1 for i in report.evidence if i.get("tool") != "detector"),
            fallback=report.fallback, triage_fallback=report.triage_fallback,
            failed=_failed(report), error=None, report=str(path.relative_to(out_root)),
        )
    except Exception as exc:  # noqa: BLE001 - any failure is a data point, not a crash
        row.update(
            cause=None, confidence=None, correct=False, claims_total=0, claims_verified=0,
            tool_calls=0, fallback=None, triage_fallback=None, failed=True,
            error=agent._why(exc), report=None,
        )
    finally:
        conn.close()
    row["wall_s"] = round(time.monotonic() - start, 2)
    after_model, after_all = quota.spent(ledger, model), quota.spent_total(ledger)
    row["cost_usd"] = round(after_model - before_model, 6)  # the model's own provider
    row["tool_cost_usd"] = round((after_all - before_all) - (after_model - before_model), 6)
    # ponytail: token counts are not kept by the agent; cost is the ledger delta
    return row


def summarize(rows: list[dict]) -> dict:
    """Per-model numbers over the model's rows."""
    out = {}
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        out[model] = {"all": _stats(mine)}
        for event in dict.fromkeys(r["event"] for r in mine):
            out[model][event] = _stats([r for r in mine if r["event"] == event])
    return out


def _stats(rows: list[dict]) -> dict:
    correct = [r for r in rows if r["correct"]]
    claims = sum(r["claims_total"] for r in rows)
    return {
        "runs": len(rows),
        "correct": len(correct),
        "mean_confidence_correct": (
            round(statistics.fmean(r["confidence"] for r in correct), 3) if correct else None
        ),
        "verified_claim_rate": (
            round(sum(r["claims_verified"] for r in rows) / claims, 3) if claims else None
        ),
        "median_latency_s": round(statistics.median(r["wall_s"] for r in rows), 1),
        "mean_cost_usd": round(statistics.fmean(r["cost_usd"] for r in rows), 4),
        "failures": sum(1 for r in rows if r["failed"]),
    }


def _fmt(v, spec="") -> str:
    return "-" if v is None else format(v, spec)


def render_tables(result: dict) -> str:
    """One Markdown table per model: a row per event and an `all` row."""
    parts = []
    for model, groups in result["summary"].items():
        parts += [
            f"### {model}", "",
            "| event | accuracy | mean confidence (correct) | verified claims | "
            "median latency (s) | mean cost (USD) | failures |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, s in groups.items():
            parts.append(
                f"| {name} | {s['correct']}/{s['runs']} | {_fmt(s['mean_confidence_correct'])} | "
                f"{_fmt(s['verified_claim_rate'], '.0%')} | {s['median_latency_s']} | "
                f"{s['mean_cost_usd']:.4f} | {s['failures']} |"
            )
        parts.append("")
    return "\n".join(parts)


def run(events, models, runs, *, ledger, transport=None, out_root: Path | None = None) -> dict:
    """Run the comparison, save it under `<data>/eval/` and return the result dict."""
    events, models = list(events), list(models)
    preflight(events, models, runs, ledger)
    out_root = out_root or config.data_dir() / "eval"
    rows = [
        _one(model, event, n, ledger, out_root, transport)
        for model in models for event in events for n in range(1, runs + 1)
    ]
    now = datetime.now(UTC)
    result = {
        "started_at": now.isoformat(timespec="seconds"),
        "events": events, "models": models, "runs": runs,
        "ground_truth": {
            e: {"cause": GROUND_TRUTH[e][0], "why": GROUND_TRUTH[e][1]} for e in events
        },
        "results": rows, "summary": summarize(rows),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (out_root / f"{now.strftime('%Y%m%dT%H%M%SZ')}.json").write_text(text)
    (out_root / "latest.json").write_text(text)
    return result
