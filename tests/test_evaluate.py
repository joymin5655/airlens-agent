import json

import httpx
import pytest

from airlens_agent import agent, cli, config, evaluate, http, llm, quota, store
from conftest import make_candidate

FIRE, DUST = "fire-2025-03", "dust-2026-02"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for var in ("MODEL_API_KEY", "BASETEN_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(var, "FAKE-KEY-123456")
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(http, "_sleep", lambda _s: None)
    monkeypatch.setattr(cli, "_ensure_images", lambda *a, **k: None)  # no network
    for name in (FIRE, DUST):
        conn = store.connect(config.event_db_path(name))
        store.save_candidates(conn, [make_candidate(f"{name}-c1")])
        conn.close()


def _body(content):
    message = {"role": "assistant", "content": content}
    return {
        "model": "fake-model",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
    }


def _transport(cause, hosts=None, seen=None):
    """Triage, one-step investigation (no tool calls), then a report that cites E1."""
    triage = json.dumps({"category": cause, "hypotheses": [], "checks": []})
    report = json.dumps({
        "headline": "h", "cause": cause, "confidence": 0.8, "summary": "s",
        "claims": [{"text": "Peak PM10 was 300", "evidence_ids": ["E1"]}],
        "alternatives": [], "limitations": [],
    })
    queue = [_body(triage), _body("done"), _body(report)]

    def handler(request):
        if seen is not None:
            seen.append(request.url.host)
        if not queue:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=queue.pop(0))

    return httpx.MockTransport(handler)


@pytest.fixture
def ledger():
    conn = store.connect(config.live_db_path())
    yield conn
    conn.close()


def test_ground_truth_scoring_and_accounting(ledger):
    result = evaluate.run([FIRE], ["meta"], 1, ledger=ledger, transport=_transport("dust"))
    row = result["results"][0]
    assert row["truth"] == "wildfire_smoke" and row["cause"] == "dust" and not row["correct"]
    assert row["claims_verified"] == 1 and row["claims_total"] == 1 and row["tool_calls"] == 0
    # 3 calls of 1000 in / 1000 out at $0.10 / $0.20 per 1M
    assert row["cost_usd"] == pytest.approx(3 * (1000 * 0.10 + 1000 * 0.20) / 1e6)
    assert row["tool_cost_usd"] == 0 and row["failed"] is False and row["error"] is None
    s = result["summary"]["meta"]["all"]
    assert s["correct"] == 0 and s["runs"] == 1 and s["verified_claim_rate"] == 1.0


def test_correct_answer_scores_and_uses_the_models_own_provider(ledger):
    hosts: list[str] = []
    result = evaluate.run(
        [DUST], ["openai"], 1, ledger=ledger, transport=_transport("dust", seen=hosts)
    )
    row = result["results"][0]
    assert row["correct"] and row["confidence"] == 0.8
    assert set(hosts) == {"api.openai.com"}  # triage and investigation both on openai
    assert quota.spent(ledger, "openai") > 0 and quota.spent(ledger, "meta") == 0


def test_a_failed_call_is_a_failure_and_never_falls_back(ledger):
    hosts: list[str] = []

    def handler(request):
        hosts.append(request.url.host)
        return httpx.Response(500, text="down")

    result = evaluate.run(
        [DUST], ["meta"], 1, ledger=ledger, transport=httpx.MockTransport(handler)
    )
    row = result["results"][0]
    assert row["failed"] and not row["correct"]
    assert set(hosts) == {"api.meta.ai"}  # no switch to baseten
    assert result["summary"]["meta"]["all"]["failures"] == 1


def test_a_missing_key_is_refused_before_any_call(ledger, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(evaluate.EvalError, match="OPENAI_API_KEY"):
        evaluate.run([DUST], ["openai"], 1, ledger=ledger, transport=_transport("dust"))


def test_a_budget_that_cannot_cover_the_runs_is_refused(ledger):
    quota.charge(ledger, "openai", config.SPEND_CAPS["openai"] - 0.01)
    with pytest.raises(evaluate.EvalError, match="remaining budget"):
        evaluate.run([DUST], ["openai"], 1, ledger=ledger, transport=_transport("dust"))
    assert evaluate.estimate_investigation_usd("openai") > 0.01


def test_the_openai_cap_cannot_exceed_five_dollars():
    assert config.SPEND_CAPS["openai"] <= 5.0


def test_reports_and_results_go_under_eval_not_the_demo_reports(ledger, tmp_path):
    result = evaluate.run([DUST], ["meta"], 2, ledger=ledger, transport=_transport("dust"))
    eval_dir = tmp_path / "eval"
    assert (eval_dir / "latest.json").exists()
    stamped = [p for p in eval_dir.glob("*.json") if p.name != "latest.json"]
    assert len(stamped) == 1 and stamped[0].read_text() == (eval_dir / "latest.json").read_text()
    for row in result["results"]:
        if row["report"]:
            path = eval_dir / row["report"]
            assert path.parent == eval_dir / "reports" / "meta" / DUST and path.exists()
    assert not (tmp_path / "events" / DUST / "reports").exists()
    assert str(tmp_path) not in (eval_dir / "latest.json").read_text()


def test_table_rendering():
    rows = [
        {"event": DUST, "model": "meta", "run": r, "correct": ok, "confidence": 0.8,
         "claims_total": 2, "claims_verified": 1, "wall_s": w, "cost_usd": 0.01,
         "tool_cost_usd": 0.0025, "failed": not ok}
        for r, ok, w in ((1, True, 10.0), (2, False, 20.0))
    ]
    text = evaluate.render_tables({"summary": evaluate.summarize(rows)})
    assert "### meta" in text
    # cost per case counts paid tools too (SAM 3.1 is billed to Meta whichever model runs)
    assert f"| {DUST} | 1/2 | 0.8 | 50% | 15.0 | 0.0125 | 1 |" in text
    assert "| all | 1/2 |" in text


def test_cli_eval_runs_and_prints_tables(ledger, monkeypatch, capsys):
    monkeypatch.setattr(
        agent, "triage", lambda cand, **kw: agent.Triage(category="dust", hypotheses=[], checks=[])
    )
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("no real calls"))
    result = {"summary": {"meta": {"all": {
        "runs": 1, "correct": 1, "mean_confidence_correct": 0.9, "verified_claim_rate": 1.0,
        "median_latency_s": 1.0, "mean_cost_usd": 0.01, "failures": 0}}}, "results": [{}]}
    seen = {}

    def fake_run(events, models, runs, *, ledger):
        seen.update(events=events, models=models, runs=runs)
        return result

    monkeypatch.setattr(evaluate, "run", fake_run)
    assert cli.main(["eval", "--models", "meta,openai", "--runs", "1", "--events", DUST]) == 0
    assert seen == {"events": [DUST], "models": ["meta", "openai"], "runs": 1}
    assert "### meta" in capsys.readouterr().out
