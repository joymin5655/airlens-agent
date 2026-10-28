import httpx
import pytest

from airlens_agent import http, quota

FAKE_KEY = "FAKE-KEY-abc123/xyz=="


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DATA_GO_KR_SERVICE_KEY", FAKE_KEY)
    monkeypatch.setattr(http, "_sleep", lambda _s: None)


def _transport(statuses):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        return httpx.Response(status, json={"ok": status == 200})

    return httpx.MockTransport(handler), calls


def test_retries_5xx_then_succeeds():
    transport, calls = _transport([503, 429, 200])
    resp = http.get("https://x.test/a", {"serviceKey": FAKE_KEY}, op="t.op", transport=transport)
    assert resp.status_code == 200 and len(calls) == 3


def test_retries_network_error():
    n = {"i": 0}

    def handler(request):
        n["i"] += 1
        if n["i"] < 3:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200)

    resp = http.get("https://x.test/a", op="t.op", transport=httpx.MockTransport(handler))
    assert resp.status_code == 200 and n["i"] == 3


def test_error_text_never_contains_key():
    transport, _ = _transport([500])
    with pytest.raises(http.HttpError) as err:
        http.get("https://x.test/a", {"serviceKey": FAKE_KEY}, op="t.op", retries=1,
                 transport=transport)
    assert FAKE_KEY not in str(err.value)
    transport, _ = _transport([404])
    with pytest.raises(http.HttpError) as err:
        http.get("https://x.test/a", {"serviceKey": FAKE_KEY}, op="t.op", transport=transport)
    assert FAKE_KEY not in str(err.value)


def test_redact_replaces_key_and_query_param():
    assert FAKE_KEY not in http.redact(f"bad {FAKE_KEY} here")
    assert "SECRET" not in http.redact("https://h/?serviceKey=SECRET&a=1")


def test_attempts_counted_in_ledger(conn):
    transport, _ = _transport([500, 200])
    http.get("https://x.test/a", op="firms.test", conn=conn, transport=transport)
    assert quota.usage(conn)["firms.test"]["used"] == 2
