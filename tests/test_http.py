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


META_KEY = "FAKE-META-KEY-9f8e7d"
BASETEN_KEY = "FAKE-BASETEN-KEY-1a2b3c"


def _post_env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", META_KEY)
    monkeypatch.setenv("BASETEN_API_KEY", BASETEN_KEY)


def test_post_json_retries_429_and_503_then_succeeds(monkeypatch, conn):
    _post_env(monkeypatch)
    transport, calls = _transport([429, 503, 200])
    resp = http.post_json("https://x.test/p", {"a": 1}, headers={"Authorization": "Bearer x"},
                          op="meta.chat", conn=conn, retries=2, transport=transport)
    assert resp.status_code == 200 and len(calls) == 3
    assert quota.usage(conn)["meta.chat"]["used"] == 3


def test_post_json_does_not_retry_other_errors(monkeypatch):
    _post_env(monkeypatch)
    for status in (400, 401, 500):
        transport, calls = _transport([status, 200])
        with pytest.raises(http.HttpError) as err:
            http.post_json("https://x.test/p", {}, headers={}, op="meta.chat", transport=transport)
        assert f"HTTP {status}" in str(err.value) and len(calls) == 1


def test_post_json_gives_up_after_retries(monkeypatch):
    _post_env(monkeypatch)
    transport, calls = _transport([503])
    with pytest.raises(http.HttpError, match="failed after 2 attempts"):
        http.post_json("https://x.test/p", {}, headers={}, op="meta.chat", transport=transport)
    assert len(calls) == 2


def test_post_json_error_redacts_both_model_keys(monkeypatch):
    _post_env(monkeypatch)

    def handler(request):
        return httpx.Response(400, text=f"bad key {META_KEY} and {BASETEN_KEY} " + "x" * 1000)

    with pytest.raises(http.HttpError) as err:
        http.post_json("https://x.test/p", {}, headers={}, op="meta.chat",
                       transport=httpx.MockTransport(handler))
    text = str(err.value)
    assert META_KEY not in text and BASETEN_KEY not in text and len(text) < 400
    assert META_KEY not in http.redact(META_KEY) and BASETEN_KEY in http._secrets()


@pytest.mark.parametrize("key", [META_KEY, BASETEN_KEY])
def test_post_json_error_redacts_a_key_cut_by_the_snippet_limit(monkeypatch, key):
    _post_env(monkeypatch)
    body = "x" * 293 + key + " tail"  # the key starts 7 characters before the 300-character cut

    def handler(request):
        return httpx.Response(400, text=body)

    with pytest.raises(http.HttpError) as err:
        http.post_json("https://x.test/p", {}, headers={}, op="meta.chat",
                       transport=httpx.MockTransport(handler))
    text = str(err.value)
    # cutting first would leave a key prefix that redact() can no longer recognise
    assert key[:6] not in text and "***" in text


# --- retries only when the request certainly never reached the server -------------------------


def _flaky(exc_type, message="boom"):
    """Transport that fails once with `exc_type`, then answers 200; counts the attempts."""
    n = {"i": 0}

    def handler(request):
        n["i"] += 1
        if n["i"] == 1:
            raise exc_type(message, request=request)
        return httpx.Response(200)

    return httpx.MockTransport(handler), n


@pytest.mark.parametrize("exc", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout])
def test_post_json_retries_errors_where_nothing_was_sent(monkeypatch, conn, exc):
    _post_env(monkeypatch)
    transport, n = _flaky(exc)
    resp = http.post_json("https://x.test/p", {}, headers={}, op="meta.chat", conn=conn,
                          transport=transport)
    assert resp.status_code == 200 and n["i"] == 2
    assert quota.usage(conn)["meta.chat"]["used"] == 2  # every attempt is reserved


@pytest.mark.parametrize("exc", [
    httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.WriteError,
    httpx.RemoteProtocolError, httpx.DecodingError,
])
def test_post_json_does_not_retry_when_the_request_may_have_been_billed(monkeypatch, conn, exc):
    _post_env(monkeypatch)
    transport, n = _flaky(exc, f"gave up talking to the host with {META_KEY}")
    with pytest.raises(http.HttpError, match=exc.__name__) as err:
        http.post_json("https://x.test/p", {}, headers={}, op="meta.chat", conn=conn,
                       retries=3, transport=transport)
    assert n["i"] == 1 and quota.usage(conn)["meta.chat"]["used"] == 1
    assert META_KEY not in str(err.value) and err.value.__cause__ is None


def test_get_still_retries_read_timeouts():
    transport, n = _flaky(httpx.ReadTimeout)  # a free GET may be repeated; only paid POSTs may not
    assert http.get("https://x.test/a", op="t.op", transport=transport).status_code == 200
    assert n["i"] == 2
