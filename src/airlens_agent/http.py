"""HTTP GET/POST with retries, key redaction and per-op quota accounting."""

import logging
import re
import sqlite3
import time
from urllib.parse import quote

import httpx

from . import config, quota

for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

_sleep = time.sleep  # patched in tests
_KEY_PARAM = re.compile(r"(?i)((?:service_?key|map_?key|api_?key|key)=)[^&\s\"']+")
# Errors raised before any byte of the request was sent; a retry cannot bill a call twice.
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class HttpError(RuntimeError):
    pass


def _secrets() -> list[str]:
    values = []
    for getter in (
        config.data_go_kr_key, config.firms_map_key, config.model_api_key, config.baseten_api_key
    ):
        try:
            values.append(getter())
        except config.MissingKeyError:
            pass
    return [v for val in values for v in {val, quote(val, safe=""), quote(val)} if v]


def redact(text: str) -> str:
    """Replace configured key values (and key=... query params) with ***."""
    for secret in sorted(_secrets(), key=len, reverse=True):
        text = text.replace(secret, "***")
    return _KEY_PARAM.sub(r"\1***", text)


def get(
    url: str,
    params: dict | None = None,
    *,
    op: str,
    conn: sqlite3.Connection | None = None,
    timeout: float = 30.0,
    retries: int = 3,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Response:
    """GET with backoff on network errors, 5xx and 429.

    With `conn`, every attempt is counted against `op` in the quota ledger first
    (QuotaExceeded stops capped ops). Errors never contain key values.
    """
    last = "no attempt made"
    for attempt in range(retries + 1):
        if conn is not None:
            quota.reserve(conn, op)
        try:
            with httpx.Client(timeout=timeout, transport=transport) as client:
                resp = client.get(url, params=params)
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}"
            elif resp.status_code >= 400:
                raise HttpError(redact(f"{op}: HTTP {resp.status_code}"))
            else:
                return resp
        if attempt < retries:
            _sleep(min(2.0**attempt, 10.0))
    raise HttpError(redact(f"{op}: failed after {retries + 1} attempts ({last})"))


def post_json(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str],
    op: str,
    conn: sqlite3.Connection | None = None,
    timeout: float = 120.0,
    retries: int = 1,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Response:
    """POST a JSON body for a paid call.

    Retries only when the request certainly never reached the server (connect error, connect or
    pool timeout) and on 429/503. Any other transport error (a read timeout, a broken or
    undecodable reply) raises HttpError at once: the provider may already have billed the call.

    Every attempt is counted against `op` first. Other status >= 400 raises HttpError with the
    status and a redacted, truncated body snippet; errors never contain key values.
    """
    last = "no attempt made"
    for attempt in range(retries + 1):
        if conn is not None:
            quota.reserve(conn, op)
        try:
            with httpx.Client(timeout=timeout, transport=transport) as client:
                resp = client.post(url, json=payload, headers=headers)
        except _NOT_SENT as exc:
            last = f"{type(exc).__name__}: {exc}"
        except httpx.HTTPError as exc:
            raise HttpError(
                redact(f"{op}: {type(exc).__name__}: {exc} (not retried: the call may be billed)")
            ) from None
        else:
            if resp.status_code in (429, 503):
                last = f"HTTP {resp.status_code}"
            elif resp.status_code >= 400:
                # redact the whole body, then cut: a key cut in half would no longer match
                raise HttpError(f"{op}: HTTP {resp.status_code}: {redact(resp.text)[:300]}")
            else:
                return resp
        if attempt < retries:
            _sleep(min(2.0**attempt, 10.0))
    raise HttpError(redact(f"{op}: failed after {retries + 1} attempts ({last})"))
