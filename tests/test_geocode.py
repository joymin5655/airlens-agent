import httpx

from airlens_agent.sources import geocode

SEOUL = {
    "results": [
        {"name": "Seoul", "latitude": 37.566, "longitude": 126.9784, "country": "South Korea",
         "country_code": "KR", "admin1": "Seoul", "timezone": "Asia/Seoul"},
    ]
}


def transport(payloads: dict, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.params["name"]
        seen.append(name)
        return httpx.Response(200, json=payloads.get(name, {"generationtime_ms": 0.3}))

    return httpx.MockTransport(handler)


def test_search_parses_and_caches(conn):
    seen: list = []
    t = transport({"Seoul": SEOUL}, seen)
    first = geocode.search(conn, "Seoul", transport=t)
    again = geocode.search(conn, "  seoul ", transport=t)  # same cache key
    assert first == again
    assert first[0]["lat"] == 37.566 and first[0]["country_code"] == "KR"
    assert first[0]["admin1"] == "Seoul"
    assert seen == ["Seoul"]


def test_unknown_place_is_empty_and_cached(conn):
    seen: list = []
    t = transport({}, seen)
    assert geocode.search(conn, "Qwertyville", transport=t) == []
    assert geocode.search(conn, "Qwertyville", transport=t) == []
    assert seen == ["Qwertyville"]


def test_comma_miss_retries_with_the_head(conn):
    seen: list = []
    t = transport({"Seoul": SEOUL}, seen)
    results = geocode.search(conn, "Seoul, South Korea", transport=t)
    assert results[0]["name"] == "Seoul"
    assert seen == ["Seoul, South Korea", "Seoul"]


def test_blank_place_makes_no_call(conn):
    seen: list = []
    assert geocode.search(conn, "   ", transport=transport({}, seen)) == []
    assert seen == []
