from datetime import date, datetime

import httpx
import pytest

from airlens_agent import http, quota
from airlens_agent.config import KST
from airlens_agent.sources import airkorea

# Synthetic payloads shaped like the real API responses (invented numbers).
REALTIME = {"response": {"body": {"totalCount": 5, "items": [
    {"sidoName": "서울", "stationName": "중구", "dataTime": "2026-01-02 24:00",
     "pm10Value": "40", "pm25Value": "-", "o3Value": "0.030", "no2Value": "0.020",
     "coValue": "0.5", "so2Value": "0.002", "pm10Flag": None, "pm25Flag": None},
    {"sidoName": "부산", "stationName": "중구", "dataTime": "2026-01-02 10:00",
     "pm10Value": "-", "pm25Value": "", "o3Value": "-", "no2Value": "-",
     "coValue": "-", "so2Value": "-", "pm10Flag": "통신장애", "pm25Flag": "점검및교정"},
    {"sidoName": "전남광주", "stationName": "상무", "dataTime": "2026-01-02 10:00",
     "pm10Value": "12", "pm25Value": "5", "pm10Flag": "점검및교정"},
    {"sidoName": "경북", "stationName": "안동", "dataTime": None, "pm10Value": "9"},
]}}}


def test_parse_realtime_values_flags_and_keys():
    obs = {o.station_key: o for o in airkorea.parse_realtime(REALTIME)}
    assert set(obs) == {"서울|중구", "부산|중구", "전남광주|상무"}  # null dataTime skipped
    seoul = obs["서울|중구"]
    assert seoul.pm10 == 40.0 and seoul.pm25 is None and seoul.o3 == 0.03
    assert seoul.ts == datetime(2026, 1, 3, 0, 0, tzinfo=KST)  # hour 24 -> next day 00:00
    busan = obs["부산|중구"]
    assert busan.pm10 is None and busan.pm25 is None
    assert busan.flags == {"pm10": "통신장애", "pm25": "점검및교정"}
    assert obs["전남광주|상무"].pm10 is None  # flagged value dropped
    assert obs["전남광주|상무"].flags == {"pm10": "점검및교정"}


FORECAST = {"response": {"body": {"items": [
    {"dataTime": "2026-01-02 05시 발표", "informCode": "PM10", "informData": "2026-01-02",
     "informOverall": "overall", "informCause": "cause",
     "informGrade": "서울 : 보통,제주 : 좋음,경기남부 : 나쁨"},
]}}}


def test_parse_forecast():
    (f,) = airkorea.parse_forecast(FORECAST, date(2026, 1, 2))
    assert f.issued_at == datetime(2026, 1, 2, 5, tzinfo=KST)
    assert f.target_date == date(2026, 1, 2) and f.pollutant == "PM10"
    assert f.grades == {"서울": "보통", "제주": "좋음", "경기남부": "나쁨"}


STATIONS = {"response": {"body": {"items": [
    # dmX holds latitude here, as in the real response
    {"stationName": "중구", "addr": "서울 중구 덕수궁길 15", "dmX": "37.564", "dmY": "126.975"},
    # documented order (dmX = longitude)
    {"stationName": "안동", "addr": "경상북도 안동시 서동문로 1", "dmX": "128.73", "dmY": "36.56"},
    {"stationName": "상무", "addr": "전남광주통합특별시 서구 상무중앙로 1", "dmX": "35.15",
     "dmY": "126.85"},
    {"stationName": "충주", "addr": "충청북도 충주시 1", "dmX": "36.9", "dmY": "127.9"},
    {"stationName": "노좌표", "addr": "제주특별자치도 제주시 1", "dmX": None, "dmY": None},
]}}}


def test_parse_stations_province_and_swapped_coords():
    st = {s.key: s for s in airkorea.parse_stations(STATIONS)}
    assert set(st) == {"서울|중구", "경북|안동", "전남광주|상무", "충북|충주", "제주|노좌표"}
    assert (st["서울|중구"].lat, st["서울|중구"].lon) == (37.564, 126.975)
    assert (st["경북|안동"].lat, st["경북|안동"].lon) == (36.56, 128.73)
    assert st["제주|노좌표"].lat is None


def test_short_sido_variants():
    cases = {"서울특별시 중구": "서울", "경기도 수원": "경기", "광주광역시 서구": "전남광주",
             "전라남도 목포": "전남광주", "전북특별자치도 전주": "전북", "강원도 춘천": "강원",
             "세종특별자치시": "세종", "": None, "미상 어딘가": None}
    for addr, expected in cases.items():
        assert airkorea.short_sido(addr) == expected, addr


FAKE_KEY = "FAKE-DATA-GO-KR-KEY/0123+456789=="
XML_ERROR = (
    "<OpenAPI_ServiceResponse><cmmMsgHeader><errMsg>SERVICE ERROR</errMsg>"
    "<returnReasonCode>30</returnReasonCode></cmmMsgHeader></OpenAPI_ServiceResponse>"
)


def _reply(**kw) -> httpx.Response:
    return httpx.Response(200, **kw)


def _header(code: str, msg: str = "") -> dict:
    return {"response": {"header": {"resultCode": code, "resultMsg": msg},
                         "body": {"items": []}}}


def test_result_code_error_raises(monkeypatch):
    monkeypatch.setenv("DATA_GO_KR_SERVICE_KEY", FAKE_KEY)
    resp = _reply(json=_header("22", "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR"))
    with pytest.raises(http.HttpError, match="22"):
        airkorea._payload(resp, "airkorea.realtime")


def test_nodata_and_normal_codes_pass():
    assert airkorea.parse_realtime(airkorea._payload(_reply(json=_header("03")), "op")) == []
    assert airkorea._payload(_reply(json=_header("00", "NORMAL_CODE")), "op")
    assert airkorea._payload(_reply(json=REALTIME), "op") is not None  # no header at all


def test_xml_gateway_error_raises():
    with pytest.raises(http.HttpError, match="non-JSON"):
        airkorea._payload(_reply(text=XML_ERROR), "airkorea.forecast")


def test_non_json_body_is_redacted(monkeypatch):
    monkeypatch.setenv("DATA_GO_KR_SERVICE_KEY", FAKE_KEY)
    with pytest.raises(http.HttpError) as exc:
        airkorea._payload(_reply(text=f"denied for {FAKE_KEY}"), "airkorea.stations")
    assert "***" in str(exc.value) and FAKE_KEY not in str(exc.value)


def test_capped_realtime_makes_no_request(conn, monkeypatch):
    monkeypatch.setenv("DATA_GO_KR_SERVICE_KEY", FAKE_KEY)

    def no_client(*_a, **_kw):
        raise AssertionError("an HTTP request was made past the daily cap")

    monkeypatch.setattr(http.httpx, "Client", no_client)
    for _ in range(30):
        quota.reserve(conn, "airkorea.realtime")
    with pytest.raises(quota.QuotaExceeded):
        airkorea.fetch_realtime_nation(conn)
