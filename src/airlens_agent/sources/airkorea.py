"""AirKorea (data.go.kr) live connectors: nationwide realtime, forecast bulletin, station list."""

import re
import sqlite3
from datetime import date, datetime, timedelta

from .. import config, http, store
from ..config import KST
from ..models import Forecast, Observation, Station

BASE = "https://apis.data.go.kr/B552584"
REALTIME_URL = f"{BASE}/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty"
FORECAST_URL = f"{BASE}/ArpltnInforInqireSvc/getMinuDustFrcstDspth"
STATIONS_URL = f"{BASE}/MsrstnInfoInqireSvc/getMsrstnList"

_POLLUTANTS = ("pm10", "pm25", "o3", "no2", "co", "so2")

# Full province names in station `addr` -> the short `sidoName` used by the realtime API.
# Since 2026 the realtime API reports Jeonnam and Gwangju together as "전남광주".
_SIDO_PREFIXES = {
    "서울": "서울", "부산": "부산", "대구": "대구", "인천": "인천", "대전": "대전",
    "울산": "울산", "세종": "세종", "경기": "경기", "강원": "강원", "제주": "제주",
    "충청북도": "충북", "충북": "충북", "충청남도": "충남", "충남": "충남",
    "전라북도": "전북", "전북": "전북", "경상북도": "경북", "경북": "경북",
    "경상남도": "경남", "경남": "경남", "전라남도": "전남광주", "전남": "전남광주",
    "광주": "전남광주",
}


def station_key(sido: str, name: str) -> str:
    return f"{sido}|{name}"


def _num(value: str | None) -> float | None:
    if value is None or value.strip() in ("", "-"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_data_time(text: str) -> datetime:
    """'YYYY-MM-DD HH:MM' in KST; hour 24 means 00:00 of the next day."""
    day, _, hm = text.strip().partition(" ")
    hour, minute = (int(x) for x in hm.split(":"))
    base = datetime.fromisoformat(day).replace(tzinfo=KST)
    return base + timedelta(hours=hour, minutes=minute)


def _items(resp_json: dict) -> list[dict]:
    body = (resp_json.get("response") or {}).get("body") or {}
    items = body.get("items") or []
    return items if isinstance(items, list) else items.get("item", [])


def parse_realtime(payload: dict) -> list[Observation]:
    """Items with a null dataTime are skipped; flagged values become None plus a flag note."""
    out = []
    for it in _items(payload):
        if not it.get("dataTime") or not it.get("sidoName") or not it.get("stationName"):
            continue
        values, flags = {}, {}
        for p in _POLLUTANTS:
            flag = (it.get(f"{p}Flag") or "").strip()
            if flag:
                flags[p] = flag
                values[p] = None
            else:
                values[p] = _num(it.get(f"{p}Value"))
        out.append(
            Observation(
                station_key=station_key(it["sidoName"], it["stationName"]),
                ts=parse_data_time(it["dataTime"]),
                flags=flags,
                source="airkorea",
                **values,
            )
        )
    return out


def fetch_realtime_nation(conn: sqlite3.Connection) -> list[Observation]:
    """One call returns every station (~672)."""
    params = {
        "serviceKey": config.data_go_kr_key(), "returnType": "json", "numOfRows": 1000,
        "pageNo": 1, "sidoName": "전국", "ver": "1.0",
    }
    resp = http.get(REALTIME_URL, params, op="airkorea.realtime", conn=conn)
    return parse_realtime(resp.json())


def parse_grades(text: str | None) -> dict[str, str]:
    """'서울 : 보통,제주 : 좋음' -> {'서울': '보통', '제주': '좋음'}."""
    grades = {}
    for part in (text or "").split(","):
        region, sep, grade = part.partition(":")
        if sep and region.strip():
            grades[region.strip()] = grade.strip()
    return grades


def parse_forecast(payload: dict, day: date) -> list[Forecast]:
    out = []
    for it in _items(payload):
        m = re.match(r"(\d{4}-\d{2}-\d{2})\s*(\d{1,2})", it.get("dataTime") or "")
        issued = (
            datetime.fromisoformat(m.group(1)).replace(hour=int(m.group(2)) % 24, tzinfo=KST)
            if m
            else datetime(day.year, day.month, day.day, tzinfo=KST)
        )
        target = it.get("informData") or day.isoformat()
        out.append(
            Forecast(
                issued_at=issued,
                target_date=date.fromisoformat(target),
                pollutant=it.get("informCode", ""),
                overall=it.get("informOverall", ""),
                cause=it.get("informCause", ""),
                grades=parse_grades(it.get("informGrade")),
            )
        )
    return out


def fetch_forecast(conn: sqlite3.Connection, day: date) -> list[Forecast]:
    out: list[Forecast] = []
    for code in ("PM10", "PM25"):
        params = {
            "serviceKey": config.data_go_kr_key(), "returnType": "json", "numOfRows": 100,
            "pageNo": 1, "searchDate": day.isoformat(), "InformCode": code,
        }
        resp = http.get(FORECAST_URL, params, op="airkorea.forecast", conn=conn)
        out += parse_forecast(resp.json(), day)
    return out


def short_sido(addr: str) -> str | None:
    """Province short name from a station address ('경상북도 ...' -> '경북')."""
    token = (addr or "").split(maxsplit=1)[0] if (addr or "").strip() else ""
    if token.startswith("전남광주"):
        return "전남광주"
    for prefix in sorted(_SIDO_PREFIXES, key=len, reverse=True):
        if token.startswith(prefix):
            return _SIDO_PREFIXES[prefix]
    return None


def _latlon(item: dict) -> tuple[float | None, float | None]:
    """dmX/dmY are swapped vs the docs in places, so decide by value range."""
    a, b = _num(item.get("dmX")), _num(item.get("dmY"))
    for lat, lon in ((a, b), (b, a)):
        if lat is not None and lon is not None and 33 <= lat <= 39 and 124 <= lon <= 132:
            return lat, lon
    return None, None


def parse_stations(payload: dict) -> list[Station]:
    out = []
    for it in _items(payload):
        sido = short_sido(it.get("addr", ""))
        name = it.get("stationName")
        if not sido or not name:
            continue
        lat, lon = _latlon(it)
        out.append(
            Station(
                key=station_key(sido, name), name=name, sido=sido, lat=lat, lon=lon,
                code=None, source="airkorea-msrstn",
                network=(it.get("mangName") or "").strip() or None,
            )
        )
    return out


def fetch_stations(conn: sqlite3.Connection) -> list[Station]:
    params = {
        "serviceKey": config.data_go_kr_key(), "returnType": "json", "numOfRows": 1000, "pageNo": 1,
    }
    resp = http.get(STATIONS_URL, params, op="airkorea.stations", conn=conn)
    stations = parse_stations(resp.json())
    store.upsert_stations(conn, stations)
    return stations
