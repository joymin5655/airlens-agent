"""Shared dataclasses. Timestamps are tz-aware; observation hours are KST."""

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Station:
    key: str  # f"{sido}|{name}"
    name: str
    sido: str  # short province name, e.g. "서울", "경북"
    lat: float | None
    lon: float | None
    code: str | None
    source: str


@dataclass
class Observation:
    station_key: str
    ts: datetime  # tz-aware KST, hour resolution
    pm10: float | None = None
    pm25: float | None = None
    o3: float | None = None
    no2: float | None = None
    co: float | None = None
    so2: float | None = None
    flags: dict[str, str] = field(default_factory=dict)  # e.g. {"pm10": "통신장애"}
    source: str = "airkorea"


@dataclass
class Forecast:
    issued_at: datetime
    target_date: date
    pollutant: str  # "PM10" | "PM25"
    overall: str
    cause: str
    grades: dict[str, str]  # region -> grade
    source: str = "airkorea"


@dataclass
class Hotspot:
    lat: float
    lon: float
    ts_utc: datetime
    satellite: str
    source: str
    frp: float | None = None
    confidence: str | None = None
    daynight: str | None = None


@dataclass
class GridSample:
    """One grid point/time with named variables (wind, CAMS concentrations, ...)."""

    lat: float
    lon: float
    ts: datetime
    source: str
    variables: dict[str, float | None] = field(default_factory=dict)


@dataclass
class Candidate:
    event_id: str
    ts: datetime
    kind: str  # "regional" | "local"
    hint: str  # "coarse" | "fine" | "mixed"
    station_keys: list[str]
    centroid_lat: float
    centroid_lon: float
    bbox: tuple[float, float, float, float]  # west, south, east, north
    peak_pm10: float | None
    peak_pm25: float | None
    ratio_pm25_pm10: float | None
    severity: float
    features: dict = field(default_factory=dict)
