"""AirKorea confirmed-data Excel archive for past-event replay.

KOGL Type 3 (attribution, no modification): downloads stay under data/ (git-ignored) and
tests use synthetic files only. The hour in 측정일시 marks the END of the interval.
"""

import re
import sqlite3
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from .. import quota
from ..config import KST
from ..models import Observation, Station
from .airkorea import short_sido, station_key

PAGE_URL = "https://www.airkorea.or.kr/web/last_amb_hour_data?pMENU_NO=123"
DOWNLOAD_URL = "https://www.airkorea.or.kr/jfile/readDownloadFile.do"
# Known file ids: 2001-2025 share one id (seq = year - 2000); 2026 has its own.
_FALLBACK = {"default": "1657eebba9237", 2026: "1658eebba9258"}
OP = "airkorea.archive"
SOURCE = "airkorea-archive"

_COLUMNS = {
    "지역": "region", "망": "network", "측정소코드": "code", "측정소명": "name",
    "측정일시": "ts", "SO2": "so2", "CO": "co", "O3": "o3", "NO2": "no2",
    "PM10": "pm10", "PM25": "pm25", "주소": "addr",
}
_VALUE_COLS = ("so2", "co", "o3", "no2", "pm10", "pm25")


def _fallback_params(year: int) -> dict:
    if year in _FALLBACK:
        return {"fileId": _FALLBACK[year], "fileSeq": year - 2000}
    return {"fileId": _FALLBACK["default"], "fileSeq": year - 2000}


def _params_from_page(year: int, html: str) -> dict | None:
    """Find the download link for `year` in the page, if the markup still has one."""
    for m in re.finditer(r"readDownloadFile\.do\?[^\"'<>\s]*", html):
        link = m.group(0).replace("&amp;", "&")
        id_m, seq_m = re.search(r"fileId=(\w+)", link), re.search(r"fileSeq=(\d+)", link)
        if id_m and seq_m and int(seq_m.group(1)) == year - 2000:
            return {"fileId": id_m.group(1), "fileSeq": int(seq_m.group(1))}
    return None


def _valid_zip(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path, metadata_encoding="cp949") as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def download_year(
    year: int, dest_dir: Path, conn: sqlite3.Connection | None = None
) -> Path:
    """Download the yearly archive ZIP into dest_dir and return its path.

    Skips the download when a valid zip is already there. Counted in the ledger as
    op `airkorea.archive` (uncapped) when `conn` is given.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{year}.zip"
    if _valid_zip(path):
        return path
    tmp = path.with_suffix(".zip.part")
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        params = None
        try:
            page = client.get(PAGE_URL)
            if page.status_code == 200:
                params = _params_from_page(year, page.text)
        except httpx.HTTPError:
            pass
        params = params or _fallback_params(year)
        if conn is not None:
            quota.reserve(conn, OP)
        with client.stream("GET", DOWNLOAD_URL, params=params) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
    tmp.replace(path)
    if not _valid_zip(path):
        path.unlink()
        raise RuntimeError(f"archive {year}: downloaded file is not a valid zip")
    return path


def _month_member(zf: zipfile.ZipFile, year: int, month: int) -> str:
    want = re.compile(rf"{year}\s*년\s*0?{month}\s*월")
    for info in zf.infolist():
        if want.search(info.filename) and info.filename.lower().endswith(".xlsx"):
            return info.filename
    raise FileNotFoundError(f"no member for {year}-{month:02d} in {zf.filename}")


def extract_month(zip_path: Path, year: int, month: int, dest_dir: Path | None = None) -> Path:
    """Extract only the needed month to <dest_dir>/<year>-<MM>.xlsx."""
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir) if dest_dir else zip_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{year}-{month:02d}.xlsx"
    with zipfile.ZipFile(zip_path, metadata_encoding="cp949") as zf:
        with zf.open(_month_member(zf, year, month)) as src, open(out, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    return out


def parse_ts(value) -> datetime:
    """'YYYYMMDDHH' (hour = END of interval, 24 = next day 00:00 KST) -> tz-aware KST."""
    text = str(int(value)) if not isinstance(value, str) else value.strip()
    base = datetime(int(text[:4]), int(text[4:6]), int(text[6:8]), tzinfo=KST)
    return base + timedelta(hours=int(text[8:10]))


def _clean(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return None if num <= -999 else num


def read_frame(xlsx_path: Path) -> pd.DataFrame:
    """Read one month xlsx with normalised column names (raises if columns are missing)."""
    df = pd.read_excel(xlsx_path, dtype={"측정일시": str, "측정소코드": str})
    df = df.rename(columns={c: _COLUMNS[c] for c in df.columns if c in _COLUMNS})
    missing = {"name", "ts", "region", "pm10", "pm25"} - set(df.columns)
    if missing:
        raise ValueError(f"{xlsx_path}: missing columns {sorted(missing)}; got {list(df.columns)}")
    return df


def _station_from_row(row) -> Station | None:
    sido = short_sido(str(row.get("region") or "")) or short_sido(str(row.get("addr") or ""))
    name = str(row.get("name") or "").strip()
    if not sido or not name:
        return None
    code = row.get("code")
    network = row.get("network")
    return Station(
        key=station_key(sido, name), name=name, sido=sido, lat=None, lon=None,
        code=None if code is None or pd.isna(code) else str(code).strip(),
        source=SOURCE,
        network=None if network is None or pd.isna(network) else str(network).strip(),
    )


def parse_frame(df: pd.DataFrame) -> tuple[list[Observation], list[Station]]:
    """Rows -> observations (-999 -> None) and the distinct stations seen."""
    obs: list[Observation] = []
    stations: dict[str, Station] = {}
    for row in df.to_dict("records"):
        st = _station_from_row(row)
        if st is None or row.get("ts") is None or pd.isna(row["ts"]):
            continue
        stations.setdefault(st.key, st)
        values = {c: _clean(row.get(c)) for c in _VALUE_COLS}
        obs.append(
            Observation(
                station_key=st.key, ts=parse_ts(row["ts"]), source=SOURCE, **values
            )
        )
    return obs, list(stations.values())


def read_month_full(
    zip_path: Path, year: int, month: int
) -> tuple[list[Observation], list[Station]]:
    """Extract and parse one month; returns observations and the stations (code, network)."""
    xlsx = extract_month(zip_path, year, month)
    return parse_frame(read_frame(xlsx))


def read_month(zip_path: Path, year: int, month: int) -> list[Observation]:
    """Parse one month of hourly observations from the archive (-999 means missing)."""
    return read_month_full(zip_path, year, month)[0]


def copy_live_stations(live: sqlite3.Connection, target: sqlite3.Connection) -> int:
    """Copy the stations table (coordinates, network) from the live DB into `target`."""
    cols = "key, name, sido, lat, lon, code, source, network"
    rows = live.execute(f"SELECT {cols} FROM stations").fetchall()
    target.executemany(f"INSERT OR REPLACE INTO stations ({cols}) VALUES (?,?,?,?,?,?,?,?)", rows)
    target.commit()
    return len(rows)


def merge_archive_stations(conn: sqlite3.Connection, stations: list[Station]) -> dict:
    """Fill code/network of coordinate-carrying live stations from the archive; add the
    archive-only stations (no coordinates). Returns the join report."""
    known = {r[0] for r in conn.execute("SELECT key FROM stations")}
    matched, unmatched = 0, []
    for st in stations:
        if st.key in known:
            matched += 1
            conn.execute(
                "UPDATE stations SET code = COALESCE(?, code), "
                "network = COALESCE(network, ?) WHERE key = ?",
                (st.code, st.network, st.key),
            )
        else:
            unmatched.append(st.name)
            conn.execute(
                "INSERT INTO stations (key, name, sido, lat, lon, code, source, network) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (st.key, st.name, st.sido, None, None, st.code, st.source, st.network),
            )
    conn.commit()
    total = len(stations)
    return {
        "archive_stations": total,
        "matched": matched,
        "join_rate": round(matched / total, 4) if total else None,
        "unmatched_sample": sorted(unmatched)[:20],
    }
