"""Archive parsing on a tiny synthetic zip (the real data is KOGL type 3: never committed)."""

import zipfile
from datetime import datetime

import openpyxl
import pytest

from airlens_agent import store
from airlens_agent.config import KST
from airlens_agent.sources import airkorea_archive as arc

HEADER = ["지역", "망", "측정소코드", "측정소명", "측정일시", "SO2", "CO", "O3", "NO2", "PM10",
          "PM25", "주소"]


class Cp949Info(zipfile.ZipInfo):
    """A member whose name is stored as raw cp949 bytes (no UTF-8 flag), like the real archive."""

    def _encodeFilenameFlags(self):
        return self.filename.encode("cp949"), self.flag_bits


def make_zip(tmp_path, rows):
    xlsx = tmp_path / "src.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(HEADER)
    for row in rows:
        ws.append(row)
    wb.save(xlsx)
    path = tmp_path / "2026.zip"
    with zipfile.ZipFile(path, "w") as zf:
        info = Cp949Info("2026년 2월.xlsx")
        zf.writestr(info, xlsx.read_bytes())
    return path


ROWS = [
    ["서울 중구", "도시대기", 111121, "중구", 2026022101, 0.003, 0.5, 0.03, 0.02, 55, 43,
     "서울 중구 덕수궁길 15"],
    ["서울 중구", "도시대기", 111121, "중구", 2026022124, 0.003, 0.5, 0.03, 0.02, -999, 40,
     "서울 중구 덕수궁길 15"],
    ["광주 서구", "도로변대기", 156111, "상무", 2026022201, -999, -999, -999, -999, 80, -999,
     "광주 서구 1"],
]


def test_zip_member_is_found_with_cp949_name_and_month_extracted(tmp_path):
    zip_path = make_zip(tmp_path, ROWS)
    xlsx = arc.extract_month(zip_path, 2026, 2, tmp_path / "out")
    assert xlsx.name == "2026-02.xlsx" and xlsx.exists()
    with pytest.raises(FileNotFoundError):
        arc.extract_month(zip_path, 2026, 3, tmp_path / "out")


def test_read_month_hour_24_and_missing_values(tmp_path):
    obs = arc.read_month(make_zip(tmp_path, ROWS), 2026, 2)
    by = {(o.station_key, o.ts): o for o in obs}
    assert by[("서울|중구", datetime(2026, 2, 21, 1, tzinfo=KST))].pm10 == 55
    last = by[("서울|중구", datetime(2026, 2, 22, 0, tzinfo=KST))]  # hour 24 -> next day 00:00
    assert last.pm10 is None and last.pm25 == 40
    gwangju = by[("전남광주|상무", datetime(2026, 2, 22, 1, tzinfo=KST))]
    assert gwangju.pm10 == 80 and gwangju.pm25 is None and gwangju.so2 is None


def test_stations_keep_network_and_code_and_join_live(tmp_path, conn):
    _, stations = arc.read_month_full(make_zip(tmp_path, ROWS), 2026, 2)
    assert {s.key: s.network for s in stations} == {"서울|중구": "도시대기",
                                                    "전남광주|상무": "도로변대기"}
    store.upsert_stations(conn, [arc.Station("서울|중구", "중구", "서울", 37.5, 126.9, None,
                                             "airkorea-msrstn", None)])
    report = arc.merge_archive_stations(conn, stations)
    assert report["matched"] == 1 and report["unmatched_sample"] == ["상무"]
    row = conn.execute("SELECT code, network, lat FROM stations WHERE key='서울|중구'").fetchone()
    assert row == ("111121", "도시대기", 37.5)


def test_parse_ts_hour_24():
    assert arc.parse_ts("2026022824") == datetime(2026, 3, 1, 0, tzinfo=KST)
    assert arc.parse_ts(2026022201) == datetime(2026, 2, 22, 1, tzinfo=KST)


def test_download_skips_valid_zip(tmp_path):
    path = make_zip(tmp_path, ROWS)
    assert arc.download_year(2026, tmp_path) == path  # valid zip present: no network call


def test_link_from_page_and_fallback():
    html = '<a href="/jfile/readDownloadFile.do?fileId=abc123&amp;fileSeq=25">2025</a>'
    assert arc._params_from_page(2025, html) == {"fileId": "abc123", "fileSeq": 25}
    assert arc._params_from_page(2024, html) is None
    assert arc._fallback_params(2026) == {"fileId": "1658eebba9258", "fileSeq": 26}
    assert arc._fallback_params(2025) == {"fileId": "1657eebba9237", "fileSeq": 25}


def test_live_station_parser_reads_network_and_store_migrates(tmp_path):
    import sqlite3

    from airlens_agent.sources import airkorea

    payload = {"response": {"body": {"items": [
        {"stationName": "중구", "addr": "서울 중구 1", "dmX": "37.5", "dmY": "127.0",
         "mangName": "도로변대기"}]}}}
    assert airkorea.parse_stations(payload)[0].network == "도로변대기"
    old = tmp_path / "old.sqlite"  # a DB created before the network column existed
    db = sqlite3.connect(old)
    db.execute("CREATE TABLE stations (key TEXT PRIMARY KEY, name TEXT, sido TEXT, lat REAL, "
               "lon REAL, code TEXT, source TEXT)")
    db.commit()
    db.close()
    conn = store.connect(old)
    store.upsert_stations(conn, airkorea.parse_stations(payload))
    assert conn.execute("SELECT network FROM stations").fetchone() == ("도로변대기",)
