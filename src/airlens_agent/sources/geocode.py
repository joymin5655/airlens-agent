"""Open-Meteo geocoding: place name -> coordinates, country and admin names.

Geocoding data by Open-Meteo.com (CC BY 4.0, GeoNames). No key. Results are cached in the DB, so a
repeated place costs no call.
"""

import json
import sqlite3

from .. import http

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
MAX_RESULTS = 3
SCHEMA = "CREATE TABLE IF NOT EXISTS geocode_cache (query TEXT PRIMARY KEY, results TEXT NOT NULL)"


def _place(item: dict) -> dict:
    return {
        "name": item.get("name"),
        "lat": item.get("latitude"),
        "lon": item.get("longitude"),
        "country": item.get("country"),
        "country_code": item.get("country_code"),
        "admin1": item.get("admin1"),
        "admin2": item.get("admin2"),
        "timezone": item.get("timezone"),
    }


def _search(conn: sqlite3.Connection, name: str, transport) -> list[dict]:
    key = " ".join(name.lower().split())
    row = conn.execute("SELECT results FROM geocode_cache WHERE query = ?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    params = {"name": name, "count": MAX_RESULTS, "language": "en", "format": "json"}
    data = http.get(
        GEOCODE_URL, params, op="openmeteo.geocode", conn=conn, transport=transport
    ).json()
    results = [_place(i) for i in (data.get("results") or []) if isinstance(i, dict)]
    # a miss is cached too: the same typo must not cost a call each time
    conn.execute("INSERT OR REPLACE INTO geocode_cache VALUES (?, ?)", (key, json.dumps(results)))
    conn.commit()
    return results


def search(conn: sqlite3.Connection, place: str, *, transport=None) -> list[dict]:
    """Up to three matches, best first; [] when the place is unknown.

    "Seoul, South Korea" finds nothing on the API (it matches the name only), so a miss with a
    comma is retried with the part before it.
    ponytail: the part after the comma is not used to rank the matches (upgrade: filter by it).
    """
    conn.execute(SCHEMA)
    place = " ".join(place.split())
    if not place:
        return []
    results = _search(conn, place, transport)
    if not results and "," in place:
        head = place.split(",", 1)[0].strip()
        if head:
            results = _search(conn, head, transport)
    return results
