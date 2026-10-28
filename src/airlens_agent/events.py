"""Past events used for replay."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import KST


@dataclass
class Event:
    name: str
    start: datetime
    end: datetime
    focus_bbox: tuple[float, float, float, float]  # west, south, east, north
    key_day: datetime
    expected_hint: str  # "coarse" | "fine" | "mixed"

    @property
    def replay_start(self) -> datetime:
        """The first 7 days of the window are baseline only."""
        return self.start + timedelta(days=7)


EVENTS: dict[str, Event] = {
    # Uiseong wildfire started 2025-03-22; the satellite image that day shows two smoke plumes.
    "fire-2025-03": Event(
        name="fire-2025-03",
        start=datetime(2025, 3, 15, 0, tzinfo=KST),
        end=datetime(2025, 3, 28, 23, tzinfo=KST),
        focus_bbox=(127.5, 35.0, 130.0, 37.5),
        key_day=datetime(2025, 3, 22, tzinfo=KST),
        expected_hint="fine",
    ),
    # Seoul PM10 reached 311 ug/m3 on 2026-02-22.
    "dust-2026-02": Event(
        name="dust-2026-02",
        start=datetime(2026, 2, 14, 0, tzinfo=KST),
        end=datetime(2026, 2, 24, 23, tzinfo=KST),
        focus_bbox=(125.5, 36.5, 128.0, 38.3),
        key_day=datetime(2026, 2, 22, tzinfo=KST),
        expected_hint="coarse",
    ),
}


def get_event(name: str) -> Event:
    try:
        return EVENTS[name]
    except KeyError:
        raise SystemExit(f"unknown event {name!r}; choose from {sorted(EVENTS)}") from None
