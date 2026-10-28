"""Past events used for replay (owner: implementer C)."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    name: str
    start: datetime
    end: datetime
    focus_bbox: tuple[float, float, float, float]  # west, south, east, north
    key_day: datetime
    expected_hint: str  # "coarse" | "fine" | "mixed"


EVENTS: dict[str, Event] = {}  # filled in by C: 2025-03 Gyeongsang wildfire, 2026-02 Asian dust
