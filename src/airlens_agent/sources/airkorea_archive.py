"""AirKorea confirmed-data Excel archive for past-event replay (owner: implementer C)."""

from pathlib import Path

from ..models import Observation


def download_year(year: int, dest_dir: Path) -> Path:
    """Download the yearly archive ZIP into dest_dir and return its path."""
    raise NotImplementedError


def read_month(zip_path: Path, year: int, month: int) -> list[Observation]:
    """Parse one month of hourly observations from the archive (-999 means missing)."""
    raise NotImplementedError
