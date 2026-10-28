"""NASA GIBS true-colour satellite images (owner: implementer B)."""

from datetime import date
from pathlib import Path


def fetch_image(
    bbox: tuple[float, float, float, float],
    day: date,
    layer: str,
    *,
    width: int = 1024,
    out_dir: Path,
) -> Path:
    """Download a WMS image of `layer` for bbox (west, south, east, north) on `day`
    into out_dir and return the file path."""
    raise NotImplementedError
