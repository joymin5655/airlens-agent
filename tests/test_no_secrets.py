import subprocess
from pathlib import Path

import pytest

from airlens_agent import config

ROOT = Path(__file__).resolve().parents[1]


def _files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout.decode()
    return [ROOT / p for p in out.split("\0") if p]


def test_no_key_values_in_repo_files():
    keys = []
    for getter in (config.data_go_kr_key, config.firms_map_key):
        try:
            keys.append(getter())
        except config.MissingKeyError:
            pass
    if not keys:
        pytest.skip("no keys configured")
    leaked = []
    for path in _files():
        if not path.is_file():
            continue
        data = path.read_bytes()
        if any(k.encode() in data for k in keys):
            leaked.append(str(path.relative_to(ROOT)))
    assert not leaked, f"key value found in: {leaked}"
