from pathlib import Path

from airlens_agent import config


def test_env_files_ignore_the_cwd(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DATA_GO_KR_SERVICE_KEY=decoy\n")
    monkeypatch.chdir(tmp_path)
    files = config.env_files()
    assert files
    assert not any(tmp_path in f.parents for f in files)
    assert set(files) <= {config.PROJECT_ROOT / ".env", config.MAIN_ROOT / ".env"}


def test_data_dir_does_not_follow_the_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("AIRLENS_AGENT_DATA", raising=False)
    monkeypatch.chdir(tmp_path)
    assert config.data_dir() == config.MAIN_ROOT / "data"


def test_data_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path / "d"))
    assert config.data_dir() == Path(tmp_path / "d")
