from airlens_agent import cli, http
from airlens_agent.sources import airkorea, firms


def test_fetch_now_continues_when_firms_is_down(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AIRLENS_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(airkorea, "fetch_realtime_nation", lambda conn: [])

    def down(conn, bbox):
        raise http.HttpError("firms.area: HTTP 503")

    monkeypatch.setattr(firms, "fetch_hotspots", down)
    assert cli.cmd_fetch_now(None) == {
        "observations": 0, "latest_ts": None, "hotspots": None, "candidates": 0}
    assert "[firms] skipped" in capsys.readouterr().err
