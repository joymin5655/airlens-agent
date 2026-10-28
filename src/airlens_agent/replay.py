"""Past-event data download and replay (owner: implementer C)."""


def fetch_event(name: str) -> None:
    """Download and store all data for event `name` into config.event_db_path(name)."""
    raise NotImplementedError


def run_replay(name: str) -> dict:
    """Run detection hour by hour over the event window and return a summary dict."""
    raise NotImplementedError
