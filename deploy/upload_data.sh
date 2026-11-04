#!/usr/bin/env bash
# One-time upload of the past-event data (and the local spend history) to the Modal Volume.
# Run from the repository root; `modal` must be logged in. data/ stays out of git.
set -euo pipefail

VOLUME=airlens-agent-data
SNAP=$(mktemp -d)

modal volume create "$VOLUME" 2>/dev/null || true  # already exists is fine

# Complete standalone copies: a SQLite file with a -wal/-shm next to it is not safe to copy as is.
for name in fire-2025-03 dust-2026-02; do
  sqlite3 "data/events/$name.sqlite" ".backup '$SNAP/$name.sqlite'"
  modal volume put --force "$VOLUME" "$SNAP/$name.sqlite" "/events/$name.sqlite"
  modal volume put --force "$VOLUME" "data/events/$name" "/events/$name"  # images, reports, ...
done

# The live DB carries the all-time model spend (the budget caps count it) and the station list.
sqlite3 data/live.sqlite ".backup '$SNAP/live.sqlite'"
modal volume put --force "$VOLUME" "$SNAP/live.sqlite" /live.sqlite

# The saved example answers (shown when the daily question cap is reached) and the latest model
# comparison, when they have been made (`airlens-agent ask --build-examples`, `airlens-agent eval`).
if [ -f data/ask/examples.json ]; then
  modal volume put --force "$VOLUME" data/ask/examples.json /ask/examples.json
fi
if [ -f data/eval/latest.json ]; then
  modal volume put --force "$VOLUME" data/eval/latest.json /eval/latest.json
fi

rm -rf "$SNAP"
modal volume ls "$VOLUME" /events
