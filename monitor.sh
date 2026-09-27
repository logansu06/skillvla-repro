#!/usr/bin/env bash
# Live training dashboard: ./monitor.sh [-n SECONDS] [--detail] [--once] [--log FILE]
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/.venv/bin/python" "$DIR/scripts/monitor.py" "$@"
