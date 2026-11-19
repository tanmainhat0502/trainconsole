#!/bin/bash
# Start the Train Console API on :9080
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

PYTHON="${TC_CONSOLE_PYTHON:-python3}"
PORT="${TC_CONSOLE_PORT:-9080}"

exec "$PYTHON" -m uvicorn trainconsole.app.main:app --host 0.0.0.0 --port "$PORT"
