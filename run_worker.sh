#!/bin/bash
# Run the Train Console worker once (no supervisor).
# Prefer ./deploy/tc_worker.sh for a long-running deployment.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

PYTHON="${TC_WORKER_PYTHON:-python3}"
exec "$PYTHON" -m trainconsole.worker --poll 5
