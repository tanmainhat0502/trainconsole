#!/bin/bash
# Keep the Train Console worker alive.
# Usage: ./deploy/tc_worker.sh
# Set TC_WORKER_PYTHON if "python3" is not the interpreter with the console deps.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/.env" ]; then
    set -a
    . "$ROOT/.env"
    set +a
fi
PYTHON="${TC_WORKER_PYTHON:-python3}"
LOG="$ROOT/var/tc_worker.log"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/var"

# Only one supervisor may run.
exec 9>"$ROOT/var/tc_worker.lock"
if ! flock -n 9; then
    exit 0
fi

while true; do
    if ! pgrep -f -- '[-]m trainconsole.worker --poll' >/dev/null; then
        "$PYTHON" -m trainconsole.worker --poll 5 >>"$LOG" 2>&1
        code=$?
        echo "[supervisor] worker exited (code $code) at $(date -Is); restarting in 5s" >>"$LOG"
    fi
    sleep 5
done
