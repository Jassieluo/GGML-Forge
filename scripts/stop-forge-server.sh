#!/usr/bin/env bash
set -euo pipefail

PID_FILE=/root/logs/forge-server.pid
BINARY=/root/GGML-Forge/build-x64-linux-cuda-release/bin/forge-server
if [[ ! -f "$PID_FILE" ]]; then
    echo "forge-server is not running (no PID file)"
    exit 0
fi

pid=$(<"$PID_FILE")
executable=""
if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -e "/proc/$pid/exe" ]]; then
    executable=$(readlink "/proc/$pid/exe")
    executable=${executable% (deleted)}
fi
if [[ "$executable" != "$BINARY" ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "forge-server is not running (stale PID file removed)"
    rm -f "$PID_FILE"
    exit 0
fi

kill "$pid"
for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "forge-server stopped"
        exit 0
    fi
    sleep 1
done

echo "forge-server did not stop within 30 seconds (PID $pid)" >&2
exit 1
