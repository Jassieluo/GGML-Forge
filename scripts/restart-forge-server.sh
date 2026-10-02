#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$SCRIPT_DIR/stop-forge-server.sh"
"$SCRIPT_DIR/start-forge-server.sh"
