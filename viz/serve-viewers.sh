#!/usr/bin/env bash
# Start a tiny static HTTP server pointed at the Phase 3 viewer directory.
#
# Usage:
#   viz/serve-viewers.sh                 # default port 8765, dir results/viz_phase3_html
#   viz/serve-viewers.sh 9000            # start probing from port 9000
#   viz/serve-viewers.sh 9000 some/dir   # custom port + dir
#
# If the requested port is already bound (e.g. another local server is up),
# we scan upward for the next free port instead of failing. This avoids
# the EADDRINUSE error from `python -m http.server` when a previous run
# is still alive or another tool occupies the default.
#
# The viewers' Plotly bundle is loaded from a CDN, so the box running the
# server needs outbound internet for the plots to render.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
START_PORT="${1:-8765}"
DIR="${2:-results/viz_phase3_html}"
MAX_TRIES=20
INDEX_BUILDER="$SCRIPT_DIR/build-viewer-index-html.py"
MODAL_DIR="results/modal_larger_geometry"

cd "$REPO_ROOT"

if ! [[ "$START_PORT" =~ ^[0-9]+$ ]] || [[ "$START_PORT" -lt 1 || "$START_PORT" -gt 65535 ]]; then
    echo "error: port must be an integer in 1..65535, got '$START_PORT'" >&2
    exit 2
fi

if [[ ! -d "$DIR" ]]; then
    echo "error: directory $DIR does not exist" >&2
    exit 2
fi

if [[ -f "$INDEX_BUILDER" ]]; then
    python3 "$INDEX_BUILDER" --dir "$DIR" --modal-dir "$MODAL_DIR" >/dev/null
fi

# Probe upward for a free port. We open and immediately close a TCP socket
# to test bindability — there is a tiny race window before exec'ing the
# real server, but for a local dev tool this is acceptable.
PORT="$START_PORT"
TRIES=0
while ! python3 -c "
import socket, sys
s = socket.socket()
try:
    s.bind(('127.0.0.1', $PORT))
except OSError:
    sys.exit(1)
finally:
    s.close()
" 2>/dev/null; do
    PORT=$((PORT + 1))
    TRIES=$((TRIES + 1))
    if [[ "$PORT" -gt 65535 ]]; then
        echo "error: no free port found before 65535" >&2
        exit 3
    fi
    if [[ "$TRIES" -ge "$MAX_TRIES" ]]; then
        echo "error: no free port found in $START_PORT..$((START_PORT + MAX_TRIES - 1))" >&2
        exit 3
    fi
done

if [[ "$PORT" -ne "$START_PORT" ]]; then
    echo "note: port $START_PORT was busy, using $PORT instead"
fi

echo "Serving $DIR at http://127.0.0.1:$PORT/"
echo "Open in browser: http://127.0.0.1:$PORT/index.html"
echo "Stop with Ctrl-C."
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$DIR"
