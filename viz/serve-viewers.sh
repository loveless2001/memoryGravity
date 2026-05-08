#!/usr/bin/env bash
# Start a tiny static HTTP server pointed at the Phase 3 viewer directory.
#
# Usage:
#   viz/serve-viewers.sh                 # default port 8765, dir results/viz_phase3_html
#   viz/serve-viewers.sh 9000            # custom port
#   viz/serve-viewers.sh 9000 some/dir   # custom port + dir
#
# The viewers' Plotly bundle is loaded from a CDN, so the box running the
# server needs outbound internet for the plots to render.

set -euo pipefail

PORT="${1:-8765}"
DIR="${2:-results/viz_phase3_html}"

if [[ ! -d "$DIR" ]]; then
    echo "error: directory $DIR does not exist" >&2
    exit 2
fi

echo "Serving $DIR at http://127.0.0.1:$PORT/"
echo "Open in browser: http://127.0.0.1:$PORT/index.html"
echo "Stop with Ctrl-C."
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$DIR"
