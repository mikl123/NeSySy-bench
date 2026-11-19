#!/usr/bin/env bash
# Representative subset of the paper's results (about 7 hours on a laptop).
# Usage: ./run_subset.sh [fig3 fig4 fig7 fig8 fig9 tab2 tab4] [--resume] [--out DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -x "$HERE/.venv/bin/python" ] || { echo "Run ./install.sh first." >&2; exit 1; }
exec "$HERE/.venv/bin/python" "$HERE/run.py" subset "$@"
