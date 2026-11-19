#!/usr/bin/env bash
# All runs behind the paper's results (several days on a laptop; see README.md
# for the duration of each result). Select results to run only part of it.
# Usage: ./run_full.sh [fig3 fig4 fig7 fig8 fig9 fig10 tab1 tab2 tab3 tab4] [--resume] [--out DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -x "$HERE/.venv/bin/python" ] || { echo "Run ./install.sh first." >&2; exit 1; }
exec "$HERE/.venv/bin/python" "$HERE/run.py" full "$@"
