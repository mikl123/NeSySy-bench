#!/usr/bin/env bash
# Smoke test (about 15 minutes): one Boolean, one real-valued and one
# method-comparison run; then the paper's figures and tables are drawn from them.
# Usage: ./smoke_test.sh [--resume] [--out DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -x "$HERE/.venv/bin/python" ] || { echo "Run ./install.sh first." >&2; exit 1; }
exec "$HERE/.venv/bin/python" "$HERE/run.py" smoke "$@"
