#!/usr/bin/env bash
# Install the pinned dependencies of NeSySy into .venv (in this directory).
#
#   ./install.sh                     install (offline from ./wheels if that directory exists,
#                                    otherwise from PyPI)
#   PYTHON=python3.13 ./install.sh   use a specific interpreter (Python >= 3.12)
#
# NeSySy itself runs from this source tree; nothing is installed system-wide.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE"
PYTHON="${PYTHON:-python3}"
VENV="$HERE/.venv"
WHEELS="$HERE/wheels"
PIP_VERSION="25.1.1"
PISHIELD="pishield @ git+https://github.com/mikl123/PiShield.git@3a7dae2af0b0eecff0fe19f3d6d229b5453b577b"

if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
    echo "NeSySy needs Python >= 3.12; '$PYTHON' is $("$PYTHON" --version 2>&1 || echo missing)." >&2
    exit 1
fi
echo "Using $("$PYTHON" --version) ($(command -v "$PYTHON")), installing into $VENV"

# 1. Virtual environment. On Ubuntu without the python3-venv package, create it
#    without pip and bootstrap pip from the bundled wheel (or from PyPI).
if ! "$PYTHON" -m venv "$VENV" 2>/dev/null; then
    rm -rf "$VENV"
    "$PYTHON" -m venv --without-pip "$VENV"
    PIP_WHEEL="$(ls "$WHEELS"/pip-*.whl 2>/dev/null | head -n 1 || true)"
    if [ -n "$PIP_WHEEL" ]; then
        "$VENV/bin/python" "$PIP_WHEEL/pip" install --quiet --no-index "$PIP_WHEEL"
    else
        "$VENV/bin/python" -c "import urllib.request as u; u.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '$VENV/get-pip.py')"
        "$VENV/bin/python" "$VENV/get-pip.py" --quiet "pip==$PIP_VERSION"
    fi
fi
PIP=("$VENV/bin/python" -m pip)

# 2. Dependencies, pinned to the versions used for the paper.
if [ -d "$WHEELS" ]; then
    echo "Installing from the bundled wheels in $WHEELS (offline)"
    "${PIP[@]}" install --no-index --find-links "$WHEELS" \
        -r "$HERE/requirements.txt" -c "$HERE/constraints.txt"
else
    echo "Installing from PyPI (network access needed)"
    if [ "$(uname -s)" = "Linux" ]; then
        # CPU-only PyTorch: the default Linux wheel pulls several GB of CUDA libraries.
        # Its dependencies are installed below, pinned by constraints.txt.
        "${PIP[@]}" install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0"
    fi
    "${PIP[@]}" install "$PISHIELD" -c "$HERE/constraints.txt"
    "${PIP[@]}" install -r "$HERE/requirements.txt" -c "$HERE/constraints.txt"
fi

# 3. Check that everything imports, including NeSySy from this source tree.
(cd "$ROOT" && "$VENV/bin/python" - <<'EOF'
import cnfgen, deeplog, mlflow, nbconvert, pishield, pysat, sklearn, torch, z3
import nesy_approaches.methods, nesy_data_gen.constraints
print(f"OK: torch {torch.__version__}, z3 {z3.get_version_string()}, mlflow {mlflow.__version__}")
EOF
)
echo "Installation finished. Next: ./smoke_test.sh"
