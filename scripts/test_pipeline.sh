#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at: $PYTHON_BIN"
  echo "Set PYTHON_BIN to override, e.g. PYTHON_BIN=python3"
  exit 1
fi

echo "[1/2] Compile check"
"$PYTHON_BIN" -m compileall extractai

echo "[2/2] Unit tests"
"$PYTHON_BIN" -m unittest discover -s tests -v

echo "All checks passed."
