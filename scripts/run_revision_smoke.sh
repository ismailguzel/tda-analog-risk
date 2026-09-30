#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${PYTHON_BIN:-}" && -x ".venv/bin/python" ]]; then
	PYTHON_BIN=".venv/bin/python"
else
	PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
if [[ -z "${PYTHONPYCACHEPREFIX:-}" ]]; then
	PYTHONPYCACHEPREFIX="$(mktemp -d "${TMPDIR:-/tmp}/tda-pycache.XXXXXX")"
	export PYTHONPYCACHEPREFIX
fi
"$PYTHON_BIN" scripts/check_revision_workspace.py
"$PYTHON_BIN" -m compileall -q tda_risk scripts
printf '%s\n' '[OK] Python sources compile.'
