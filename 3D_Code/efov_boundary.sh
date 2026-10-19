#!/usr/bin/env bash
# Run from any directory. Extra arguments are passed directly to the readable Python runner.
set -euo pipefail
scriptFolder="$(cd "$(dirname "$0")" && pwd)"
pythonExecutable="${PYTHON:-python3}"
export MPLBACKEND=Agg
exec "$pythonExecutable" "$scriptFolder/run_boundary.py" --np "${NP:-1}" "$@"
