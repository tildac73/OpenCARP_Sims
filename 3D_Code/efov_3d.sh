#!/usr/bin/env bash
# Run the four checkpointed stages, then electrograms, ground truth and maps.
# Usage from any directory: bash /path/to/Masters_V2/3D_Code/efov_3d.sh
# PYTHON=/path/to/python selects the environment containing carputils and SciPy.
# SCENARIO=protocol (default: s1,s2,s3,s4), pre, ablate, reinduce, or all (7 stages).
# SKIP_SIMS=1 reprocesses existing outputs; MAPS=0 skips contribution maps.
# NP=1, T_LIFT=1950, CONFIG=slab3d_default.json are the remaining defaults.
set -euo pipefail
trap 'echo "FAILED at line $LINENO; see the error above." >&2' ERR
export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PIPE="$ROOT/Shared_Pipeline"
PYTHON="${PYTHON:-python3}"
SCENARIO="${SCENARIO:-protocol}"
T_LIFT="${T_LIFT:-1950}"
NP="${NP:-1}"
MAPS="${MAPS:-1}"
SKIP_SIMS="${SKIP_SIMS:-0}"
CONFIG="${CONFIG:-slab3d_default.json}"

case "$SCENARIO" in
  protocol) STAGE_NAMES=(s1_pace s2_block s3_reentry s4_ablate_terminate) ;;
  pre) STAGE_NAMES=(s1_pace s2_block s3_reentry) ;;
  ablate) STAGE_NAMES=(s4_ablate_terminate) ;;
  reinduce) STAGE_NAMES=(post_s1 post_s2 post_s3) ;;
  all) STAGE_NAMES=(s1_pace s2_block s3_reentry s4_ablate_terminate post_s1 post_s2 post_s3) ;;
  *) echo "Unknown SCENARIO: $SCENARIO" >&2; exit 2 ;;
esac
for flag in "$MAPS" "$SKIP_SIMS"; do
  [[ "$flag" == 0 || "$flag" == 1 ]] || { echo 'MAPS and SKIP_SIMS must be 0 or 1' >&2; exit 2; }
done
if [[ -n "${TUNE_CV:-}" ]]; then
  echo 'TUNE_CV is disabled: the old tuner did not include cAF remodelling.' >&2
  exit 2
fi
[[ "$CONFIG" = /* ]] || CONFIG="$PIPE/configs/$CONFIG"

# Check dependencies and configuration before spending time on simulations.
"$PYTHON" - "$CONFIG" "$NP" "$T_LIFT" "$SKIP_SIMS" "$MAPS" <<'PY'
import importlib, json, math, sys
config, np_, lift, skip, maps = sys.argv[1:]
if int(np_) < 1 or not math.isfinite(float(lift)) or not 1802 < float(lift) < 3500:
    raise ValueError('NP must be positive; T_LIFT must be between 1802 and 3500 ms')
for module in ['numpy', 'scipy', 'pandas'] + (['carputils'] if skip == '0' else []) + (['matplotlib'] if maps == '1' else []):
    importlib.import_module(module)
with open(config) as stream:
    cfg = json.load(stream)
if cfg.get('sweep'):
    raise ValueError('The 3D launcher takes one configuration at a time; remove sweep')
if cfg.get('wang_export', {}).get('enabled', False):
    raise ValueError('Clinical Wang export needs real patient/location metadata; use the 3D CSV outputs')
PY
JOB_NAME="$("$PYTHON" -c 'import sys; print(f"reentry_tlift{float(sys.argv[1]):g}_np{int(sys.argv[2])}")' "$T_LIFT" "$NP")"
JOB="$SCRIPT_DIR/runs/$JOB_NAME"
if [[ "$SKIP_SIMS" == 0 ]]; then
  echo "=== Simulations: $SCENARIO -> $JOB"
  "$PYTHON" "$SCRIPT_DIR/run.py" --np "$NP" --scenario "$SCENARIO" --t-lift "$T_LIFT" < /dev/null
fi

STAGES=()
for stage in "${STAGE_NAMES[@]}"; do
  directory="$JOB/$stage"
  for file in egm.npz vm.igb meta.json meshname.txt parameters.par; do
    [[ -s "$directory/$file" ]] || { echo "Missing output: $directory/$file" >&2; exit 1; }
  done
  STAGES+=("$directory")
done

# Exact run lists prevent old reinduction outputs leaking into a four-stage batch.
mkdir -p "$JOB/configs"
CFG="$JOB/configs/$(basename "$CONFIG")"
OUT="$("$PYTHON" - "$CONFIG" "$CFG" "$ROOT/results" "$JOB_NAME" "$SCENARIO" "${STAGES[@]}" <<'PY'
import json, pathlib, sys
source, destination, results, job, scenario, *stages = sys.argv[1:]
with open(source) as stream:
    cfg = json.load(stream)
cfg.update(runs=stages, out_dir=results)
cfg['name'] = f"{cfg['name']}__{job}__{scenario}"
# Post-only analysis still calibrates against the matching pre-ablation recording.
if scenario in ('ablate', 'reinduce') and cfg.get('gain', {}).get('mode', 'auto') == 'auto':
    cfg['calibration_runs'] = [str(pathlib.Path(stages[0]).parent / 's3_reentry')]
with open(destination, 'w') as stream:
    json.dump(cfg, stream, indent=2)
print(pathlib.Path(results) / cfg['name'])
PY
)"
echo '=== Electrograms'
"$PYTHON" "$PIPE/egm.py" "$CFG"
echo '=== Ground truth'
"$PYTHON" "$PIPE/ground_truth.py" "$CFG"
if [[ "$MAPS" == 1 ]]; then
  echo '=== Contribution maps'
  "$PYTHON" "$PIPE/gt_maps.py" "$CFG" "$OUT/maps" "${STAGES[@]}"
fi
echo "Completed successfully -> $OUT"
