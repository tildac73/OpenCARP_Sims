#!/usr/bin/env bash
# Run the four checkpointed stages, then electrograms, ground truth, maps and thesis figures.
# Usage from any directory: bash /path/to/Masters_V2/3D_Code/efov_3d.sh
# PYTHON=/path/to/python selects the environment containing carputils and SciPy.
# SCENARIO=protocol (default: s1,s2,s3,s4), pre, ablate, reinduce, or all (7 stages).
# RESUME=1 (default) reuses completed stages and reruns incomplete ones; RESUME=0 refuses
#   to touch existing stage output. SKIP_SIMS=1 reprocesses existing outputs only.
# MAPS=0 skips contribution maps; FIGURES=0 skips thesis figures; STRICT_FIGURES=1 fails
#   the run if any figure fails. NP=1, T_LIFT=1950, CONFIG=slab3d_default.json are defaults.
set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit $?); see the error above. Log: ${LOG:-none}" >&2' ERR
export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"  # headless remote machines have no display
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PIPE="$ROOT/Shared_Pipeline"
PYTHON="${PYTHON:-python3}"
SCENARIO="${SCENARIO:-protocol}"
T_LIFT="${T_LIFT:-1950}"
NP="${NP:-1}"
MAPS="${MAPS:-1}"
FIGURES="${FIGURES:-1}"
STRICT_FIGURES="${STRICT_FIGURES:-0}"
SKIP_SIMS="${SKIP_SIMS:-0}"
RESUME="${RESUME:-1}"
CONFIG="${CONFIG:-slab3d_default.json}"

case "$SCENARIO" in
  protocol) STAGE_NAMES=(s1_pace s2_block s3_reentry s4_ablate_terminate) ;;
  pre) STAGE_NAMES=(s1_pace s2_block s3_reentry) ;;
  ablate) STAGE_NAMES=(s4_ablate_terminate) ;;
  reinduce) STAGE_NAMES=(post_s1 post_s2 post_s3) ;;
  all) STAGE_NAMES=(s1_pace s2_block s3_reentry s4_ablate_terminate post_s1 post_s2 post_s3) ;;
  *) echo "Unknown SCENARIO: $SCENARIO" >&2; exit 2 ;;
esac
for flag in "$MAPS" "$SKIP_SIMS" "$RESUME" "$FIGURES" "$STRICT_FIGURES"; do
  [[ "$flag" == 0 || "$flag" == 1 ]] || { echo 'MAPS, SKIP_SIMS, RESUME, FIGURES and STRICT_FIGURES must be 0 or 1' >&2; exit 2; }
done
[[ "$NP" =~ ^[0-9]+$ ]] || { echo "NP must be a positive integer, got '$NP'" >&2; exit 2; }
if [[ -n "${TUNE_CV:-}" ]]; then
  echo 'TUNE_CV is disabled: the old tuner did not include cAF remodelling.' >&2
  exit 2
fi
command -v "$PYTHON" > /dev/null || { echo "Python not found: $PYTHON (set PYTHON=...)" >&2; exit 2; }
[[ "$CONFIG" = /* ]] || CONFIG="$PIPE/configs/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }

# Check dependencies, simulator executables and configuration before spending time on simulations.
"$PYTHON" - "$CONFIG" "$NP" "$T_LIFT" "$SKIP_SIMS" "$MAPS" "$FIGURES" "$SCRIPT_DIR" "$SCENARIO" <<'PY'
import importlib, json, math, os, shutil, sys
config, np_, lift, skip, maps, figures, here, scenario = sys.argv[1:]
if sys.version_info < (3, 8):
    sys.exit(f'Python 3.8 or newer is required, found {sys.version.split()[0]}')
if int(np_) < 1 or not math.isfinite(float(lift)) or not 1802 < float(lift) < 3500:
    sys.exit('NP must be positive; T_LIFT must be between 1802 and 3500 ms')
need = ['numpy', 'scipy', 'pandas'] + (['carputils'] if skip == '0' else [])
need += ['matplotlib'] if '1' in (maps, figures) else []
missing = []
for module in need:
    try:
        importlib.import_module(module)
    except ImportError as error:
        missing.append(f'{module} ({error})')
if missing:
    sys.exit('Missing Python modules: ' + ', '.join(missing) + '. See 3D_Code/requirements.txt')
with open(config) as stream:
    cfg = json.load(stream)
if cfg.get('sweep'):
    sys.exit('The 3D launcher takes one configuration at a time; remove sweep')
if cfg.get('wang_export', {}).get('enabled', False):
    sys.exit('Clinical Wang export needs real patient/location metadata; use the 3D CSV outputs')
if skip == '0':
    # Simulator executables are checked by run.py: carputils only exposes settings.execs
    # inside its carpexample decorator. Parameter files are checked here, before any stage runs.
    pars = ['common.par', 's1_pace.par', 's2_block.par', 's3_reentry.par']
    pars += ['scar.par', 's4_ablate_terminate.par'] if scenario != 'pre' else []
    absent = [p for p in pars if not os.path.isfile(os.path.join(here, p))]
    if absent:
        sys.exit('Missing parameter files in 3D_Code: ' + ', '.join(absent))
    free = shutil.disk_usage(here).free / 1e9
    if free < 20:
        print(f'WARNING: only {free:.0f} GB free; a full run needs roughly 15-20 GB', file=sys.stderr)
print('Preflight OK')
PY
JOB_NAME="$("$PYTHON" -c 'import sys; print(f"reentry_tlift{float(sys.argv[1]):g}_np{int(sys.argv[2])}")' "$T_LIFT" "$NP")"
JOB="$SCRIPT_DIR/runs/$JOB_NAME"

# Log everything (stdout and stderr stay separate on the terminal) and record the environment.
mkdir -p "$JOB/logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$JOB/logs/pipeline_${SCENARIO}_${STAMP}.log"
exec > >(tee -a "$LOG") 2> >(tee -a "$LOG" >&2)
{
  echo "date: $(date)"; echo "host: $(hostname)"; echo "user: $(id -un)"
  echo "command: SCENARIO=$SCENARIO T_LIFT=$T_LIFT NP=$NP SKIP_SIMS=$SKIP_SIMS RESUME=$RESUME MAPS=$MAPS FIGURES=$FIGURES CONFIG=$CONFIG"
  echo "git: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a) $(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted files"
  echo "python: $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
  "$PYTHON" - <<'PY'
import importlib
for m in ('numpy', 'scipy', 'pandas', 'matplotlib', 'carputils'):
    try:
        mod = importlib.import_module(m)
        print(f"{m}: {getattr(mod, '__version__', 'installed')}")
    except Exception as error:
        print(f'{m}: not available ({type(error).__name__})')
try:
    from carputils import settings
    print('openCARP:', getattr(settings.execs, 'CARP', '?'))
    print('mesher:', getattr(settings.execs, 'MESHER', '?'))
except Exception:
    pass
PY
} > "$JOB/logs/environment_${STAMP}.txt" 2>&1
echo "=== Log: $LOG"

SECONDS=0
step() { echo "=== $1 ($(date +%H:%M:%S), elapsed $((SECONDS / 60)) min)"; }

if [[ "$SKIP_SIMS" == 0 ]]; then
  step "Simulations: $SCENARIO -> $JOB"
  RESUME_FLAG=()
  [[ "$RESUME" == 1 ]] && RESUME_FLAG=(--resume)
  "$PYTHON" "$SCRIPT_DIR/run.py" --np "$NP" --scenario "$SCENARIO" --t-lift "$T_LIFT" "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}" < /dev/null
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
step 'Electrograms'
"$PYTHON" "$PIPE/egm.py" "$CFG"
step 'Ground truth'
"$PYTHON" "$PIPE/ground_truth.py" "$CFG"
if [[ "$MAPS" == 1 ]]; then
  step 'Contribution maps'
  "$PYTHON" "$PIPE/gt_maps.py" "$CFG" "$OUT/maps" "${STAGES[@]}"
fi
FIG_NOTE=""
if [[ "$FIGURES" == 1 ]]; then
  step 'Figures'
  STRICT=()
  [[ "$STRICT_FIGURES" == 1 ]] && STRICT=(--strict)
  "$PYTHON" "$SCRIPT_DIR/figures.py" --results "$OUT" --config "$CFG" "${STRICT[@]+"${STRICT[@]}"}" "${STAGES[@]}"
  if [[ -s "$OUT/figures/failures.log" ]]; then
    FIG_NOTE=" ($(grep -c '^=== ' "$OUT/figures/failures.log") figure(s) failed: see $OUT/figures/failures.log)"
  fi
fi
cp "$LOG" "$OUT/" 2>/dev/null || true
echo "Completed successfully in $((SECONDS / 60)) min -> $OUT${FIG_NOTE}"
