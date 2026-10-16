#!/bin/bash
# 3D micro-reentry batch: simulations -> electrograms -> ground truth -> 3D contribution maps.
#
#   1. run.py          pre (s1-s3), ablate (s4) and reinduce (post_s1-s3) stages, into
#                      3D_Code/runs/reentry_tlift<T_LIFT>[_cv<TUNE_CV>]_np<NP>/<stage>/
#   2. egm.py          -> results/<config>__<job>/  points.csv, run_summary.json
#   3. ground_truth.py -> results/<config>__<job>/  groundtruth.csv, points_with_gt.csv
#   4. gt_maps.py      -> results/<config>__<job>/maps/  per stage: top view and side (depth)
#                      view .png, and per catheter placement a 3D .vtk volume (+ electrodes)
#                      to open in ParaView
#
# Start from the repo folder with:   nohup bash 3D_Code/efov_3d.sh > log_3d.txt 2>&1 < /dev/null &
# Options (environment variables):
#   T_LIFT=<ms>      time the exit block is lifted (default 1950; set from the dry run)
#   SCENARIO=<name>  all (default) | pre | ablate | reinduce   (ablate needs pre's checkpoints)
#   TUNE_CV=<m/s>    retune conductivities with tuneCV (default: common.par's values)
#   NP=<n>           openCARP processes (default 1)
#   SKIP_SIMS=1      skip step 1 and re-process the existing runs for these settings
#   CONFIG=<file>    Shared_Pipeline config (default slab3d_default.json)
#   MAPS=0           skip step 4
set -u
export PYTHONUNBUFFERED=1              # write Python output to the log immediately
REPO="$(cd "$(dirname "$0")" && pwd)"            # 3D_Code
ROOT="$(dirname "$REPO")"                        # repo root
PIPE="$ROOT/Shared_Pipeline"
T_LIFT="${T_LIFT:-1950}"
SCENARIO="${SCENARIO:-all}"
TUNE_CV="${TUNE_CV:-}"
NP="${NP:-1}"
CONFIG="${CONFIG:-slab3d_default.json}"
MAPS="${MAPS:-1}"

# run folder, named as run.py's jobID does
JOB_NAME=$(python3 -c "import sys; t, cv, np_ = sys.argv[1:]
print('reentry_tlift{:g}{}_np{}'.format(float(t), '_cv{:g}'.format(float(cv)) if cv else '', np_))" \
           "$T_LIFT" "$TUNE_CV" "$NP")
JOB="$REPO/runs/$JOB_NAME"

# ---------------- 1. simulations ----------------
if [ "${SKIP_SIMS:-0}" != 1 ]; then
  echo "=== simulations: $SCENARIO, T_LIFT $T_LIFT -> $JOB  $(date)"
  cd "$ROOT"
  python3 "$REPO/run.py" --np "$NP" --scenario "$SCENARIO" --t-lift "$T_LIFT" \
          ${TUNE_CV:+--tune-cv "$TUNE_CV"} < /dev/null \
    || echo "FAILED: run.py (processing whatever stages finished)"
  echo "SIMULATIONS DONE $(date)"
fi

STAGES=()
for D in "$JOB"/*/; do
  [ -f "$D/egm.npz" ] && STAGES+=( "${D%/}" )
done
if [ ${#STAGES[@]} -eq 0 ]; then
  echo "No stages with egm.npz in $JOB - nothing to process"; exit 1
fi
echo "Stages: $(for D in "${STAGES[@]}"; do printf '%s ' "$(basename "$D")"; done)"

# ---------------- 2-4. electrograms, ground truth, 3D maps ----------------
# The config is copied with absolute paths: this job's stages in, results/<name>__<job>/ out
mkdir -p "$JOB/configs"
CFG="$JOB/configs/$CONFIG"
python3 - "$PIPE/configs/$CONFIG" "$CFG" "$JOB/*" "$ROOT/results" "$JOB_NAME" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg['runs'], cfg['out_dir'] = sys.argv[3], sys.argv[4]
cfg['name'] = '{}__{}'.format(cfg['name'], sys.argv[5])
json.dump(cfg, open(sys.argv[2], 'w'), indent=1)
EOF
NAME=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$CFG")
OUT="$ROOT/results/$NAME"

echo "=== electrograms  $(date)"
python3 "$PIPE/egm.py" "$CFG" || { echo "FAILED: egm.py"; exit 1; }

echo "=== ground truth  $(date)"
python3 "$PIPE/ground_truth.py" "$CFG" || echo "FAILED: ground_truth.py"

if [ "$MAPS" = 1 ]; then
  echo "=== contribution maps (top, depth, 3D volumes)  $(date)"
  python3 "$PIPE/gt_maps.py" "$CFG" "$OUT/maps" "${STAGES[@]}" || echo "FAILED: gt_maps.py"
fi
echo "ALL DONE $(date)  -> $OUT"
