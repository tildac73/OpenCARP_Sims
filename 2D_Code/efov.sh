#!/bin/bash
# 0.25 mm runs: 3 CVs x {pre, post} x 3 angles = 18 runs.
# Start from the repo folder with:   nohup bash efov_overnight.sh > log.txt 2>&1 < /dev/null &
# The first run at each CV tunes conductivity and caches it in tuned_cv.csv; the rest reuse it.
set -u
NP=1                                   # 1 core: mpirun on this machine does not match openCARP's MPI
REPO="$(cd "$(dirname "$0")" && pwd)"

for CV in 0.25 0.50 0.60; do
  TEND=$(awk -v c=$CV 'BEGIN{t=81/c+20; if (t<250) t=250; printf "%d", t}')
  mkdir -p "$REPO/cv$CV" && cd "$REPO/cv$CV"
  for STATE in pre post; do
    for ANGLE in 0 45 90; do
      echo "=== CV $CV  $STATE  angle $ANGLE  tend $TEND  $(date)"
      python3 "$REPO/run.py" --state $STATE --angle $ANGLE --dx 0.25 \
              --tune-cv $CV --tend $TEND --np $NP < /dev/null \
        || echo "FAILED: CV $CV $STATE $ANGLE"
    done
  done
done
echo "ALL DONE $(date)"
cat "$REPO/tuned_cv.csv"