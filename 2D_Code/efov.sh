#!/bin/bash
# Overnight 0.25 mm runs: 3 CVs x {pre, post} x 3 angles = 18 runs.
# The first run at each CV runs tuneCV and caches gi/ge in tuned_cv.csv; the rest reuse it.
set -u
NP=1
REPO="$(cd "$(dirname "$0")" && pwd)"

for CV in 0.40 0.65 0.90; do
  mkdir -p "$REPO/cv$CV" && cd "$REPO/cv$CV"
  for STATE in pre post; do
    for ANGLE in 0 45 90; do
      echo "=== CV $CV  $STATE  angle $ANGLE  $(date)"
      python3 "$REPO/run.py" --state $STATE --angle $ANGLE --dx 0.25 \
              --tune-cv $CV --np $NP \
        || echo "FAILED: CV $CV $STATE $ANGLE"
    done
  done
done
echo "ALL DONE $(date)"
cat "$REPO/tuned_cv.csv"