#!/bin/bash
# Full EFOV batch: simulations -> electrograms -> ground truth -> contribution maps.
#
#   1. run.py       3 CVs x {pre, post} x 3 angles = 18 runs (0.25 mm), into runs/<batch>/cv<CV>/
#                   (the first run at each CV tunes conductivity and caches it in tuned_cv.csv)
#   2. meta.json    per run: the gi/ge it actually used (from its parameters.par)
#   3. egm.py       -> results/<config>/  points.csv + xlsx files for EFOV_analysis_pipeline.Rmd
#      ground_truth.py -> results/<config>/  groundtruth.csv, points_with_gt.csv (fov80 per recording)
#   4. gt_maps.py   -> results/<config>/maps/  contribution maps (.png, .npz, summary .csv)
#   5. EFOV_analysis_pipeline.Rmd -> results/<config>/EFOV_analysis_pipeline.html (.md without
#                   pandoc) + R_output/ (tables, figures, cache) + R_log.txt
#   (sweep configs: egm.py and the R analysis write results/<config>__<value>/,
#    ground truth and maps results/<config>/)
#
# Start from the repo folder with:   nohup bash 2D_Code/efov.sh > log.txt 2>&1 < /dev/null &
# Options (environment variables):
#   BATCH=<name>     batch folder name (default: today's date); reuse a name to add to / redo a batch
#   SKIP_SIMS=1      skip step 1 and re-process an existing batch (needs BATCH)
#   CONFIGS="..."    Shared_Pipeline configs to process (default: clin_default.json)
#   MAPS=0           skip step 4
#   RUN_R=0          skip step 5 (needs Rscript, the Rmd's R packages, and JAGS for mcp)
set -u
NP=1                                   # 1 core: mpirun on this machine does not match openCARP's MPI
REPO="$(cd "$(dirname "$0")" && pwd)"            # 2D_Code
ROOT="$(dirname "$REPO")"                        # repo root (EFOV_analysis_pipeline.Rmd lives here)
PIPE="$ROOT/Shared_Pipeline"
BATCH="${BATCH:-$(date +%Y-%m-%d)}"
BATCH_DIR="$REPO/runs/$BATCH"
CONFIGS="${CONFIGS:-clin_default.json}"
MAPS="${MAPS:-1}"
RUN_R="${RUN_R:-1}"

# ---------------- 1. simulations ----------------
if [ "${SKIP_SIMS:-0}" != 1 ]; then
  for CV in 0.25 0.50 0.60; do
    TEND=$(awk -v c=$CV 'BEGIN{t=81/c+20; if (t<250) t=250; printf "%d", t}')
    mkdir -p "$BATCH_DIR/cv$CV" && cd "$BATCH_DIR/cv$CV"
    for STATE in pre post; do
      for ANGLE in 0 45 90; do
        echo "=== CV $CV  $STATE  angle $ANGLE  tend $TEND  $(date)"
        python3 "$REPO/run.py" --state $STATE --angle $ANGLE --dx 0.25 \
                --tune-cv $CV --tend $TEND --np $NP < /dev/null \
          || echo "FAILED: CV $CV $STATE $ANGLE"
      done
    done
  done
  cd "$ROOT"
  echo "SIMULATIONS DONE $(date)"
  cat "$REPO/tuned_cv.csv"
fi

RUNS=( "$BATCH_DIR"/cv*/20*_simple_* )
if [ ! -e "${RUNS[0]}" ]; then
  echo "No run folders in $BATCH_DIR - nothing to process"; exit 1
fi

# ---------------- 2. per-run metadata ----------------
# ground_truth.py takes gi from meta.json if present (otherwise from Shared_Pipeline/tuned_cv.csv)
for RUN in "${RUNS[@]}"; do
  [ -f "$RUN/parameters.par" ] || continue
  awk '$1 == "gregion[0].g_il" {gi = $3} $1 == "gregion[0].g_el" {ge = $3}
       END {if (gi != "") printf "{\"gi\": %s, \"ge\": %s}\n", gi, ge}' \
      "$RUN/parameters.par" > "$RUN/meta.json"
  [ -s "$RUN/meta.json" ] || rm -f "$RUN/meta.json"
done

# ---------------- 3-4. electrograms, ground truth, maps ----------------
# The Shared_Pipeline scripts resolve relative paths against Shared_Pipeline/, so each config is
# copied with absolute paths: this batch's runs in, results/<name>/ at the repo root out
# (where the Rmd's DATA_DIR = "results/<name>" expects them).
mkdir -p "$BATCH_DIR/configs"
for CONF in $CONFIGS; do
  SRC="$PIPE/configs/$CONF"
  CFG="$BATCH_DIR/configs/$CONF"
  python3 - "$SRC" "$CFG" "$BATCH_DIR/cv*/20*_simple_*" "$ROOT/results" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg['runs'], cfg['out_dir'] = sys.argv[3], sys.argv[4]
json.dump(cfg, open(sys.argv[2], 'w'), indent=1)
EOF
  echo "=== $CONF: electrograms  $(date)"
  python3 "$PIPE/egm.py" "$CFG" || { echo "FAILED: egm.py $CONF"; continue; }

  # ground truth and maps don't depend on gain or noise: for a sweep config they are computed
  # once, into results/<name>/ (the egm.py outputs are in results/<name>__<value>/)
  NAME=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$CFG")
  OUT="$ROOT/results/$NAME"

  echo "=== $CONF: ground truth  $(date)"
  python3 "$PIPE/ground_truth.py" "$CFG" || echo "FAILED: ground_truth.py $CONF"

  if [ "$MAPS" = 1 ]; then
    echo "=== $CONF: contribution maps  $(date)"
    python3 "$PIPE/gt_maps.py" "$CFG" "$OUT/maps" "${RUNS[@]}" || echo "FAILED: gt_maps.py $CONF"
  fi

  # ---------------- 5. R analysis ----------------
  # EFOV_analysis_pipeline.Rmd once per egm.py output folder (one per sweep value):
  # report -> <folder>/EFOV_analysis_pipeline.html, tables/figures/cache -> <folder>/R_output/
  if [ "$RUN_R" = 1 ]; then
    if ! command -v Rscript > /dev/null; then
      echo "SKIPPED R analysis: Rscript not found"
    else
      for DATA in "$ROOT/results/$NAME" "$ROOT/results/$NAME"__*; do
        [ -f "$DATA/merged_rectangle_summary_euclidean.xlsx" ] || continue
        echo "=== $CONF: R analysis on $(basename "$DATA")  $(date)"
        # HTML needs pandoc (bundled with RStudio, often missing for plain Rscript): without it,
        # knit to Markdown. Both stop at the first error instead of writing it into the report.
        EFOV_DATA_DIR="$DATA" EFOV_OUT_DIR="$DATA/R_output" EFOV_ROOT="$ROOT" Rscript -e '
          d <- Sys.getenv("EFOV_DATA_DIR"); setwd(Sys.getenv("EFOV_ROOT"))
          rmd <- file.path(getwd(), "EFOV_analysis_pipeline.Rmd")
          knitr::opts_chunk$set(error = FALSE)
          if (rmarkdown::pandoc_available("1.12.3")) {
            rmarkdown::render(rmd, output_dir = d, knit_root_dir = getwd(),
                              intermediates_dir = tempdir(), envir = new.env(), quiet = TRUE)
          } else {
            message("pandoc not found: writing a Markdown report instead of HTML")
            knitr::opts_chunk$set(fig.path = file.path(d, "R_output", "report_figures", ""))
            knitr::knit(rmd, output = file.path(d, "EFOV_analysis_pipeline.md"),
                        envir = new.env(), quiet = TRUE)
          }' \
          > "$DATA/R_log.txt" 2>&1 \
          && echo "    -> $(ls "$DATA"/EFOV_analysis_pipeline.{html,md} 2>/dev/null | head -1)" \
          || { echo "FAILED: R analysis on $(basename "$DATA") - last lines of $DATA/R_log.txt:";
               tail -15 "$DATA/R_log.txt"; }
      done
    fi
  fi
done
echo "ALL DONE $(date)"
