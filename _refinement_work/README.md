# Part 1 refinement (27 Oct 2026)

Reprocessing of `boundaryPilotV2 2` without re-simulation. Findings: "Part 1 refinement - findings log" doc.
Decisions: `docs/decision_log.md` (DL-2026-10-27-01 to -04). Checks: `docs/verification_log.md`.

The driver imports the unchanged `3D_Code` and `Shared_Pipeline` functions and writes to a new
folder per run; the raw run folder is never modified. Run from the repository root.

| Output folder | Command | Purpose |
| --- | --- | --- |
| `run_w0_400_repro` | `python3 _refinement_work/run_window.py measure --out _refinement_work/run_w0_400_repro --window 0 400` then `contrib-intact` | Reproduction of the original analysis |
| `run_w25_100` | `measure`, `contrib-intact`, `contrib-lesion`, `finalise` with `--window 25 100` | Declared window: PP, FOV80, change points |
| `run_w30_100_unisens` | `measure`, `changepoints-only` with `--window 30 100` | Unipolar window-start sensitivity |
| `run_w25_100_noise<SD>` | `measure`, `changepoints-only` with `--window 25 100 --noise-sd <SD>` for SD 0.001, 0.0025, 0.005, 0.01, 0.02 | Noise-level sweep |
| `noise_sweep_summary.csv` | analysis snippet recorded in the findings log, Step 3 | Noise sweep summary |
| `bayes_w25_100`, `bayes_original_long` | `Rscript _refinement_work/bayes_refit.R <run> <out>` (Mac, mcp 0.3.4) | Bayesian refits, 15,000 / 5,000 iterations |

Stages for `run_window.py`: `measure`, `contrib-intact`, `contrib-lesion`, `finalise`, `changepoints-only`.
Fixed settings: gain held at 0.461584; physiology not rerun (original pass carried over).

`shim/scipy` is used only where scipy cannot be installed (analysis VM); `test_shim.py` checks it against scipy.
Data slices (`*_vm_a.npy` etc.) were scratch copies and are not tracked; delete them freely.

Regression test: `python3 -m unittest tests.test_v2_reference -v` (`EFOV_RUN_SLOW=1` adds the FOV80 check).
