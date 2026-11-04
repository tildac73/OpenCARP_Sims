# eFOV final model (v4-reference, branch finalModelV2)

In-silico reconstruction of Wang et al.'s HD Grid effective-field-of-view study, with the exact
contributing tissue of every electrogram as ground truth. Plan and decisions: project document
`claude/final_model_plan_2026-11-03.md`; `docs/decision_log.md` (DL-2026-11-03-02, FM-01 to FM-22; DL-2026-11-03-03, FM-23 to FM-30).

Forward model: openCARP monodomain Vm; electrograms by reciprocity from one openCARP Laplace lead field per
electrode with the sigma_i source (FM-23). Lesion: 5.6 mm ablation strip with isolated viable tissue beyond
(FM-24). Amplitudes in model units, never calibrated to clinical values (FM-25).

## Files that define the model
| File | Holds |
| --- | --- |
| `config/model.par` | every openCARP setting (solver, time, cells, conductivities, lesion, bath, stimuli, outputs) |
| `config/leadfield.par` | the openCARP Laplace solve used for every electrode's lead field (FM-14, FM-23) |
| `config/model.json` | `sweep`: variables, baselines, planned values; `fixed`: mesh, bath, electrodes, analysis |
| `config/tuning.json` | beta tuned for a mesh and CV target (written by `efov tune --write-par`) |
| `config/sweeps/oat_baseline.json` | the one-at-a-time sweep |
| `reference/EFOV_analysis_pipeline.Rmd` | Wang's analysis, unedited (MD5 c739f13e6d07996b81d6f733f04f503d) |

## Run order on the workstation (efov-runner)
```bash
export MPIEXEC_BIN=/usr/local/lib/opencarp/lib/petsc/bin/mpiexec
python3 -m pytest tests -q                                    # unit tests (no openCARP)
python3 checks/container_checks.py --out runs/checks --np 2   # C1, W0 (Laplace cost), C4, C5, C8 on a small slab
python3 -m efov plan                                          # mesh size, lead fields per condition, assumptions
python3 -m efov tune --out runs/tune --np 24 --write-par      # W1: beta at 0.25 / 0.1 mm
nohup python3 -m efov run --out runs/v4_reference --np 24 > runs/v4_reference.log 2>&1 &     # W2, inferior
nohup python3 -m efov sweep config/sweeps/oat_baseline.json --out runs/oat --np 24 > runs/oat.log 2>&1 &
```
If W0 fails, set `fixed.groundTruth.leadFieldMethod` to `"python"` (pre-declared fallback, FM-14).

## Layout
`efov/` one module per stage (`python -m efov --help`); `efov/opencarp/slab_experiment.py` the carputils
experiment; `R/wang_replication.R` Wang's Rmd on one simulated slab; `checks/` openCARP checks;
`notebooks/` the same stages step by step (`python notebooks/build_notebooks.py` regenerates them);
`tests/` unit and synthetic end-to-end tests (`EFOV_RUN_OPENCARP=1` enables the real openCARP test).

## Outputs of a run (`runs/<name>/`)
`simulations/<condition>/run.par` resolved openCARP file; `physiology/`, `leadFields/` (which cached lead fields
the run used; the files are in `leadFieldCache/<conductor fingerprint>/`), `electrograms/`,
`measurements/` (points with PP and both peak frequencies, amplitudes.json, pfNoiseRobustness.csv), `export/` (Wang-format datasets per gap, noise realisation and PF definition), `changePoints/`
(Wang's tables per dataset), `groundTruth/` (summary.csv, profiles.npz, fields_*.npz, shares_*.vtk),
`report.md`, `acceptance.json`, `figures/`.
