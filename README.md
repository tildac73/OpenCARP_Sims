# eFOV final model (v4-reference, branch finalModelV2)

In-silico reconstruction of Wang et al.'s HD Grid effective-field-of-view study, with the exact
contributing tissue of every electrogram as ground truth. Plan and decisions: project document
`claude/final_model_plan_2026-11-03.md`; `docs/decision_log.md` (DL-2026-11-03-02, FM-01 to FM-22; DL-2026-11-03-03, FM-23 to FM-30).

Forward model: openCARP monodomain Vm; electrograms by reciprocity from one lead field per electrode with the
sigma_i source (FM-23). Lead fields are our P1 solve on openCARP's run mesh (FM-31: openCARP 19.0.0's Laplace
mode ignores current stimuli, so the pre-declared fallback of FM-14 is used). Lesion: 5.6 mm ablation strip with isolated viable tissue beyond
(FM-24). Amplitudes in model units, never calibrated to clinical values (FM-25).

## Files that define the model
| File | Holds |
| --- | --- |
| `config/model.par` | every openCARP setting (solver, time, cells, conductivities, lesion, bath, stimuli, outputs) |
| `config/leadfield.par` | the openCARP Laplace route for lead fields (FM-14); not used while `leadFieldMethod = python` (FM-31) |
| `config/model.json` | `sweep`: variables, baselines, planned values; `fixed`: mesh, bath, electrodes, analysis |
| `config/tuning.json` | beta tuned for a mesh and CV target (written by `efov tune --write-par`; production mesh done 4 Nov, FM-32) |
| `config/sweeps/oat_baseline.json` | the one-at-a-time sweep |
| `reference/EFOV_analysis_pipeline.Rmd` | Wang's analysis, unedited (MD5 c739f13e6d07996b81d6f733f04f503d) |

## Run order on the workstation (efov-runner)
```bash
export MPIEXEC_BIN=/usr/local/lib/opencarp/lib/petsc/bin/mpiexec
python3 -m pytest tests -q                                    # unit tests (no openCARP)
python3 checks/container_checks.py --out runs/checks --np 2   # C1, W0, C4, C4p, C5, C8 on a small slab (about 30 min)
python3 -m efov plan                                          # mesh size, lead fields per condition, assumptions
nohup python3 -m efov run --out runs/v4_reference --np 24 > runs/v4_reference.log 2>&1 &     # production reference run
python3 -m efov status --out runs/v4_reference
nohup python3 -m efov sweep config/sweeps/oat_baseline.json --out runs/oat --np 24 > runs/oat.log 2>&1 &
```
beta is already tuned for the production mesh (config/tuning.json, FM-32); `efov tune --write-par` is needed
only after a change to the mesh, CV target, ionic model or conductivities. `--np` sets both the openCARP MPI
processes and the lead-field worker processes.

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
