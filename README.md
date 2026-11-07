# eFOV final model (v4-reference, branch finalModelV2)

In-silico reconstruction of Wang et al.'s HD Grid effective-field-of-view study, with the exact
contributing tissue of every electrogram as ground truth. Plan and decisions: project document
`claude/final_model_plan_2026-11-03.md`; `docs/decision_log.md` (DL-2026-11-03-02, FM-01 to FM-22; DL-2026-11-03-03, FM-23 to FM-30).

Forward model: openCARP monodomain Vm; electrograms by reciprocity from one lead field per electrode with the
sigma_i source (FM-23). Each lead field is an openCARP Laplace solve with the electrode held at 1 mV and the
corner grounded, divided by the current the electrode draws (equipotential electrode, FM-34). Lesion: 5.6 mm ablation strip with isolated viable tissue beyond
(FM-24). Amplitudes in model units, never calibrated to clinical values (FM-25).

## Files that define the model
| File | Holds |
| --- | --- |
| `config/model.par` | every openCARP setting (solver, time, cells, conductivities, lesion, bath, stimuli, outputs) |
| `config/leadfield.par` | the openCARP Laplace solve of every electrode's lead field: electrode at 1 mV, corner ground (FM-14, FM-34) |
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
```
Production: the base model, its figures, then every experiment, from one notebook:
```bash
nohup jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 notebooks/run_all.ipynb \
      --output run_all_executed.ipynb > runs/run_all.log 2>&1 &
```
The same work without the notebook (same folders, so either resumes the other's completed stages):
```bash
nohup python3 -m efov sweep config/sweeps/oat_baseline.json --out runs/production --np 24 > runs/production.log 2>&1 &
tail runs/production.log; cat runs/production/sweepStatus.json   # progress; a failed variant carries its error
```
beta is already tuned for the production mesh (config/tuning.json, FM-32); `efov tune --write-par` is needed
only after a change to the mesh, CV target, ionic model or conductivities. `--np` sets both the openCARP MPI
processes and the lead-field worker processes.

## Layout
`efov/` one module per stage (`python -m efov --help`); `efov/opencarp/slab_experiment.py` the carputils
experiment; `R/wang_replication.R` Wang's Rmd on one simulated slab; `checks/` openCARP checks;
`notebooks/run_all.ipynb` one notebook that runs the base model with its figures, then every experiment of the
sweep file against it, then the noise-doubling test and the summary (`python notebooks/build_notebooks.py`
regenerates it; `config/sweeps/test_small.json` is its short test sweep);
`tests/` unit and synthetic end-to-end tests (`EFOV_RUN_OPENCARP=1` enables the real openCARP test).

## Outputs of a run (`runs/<name>/`)
`simulations/<condition>/run.par` resolved openCARP file; `physiology/`, `leadFields/` (which cached lead fields
the run used; the files are in `leadFieldCache/<conductor fingerprint>/`), `electrograms/`,
`measurements/` (points with PP and both peak frequencies, amplitudes.json, pfNoiseRobustness.csv), `export/` (Wang-format datasets per gap, noise realisation and PF definition), `changePoints/`
(Wang's tables per dataset), `groundTruth/` (summary.csv, profiles.npz, fields_*.npz, shares_*.vtk),
`report.md`, `acceptance.json`, `figures/`.
