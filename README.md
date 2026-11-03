# eFOV final model (v3-reference)

A controlled 3D atrial slab with a straight transmural lesion boundary, simulated in openCARP, and the
end-to-end analysis of the electrograms an Advisor-HD-Grid-like catheter would record over it:
clinical-style voltage change points, compared with the exact tissue contribution to every recording.

Everything scientific is set in **`config/final_model.json`**; the model and its reasons are described in
**`docs/final_model.md`**. Earlier development code (2D sheet, pilot, readiness kit, final settings run)
is on the `pilot_v2` branch and in the git history.

## Layout

```
config/final_model.json      every value of the model and analysis, with provenance
config/test_small.json       small version for checking the code (minutes, 2 cores); not a result
efov/                        the package, one module per step (see efov/__init__.py)
  opencarp/slab_experiment.py   carputils experiment: one openCARP run
  opencarp/solver.par           fixed openCARP solver settings
notebooks/                   step-by-step notebooks 00-06 (same stages as the command line)
R/bayesian_change_points.R   optional mcp fits of the primary profiles
tests/                       unit tests and an opt-in real-openCARP end-to-end test
docs/                        model description, decision and verification logs, parameter sources
reference/                   the clinical analysis the change-point method mirrors
```

## Run it (workstation, inside `efov-runner`)

```bash
pip install pyamg                           # if not in the image: fast large extracellular solves
export MPIEXEC_BIN=/usr/local/lib/opencarp/lib/petsc/bin/mpiexec

python -m efov plan                                       # size, restart time, unconfirmed values
python -m efov import-production-model 3D_Code/runs/final_v2/productionModel.json \
       --write config/final_model.json                    # once: values chosen by the final settings run
python -m efov dry --out runs/v3_reference --np 24        # check the openCARP commands
nohup python -m efov run --out runs/v3_reference --np 24 > runs/v3_reference.log 2>&1 &
python -m efov status --out runs/v3_reference
```

Or open `notebooks/00_model_and_settings.ipynb` and work through 00-06. Notebooks and command line
use the same stages and the same output folder; either can resume the other.

Stages: `preflight -> conditioning -> simulation -> physiology -> window -> electrograms ->
measurements -> changePoints -> groundTruth -> report`. Each writes `runs/<name>/stages/<stage>.json`
with a fingerprint of the settings it used. A rerun skips completed stages and **refuses outputs made
with different settings** (use a new output folder). Analysis-only changes (for example change-point
settings) invalidate only the stages that use them.

## Outputs (`runs/<name>/`)

| Path | Content |
| --- | --- |
| `model.json` | the configuration used |
| `simulations/<condition>/` | openCARP output (vm.igb, phie.igb on the electrode layer, LATs), runRecord.json with the exact commands, log |
| `physiology/` | per-beat checks and criteria |
| `electrograms/` | window.json, unipolar electrograms (npz), forwardModelCheck.json |
| `measurements/` | calibration, points (every recording, placement, noise realisation, with weights), traces, filter and profile checks |
| `changePoints/` | fits, model selection, fit data, predictions, method.json |
| `groundTruth/` | contributionField.npz (signed shares on 0.5 mm voxels), summary.csv, checks.json |
| `acceptance.json`, `report.md`, `figures/` | summary |

## Tests

```bash
python -m pytest tests -q                                  # unit tests (no openCARP needed)
EFOV_RUN_OPENCARP=1 python -m pytest tests/test_end_to_end.py -q   # real openCARP, small config, ~20 min on 2 cores
```
