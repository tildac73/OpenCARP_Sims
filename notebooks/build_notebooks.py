"""Write the step-by-step notebooks (run once after editing this file: python notebooks/build_notebooks.py).

Every notebook starts with the same settings cell: the config file, the output folder and the MPI
process count. They can be overridden with the environment variables EFOV_CONFIG, EFOV_OUT, EFOV_NP
(used to test the notebooks on config/test_small.json).
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

SETTINGS = '''\
import os, sys
from pathlib import Path
REPOSITORY = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPOSITORY))

# ---- settings: edit here, or set EFOV_CONFIG / EFOV_OUT / EFOV_NP ----
CONFIG = os.environ.get("EFOV_CONFIG", str(REPOSITORY / "config" / "final_model.json"))
OUT    = os.environ.get("EFOV_OUT",    str(REPOSITORY / "runs" / "v3_reference"))
NP     = int(os.environ.get("EFOV_NP", "24"))      # MPI processes for openCARP
# openCARP's own MPI launcher (efov-runner image); needed when NP > 1
os.environ.setdefault("MPIEXEC_BIN", "/usr/local/lib/opencarp/lib/petsc/bin/mpiexec")

import json
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display
from efov.config import loadConfig
from efov.pipeline import Run

config = loadConfig(CONFIG)
run = Run(config, OUT, processCount=NP)
print(f"{config['modelName']} -> {run.out}")
run.status()'''


def notebook(name, cells):
    nb = new_notebook()
    nb.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
    nb.cells = [new_markdown_cell(cells[0])] + [new_code_cell(SETTINGS)] + [
        new_markdown_cell(c[3:]) if c.startswith("md:") else new_code_cell(c) for c in cells[1:]]
    nbformat.write(nb, HERE / name)


notebook("00_model_and_settings.ipynb", [
    """# 0. The final model and its settings

The model is defined by one file, `config/final_model.json`; the code sets no scientific value.
This notebook shows that configuration, which values still await confirmation from the workstation
final settings run, and the openCARP commands it will produce. Nothing is simulated here.

**Order of the notebooks**

| Notebook | Stages | What it does |
| --- | --- | --- |
| 01 | preflight, conditioning, simulation | openCARP runs (hours on the workstation) |
| 02 | physiology, window | activation checks; measurement window |
| 03 | electrograms | unipolar phi_e at every electrode; forward-model check |
| 04 | measurements, changePoints | calibration, recordings, PP, clinical change points |
| 05 | groundTruth | exact contributing-tissue fields |
| 06 | report | acceptance, report.md, figures |

The same stages run in one command: `python -m efov run --out runs/v3_reference --np 24`.
Every stage is resumable and refuses outputs made with different settings.""",
    "md:## The configuration, section by section",
    """for section in ["tissue", "lesion", "bath", "pacing", "numerics", "electrodes"]:
    values = {k: v for k, v in config[section].items() if not k.startswith("_")}
    display(Markdown(f"**{section}**"), pd.Series(values, dtype=object).to_frame("value"))""",
    "md:## Provenance: where each consequential value comes from\n\n`awaitingWorkstation` values are the expected rule outcomes; replace them with the workstation's `productionModel.json` (next cell).",
    """pd.DataFrame(config["provenance"]).T.drop(index="_about", errors="ignore")""",
    """# Import the rule-chosen values from the final settings run (pilot_v2: 3D_Code/runs/final_v2/productionModel.json):
# from efov.config import importProductionModel
# importProductionModel("path/to/productionModel.json", CONFIG, outputPath=CONFIG)""",
    "md:## Size of the run",
    """from efov import config as cfg
print(f"mesh: {cfg.meshNodeCount(config):,} nodes (tissue + bath)")
print(f"placements: {len(cfg.centreYmm(config)) * len(config['electrodes']['gapsMm'])}")
print(f"prepacing restart: {cfg.restartTimeMs(config)} ms of {config['pacing']['beatCount'] * config['pacing']['cycleLengthMs']} ms")""",
    "md:## openCARP commands (carputils `--dry`): check them before the long run",
    """from efov import simulation
state = run.runStage("conditioning")["state"]
dry = simulation.runCondition(config, "lesion", state, run.out / "dry", NP, dry=True)["dryRun"]
print("\\n".join(line for line in dry.splitlines() if line.strip())[-6000:])""",
])

notebook("01_simulate.ipynb", [
    """# 1. Simulate: conditioning and the two openCARP runs

* **preflight** records software versions and the repository commit, and checks that the chosen
  solver flavour (boomeramg) works on this openCARP build (else petsc).
* **conditioning** paces one Courtemanche cell 1,000 times with `bench` (limit-cycle state).
* **simulation** runs `efov/opencarp/slab_experiment.py` (carputils) for the intact and lesion slab:
  monodomain prepacing to the restart time, then pseudo-bidomain with the bath.

On the workstation the full-size runs take hours. Running them from the terminal is safer than a
notebook kernel: `nohup python -m efov run --out runs/v3_reference --np 24 --until simulation > runs/sim.log 2>&1 &`.
This notebook then simply reuses the finished runs.""",
    """run.runStage("preflight")""",
    """run.runStage("conditioning")""",
    """result = run.runStage("simulation")
pd.DataFrame(result["timing"]).T.join(pd.DataFrame(result["nodes"]).T)""",
    "md:Each run folder keeps the exact openCARP commands (`runRecord.json`), the spec (`<condition>.json`) and the full log.",
    """record = json.loads((run.jobs["lesion"] / "runRecord.json").read_text())
print(" ".join(record["commands"]["run"])[:3000])""",
])

notebook("02_physiology_and_window.ipynb", [
    """# 2. Physiology checks and the measurement window

Checks on the last three beats in a viable region (acceptance.physiology): one activation per beat,
repolarisation inside the response window, quiet lesion core, planar wave along +x, stable latency,
amplitude and APD90, and conduction velocity within 2 % of 0.90 m/s. Engineering criteria, not
validated biological limits.

The measurement window is then derived from the intact last-beat activation of the top surface under
the grid (DL-2026-10-28-08), before any electrogram is read.""",
    """physiology = run.runStage("physiology")
pd.DataFrame({c: v["criteria"] for c, v in physiology["conditions"].items()})""",
    """pd.read_csv(run.out / "physiology" / "physiologyChecks.csv")""",
    """{c: round(v["lastBeatSpeedMPerS"], 4) for c, v in physiology["conditions"].items()}""",
    """window = run.runStage("window")
window""",
    """from efov.report import activationFigure
display(Image(str(activationFigure(run, run.folder("figures")))))""",
])

notebook("03_electrograms.ipynb", [
    """# 3. Electrograms and the forward-model check

Unipolar signal of each electrode = openCARP's phi_e (referenced to the grounded bath corner),
interpolated at its 8 surface samples and averaged. Only the last beat's window plus twice the filter
history is extracted.

The forward-model check compares openCARP's phi_e with an independent finite-element solve of the same
discrete problem (sigma_m source, same mesh, conductivities and ground) at every electrode sample.
It tests the configuration, not the physiology, and stops the run if it fails.""",
    """result = run.runStage("electrograms")
pd.DataFrame(result["forwardModelCheck"]).T""",
    """egm = np.load(run.out / "electrograms" / "intact.npz")
print({k: egm[k].shape for k in egm.files})""",
    """import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 3))
for condition in ("intact", "lesion"):
    e = np.load(run.out / "electrograms" / f"{condition}.npz")
    p = int(np.argmin(np.abs(e["placements"][:, 1]) + e["placements"][:, 2]))   # centre y = 0, smallest gap
    ax.plot(e["timeMs"], e["uni"][p, 5], label=f"{condition} E5 (model units, before gain)")
ax.set(xlabel="time (ms)", ylabel="phi_e"); ax.legend(); plt.show()""",
])

notebook("04_measurements_and_change_points.ipynb", [
    """# 4. Measurements and clinical-style change points

1. Gain: the median of the nine selected bipoles at the intact reference placement is set to
   1.44 mV (imposed normalisation, not validation).
2. Per condition and noise realisation (0 = clean, then 3 x 5 uV Gaussian): unipole E5 unfiltered;
   bipole E5-E6|E9 and omnipoles E5-E6-E9, E9-E10-E5 band-passed 30-450 Hz (causal); peak-to-peak in
   the window. Selected weights are saved and applied to the clean signals.
3. Filter-history and profile checks (stable far plateau, lesion transition).
4. Change points: decay-to-plateau fits as in the clinical analysis, one slab, profile least squares.
   Results are eligible for interpretation only if every prerequisite passed and the matched intact
   control has no supported change point.""",
    """measures = run.runStage("measurements")
measures["calibration"], measures["checks"]""",
    """pd.read_csv(run.out / "measurements" / "profileChecks.csv")""",
    """changes = run.runStage("changePoints")
print("prerequisites:", changes["prerequisites"])
pd.DataFrame(changes["primary"])""",
    """from efov.report import profileFigure
display(Image(str(profileFigure(run, run.folder("figures")))))""",
    "md:### Optional: Bayesian change points (R, mcp + JAGS; available in the efov-runner image)",
    """# import subprocess
# subprocess.run(["Rscript", str(REPOSITORY / "R" / "bayesian_change_points.R"), str(run.out)], check=True)
# pd.read_csv(run.out / "changePoints" / "bayesian" / "bayesianChangePoints.csv")""",
])

notebook("05_ground_truth.ipynb", [
    """# 5. Ground truth: the exact contributing-tissue field

For each ground-truth recording (clean, catheter centres in `groundTruth.centresYmm`, both gaps and
conditions) the reciprocity lead field gives every element's signed share of the signal energy in the
window. Shares sum to 1; nothing is cut off; negative shares (cancellation) are kept. Fields are summed
into 0.5 mm voxels (`groundTruth/contributionField.npz`).

Checks, enforced: shares sum to 1, the field reproduces the measured clean electrogram, and its scale
equals the calibration gain.""",
    """checks = run.runStage("groundTruth")
checks""",
    """summary = pd.read_csv(run.out / "groundTruth" / "summary.csv")
summary""",
    """from efov.report import fieldFigure
path = fieldFigure(run, run.folder("figures"))
if path: display(Image(str(path)))""",
    """import matplotlib.pyplot as plt
field = np.load(run.out / "groundTruth" / "contributionField.npz")
step = float(field["curveStepMm"])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for k, recording in enumerate(field["recordingId"]):
    if ":0:" in recording and "intact" in recording:
        axes[0].plot(np.arange(field["lateralCumulativeShare"].shape[1]) * step, field["lateralCumulativeShare"][k], lw=0.8)
        axes[1].plot(np.arange(field["depthCumulativeShare"].shape[1]) * step, field["depthCumulativeShare"][k], lw=0.8)
axes[0].set(xlabel="lateral distance from anchor (mm)", ylabel="cumulative share", title="intact recordings")
axes[1].set(xlabel="depth below the catheter-facing surface (mm)", title="descriptive curves, not a radius")
plt.show()""",
])

notebook("06_report.ipynb", [
    """# 6. Report and acceptance

`acceptance.json` lists every check. **modelAccepted** = every implementation, physiology and
profile check passed. **productionReady** additionally needs every configuration value confirmed
(provenance). `report.md` and `figures/` summarise the run.""",
    """result = run.runStage("report")
result""",
    """display(Markdown((run.out / "report.md").read_text()))""",
])

print("notebooks written to", HERE)
