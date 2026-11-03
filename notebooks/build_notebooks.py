"""Write the step-by-step notebooks (run once after editing this file: python notebooks/build_notebooks.py).

Every notebook starts with the same settings cell: the config file, the output folder and the MPI
process count. They can be overridden with the environment variables EFOV_CONFIG, EFOV_OUT, EFOV_NP
(used to test the notebooks on config/test_small.json). Stage names and outputs: efov/pipeline.py.
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
CONFIG = os.environ.get("EFOV_CONFIG", str(REPOSITORY / "config" / "model.json"))
OUT    = os.environ.get("EFOV_OUT",    str(REPOSITORY / "runs" / "v4_reference"))
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

Two files define the model (plan claude/final_model_plan_2026-11-03.md, FM-01, FM-02):
`config/model.par` holds every openCARP setting; `config/model.json` holds the sweep variables
(baseline and planned values) and the settings openCARP cannot read. Nothing is simulated here.

| Notebook | Stages | What it does |
| --- | --- | --- |
| 01 | preflight, conditioning, tune, simulation | beta tuning and the openCARP runs (hours on the workstation) |
| 02 | physiology, window | activation checks; per-placement measurement windows |
| 03 | leadFields, electrograms | openCARP lead field per electrode; unipolar signals by reciprocity (sigma_i source) |
| 04 | measurements, export, changePoints | acquisition chain, PP, peak frequency; Wang's code |
| 05 | groundTruth | contributing tissue: classes, depth profile, 3D fields |
| 06 | report | acceptance, report.md, figures |
| 07 | sweeps | the one-at-a-time sweep |

One command runs the same stages: `python -m efov run --out runs/v4_reference --np 24`.""",
    "md:## The openCARP parameter file (every simulator setting)",
    """print(open(config["modelPar"]).read())""",
    "md:## Sweep variables: baseline and planned values",
    """pd.DataFrame({k: dict(baseline=v["baseline"], planned=v["planned"], actsOn=v.get("actsOn", ""))
              for k, v in config["sweep"].items()}).T""",
    "md:## Size of the run and assumptions in force",
    """import subprocess
print(subprocess.run([sys.executable, "-m", "efov", "plan", "--config", CONFIG], cwd=REPOSITORY, capture_output=True, text=True).stdout)""",
])

notebook("01_simulate.ipynb", [
    """# 1. Simulate: conditioning, beta tuning and the two openCARP runs

* **preflight**: software versions, repository commit, solver flavour, R availability.
* **conditioning**: one cell paced 1,000 times with `bench` (limit cycle).
* **tune**: beta for the CV target on a strip at the production mesh (FM-18); uses `config/tuning.json`
  if `python -m efov tune --write-par` was run for this mesh and target.
* **simulation**: `efov/opencarp/slab_experiment.py` (carputils) for the intact and lesion slab. Every
  run is monodomain (FM-23) and reads one resolved parameter file (`simulations/<condition>/run.par`). In the
  lesion run a 5.6 mm ablation strip separates the paced tissue from isolated viable tissue (FM-24).

Long runs are safer from the terminal: `nohup python -m efov run --out runs/v4_reference --np 24 --until simulation > runs/sim.log 2>&1 &`.""",
    """run.runStage("preflight")""",
    """run.runStage("conditioning")""",
    """run.runStage("tune")""",
    """result = run.runStage("simulation")
pd.DataFrame(result["timing"]).T.join(pd.DataFrame(result["nodes"]).T)""",
    "md:The resolved parameter file of the lesion run: model.par with this run's overrides in place, then the run-specific paths.",
    """print((run.jobs["lesion"] / "run.par").read_text()[-3000:])""",
])

notebook("02_physiology_and_window.ipynb", [
    """# 2. Physiology checks and the measurement window

On catheter-facing tissue nodes of the last three beats: one activation per beat, repolarisation inside
the response window, quiet lesion core, planar wave in the intended direction, stable latency,
amplitude and APD90. The surface CV is reported; the CV gate is the tune stage. Engineering criteria,
not validated biological limits. Each placement's window is derived from the intact activation under
it before any electrogram is read (DL-2026-10-28-08, FM-28).""",
    """physiology = run.runStage("physiology")
pd.DataFrame({c: v["criteria"] for c, v in physiology["conditions"].items()})""",
    """pd.read_csv(run.shared / "physiology" / "physiologyChecks.csv")""",
    """window = run.runStage("window")
window""",
    """from efov.report import activationFigure
display(Image(str(activationFigure(run, run.folder("figures")))))""",
])

notebook("03_electrograms.ipynb", [
    """# 3. Electrograms

Each electrode is the set of bath mesh nodes inside its 1 mm cylinder (FM-14). Every distinct electrode
gets an openCARP Laplace lead field z (cached per conductor); its unipolar signal is phi(t) = -z^T S_i Vm(t),
the bidomain extracellular potential with the sigma_i source recovered from the monodomain Vm (FM-23).
Only the last beat's windows plus twice the filter history are computed.""",
    """run.runStage("leadFields")""",
    """run.runStage("electrograms")""",
    """egm = np.load(run.out / "electrograms" / "intact.npz")
print({k: egm[k].shape for k in egm.files})""",
])

notebook("04_measurements_and_change_points.ipynb", [
    """# 4. Measurements, peak frequency and Wang's change points

Acquisition chain (FM-09, FM-22): resample to 2 kHz, Gaussian noise per electrode, 30-500 Hz band-pass
for bipoles and omnipoles. Peak frequency (FM-10): Morlet wavelet approximation of EnSite OT Near
Field, two co-primary definitions (FM-27); bipolar/omnipolar PP within +-30 ms of the peak-frequency
time (FM-11). Amplitudes stay in model units; Wang's 1.44 mV is shown beside them, never applied (FM-25).
The export writes Wang's EnSite columns; her Rmd then runs unchanged except six one-slab functions
(R/wang_replication.R, FM-12).""",
    """measures = run.runStage("measurements")
measures["amplitudes"], measures["checks"]""",
    """points = pd.read_csv(run.out / "measurements" / "lesionPoints.csv")
points[points.noiseRepeat == 0].groupby("kind")[["ppMv", "pfHz", "pfGlobalHz"]].describe().T""",
    """run.runStage("export")""",
    """changes = run.runStage("changePoints")
print("eligible:", changes["eligibleForInterpretation"], changes["prerequisites"])
pd.read_csv(run.out / "changePoints" / "SuppResults5_change_points_all.csv")""",
    """from efov.report import profileFigure
display(Image(str(profileFigure(run, run.folder("figures"), "pfHz", "peak frequency (Hz)", "03_peak_frequency_profiles.png", log=False))))""",
])

notebook("05_ground_truth.ipynb", [
    """# 5. Ground truth: the contributing tissue

For every recording at the ground-truth placements, each tissue element's signed share of the signal
energy, from the same openCARP lead fields as the electrograms (FM-14, FM-23). Shares sum to 1; nothing is cut off; negative
shares (cancellation) are kept. Reported: near/far (4 mm convention, FM-30) x endocardial/epicardial classes (FM-15), the
transmural profile per 0.1 mm layer, lateral reach, shape descriptors, and the full 3D fields
(`groundTruth/fields_<condition>.npz`, `groundTruth/shares_<condition>_gap<g>mm.vtk` for ParaView).""",
    """checks = run.runStage("groundTruth")
checks""",
    """summary = pd.read_csv(run.out / "groundTruth" / "summary.csv")
classes = [c for c in summary.columns if c.startswith("mass_")]
summary.groupby(["condition", "kind", "gapMm"])[classes].median()""",
    """from efov.groundtruth_figures import makeGroundTruthFigures
for path in makeGroundTruthFigures(run, run.folder("figures")):
    display(Image(str(path)))""",
])

notebook("06_report.ipynb", [
    """# 6. Report and acceptance

`acceptance.json` lists every check; **modelAccepted** = tuning, ground-truth share sum,
physiology and profile checks all passed. `report.md` and `figures/` summarise the run; assumptions
still in force are listed.""",
    """result = run.runStage("report")
result""",
    """display(Markdown((run.out / "report.md").read_text()))""",
])

notebook("07_sweeps.ipynb", [
    """# 7. The one-at-a-time sweep (FM-17)

`config/sweeps/oat_baseline.json` lists the variables; each takes each planned value from
`config/model.json` alone, plus the named 'roof' region. Run from the terminal:

`nohup python -m efov sweep config/sweeps/oat_baseline.json --out runs/oat --np 24 > runs/oat.log 2>&1 &`

Amplitudes are in model units in every variant (FM-25). Variants that change only electrodes, noise or
analysis reuse the baseline simulation; lead fields are shared by every variant with the same conductor
(FM-23); the others compact their raw output after their report (FM-20). Failed variants are recorded
and never pooled. `noiseDoubling.csv` is the pre-registered primary test (FM-25): the change-point shift
per doubling of noise.""",
    """path = REPOSITORY / "runs" / "oat" / "noiseDoubling.csv"
pd.read_csv(path) if path.exists() else "noiseDoubling.csv not written yet\"""",
    """SWEEP_OUT = REPOSITORY / "runs" / "oat"
status = json.loads((SWEEP_OUT / "sweepStatus.json").read_text()) if (SWEEP_OUT / "sweepStatus.json").exists() else {}
pd.DataFrame({k: dict(status=v["status"], overrides=v.get("overrides")) for k, v in status.items()}).T""",
    """# Pool the inferior baseline and the roof variant for Wang's regional comparison:
# subprocess.run([sys.executable, "-m", "efov", "wang-combine", str(SWEEP_OUT / "variants" / "baseline"),
#                 str(SWEEP_OUT / "variants" / "roof"), "--out", str(SWEEP_OUT / "wang_regions")], cwd=REPOSITORY)""",
])

print("notebooks written to", HERE)
