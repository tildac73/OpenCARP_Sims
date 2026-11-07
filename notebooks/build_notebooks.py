"""Write notebooks/run_all.ipynb (run after editing this file: python notebooks/build_notebooks.py).

One notebook runs everything in order: Part 1 the base model (every stage, its checks and figures),
Part 2 each parameter-changing experiment of the sweep file, compared with the base model, and Part 3
the pre-registered noise-doubling test and the combined summary.

The base model is the sweep's baseline: it runs in <OUT>/variants/baseline with the sweep's shared
simulation and lead-field caches, so `python -m efov sweep` and this notebook use the same folders and
reuse each other's completed stages. Settings come from the first code cell or from the environment
variables EFOV_SWEEP, EFOV_OUT, EFOV_NP (used to test the notebook on config/sweeps/test_small.json).

Part 2 has one section per experiment of config/sweeps/oat_baseline.json, written when this file is
run; the planned values themselves are read from the model config when the notebook runs. An experiment
that is not in the sweep file in use is skipped with a message. Stage names: efov/pipeline.py.
"""
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
SWEEP_FILE = REPOSITORY / "config" / "sweeps" / "oat_baseline.json"

SETTINGS = '''\
import os, sys
from pathlib import Path
REPOSITORY = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPOSITORY))

# ---- settings: edit here, or set EFOV_SWEEP / EFOV_OUT / EFOV_NP ----
SWEEP = os.environ.get("EFOV_SWEEP", str(REPOSITORY / "config" / "sweeps" / "oat_baseline.json"))
OUT   = os.environ.get("EFOV_OUT",   str(REPOSITORY / "runs" / "production"))
NP    = int(os.environ.get("EFOV_NP", "24"))     # openCARP MPI processes and lead-field worker processes
# openCARP's own MPI launcher (efov-runner image); needed when NP > 1
os.environ.setdefault("MPIEXEC_BIN", "/usr/local/lib/opencarp/lib/petsc/bin/mpiexec")

import json
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display
from efov.sweep import Sweep, experimentFigure, experimentTable, noiseDoublingTable

pd.set_option("display.max_columns", 60, "display.width", 250)
sweep = Sweep(SWEEP, OUT, processCount=NP)
run = sweep.baseline                   # the base model = the sweep's baseline
config = run.config
FIGURES = Path(OUT) / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
print(f"model {config['modelName']} ({sweep.modelConfigPath.name}); sweep {Path(SWEEP).name}; output {sweep.out}")
print(f"{len(sweep.plan) - 1} variants planned besides the baseline")
run.status()'''

EXPERIMENT = '''\
VARIABLE = {variable!r}
names = sweep.namesFor(VARIABLE)
if not names:
    print(f"{{VARIABLE}} is not in {{Path(SWEEP).name}}: skipped")
else:
    for name in names:
        print(f"--- {{name}}: overrides {{sweep.plan[name]}}", flush=True)
        entry = sweep.runVariant(name)
        print(f"    {{entry['status']}}" + (f": {{entry['error']}}" if entry["status"] != "complete" else ""), flush=True)
    display(experimentTable(sweep, VARIABLE))
    figure = experimentFigure(sweep, VARIABLE, FIGURES / f"experiment_{{VARIABLE}}.png")
    if figure:
        display(Image(str(figure)))'''


def experimentText(variable, model, sweep):
    if variable in sweep.get("named", {}):
        overrides = sweep["named"][variable]
        return (f"### Named variant `{variable}`\n\nOverrides: `{json.dumps(overrides)}`. "
                f"{sweep.get('_about', '')}")
    spec = model["sweep"][variable]
    planned = ", ".join(f"`{json.dumps(v)}`" for v in spec["planned"]) or "none"
    source = spec.get("_source", spec.get("_about", ""))
    return (f"### Experiment: `{variable}`\n\nBaseline `{json.dumps(spec['baseline'])}`; values {planned}; acts on "
            f"*{spec.get('actsOn', '')}*.\n\n{source}")


def build():
    sweep = json.loads(SWEEP_FILE.read_text())
    model = json.loads((SWEEP_FILE.parent / sweep["modelConfig"]).resolve().read_text())
    md, code = new_markdown_cell, new_code_cell
    cells = [
        md("""# eFOV final model: base model, then every parameter experiment

One notebook, run top to bottom (plan `claude/final_model_plan_2026-11-03.md`; decisions FM-01 to FM-33 in
`docs/decision_log.md`).

* **Part 1 - base model.** The inferior-wall reference: conditioning, beta, the two openCARP runs (intact and
  lesion), physiology checks, measurement windows, lead fields, electrograms, measurements, Wang's change
  points, the ground truth, the report and its figures.
* **Part 2 - experiments.** Each sweep variable takes each planned value alone (one at a time), plus the named
  'roof' region. Each section runs its variants and compares them with the base model.
* **Part 3 - summary.** The pre-registered primary test (change-point shift per doubling of noise, FM-25) and
  one table of every variant.

Every stage records the settings it used; a rerun reuses completed stages and refuses stale ones, so the
notebook can be stopped and restarted. A failed variant is recorded with its error and the notebook carries
on; failures are never pooled. The same runs from a terminal: `python -m efov sweep config/sweeps/oat_baseline.json
--out runs/production --np 24`. Running the whole notebook takes days on the workstation; for an unattended
run, execute it with `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1
notebooks/run_all.ipynb --output run_all_executed.ipynb` under `nohup`."""),
        code(SETTINGS),

        md("""# Part 1. The base model

## 1.1 What defines it
`config/model.par` holds every openCARP setting; `config/model.json` the sweep variables (baseline and
planned values) and settings openCARP cannot read (FM-01, FM-02). Nothing is simulated in this section."""),
        code('''print(open(config["modelPar"]).read())'''),
        code('''pd.DataFrame({k: dict(baseline=v["baseline"], planned=v["planned"], actsOn=v.get("actsOn", ""))
              for k, v in config["sweep"].items()}).T'''),
        code('''import subprocess
print(subprocess.run([sys.executable, "-m", "efov", "plan", "--config", str(sweep.modelConfigPath)], cwd=REPOSITORY,
                     capture_output=True, text=True).stdout)'''),

        md("""## 1.2 Simulate: conditioning, beta and the two openCARP runs
* **preflight**: software versions, repository commit, solver flavour, R availability.
* **conditioning**: one cell paced 1,000 times with `bench` (limit cycle).
* **tune**: beta for the CV target along fibres at the production mesh (FM-18, FM-32); taken from
  `config/tuning.json` when it holds this mesh and target.
* **simulation**: `efov/opencarp/slab_experiment.py` (carputils), intact and lesion slab, monodomain (FM-23),
  one resolved parameter file per run; in the lesion run a 5.6 mm ablation strip separates the paced tissue
  from isolated viable tissue (FM-24)."""),
        code('''run.runStage("preflight")'''),
        code('''run.runStage("conditioning")'''),
        code('''run.runStage("tune")'''),
        code('''result = run.runStage("simulation")
pd.DataFrame(result["timing"]).T.join(pd.DataFrame(result["nodes"]).T)'''),
        md("The resolved parameter file of the lesion run: model.par with this run's overrides in place, then the run paths."),
        code('''print((run.jobs["lesion"] / "run.par").read_text()[-3000:])'''),

        md("""## 1.3 Physiology checks and measurement windows
Catheter-facing tissue nodes, last three beats: one activation per beat, repolarisation inside the response
window, quiet lesion core, planar wave in the intended direction, stable latency, amplitude and APD90.
Engineering criteria, not validated biological limits. The slab surface CV is reported, not gated: with bilayer
fibres it is below the 0.90 m/s along-fibre target (about 0.75 m/s, FM-33). Each placement's window comes from
the intact activation under it, before any electrogram is read (DL-2026-10-28-08, FM-28)."""),
        code('''physiology = run.runStage("physiology")
display(pd.DataFrame({c: v["criteria"] for c, v in physiology["conditions"].items()}))
{c: round(v["lastBeatSurfaceSpeedMPerS"], 3) for c, v in physiology["conditions"].items()}'''),
        code('''pd.read_csv(run.shared / "physiology" / "physiologyChecks.csv")'''),
        code('''window = run.runStage("window")
{k: v for k, v in window.items() if k != "windowsMs"}'''),

        md("""## 1.4 Lead fields and electrograms
Each electrode is the set of bath mesh nodes inside its 1 mm cylinder (FM-14), treated as a conductor: one
shared potential, no net current while recording (FM-34). Every distinct electrode gets an openCARP Laplace run
(`config/leadfield.par`): its nodes held at 1 mV, the corner grounded, operator sigma_i + sigma_e in tissue and
g_bath in the bath. Dividing the potential u by the current the electrode draws, I = u^T K u, gives its lead field
z = u / I. The unipolar signal is phi(t) = -z^T S_i Vm(t), the extracellular potential with the sigma_i source
recovered from the monodomain Vm (FM-23). The Laplace runs go `--np` at a time, one openCARP process each;
lead fields are cached per conductor and reused by later experiments."""),
        code('''run.runStage("leadFields")'''),
        code('''run.runStage("electrograms")
egm = np.load(run.out / "electrograms" / "intact.npz")
print({k: egm[k].shape for k in egm.files})'''),

        md("""## 1.5 Measurements, export and Wang's change points
Acquisition chain (FM-09, FM-22): resample to 2 kHz, Gaussian noise per electrode, 30-500 Hz band-pass for
bipoles and omnipoles. Peak frequency: two co-primary definitions (FM-10, FM-27); bipolar/omnipolar PP within
+-30 ms of the peak-frequency time (FM-11). Amplitudes stay in model units; Wang's 1.44 mV is shown beside them,
never applied (FM-25). Wang's Rmd runs unchanged except six one-slab functions (R/wang_replication.R, FM-12)."""),
        code('''measures = run.runStage("measurements")
display(pd.DataFrame(measures["amplitudes"]["modelFarViable"]))
measures["checks"]'''),
        code('''run.runStage("export")
changes = run.runStage("changePoints")
print("eligible for interpretation:", changes["eligibleForInterpretation"], changes["prerequisites"])
pd.read_csv(run.out / "changePoints" / "SuppResults5_change_points_all.csv")'''),

        md("""## 1.6 Ground truth: the contributing tissue
For every recording at the ground-truth placements, each tissue element's signed share of the signal energy,
from the same lead fields as the electrograms. Shares sum to 1; cancellation is kept. Classes: near/far
(4 mm convention, FM-30) x endocardial/epicardial half (FM-15); transmural and lateral profiles; 3D fields
(`groundTruth/fields_<condition>.npz`, `groundTruth/shares_<condition>_gap<g>mm.vtk` for ParaView)."""),
        code('''checks = run.runStage("groundTruth")
summary = pd.read_csv(run.out / "groundTruth" / "summary.csv")
classes = [c for c in summary.columns if c.startswith("mass_")]
display(summary.groupby(["condition", "kind", "gapMm"])[classes].median())
checks'''),

        md("""## 1.7 Report, acceptance and the base-model figures
`acceptance.json` lists every check; **modelAccepted** requires tuning, the ground-truth share sum, physiology
and the profile checks to pass. Change points are interpreted only if `eligibleForInterpretation` is true."""),
        code('''report = run.runStage("report")
run.runStage("compact")                 # the baseline is never compacted (it is shared by later experiments)
sweep.runBaseline()                     # records the baseline as complete (every stage is reused)
display(Markdown((run.out / "report.md").read_text()))'''),
        code('''for path in report["figures"]:
    print(Path(path).name)
    display(Image(path))'''),

        md("""# Part 2. Parameter experiments

Each section runs one experiment's variants, then shows them beside the base model: a table of headline
outcomes and a figure (`<OUT>/figures/experiment_<variable>.png`). Headline outcomes (definitions in
`efov.sweep.variantSummary`): acceptance and eligibility flags; Wang's voltage and peak-frequency change points
on the clean data and their median over the noisy realisations (posterior mean, mm from the scar boundary);
median far-field PP (model units); and, from the
ground truth, the median near-field and epicardial |c|-mass fractions. Every variant has its full run folder
(`<OUT>/variants/<name>/`: report.md, figures, tables) for detail.

Variants that change only electrodes, noise or analysis reuse the base-model simulation; lead fields are shared
by every variant with the same conductor (FM-23); variants with their own simulation are compacted after their
report (FM-20). Change points in a variant whose `eligibleForInterpretation` is false are not interpreted.""")]

    for variable in sweep.get("oneAtATime", []) + list(sweep.get("named", {})):
        cells += [md(experimentText(variable, model, sweep)), code(EXPERIMENT.format(variable=variable))]

    cells += [
        md("""# Part 3. Summary

## 3.1 Pre-registered primary test: change-point shift per doubling of noise (FM-25)
For each of Wang's change points: the median over noisy realisations at each noise level, and the shift from
the previous (half) level. A change point that moves with the noise floor measures detectability, not
sensing volume. Criterion and interpretation as pre-registered in FM-25."""),
        code('''doubling = noiseDoublingTable(sweep.out, sweep.modelConfigPath)
doubling if doubling is not None else "No noise variants have change points yet"'''),
        md("""## 3.2 Every variant
Status of every variant (failed ones with their error) and the long table of headline outcomes, saved as
`<OUT>/sweepSummary.csv`."""),
        code('''status = pd.DataFrame({k: dict(status=v["status"], overrides=json.dumps(v.get("overrides")), error=v.get("error"))
                       for k, v in sweep.status.items()}).T
display(status)
table = sweep.summary()
table.to_csv(sweep.out / "sweepSummary.csv", index=False)
table'''),
        md("""## 3.3 Regional comparison (optional)
Pool the inferior baseline and the roof variant for Wang's regional analysis:"""),
        code('''# import subprocess
# subprocess.run([sys.executable, "-m", "efov", "wang-combine", str(sweep.out / "variants" / "baseline"),
#                 str(sweep.out / "variants" / "roof"), "--out", str(sweep.out / "wang_regions")], cwd=REPOSITORY)'''),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
    nbformat.write(nb, HERE / "run_all.ipynb")
    return HERE / "run_all.ipynb"


if __name__ == "__main__":
    print("written", build())
