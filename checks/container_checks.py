#!/usr/bin/env python3
"""Software and feasibility checks that need openCARP (and R), on the small test slab. Run once, in efov-runner.

    cd <repository>; export MPIEXEC_BIN=/usr/local/lib/opencarp/lib/petsc/bin/mpiexec
    python3 checks/container_checks.py --out runs/checks --np 2

These are the checks of the final model plan that cannot run without openCARP. Criteria were fixed in
the plan (claude/final_model_plan_2026-11-03.md, Section 4) before any result was seen. By decision
FM-26 no check compares openCARP outputs with other openCARP outputs or with our own solver.
  C1  mesher: per-axis resolution; bath above and beside the tissue only; zero bath fibres
  W0  lead fields (FM-14, FM-23, FM-34): openCARP's Laplace run holds the electrode at 1 mV and the ground
      at 0 mV, the potential is non-trivial and within [0, 1] mV (maximum principle), the electrode draws a
      positive current; time and peak memory per Laplace run and a crude production estimate.
      Diagnostics: two electrodes' transfer potentials (near-symmetric; exact only for conductors present
      in both solves), and whether this build applies extracellular current stimuli (19.0.0 does not)
  C4  openCARP reads the written fibre file: with fibres at 90 deg the x-speed falls to the transverse
      speed (ratio within 10 % of sqrt(sigma_m,t / sigma_m,l)); C4p repeats it at the production mesh
      spacing and tuned beta (the 0.5 mm test mesh under-resolves the slow transverse wave)
  C5  lesion strip (FM-24): viable tissue activates every node; neither the lesion strip nor the
      isolated tissue beyond it activates
  C8  Wang's Rmd runs through R/wang_replication.R on a synthetic dataset; her lm-replaced Table 3
      slope equals a direct least-squares fit
openCARP 19.0.0 ignores extracellular current stimuli, so the electrode is driven by a potential and
normalised by its current (config/leadfield.par, FM-34).
Every check is recorded in <out>/checks_report.json; a failure never stops the other checks.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from efov import config as cfg, leadfield, simulation  # noqa: E402
from efov.geometry import electrodeLayout  # noqa: E402
from efov.io import writeVertices, readVertices, saveJson  # noqa: E402

SMALL = REPOSITORY / "config" / "test_small.json"
DURATION_MS = 30.0


def shortRun(config, condition, folder, processCount):
    state = simulation.conditionCell(config, folder / "conditioning")
    simulation.runCondition(config, condition, state, folder / "simulations", processCount, durationMs=DURATION_MS)
    job = folder / "simulations" / condition
    return job, simulation.tissueModel(job)


def checkMesh(config, job, tissue):
    p, tags, elements = tissue["positions"], tissue["tags"], tissue["elements"]
    spacing = cfg.spacingMm(config)
    steps = [np.round(np.diff(np.unique(np.round(p[:, axis], 6))), 6) for axis in range(3)]
    wall = cfg.slabSizeMm(config)[2]
    centres = p[elements].mean(axis=1)
    result = dict(spacingFoundMm=[float(np.median(s)) for s in steps], spacingExpectedMm=spacing.tolist(),
                  bathBelowTissue=int(((tags == 0) & (centres[:, 2] < -wall / 2)).sum()),
                  bathAboveTissue=int(((tags == 0) & (centres[:, 2] > wall / 2)).sum()),
                  bathFibresZero=bool(np.all(tissue["fibres"][tags == 0] == 0)))
    result["passed"] = bool(np.allclose(result["spacingFoundMm"], spacing) and result["bathBelowTissue"] == 0
                            and result["bathAboveTissue"] > 0 and result["bathFibresZero"])
    return result


def checkLeadFields(config, job, tissue, folder, processCount):
    """W0: the openCARP Laplace lead field of two electrodes near the slab centre (FM-34), and its cost."""
    import resource
    from efov.electrograms import electrodeSets
    layout, sets = electrodeSets(config, tissue)
    order = np.argsort(np.linalg.norm(layout["uniqueCentres"][:, :2], axis=1))
    ground = readVertices(job / "ground.vtx")
    stiffness, potential = leadfield.bulkStiffness(config, tissue), leadfield.electrodePotentialMv(config)
    runs = []
    for k in order[:2]:
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        started = time.time()
        u = leadfield.laplacePotential(config, tissue["meshBase"], ground, sets[k], len(tissue["positions"]), folder / f"laplace{k}")
        seconds = time.time() - started
        z, current = leadfield.leadFieldFromPotential(u, stiffness, sets[k], ground, potential)   # raises if a property fails
        runs.append(dict(electrode=int(k), nodes=int(len(sets[k])), seconds=seconds, currentUA=current,
                         peakMemoryGB=max(before, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1e6, z=z))
    a, b = runs
    transfer = (a["z"][sets[b["electrode"]]].mean(), b["z"][sets[a["electrode"]]].mean())
    current = probeCurrentStimulus(config, tissue, ground, sets[order[0]], folder / "currentProbe")
    production = cfg.loadConfig()
    electrodes = len(electrodeLayout(production)["uniqueCentres"])
    scale = cfg.meshNodeCount(production) / len(tissue["positions"])
    seconds = float(np.mean([r["seconds"] for r in runs]))
    return dict(passed=True, nodesSmall=int(len(tissue["positions"])),
                runs=[{k: v for k, v in r.items() if k != "z"} for r in runs],
                leadFieldAtElectrodeMvPerUA=[potential / r["currentUA"] for r in runs],
                transferMvPerUA=[float(t) for t in transfer], transferRelativeDifference=float(abs(transfer[0] - transfer[1]) / abs(transfer[0])),
                currentStimulusApplied=current, productionElectrodesPerCondition=electrodes,
                productionCoreHoursPerConditionCrude=float(seconds * scale * electrodes / 3600),
                note="Crude linear scaling by node count (one openCARP process per Laplace run; mesh reading and the AMG "
                     "solve grow at least linearly). Peak memory is the largest child process so far.")


def probeCurrentStimulus(config, tissue, ground, nodes, folder):
    """Diagnostic: does this openCARP build apply an extracellular current stimulus in a Laplace run (19.0.0: no)?"""
    from efov.parfile import writeResolved
    from efov.simulation import findExecutable, runLogged
    folder.mkdir(parents=True, exist_ok=True)
    writeVertices(folder / "electrode.vtx", nodes); writeVertices(folder / "ground.vtx", ground)
    lines = {"meshname": str(tissue["meshBase"]), "simID": str(folder), "stim[0].elec.vtx_file": str(folder / "ground"),
             "stim[1].elec.vtx_file": str(folder / "electrode"), **leadfield.laplaceRegionLines(config),
             "stim[1].crct.total_current": 1}
    par = writeResolved(config["leadFieldPar"], {"stim[1].crct.type": 1}, lines, folder / "probe.par",
                        header="# W0 diagnostic: 1 uA extracellular current stimulus instead of the 1 mV potential")
    if runLogged([findExecutable("openCARP", "OPENCARP_BIN"), "+F", str(par)], folder / "probe.log") != 0:
        return None
    u, _ = leadfield.laplaceOutput(folder, len(tissue["positions"]))
    return bool(np.abs(np.delete(u, nodes)).max() > 0)


def surfaceSpeedX(tissue, job):
    from efov.io import readActivations
    nodes, times = readActivations(job / "act-thresh.dat")
    p = tissue["tissuePositions"]
    first = np.full(len(p), np.nan)
    first[nodes[::-1]] = times[::-1]
    use = np.isclose(p[:, 2], p[:, 2].max()) & (np.abs(p[:, 1] - 6) < 1.0) & (p[:, 0] > -4) & (p[:, 0] < 4) & np.isfinite(first)
    return float(1 / np.polyfit(p[use, 0], first[use], 1)[0])


def atProductionSpacing(config):
    """The small slab at the production mesh spacing, with the beta tuned for that mesh (C4 at production spacing)."""
    production = cfg.loadConfig()
    beta = cfg.tunedBetaFor(production)
    if beta is None:
        raise RuntimeError("beta is not tuned for the production mesh: run `python -m efov tune --write-par` first")
    config = cfg.withParOverrides(config, {"imp_region[0].cellSurfVolRatio": beta})   # reloads the file, so first
    config["fixed"] = json.loads(json.dumps(config["fixed"]))
    for key in ("spacingInPlaneMm", "spacingTransmuralMm"):
        config["fixed"]["tissue"][key] = production["fixed"]["tissue"][key]
    return config


def checkFibres(folder, processCount, productionSpacing=False):
    """C4 (criterion fixed in the plan: ratio within 10 % of the continuum value), on the small slab's own
    mesh; with productionSpacing the same measurement at the production mesh (reported beside it)."""
    speeds = {}
    for angle in (0.0, 90.0):
        config = cfg.loadConfig(SMALL, {"fibres": {"model": "uniform", "angleDeg": angle}})
        if productionSpacing:
            config = atProductionSpacing(config)
        job, tissue = shortRun(config, "intact", folder / f"fibres{angle:.0f}{'_production' if productionSpacing else ''}", processCount)
        speeds[angle] = surfaceSpeedX(tissue, job)
    config = cfg.loadConfig(SMALL)
    regions, _ = leadfield.principalConductivities(config)
    gi, ge = regions["tissue"]
    sigmaM = gi * ge / (gi + ge)
    expected = float(np.sqrt(sigmaM[1] / sigmaM[0]))
    ratio = speeds[90.0] / speeds[0.0]
    return dict(spacingMm=cfg.spacingMm(atProductionSpacing(cfg.loadConfig(SMALL)) if productionSpacing else config).tolist(),
                speedAlongFibresMPerS=speeds[0.0], speedAcrossFibresMPerS=speeds[90.0], ratio=ratio, expectedContinuumRatio=expected,
                relativeDifference=float((ratio - expected) / expected), passed=bool(abs(ratio - expected) / expected <= 0.10))


def checkLesionStrip(config, job, tissue):
    """Viable tissue activates; the lesion strip and the isolated tissue beyond it never do (FM-24)."""
    from efov.io import readActivations
    nodes, _ = readActivations(job / "act-thresh.dat")
    activated = np.zeros(len(tissue["tissueNodes"]), bool)
    activated[np.unique(nodes)] = True
    y = tissue["tissuePositions"][:, 1]
    boundary, width = config["fixed"]["tissue"]["boundaryYmm"], config["fixed"]["tissue"]["lesionWidthMm"]
    spacing = cfg.spacingMm(config)[0]
    viable, lesion, isolated = y > boundary + spacing, (y < boundary - spacing) & (y > boundary - width + spacing), y < boundary - width - spacing
    result = dict(viableActivatedFraction=float(activated[viable].mean()), lesionActivatedNodes=int(activated[lesion].sum()),
                  isolatedActivatedNodes=int(activated[isolated].sum()), isolatedNodes=int(isolated.sum()),
                  lesionElements=int((tissue["tags"] == 5).sum()))
    result["passed"] = bool(result["viableActivatedFraction"] == 1.0 and result["lesionActivatedNodes"] == 0
                            and result["isolatedActivatedNodes"] == 0 and result["isolatedNodes"] > 0)
    return result


def checkWangReplication(folder):
    import pandas as pd
    rscript = shutil.which("Rscript")
    if not rscript:
        return dict(passed=False, error="Rscript not found")
    rng = np.random.default_rng(3)
    rows = []
    for condition, file in (("lesion", "merged_rectangle_summary_euclidean.xlsx"), ("intact", "merged_rectangle_summary_euclidean_pre.xlsx")):
        for kind, code, b1 in (("uni", 1, -0.1), ("bi", 2, -0.25), ("omni", 3, -0.3)):
            d = np.linspace(-9.5, 14.5, 400)
            lesion = condition == "lesion"
            pp = np.exp(np.log(1.5) + (b1 * np.clip(d, 0, 10) if lesion else 0) + rng.normal(0, 0.2, d.size))
            pf = np.where(kind == "uni", np.nan, 350 * np.exp((-0.04 * np.clip(d, 0, 12) if lesion else 0)) + rng.normal(0, 5, d.size))
            rows.append(pd.DataFrame(dict(patientID=1, inf0_roof1=0, uni1_bi2_omni3=code, DistAcrossPath=d - 3, PP=pp, PFreq=pf,
                                          condition=condition, file=file)))
    data = pd.concat(rows)
    dataset = folder / "wang_dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    for file, part in data.groupby("file"):
        part.drop(columns=["condition", "file"]).to_excel(dataset / file, index=False)
    out = folder / "wang_output"
    code = simulation.runLogged([rscript, str(REPOSITORY / "R" / "wang_replication.R"), str(dataset), str(out)], folder / "wang.log")
    status = pd.read_csv(out / "chunkStatus.csv") if (out / "chunkStatus.csv").exists() else None
    if code != 0 or status is None:
        return dict(passed=False, error=f"R exit {code}; see {folder / 'wang.log'}")
    table3 = pd.read_csv(out / "tables" / "Table3_mixed_models_REML.csv")
    # Wang's 'volt' dataset: 0 < Dist < 15 mm, PP > 0.01 mV, then PP above its 99.9th percentile (over all
    # electrode kinds of the file, R quantile type 7 = numpy 'linear') removed; the bipolar slope is fitted on that.
    volt = data[data.file == "merged_rectangle_summary_euclidean.xlsx"]
    volt = volt[(volt.DistAcrossPath + 3 > 0) & (volt.DistAcrossPath + 3 < 15) & (volt.PP > 0.01)]
    volt = volt[volt.PP <= np.quantile(volt.PP, 0.999)]
    bipolar = volt[volt.uni1_bi2_omni3 == 2]
    direct = float(np.polyfit(bipolar.DistAcrossPath + 3, np.log(bipolar.PP), 1)[0])
    row = table3[(table3.Electrode == "Bipolar") & (table3.Metric == "Voltage")]
    fitted = float(row.b1.iloc[0]) if len(row) and row.model.iloc[0] == "Exponential" else float("nan")
    return dict(chunks=status.status.value_counts().to_dict(), failedChunks=status[status.status == "failed"].chunk.tolist(),
                table5Written=(out / "tables" / "Table5_bayesian_posteriors.csv").exists(),
                bipolarSlopeWang=fitted, bipolarSlopeDirect=direct,
                passed=bool((out / "tables" / "Table5_bayesian_posteriors.csv").exists() and abs(fitted - direct) < 1e-6))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPOSITORY / "runs" / "checks"))
    parser.add_argument("--np", type=int, default=2)
    parser.add_argument("--only", nargs="*", help="subset of C1 W0 C4 C4p C5 C8")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report, config = {}, cfg.loadConfig(SMALL)

    def record(name, function):
        if args.only and name not in args.only:
            return
        print(f"--- {name}", flush=True)
        try:
            report[name] = function()
        except Exception as error:
            report[name] = dict(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(f"{name}: {'PASS' if report[name].get('passed') else 'FAIL'}  " + json.dumps({k: v for k, v in report[name].items()
              if k not in ('traceback',)}, default=str)[:600], flush=True)
        saveJson(out / "checks_report.json", report)

    base = {}

    def baseRun():
        if "job" not in base:
            base["job"], base["tissue"] = shortRun(config, "lesion", out / "base", args.np)
        return base["job"], base["tissue"]

    record("C1", lambda: checkMesh(config, *baseRun()))
    record("W0", lambda: checkLeadFields(config, *baseRun(), out / "leadfields", args.np))
    record("C4", lambda: checkFibres(out, args.np))
    record("C4p", lambda: checkFibres(out, args.np, productionSpacing=True))
    record("C5", lambda: checkLesionStrip(config, *baseRun()))
    record("C8", lambda: checkWangReplication(out))
    print("\nSummary:", {k: ("PASS" if v.get("passed") else "FAIL") for k, v in report.items()})
    print(f"Report: {out / 'checks_report.json'}")
    if not report.get("W0", {}).get("openCarpLaplace", {}).get("appliesCurrent", True):
        print("W0 diagnostic: openCARP's Laplace mode did not apply the current stimulus; the lead fields use the python route (FM-14, FM-31).")


if __name__ == "__main__":
    main()
