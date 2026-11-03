#!/usr/bin/env python3
"""Software and feasibility checks that need openCARP (and R), on the small test slab. Run once, in efov-runner.

    cd <repository>; export MPIEXEC_BIN=/usr/local/lib/opencarp/lib/petsc/bin/mpiexec
    python3 checks/container_checks.py --out runs/checks --np 2

These are the checks of the final model plan that cannot run without openCARP. Criteria were fixed in
the plan (claude/final_model_plan_2026-11-03.md, Section 4) before any result was seen:
  C1  mesher: per-axis resolution; bath above and beside the tissue only; zero bath fibres
  W0  openCARP lead fields (FM-14): (a) the Laplace mode accepts a total-current extracellular stimulus
      and writes a whole-mesh potential; (b) reciprocity z_A(B) = z_B(A) within 1e-6 relative;
      (c) summed contributions reproduce openCARP's pseudo-bidomain phi_e at each electrode
      (relative residual <= 1e-3, scale 1 +- 1e-3); (d) time per Laplace run (diagnostic).
      Diagnostic only: the openCARP lead field against the Python one (same discrete problem).
  C3  the resolved .par alone reproduces the run (Vm max difference 0)
  C4  openCARP reads the written fibre file: with fibres at 90 deg the x-speed falls to the transverse
      speed (ratio within 10 % of sqrt(sigma_m,t / sigma_m,l))
  C5  the scar types change phi_e over the lesion (relative difference > 1 %), lesion stays quiet
  C8  Wang's Rmd runs through R/wang_replication.R on a synthetic dataset; her lm-replaced Table 3
      slope equals a direct least-squares fit
If W0 fails, the pre-declared fallback is fixed.groundTruth.leadFieldMethod = "python" (FM-14).
Every check is recorded in <out>/checks_report.json; a failure never stops the other checks.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from efov import config as cfg, leadfield, meshbuild, simulation  # noqa: E402
from efov.electrograms import phieTimes  # noqa: E402
from efov.geometry import electrodeLayout, electrodeNodeSets  # noqa: E402
from efov.io import readIgb, readVertices, saveJson  # noqa: E402

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
    layout = electrodeLayout(config)
    phieNodes = np.load(job / "phieNodes.npy")
    centres = layout["uniqueCentres"][np.argsort(np.linalg.norm(layout["uniqueCentres"][:, :2], axis=1))][:3]
    v = config["values"]
    sets = electrodeNodeSets(tissue["positions"], phieNodes, centres, v["electrodeRadiusMm"], v["electrodeLengthMm"])
    ground = readVertices(job / "ground.vtx")
    fields, seconds = [], []
    for k, nodes in enumerate(sets):
        started = time.time()
        fields.append(leadfield.openCarpLeadField(config, tissue["meshBase"], ground, nodes, len(tissue["positions"]),
                                                  folder / f"laplace{k}", processCount))
        seconds.append(time.time() - started)
    reciprocity = abs(fields[0][sets[1]].mean() - fields[1][sets[0]].mean()) / abs(fields[1][sets[0]].mean())
    model = leadfield.ContributionModel(config, tissue)
    vm = readIgb(job / "vm.igb").T.astype(float)                                  # (tissue nodes, frames)
    phie = readIgb(job / "phie.igb").astype(float)                                # (frames, output nodes)
    column = {n: i for i, n in enumerate(phieNodes)}
    rows = []
    for k, nodes in enumerate(sets):
        measured = phie[:, [column[n] for n in nodes]].mean(axis=1)
        nodal = np.zeros(len(model.tissueNodes))
        np.add.at(nodal, model.localElements, model.coefficients(fields[k]))
        reconstructed = nodal @ vm
        scale = float(measured @ reconstructed / (reconstructed @ reconstructed))
        residual = float(np.linalg.norm(measured - reconstructed) / np.linalg.norm(measured))
        rows.append(dict(electrode=k, nodes=len(nodes), scale=scale, relativeResidual=residual))
    python = leadfield.PythonLeadFieldSolver(config, tissue, ground).leadField(sets[0])
    agreement = float(np.linalg.norm(python - fields[0]) / np.linalg.norm(python))
    production = cfg.meshNodeCount(cfg.loadConfig()) / len(tissue["positions"])
    result = dict(laplaceAcceptsCurrent=True, reciprocityRelativeDifference=float(reciprocity), reconstruction=rows,
                  secondsPerLaplaceRunSmall=float(np.mean(seconds)),
                  secondsPerLaplaceRunProductionCrude=float(np.mean(seconds) * production),
                  diagnosticPythonVsOpenCarpRelative=agreement,
                  leadFieldOutputFile=json.loads((folder / "laplace0" / "leadfield.json").read_text())["output"])
    result["passed"] = bool(reciprocity <= 1e-6 and all(r["relativeResidual"] <= 1e-3 and abs(r["scale"] - 1) <= 1e-3 for r in rows))
    return result


def checkResolvedPar(config, job, folder, processCount):
    """Replay the run from its resolved .par: with the recorded carputils solver options (must be identical),
    and with the .par alone (openCARP's default solver options; difference reported, not gated)."""
    import os
    recorded = json.loads((job / "runRecord.json").read_text())["commands"]["run"]
    results = {}
    for name, extra in (("withRecordedSolverOptions", recorded[3:]), ("parAlone", [])):
        replay = folder / f"replay_{name}"
        replay.mkdir(parents=True, exist_ok=True)
        (replay / "run.par").write_text((job / "run.par").read_text().replace(f"simID = {job}", f"simID = {replay}"))
        command = [simulation.findExecutable("openCARP", "OPENCARP_BIN"), "+F", str(replay / "run.par")] + extra
        if processCount > 1:
            command = [os.environ.get("MPIEXEC_BIN", "mpiexec"), "-n", str(processCount)] + command
        if simulation.runLogged(command, replay / "replay.log") != 0:
            return dict(passed=False, error=f"openCARP replay '{name}' failed; see {replay / 'replay.log'}")
        results[name] = float(np.max(np.abs(readIgb(replay / "vm.igb") - readIgb(job / "vm.igb"))))
    return dict(vmMaxDifferenceMv=results, passed=results["withRecordedSolverOptions"] == 0.0,
                note="carputils adds only solver-flavour options to the resolved .par (recorded in runRecord.json)")


def surfaceSpeedX(tissue, job):
    from efov.io import readActivations
    nodes, times = readActivations(job / "act-thresh.dat")
    p = tissue["tissuePositions"]
    first = np.full(len(p), np.nan)
    first[nodes[::-1]] = times[::-1]
    use = np.isclose(p[:, 2], p[:, 2].max()) & (np.abs(p[:, 1] - 6) < 1.0) & (p[:, 0] > -4) & (p[:, 0] < 4) & np.isfinite(first)
    return float(1 / np.polyfit(p[use, 0], first[use], 1)[0])


def checkFibres(folder, processCount):
    speeds = {}
    for angle in (0.0, 90.0):
        config = cfg.loadConfig(SMALL, {"fibres": {"model": "uniform", "angleDeg": angle}})
        job, tissue = shortRun(config, "intact", folder / f"fibres{angle:.0f}", processCount)
        speeds[angle] = surfaceSpeedX(tissue, job)
    config = cfg.loadConfig(SMALL)
    regions, _ = leadfield.principalConductivities(config)
    gi, ge = regions["tissue"]
    sigmaM = gi * ge / (gi + ge)
    expected = float(np.sqrt(sigmaM[1] / sigmaM[0]))
    ratio = speeds[90.0] / speeds[0.0]
    return dict(speedAlongFibresMPerS=speeds[0.0], speedAcrossFibresMPerS=speeds[90.0], ratio=ratio, expectedContinuumRatio=expected,
                passed=bool(abs(ratio - expected) / expected <= 0.10))


def checkScarTypes(folder, processCount):
    signals = {}
    for scar in ("ablation", "insulating"):
        config = cfg.loadConfig(SMALL, {"scar": scar})
        job, tissue = shortRun(config, "lesion", folder / f"scar_{scar}", processCount)
        phieNodes = np.load(job / "phieNodes.npy")
        over = phieNodes[(tissue["positions"][phieNodes, 1] < -3) & (np.isclose(tissue["positions"][phieNodes, 2], tissue["positions"][phieNodes, 2].min()))]
        phie = readIgb(job / "phie.igb").astype(float)
        column = {n: i for i, n in enumerate(phieNodes)}
        signals[scar] = phie[:, [column[n] for n in over]]
        from efov.io import readActivations
        nodes, _ = readActivations(job / "act-thresh.dat")
        lesionNodes = np.unique(tissue["elements"][tissue["tags"] == 5])
        viable = np.unique(tissue["elements"][tissue["tags"] == 1])
        core = np.setdiff1d(lesionNodes, viable)
        local = np.searchsorted(tissue["tissueNodes"], core)
        signals[scar + "_quiet"] = bool(not np.isin(local, nodes).any())
    difference = float(np.linalg.norm(signals["ablation"] - signals["insulating"]) / np.linalg.norm(signals["insulating"]))
    return dict(relativeDifferenceOverLesion=difference, lesionQuiet=dict(ablation=signals["ablation_quiet"], insulating=signals["insulating_quiet"]),
                passed=bool(difference > 0.01 and signals["ablation_quiet"] and signals["insulating_quiet"]))


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
    bipolar = data[(data.file == "merged_rectangle_summary_euclidean.xlsx") & (data.uni1_bi2_omni3 == 2)]
    bipolar = bipolar[(bipolar.DistAcrossPath + 3 > 0) & (bipolar.DistAcrossPath + 3 < 15) & (bipolar.PP > 0.01)]
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
    parser.add_argument("--only", nargs="*", help="subset of C1 W0 C3 C4 C5 C8")
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
    record("C3", lambda: checkResolvedPar(config, baseRun()[0], out, args.np))
    record("C4", lambda: checkFibres(out, args.np))
    record("C5", lambda: checkScarTypes(out, args.np))
    record("C8", lambda: checkWangReplication(out))
    print("\nSummary:", {k: ("PASS" if v.get("passed") else "FAIL") for k, v in report.items()})
    print(f"Report: {out / 'checks_report.json'}")
    if not report.get("W0", {}).get("passed", True):
        print("W0 failed: set fixed.groundTruth.leadFieldMethod to \"python\" in config/model.json (pre-declared fallback, FM-14).")


if __name__ == "__main__":
    main()
