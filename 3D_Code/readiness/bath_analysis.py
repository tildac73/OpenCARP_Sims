"""Step 6: electrograms and ground truth with a blood bath, from Vm of completed ladder runs.

Why post-processing is valid. In openCARP's pseudo-bidomain mode Vm is the monodomain solution:
the bath changes phi_e, not Vm. So phi_e with a bath is one elliptic solve per output frame on the
Vm openCARP already produced. The bath_check stage verifies, on this build, that openCARP's own
pseudo-bidomain phi_e equals fem_extracellular's solve of the same problem; this stage then uses
the same solver on the ladder runs' Vm.

Per model (readiness_kit.json "bathAnalysis"):
  * lattice = the source run's tissue grid (same nodes) extended by a blood layer of thickness B
    on the catheter side, ground on the far bath face, outer faces insulated;
  * scar "insulating": lesion elements keep openCARP's near-zero conductivities (as the pilot);
    scar "conducting": lesion elements are passive with the healthy extracellular tensor (ASSUMPTION;
    no scar conductivity has been sourced);
  * electrograms: phi_e at every electrode sample from unfiltered Vm over the analysis frames,
    averaged per electrode, written as egm.npz into a derived run folder, then analysed by the
    unchanged pipeline (--process-only): PP, bipole/omnipolar selection, change points;
  * ground truth: reciprocity lead field per ground-truth recording (fem_extracellular), giving
    bath-aware cylinder, sphere and depth extents (contribution_metrics).
The derived folders' results/contributions.csv still use the unbounded lead field (the pipeline's
ground truth); the bath ground truth is bathGroundTruth.csv. The summary says which is which.

Limitations kept explicit: bath only on the catheter side (no epicardial medium: no sourced
conductivity); lateral bath extent = tissue extent; electrode bodies not modelled (point samples,
as in the pipeline); unipolar reference = the grounded far face.
"""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import common
import boundary_setup
import boundary_analysis as analysis
import contribution_metrics as metrics
import fem_extracellular as fem
import fibres as fibreTools
import ground_truth
import igb_io
import ladder
from boundary_geometry import signedDistance

WEIGHTS = [f"weightE{i}" for i in range(16)]


# ----------------------------------------------------------------- model
def latticeModel(runFolder, config, condition, bathMm, scar):
    """Tissue grid of the source run + blood layer; returns positions, elements, masks, tensors, ground, node map."""
    spacing = config["meshSpacingMm"]
    size = np.asarray(config["slabSizeMm"], float)
    lower = -size / 2
    upper = size / 2 + np.array([0.0, 0.0, bathMm])
    positions, elements = fem.structuredBox(lower, upper, spacing)
    runNodes = boundary_setup.readNodePositions(Path(runFolder) / "mesh" / condition)
    shape = [round((u - l) / spacing) + 1 for l, u in zip(lower, upper)]
    index = np.round((runNodes - lower) / spacing).astype(int)
    if np.any(index < 0) or np.any(index >= shape) or not np.allclose(index * spacing + lower, runNodes, atol=1e-6):
        raise ValueError("Source run mesh is not the expected regular grid")
    nodeMap = np.ravel_multi_index(index.T, shape)
    centres = positions[elements].mean(axis=1)
    tissue = centres[:, 2] < size[2] / 2
    lesion = tissue & (signedDistance(centres, config["boundary"]) > 0) if condition == "lesion" else np.zeros(len(elements), bool)
    parameters = ground_truth.readParameters(Path(runFolder) / condition / "parameters.par")
    healthyI = [float(parameters[f"gregion[0].g_i{a}"]) for a in "ltn"]
    healthyE = [float(parameters[f"gregion[0].g_e{a}"]) for a in "ltn"]
    fibreVectors = (np.tile([1.0, 0, 0], (len(elements), 1)) if config.get("fibres") is None
                    else fibreTools.fibreVectors(centres, size, config["fibres"]))
    frames = fibreTools.localFrames(fibreVectors)
    si = fibreTools.tensors(np.tile(healthyI, (len(elements), 1)), frames)
    se = fibreTools.tensors(np.tile(healthyE, (len(elements), 1)), frames)
    active = tissue.copy()
    passive = np.broadcast_to(np.eye(3), (len(elements), 3, 3)).copy()
    if condition == "lesion":
        if scar == "insulating":
            lesionI = [float(parameters[f"gregion[1].g_i{a}"]) for a in "ltn"]
            lesionE = [float(parameters[f"gregion[1].g_e{a}"]) for a in "ltn"]
            si[lesion] = fibreTools.tensors(np.tile(lesionI, (lesion.sum(), 1)), frames[lesion])
            se[lesion] = fibreTools.tensors(np.tile(lesionE, (lesion.sum(), 1)), frames[lesion])
        elif scar == "conducting":
            active &= ~lesion
            passive[lesion] = se[lesion]
        else:
            raise ValueError("scar must be 'insulating' or 'conducting'")
    ground = np.flatnonzero(np.isclose(positions[:, 2], upper[2]))
    return dict(positions=positions, elements=elements, active=active, si=si, se=se, passive=passive,
                ground=ground, nodeMap=nodeMap, tissueTop=size[2] / 2,
                lattice=dict(positions=positions, elements=elements, lower=lower, spacing=spacing, shape=shape))


def solverFor(model, bloodSPerM, method):
    passive = model["passive"].copy()
    bath = ~model["active"] & (model["positions"][model["elements"]].mean(axis=1)[:, 2] > model["tissueTop"])
    passive[bath] = bloodSPerM * np.eye(3)
    return fem.ExtracellularSolver(model["positions"], model["elements"], model["active"], model["si"], model["se"],
                                   bloodSPerM, model["ground"], method=method, passiveConductivity=passive)


# ----------------------------------------------------------------- electrograms
def bathFrameSlice(timesMs, config):
    """Frames the pipeline reads from egm.npz: the measurement window plus TWICE the filter history.

    processCondition filters from (window start - history), and checkFilterHistory re-filters from
    (window start - 2 x history) to show the history is long enough. Frames outside the slice stay
    NaN, so computing only the single-history slice made the check NaN and stopped the analysis.
    """
    rawSlice, measureSlice = analysis.measurementSlices(timesMs, config)
    measurementStart = timesMs[rawSlice][measureSlice][0]
    longer = int(np.searchsorted(timesMs, measurementStart - 2 * config["analysis"]["filterHistoryMs"]))
    return slice(max(0, min(longer, rawSlice.start)), rawSlice.stop)


def bathElectrograms(runFolder, config, condition, model, solver, rawSlice, block=50):
    """(placements, 16, frames) phi_e at electrodes for the rawSlice frames; NaN elsewhere (as electrode_sampling)."""
    source = dict(np.load(Path(runFolder) / condition / "egm.npz"))
    samples = source["uni_sample_xyz"]                                     # (placements, 16, S, 3)
    rows = fem.latticeInterpolator(model["lattice"], samples.reshape(-1, 3))
    frameIndex = np.arange(len(source["t_ms"]))[rawSlice]
    out = np.full(source["uni"].shape, np.nan, dtype=np.float32)
    vmPath = Path(runFolder) / condition / "vm.igb"
    for first in range(0, len(frameIndex), block):
        selected = frameIndex[first:first + block]
        vm = igb_io.read(vmPath, slice(selected[0], selected[-1] + 1)).T.astype(float)
        lattice = np.zeros((len(model["positions"]), vm.shape[1]))
        lattice[model["nodeMap"]] = vm
        phi = rows @ solver.solve(lattice)
        out[..., selected] = phi.reshape(*samples.shape[:3], -1).mean(axis=2)
    return source, out


def writeDerivedRun(sourceFolder, targetFolder, config, signals):
    """Copy of an analysed run whose egm.npz holds bath electrograms (Vm and meshes shared, unchanged)."""
    sourceFolder, targetFolder = Path(sourceFolder), Path(targetFolder)
    targetFolder.mkdir(parents=True, exist_ok=True)
    if not (targetFolder / "mesh").exists():
        (targetFolder / "mesh").symlink_to((sourceFolder / "mesh").resolve(), target_is_directory=True)
    for name in ["electrodes", "conditioning", "sourceSnapshot"]:
        if not (targetFolder / name).exists():
            shutil.copytree(sourceFolder / name, targetFolder / name)
    for path in list(sourceFolder.glob("*.vtx")) + list(sourceFolder.glob("*.par")) + list(sourceFolder.glob("*.log")):
        shutil.copy2(path, targetFolder / path.name)
    common.copyIfExists(sourceFolder / "manifest.json", targetFolder / "manifest.json")
    shutil.copy2(sourceFolder / "config.json", targetFolder / "config.json")
    for condition, (source, uni) in signals.items():
        destination = targetFolder / condition
        destination.mkdir(exist_ok=True)
        for name in ["vm.igb", "phie_recovery.igb"]:
            link = destination / name
            if not link.exists():
                link.symlink_to((sourceFolder / condition / name).resolve())
        for name in ["act-thresh.dat", "meshname.txt", "parameters.par"]:
            common.copyIfExists(sourceFolder / condition / name, destination / name)
        np.savez_compressed(destination / "egm.npz", **{**source, "uni": uni})
        meta = json.loads((sourceFolder / condition / "meta.json").read_text())
        meta.update(electrogramSource="readiness bath_analysis (fem_extracellular, pseudo-bidomain with blood bath)",
                    vmFrom=str(sourceFolder / condition))
        boundary_setup.saveJson(destination / "meta.json", meta)


# ----------------------------------------------------------------- ground truth
def bathGroundTruth(runFolder, derivedFolder, config, condition, model, solver):
    """Cylinder / sphere / depth extents with the bath lead field, for the ground-truth recordings."""
    results = Path(derivedFolder) / "results"
    points = pd.read_csv(results / f"{condition}Points.csv")
    points = points[(points.noiseRepeat == 0) & points.centreYmm.isin(config["analysis"]["groundTruthCentreYmm"])]
    source = dict(np.load(Path(runFolder) / condition / "egm.npz"))
    rawSlice, measureSlice = analysis.measurementSlices(source["t_ms"], config)
    sampleRate = 1000 / np.diff(source["t_ms"][:2])[0]
    centres = model["positions"][model["elements"]].mean(axis=1)
    surface = model["tissueTop"]
    geometry = type("LatticeGeometry", (), {})()
    geometry.nodePositions, geometry.elementNodes = model["positions"], model["elements"]
    rows = []
    for mode in ["uni", "differential"]:
        band = None if mode == "uni" else config["analysis"]["differentialBandHz"]
        voltage = ground_truth.readFilteredVoltage(Path(runFolder) / condition, rawSlice, band, sampleRate,
                                                   config["analysis"]["filterOrder"])[:, measureSlice].astype(float)
        lattice = np.zeros((len(model["positions"]), voltage.shape[1])); lattice[model["nodeMap"]] = voltage
        records = points[points.kind == "uni"] if mode == "uni" else points[points.kind != "uni"]
        for record in records.to_dict("records"):
            weights = np.array([record[w] for w in WEIGHTS])
            samples = source["uni_sample_xyz"][record["placement"]]
            row = None
            for electrode in np.flatnonzero(np.abs(weights) > 1e-12):
                electrodeRows = fem.latticeInterpolator(model["lattice"], samples[electrode])
                part = weights[electrode] * electrodeRows.mean(axis=0)
                row = part if row is None else row + part
            z = solver.leadField(np.asarray(row))[:, 0]
            coefficients = solver.elementCoefficients(z)
            anchor = np.array([record["anchorXmm"], record["anchorYmm"], record["anchorZmm"]])
            out = dict(recordingId=record["recordingId"], condition=condition, kind=record["kind"], label=record["label"],
                       gapMm=record["gapMm"], centreYmm=record["centreYmm"])
            for family, key, step in [("cylinder", np.linalg.norm(centres[:, :2] - anchor[:2], axis=1), 0.5),
                                      ("sphere", np.linalg.norm(centres - anchor, axis=1), 0.5),
                                      ("depth", np.clip(surface - centres[:, 2], 0, None), 0.25)]:
                found, _, _ = metrics.extents(geometry, coefficients, lattice, key, step)
                for level, value in found.items():
                    out[f"{family}{int(level * 100)}Mm"] = value
            rows.append(out)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- stage
def runModel(stageFolder, ladderFolder, spec, settings, processCount):
    folder = Path(stageFolder) / spec["name"]
    donePath = folder / "bathModel.json"
    if donePath.exists():
        return json.loads(donePath.read_text())
    sourceFolder = Path(ladderFolder) / spec["sourceRun"]
    if not (sourceFolder / "results" / "contributions.csv").exists():
        raise RuntimeError(f"Source run {spec['sourceRun']} has not been analysed")
    config = json.loads((sourceFolder / "results" / "analysisConfig.json").read_text())
    signals, solvers, models = {}, {}, {}
    for condition in ["intact", "lesion"]:
        model = latticeModel(sourceFolder, config, condition, spec["bathThicknessMm"], spec["scar"])
        solver = solverFor(model, settings["bloodConductivitySPerM"], settings["solver"])
        times = dict(np.load(sourceFolder / condition / "egm.npz"))["t_ms"]
        signals[condition] = bathElectrograms(sourceFolder, config, condition, model, solver,
                                              bathFrameSlice(times, config))
        models[condition], solvers[condition] = model, solver
    derived = folder / "run"
    writeDerivedRun(sourceFolder, derived, config, signals)
    configPath = folder / "analysisConfig.json"
    boundary_setup.saveJson(configPath, config)
    status = ladder.analyse(derived, configPath, processCount)
    truth = pd.concat([bathGroundTruth(sourceFolder, derived, config, c, models[c], solvers[c]) for c in ["intact", "lesion"]],
                      ignore_index=True)
    truth.to_csv(folder / "bathGroundTruth.csv", index=False)
    record = dict(spec=spec, analysis=status, latticeNodes=int(len(models["intact"]["positions"])),
                  solverMethod=solvers["intact"].method, source=str(sourceFolder))
    common.saveJson(donePath, record)
    return record


def fovComparison(truthA, truthB, label):
    keys = ["condition", "kind", "label", "gapMm", "centreYmm"]
    merged = truthA.merge(truthB, on=keys, suffixes=("A", "B"))
    difference = (merged.cylinder80MmB - merged.cylinder80MmA).abs()
    return dict(label=label, fov80MaxAbsMm=float(difference.max()) if len(difference) else None,
                fov80Changed=int((difference > 0).sum()), fov80Count=int(len(merged)))


def unboundedTruth(sourceFolder):
    """The pipeline's unbounded-medium FOV80 for the source run, in bathGroundTruth's layout."""
    results = Path(sourceFolder) / "results"
    contributions = pd.read_csv(results / "contributions.csv")
    points = pd.concat([pd.read_csv(results / f"{c}Points.csv") for c in ["intact", "lesion"]], ignore_index=True)
    merged = contributions.merge(points[["recordingId", "condition", "kind", "label", "gapMm", "centreYmm"]],
                                 on="recordingId", suffixes=("", "_p"))
    merged = merged.rename(columns={"fov80ReconstructionMm": "cylinder80Mm"})
    return merged[["condition", "kind", "label", "gapMm", "centreYmm", "cylinder80Mm"]]


def gateRow(row):
    """Re-apply the DL-2026-10-28-05 rule with the bath FOV80 in place of the pipeline's unbounded one."""
    t = ladder.TOLERANCES
    basic = (row["cvRelative"] <= t["cvRelative"] and row["ppRelativeMaxAll"] <= t["ppRelative"]
             and (row["fov80MaxAbsMm"] or 0) <= t["fov80Mm"])
    cpStrict = (row["changePointMaxAbsMmAllPrimary"] or 0) <= t["changePointMm"] and row["fitStatusChanges"] == 0
    cpClarified = (row["changePointMaxAbsMmSupported"] or 0) <= t["changePointMm"] and row["supportStatusChanges"] == 0
    comparable = row["changePointComparable"]
    row["passedStrict"] = bool(basic and (cpStrict or not comparable))
    row["passedClarified"] = bool(basic and (cpClarified or not comparable))
    return row


def runStage(stageFolder, ladderFolder, settings, processCount):
    stageFolder = Path(stageFolder)
    records, failures = {}, {}
    for spec in settings["models"]:
        try:
            records[spec["name"]] = runModel(stageFolder, ladderFolder, spec, settings, processCount)
        except (RuntimeError, ValueError, OSError, MemoryError) as error:
            failures[spec["name"]] = f"{type(error).__name__}: {error}"
            print(f"Bath model {spec['name']} failed: {error}", flush=True)
    comparisons = []
    for item in settings["comparisons"]:
        a, b = item["a"], item["b"]
        if b not in records or (a != "unbounded" and a not in records):
            comparisons.append(dict(label=item["label"], skipped="model not run")); continue
        try:
            if a == "unbounded":
                folderA = Path(ladderFolder) / records[b]["spec"]["sourceRun"]
                truthA = unboundedTruth(folderA)
            else:
                folderA = stageFolder / a / "run"
                truthA = pd.read_csv(stageFolder / a / "bathGroundTruth.csv")
            row = ladder.compare(folderA, stageFolder / b / "run", item["label"])
            fov = fovComparison(truthA, pd.read_csv(stageFolder / b / "bathGroundTruth.csv"), item["label"])
            row.update(fov80MaxAbsMm=fov["fov80MaxAbsMm"], fov80Changed=fov["fov80Changed"], fov80Count=fov["fov80Count"])
            row = gateRow(row)
            row["role"] = item["role"]
            row["fovSource"] = "bath lead field" + (" vs pipeline unbounded lead field" if a == "unbounded" else " (both)")
            comparisons.append(row)
        except (KeyError, ValueError, FileNotFoundError) as error:
            comparisons.append(dict(label=item["label"], error=f"{type(error).__name__}: {error}"))
    summary = dict(models=records, failures=failures, comparisons=comparisons,
                   interpretation="role 'convergence': judged against DL-2026-10-28-05; role 'effect': a physical "
                                  "difference to report (bath vs unbounded, conducting vs insulating scar), not a gate.")
    common.saveJson(stageFolder / "bathAnalysisSummary.json", summary)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, dict)} for row in comparisons]).to_csv(
        stageFolder / "bathComparisons.csv", index=False)
    return summary
