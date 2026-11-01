"""Turn carputils slab runs (slab.py) into run folders for the unchanged analysis pipeline.

The pipeline (run_boundary.py --process-only: physiology checks, electrogram measures,
bipole/omnipole selection, change points) reads pilot-style run folders. This module writes one
from an intact and a lesion slab.py run:

  mesh/          tissue-only copy of the slab.py mesh (bath removed; nodes in ascending global
                 order, which is openCARP's vm.igb and LAT order: checked 1 Nov 2026 by
                 phie_i.igb == phie.igb at those nodes)
  electrodes/    the pipeline's electrode layout (boundary_setup.writeElectrodes)
  <condition>/   vm.igb and act-thresh.dat (links to the openCARP output), parameters.par,
                 egm.npz with electrograms from either
                   * openCARP's own phi_e (phie.igb) interpolated at the electrode samples, or
                   * an extracellular solve (contribution_field.BathModel) of the same Vm,
                 computed for the analysis frames only (measurement window + twice the filter
                 history, as bath_analysis.bathFrameSlice); NaN elsewhere.
  config.json    analysis configuration, with the measurement window derived from this run's own
                 activation (ladder.deriveWindow, DL-2026-10-28-08) before any electrogram is used.

phie_recovery.igb is not produced by the production model (no unbounded recovery); the pipeline
only checks that the file exists, so it is a link to phie.igb and meta.json says so.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
for folder in ("3D_Code", "Shared_Pipeline", "3D_Code/readiness"):
    if str(REPOSITORY / folder) not in sys.path:
        sys.path.insert(0, str(REPOSITORY / folder))

import boundary_analysis as analysis  # noqa: E402
import boundary_setup  # noqa: E402
import common  # noqa: E402
import contribution_field as cf  # noqa: E402
import igb_io  # noqa: E402
import ladder  # noqa: E402
from bath_analysis import bathFrameSlice  # noqa: E402
from boundary_geometry import signedDistance  # noqa: E402

RUNNER = REPOSITORY / "3D_Code" / "run_boundary.py"


def jobRecord(jobFolder):
    return json.loads((Path(jobFolder) / "slabRun.json").read_text())


def writeTissueMesh(meshBase, outputBase, tagsOverride=None):
    positions, elements, tags, fibres = cf.readMesh(meshBase)
    tissue = tags != cf.BATH_TAG
    nodes = np.unique(elements[tissue])
    newIndex = -np.ones(len(positions), int); newIndex[nodes] = np.arange(len(nodes))
    elementsT, tagsT = newIndex[elements[tissue]], (tags[tissue] if tagsOverride is None else tagsOverride)
    outputBase = str(outputBase)
    with open(outputBase + ".pts", "w") as stream:
        stream.write(f"{len(nodes)}\n"); np.savetxt(stream, positions[nodes] * 1000.0, fmt="%.3f")
    with open(outputBase + ".elem", "w") as stream:
        stream.write(f"{len(elementsT)}\n")
        for row, tag in zip(elementsT, tagsT):
            stream.write("Tt " + " ".join(map(str, row)) + f" {int(tag)}\n")
    with open(outputBase + ".lon", "w") as stream:
        stream.write("1\n"); np.savetxt(stream, fibres[tissue], fmt="%.8f")
    return positions[nodes], elementsT, np.asarray(tagsT)


def prepare(runFolder, config, jobs, windowRule):
    """Mesh, electrodes, links and the derived window. Returns the analysis config (window filled)."""
    runFolder = Path(runFolder)
    if (runFolder / "prepared.json").exists():
        return json.loads((runFolder / "config.json").read_text())
    if runFolder.exists() and any(runFolder.iterdir()):
        raise FileExistsError(f"{runFolder} is incomplete; move it aside and rerun")
    (runFolder / "mesh").mkdir(parents=True)
    intact, lesion = jobRecord(jobs["intact"]), jobRecord(jobs["lesion"])
    pi, ei, ti, _ = cf.readMesh(intact["meshBase"])
    pl, el, tl, _ = cf.readMesh(lesion["meshBase"])
    if not (np.array_equal(ei, el) and np.allclose(pi, pl)):
        raise ValueError("Intact and lesion meshes differ in nodes or connectivity (they must differ in tags only)")
    if not np.array_equal(ti == cf.BATH_TAG, tl == cf.BATH_TAG):
        raise ValueError("Intact and lesion meshes differ in their bath elements")
    positions, elements, _ = writeTissueMesh(intact["meshBase"], runFolder / "mesh" / "slabRaw",
                                             tagsOverride=np.ones((ti != cf.BATH_TAG).sum(), int))
    shutil.copyfile(runFolder / "mesh" / "slabRaw.pts", runFolder / "mesh" / "intact.pts")
    shutil.copyfile(runFolder / "mesh" / "slabRaw.elem", runFolder / "mesh" / "intact.elem")
    shutil.copyfile(runFolder / "mesh" / "slabRaw.lon", runFolder / "mesh" / "intact.lon")
    _, _, lesionTags = writeTissueMesh(lesion["meshBase"], runFolder / "mesh" / "lesion")
    expected = signedDistance(positions[elements].mean(axis=1), config["boundary"]) > 0
    if not np.array_equal(lesionTags == cf.LESION_TAG, expected):
        raise ValueError("Lesion tags differ from the pipeline's lesion definition")
    if not np.allclose(positions.min(axis=0), -np.asarray(config["slabSizeMm"]) / 2):
        raise ValueError("Tissue mesh does not match slabSizeMm")
    boundary_setup.writeElectrodes(runFolder, config)
    for condition, job in jobs.items():
        job = Path(job).resolve()
        folder = runFolder / condition
        folder.mkdir()
        nodes = igb_io.header(job / "vm.igb")[1]
        if nodes != len(positions):
            raise ValueError(f"{condition}: vm.igb has {nodes} nodes, tissue mesh has {len(positions)}")
        for name in ["vm.igb", "act-thresh.dat"]:
            (folder / name).symlink_to(job / name)
        (folder / "phie_recovery.igb").symlink_to(job / "phie.igb")
        shutil.copy2(job / "parameters.par", folder / "parameters.par")
        (folder / "meshname.txt").write_text(str(runFolder / "mesh" / condition) + "\n")
        text = (job.parent / f"{job.name}.log").read_text(errors="replace")
        if (job / "restart.json").exists():
            # Prepaced run: two launches (prepace, restart; finalise.py checked both). The pipeline's single-launch
            # check reads the restarted run, which wrote phi_e; the full log stays beside the simulation.
            text = f"# Restarted run only; prepacing log: {job.parent / (job.name + '.log')}\n" + \
                text[text.rfind("*** Launching simulation ***"):]
        (runFolder / f"{condition}.log").write_text(text)
    (runFolder / "sourceSnapshot").mkdir()
    shutil.copy2(REPOSITORY / "3D_Code/3d_pilot.par", runFolder / "sourceSnapshot/3d_pilot.par")
    boundary_setup.saveJson(runFolder / "config.json", config)
    window = ladder.deriveWindow(runFolder, config, windowRule)
    config["analysis"]["beatWindowMs"] = window["windowMs"]
    boundary_setup.saveJson(runFolder / "config.json", config)
    boundary_setup.saveJson(runFolder / "windowDerivation.json", window)
    boundary_setup.saveJson(runFolder / "manifest.json", dict(
        model="production slab (3D_Code/production/slab.py, carputils)", jobs={k: str(v) for k, v in jobs.items()},
        distanceConvention="positive inside lesion; virtual boundary retained in intact"))
    boundary_setup.saveJson(runFolder / "prepared.json", dict(complete=True))
    return config


def analysisFrames(runFolder, config):
    times = igb_io.timeAxis(Path(runFolder) / "intact" / "vm.igb")
    return times, bathFrameSlice(times, config)


def recoveryInterpolator(meshBase, runFolder):
    positions, elements, _, _ = cf.readMesh(meshBase)
    layout = dict(np.load(Path(runFolder) / "electrodes" / "layout.npz"))
    return cf.pointInterpolator(positions, elements, layout["recoveryPoints"]), layout


def writeElectrograms(runFolder, config, condition, electrodeBlocks, source, block=50):
    """electrodeBlocks(frames) -> (placements, 16, frames) unipolar phi; writes egm.npz (NaN outside analysis frames)."""
    runFolder = Path(runFolder)
    folder = runFolder / condition
    if (folder / "meta.json").exists():
        return
    times, frames = analysisFrames(runFolder, config)
    layout = dict(np.load(runFolder / "electrodes" / "layout.npz"))
    uni = np.full((*layout["recoveryIndex"].shape[:2], len(times)), np.nan, dtype=np.float32)
    indices = np.arange(len(times))[frames]
    for first in range(0, len(indices), block):
        chosen = indices[first:first + block]
        values = electrodeBlocks(slice(chosen[0], chosen[-1] + 1))
        if not np.isfinite(values).all():
            raise ValueError(f"{condition}: non-finite extracellular potential")
        uni[..., chosen] = values
    np.savez_compressed(folder / "egm.npz", uni=uni, uni_xy=layout["centres"], uni_sample_xyz=layout["samples"],
                        d_mm=signedDistance(layout["centres"], config["boundary"]),
                        centre_xy_mm=layout["placements"][:, :2], p_gap_mm=layout["placements"][:, 2], t_ms=times)
    boundary_setup.saveJson(folder / "meta.json", dict(
        condition=condition, complete=True, electrogramSource=source,
        note="phie_recovery.igb links to openCARP phie.igb (whole mesh); electrograms are in egm.npz"))


def phieFrameOffset(jobFolder):
    """Frames by which phie.igb starts after vm.igb (prepaced runs write phi_e only after the restart)."""
    restart = Path(jobFolder) / "restart.json"
    if not restart.exists():
        return 0
    step = float(np.diff(igb_io.timeAxis(Path(jobFolder) / "vm.igb")[:2])[0])
    return int(round(json.loads(restart.read_text())["phieStartMs"] / step))


def readPhie(jobFolder, frames):
    """phie.igb frames addressed on the vm.igb time axis; refuses frames before phi_e was computed."""
    offset = phieFrameOffset(jobFolder)
    if frames.start < offset:
        raise ValueError(f"{jobFolder}: frame {frames.start} precedes the start of phie.igb (frame {offset}); "
                         "increase restartMarginMs")
    values = igb_io.read(Path(jobFolder) / "phie.igb", slice(frames.start - offset, frames.stop - offset))
    nodesFile = Path(jobFolder) / "phieNodes.npy"
    if not nodesFile.exists():
        return values
    # Restricted output: back to whole-mesh numbering, NaN where phi_e was not written. Sparse interpolation
    # multiplies stored weights only, so any electrode sample outside the written layer yields NaN and stops
    # writeElectrograms ("non-finite extracellular potential").
    nodes = np.load(nodesFile)
    total = jobRecord(jobFolder)["sets"]["nodes"]
    full = np.full((values.shape[0], total), np.nan, dtype=values.dtype)
    full[:, nodes] = values
    return full


def openCarpSignals(jobFolder, interpolator, layout):
    """openCARP phie.igb interpolated at the electrode samples, averaged over each electrode's samples."""
    index = layout["recoveryIndex"]
    return lambda frames: (interpolator @ readPhie(jobFolder, frames).T.astype(float))[index].mean(axis=2)


def electrodeRows(interpolator, layout):
    """Unique electrodes: (U x N) rows = mean of each electrode's sample rows, and the (placements, 16) map onto them.

    Overlapping electrode grids share electrodes (same samples), so U is much smaller than placements x 16.
    """
    from scipy import sparse
    index = layout["recoveryIndex"]                                  # (placements, 16, samples)
    keys = {}
    electrodeOf = np.empty(index.shape[:2], dtype=np.int64)
    for p in range(index.shape[0]):
        for e in range(index.shape[1]):
            electrodeOf[p, e] = keys.setdefault(tuple(sorted(index[p, e])), len(keys))
    unique = list(keys)
    samples = index.shape[2]
    averaging = sparse.csr_matrix((np.full(len(unique) * samples, 1.0 / samples),
                                   (np.repeat(np.arange(len(unique)), samples), np.concatenate(unique))),
                                  shape=(len(unique), interpolator.shape[0]))
    return averaging @ interpolator, electrodeOf


def femSignals(jobFolder, model, interpolator, layout, processes=1):
    """Extracellular solve of the job's Vm at the electrodes, by reciprocity (contribution_field.electrodeTransfer).

    Equals solving every frame and interpolating (tests/test_contribution_field.py), with one solve per unique
    electrode instead of one per frame.
    """
    path, cache = Path(jobFolder) / "vm.igb", {}

    def signals(frames):
        if not cache:                                              # solved on first use (skipped on resume)
            rows, cache["electrodeOf"] = electrodeRows(interpolator, layout)
            cache["transfer"] = cf.electrodeTransfer(model, rows, processes)    # (U, tissue nodes)
        return (cache["transfer"] @ igb_io.read(path, frames).T.astype(float))[cache["electrodeOf"]]
    return signals


def runPipeline(runFolder, processCount=1):
    runFolder = Path(runFolder)
    if (runFolder / "results" / "pilotStatus.json").exists():
        return json.loads((runFolder / "results" / "pilotStatus.json").read_text())
    log = runFolder / "analysis.log"
    with open(log, "w") as stream:
        code = subprocess.call([sys.executable, str(RUNNER), "--config", str(runFolder / "config.json"),
                                "--output", str(runFolder), "--process-only", "--np", str(processCount)],
                               cwd=str(REPOSITORY / "3D_Code"), stdout=stream, stderr=subprocess.STDOUT)
    status = runFolder / "results" / "pilotStatus.json"
    if not status.exists():
        tail = log.read_text(errors="replace").splitlines()[-15:]
        raise RuntimeError(f"Pipeline failed for {runFolder} (exit {code}); last lines:\n" + "\n".join(tail))
    return json.loads(status.read_text())


def groundTruth(runFolder, config, models, voxelMm, curveStepMm):
    """Contribution fields of the ground-truth recordings (both conditions); writes results/contributionField.npz."""
    import pandas as pd
    import ground_truth
    runFolder = Path(runFolder)
    output = runFolder / "results" / "contributionField.npz"
    if output.exists():
        return pd.read_csv(runFolder / "results" / "contributionFieldSummary.csv")
    gain = json.loads((runFolder / "results" / "calibration.json").read_text())["gain"]
    grid = cf.voxelGrid(config["slabSizeMm"], voxelMm)
    rows, fields, ids, lateral, depth = [], [], [], [], []
    for condition, model in models.items():
        points = pd.read_csv(runFolder / "results" / f"{condition}Points.csv")
        points = points[(points.noiseRepeat == 0) & points.centreYmm.isin(config["analysis"]["groundTruthCentreYmm"])]
        source = dict(np.load(runFolder / condition / "egm.npz"))
        rawSlice, measureSlice = analysis.measurementSlices(source["t_ms"], config)
        sampleRate = 1000 / np.diff(source["t_ms"][:2])[0]
        with np.load(runFolder / "results" / f"{condition}Traces.npz") as archive:
            traces = archive["cleanAtSelectedWeightsMv"]
        for mode in ["uni", "differential"]:
            band = None if mode == "uni" else config["analysis"]["differentialBandHz"]
            voltage = ground_truth.readFilteredVoltage(runFolder / condition, rawSlice, band, sampleRate,
                                                       config["analysis"]["filterOrder"])[:, measureSlice].astype(float)
            records = points[points.kind == "uni"] if mode == "uni" else points[points.kind != "uni"]
            for record in records.to_dict("records"):
                weights = [record[f"weightE{i}"] for i in range(16)]
                row = cf.recordingRow(model, source["uni_sample_xyz"][record["placement"]], weights)
                elementShares, phi, total = cf.shares(model, row, voltage)
                measured = traces[record["traceIndex"]].astype(float)
                scale = float(measured @ phi / (phi @ phi))
                residual = float(np.linalg.norm(measured - scale * phi) / np.linalg.norm(measured))
                anchor = [record["anchorXmm"], record["anchorYmm"], record["anchorZmm"]]
                found = cf.curves(model, elementShares, anchor, curveStepMm)
                fields.append(cf.voxelise(model, elementShares, grid).astype(np.float32))
                ids.append(record["recordingId"])
                lateral.append(found["lateral"]["cumulativeShare"]); depth.append(found["depth"]["cumulativeShare"])
                rows.append(dict(recordingId=record["recordingId"], condition=condition, kind=record["kind"],
                                 label=record["label"], gapMm=record["gapMm"], centreYmm=record["centreYmm"],
                                 shareSum=total, reconstructionResidual=residual, scaleOverGain=scale / gain,
                                 shareAbsSum=float(np.abs(elementShares).sum())))
    pad = lambda values: np.array([np.pad(v, (0, max(map(len, values)) - len(v)), constant_values=np.nan) for v in values])
    np.savez_compressed(output, recordingId=np.asarray(ids), fields=np.asarray(fields), voxelMm=voxelMm,
                        lower=grid["lower"], lateralShare=pad(lateral), depthShare=pad(depth), curveStepMm=curveStepMm,
                        source=next(iter(models.values())).source)
    table = pd.DataFrame(rows)
    table.to_csv(runFolder / "results" / "contributionFieldSummary.csv", index=False)
    return table
