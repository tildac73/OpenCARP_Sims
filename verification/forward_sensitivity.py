#!/usr/bin/env python3
"""Methodology verification, part 3: sensitivity of the forward model to the volume-conductor set-up (no openCARP).

    python3 verification/forward_sensitivity.py --out runs/verification/forward

Question. The production model surrounds the slab with 5 mm of blood, closes the outer walls with a
no-flux condition, grounds a 1 mm cube at one bath corner and leaves the epicardium insulated
(Leenknegt et al. 2023 for the bath and ground; docs/decision_log.md DL-2026-11-02-02, FM-05). None of
these was tested for convergence. This check measures how much the recorded signal and the spatial
distribution of its contributing tissue change when each is varied alone:

    bath thickness 2.5 / 5 / 10 mm;  ground at the opposite top corner;  blood below the epicardium;
    slab extended by 4 mm on every lateral side.

Method. A REDUCED model, not the production one: coarser mesh (0.5 mm in-plane, 0.25 mm transmural, 1.0 mm
wall by default), a smaller slab, and an analytic planar wave in place of an openCARP solution
(upstroke 0.5 ms, 0.9 m/s along +x; the same stand-in as tests/test_synthetic_pipeline.py). The forward
problem itself is the production one: the same P1 operators (efov.leadfield.assemble, tissueTensors),
conductivities, fibre model, electrode definition (efov.geometry.electrodeNodeSets), electrode current
(1 uA spread by the mass matrix, tests/helpers.py electrodeCurrent, as openCARP's lead-field module),
band-pass (efov.measurements.differentialFilter) and contribution definitions (efov.groundtruth.shares).
Lead fields are solved here with preconditioned conjugate gradients (relative residual 1e-10).

Reading the result. Differences are relative to the reference case (5 mm bath, corner ground, insulated
epicardium). They bound the influence of the conductor set-up on this reduced model; they are not a
convergence proof for the production mesh, and they say nothing about propagation (bath loading of Vm is
outside the monodomain model; see docs/bidomain_reference.md on the bidomainReference branch).
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPOSITORY, cfg, runChecks, variantConfig  # noqa: E402

sys.path.insert(0, str(REPOSITORY / "tests"))
import helpers  # noqa: E402  (independent reference operators used by the unit tests)

from efov import geometry, groundtruth, leadfield, measurements, meshbuild  # noqa: E402

WAVE_SPEED_M_PER_S = 0.9
ARRIVAL_AT_ORIGIN_MS = 40.0
DURATION_MS = 100.0
LEVELS = (0.5, 0.8, 0.9)


def buildMesh(config):
    """Regular Kuhn-tetrahedral slab with bath on every side, passed through the production mesh routine
    (efov.meshbuild.runMesh: insulated epicardium unless epicardialMedium is 'blood'; fibre vectors)."""
    size, spacing, bath = cfg.slabSizeMm(config), cfg.spacingMm(config), config["fixed"]["bath"]["thicknessMm"]
    lower, upper = -size / 2 - bath, size / 2 + bath
    axes = [np.linspace(lo, hi, int(round((hi - lo) / s)) + 1) for lo, hi, s in zip(lower, upper, spacing)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    shape = grid.shape[:3]
    cells = np.array(list(itertools.product(*(range(n - 1) for n in shape))))
    tets = []
    for order in itertools.permutations(range(3)):
        corner = cells.copy()
        ids = [np.ravel_multi_index(corner.T, shape)]
        for axis in order:
            corner = corner.copy(); corner[:, axis] += 1
            ids.append(np.ravel_multi_index(corner.T, shape))
        tets.append(np.stack(ids, axis=1))
    positions, elements = grid.reshape(-1, 3), np.concatenate(tets)
    centres = positions[elements].mean(axis=1)
    tags = np.where(np.all(np.abs(centres) < size / 2, axis=1), meshbuild.TISSUE_TAG, meshbuild.BATH_TAG)
    positions, elements, tags, fibres = meshbuild.runMesh(positions, elements, tags, size[2], config["values"]["epicardialMedium"],
                                                          config["values"]["fibres"])
    return dict(positions=positions, elements=elements, tags=tags, fibres=fibres, tissueNodes=np.unique(elements[tags != meshbuild.BATH_TAG]))


def oppositeCornerGround(tissue, cubeMm):
    """Nodes within cubeMm of the (min x, min y, max z) corner: the top corner diagonally opposite the configured ground."""
    p = tissue["positions"]
    corner = np.array([p[:, 0].min(), p[:, 1].min(), p[:, 2].max()])
    return np.flatnonzero((p[:, 0] <= corner[0] + cubeMm + 1e-9) & (p[:, 1] <= corner[1] + cubeMm + 1e-9) & (p[:, 2] >= corner[2] - cubeMm - 1e-9))


def solveLeadFields(config, tissue, ground, electrodeSets, tolerance=1e-10):
    """Z (mV/uA) on every node for each electrode: K z = b with z = 0 on the ground; Jacobi-preconditioned CG."""
    bulk, _ = helpers._bulkAndSource(config, tissue)
    free = np.setdiff1d(np.arange(len(tissue["positions"])), np.unique(ground))
    matrix = bulk[free][:, free].tocsr()
    inverseDiagonal = 1.0 / matrix.diagonal()
    preconditioner = LinearOperator(matrix.shape, matvec=lambda x: inverseDiagonal * x)
    fields, iterations = [], []
    for nodes in electrodeSets:
        rhs = helpers.electrodeCurrent(tissue, nodes)[free]
        count = [0]
        solution, info = cg(matrix, rhs, rtol=tolerance, atol=0.0, maxiter=50000, M=preconditioner,
                            callback=lambda _: count.__setitem__(0, count[0] + 1))
        if info != 0:
            raise RuntimeError(f"Lead-field solve did not converge (info {info}) on {matrix.shape[0]} unknowns")
        residual = float(np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs))
        z = np.zeros(len(tissue["positions"])); z[free] = solution
        fields.append(z); iterations.append(dict(iterations=count[0], relativeResidual=residual))
    return fields, iterations


def planarWave(tissue, config):
    """Vm - rest (mV) on tissue nodes at the acquisition rate: (nodes, samples), with filter history of rest before t = 0."""
    rate = config["fixed"]["analysis"]["acquisitionRateHz"]
    history = config["fixed"]["analysis"]["filterHistoryMs"]
    times = np.arange(-history, DURATION_MS + 1e-9, 1000.0 / rate)
    x = tissue["positions"][tissue["tissueNodes"], 0]
    dt = times[None, :] - (ARRIVAL_AT_ORIGIN_MS + x[:, None] / WAVE_SPEED_M_PER_S)
    vm = 110.0 * 0.5 * (1 + np.tanh(dt / 1.0)) * np.exp(-np.clip(dt, 0, None) / 150.0)
    return vm, times


def massRadii(model, mass, anchor):
    """Lateral distance from the anchor within which the stated fractions of the |c|-mass lie (mm)."""
    lateral = np.hypot(model.centres[:, 0] - anchor[0], model.centres[:, 1] - anchor[1])
    order = np.argsort(lateral)
    cumulative = np.cumsum(mass[order]) / mass.sum()
    return {f"massRadius{int(100 * level)}Mm": float(lateral[order][np.searchsorted(cumulative, level)]) for level in LEVELS}


def evaluateCase(root, name, fixedChanges, sweepOverrides, groundChoice, gapMm):
    base = {"tissue.spacingInPlaneMm": 0.5, "tissue.spacingTransmuralMm": 0.25, "electrodes.centreYmm": [0.0],
            "groundTruth.placementsYmm": [0.0]}
    base.update(fixedChanges)
    config = variantConfig(Path(root) / name, base, {"wallMm": 1.0, "gapsMm": [gapMm], **sweepOverrides})
    tissue = buildMesh(config)
    cube = config["fixed"]["bath"]["groundCubeMm"]
    if groundChoice == "configuredCorner":
        ground = meshbuild.groundNodes(tissue["positions"], tissue["elements"], tissue["tags"], cube)
    else:
        ground = oppositeCornerGround(tissue, cube)
    layout = geometry.electrodeLayout(config)
    v = config["values"]
    candidates = meshbuild.catheterSideNodes(tissue["positions"], tissue["elements"], tissue["tags"], cfg.slabSizeMm(config)[2])
    wanted = {"E5": 5, "E6": 6, "E9": 9}                                      # E5 with its +x and +y neighbours (row-major, x fastest)
    centres = {label: layout["centres"][0, index] for label, index in wanted.items()}
    sets = geometry.electrodeNodeSets(tissue["positions"], candidates, np.array(list(centres.values())), v["electrodeRadiusMm"], v["electrodeLengthMm"])
    fields, solves = solveLeadFields(config, tissue, ground, sets)
    z = {label: field[tissue["tissueNodes"]] for label, field in zip(wanted, fields)}
    model = leadfield.ContributionModel(config, tissue)
    vm, times = planarWave(tissue, config)
    from scipy.signal import sosfilt
    filtered = sosfilt(measurements.differentialFilter(config), vm, axis=-1)
    keep = times >= 0
    recordings = {
        "unipole E5": (z["E5"], vm[:, keep], centres["E5"]),
        "bipole E5-E6 (along the wave)": (z["E5"] - z["E6"], filtered[:, keep], (centres["E5"] + centres["E6"]) / 2),
        "bipole E5-E9 (across the wave)": (z["E5"] - z["E9"], filtered[:, keep], (centres["E5"] + centres["E9"]) / 2),
    }
    out = {}
    for label, (field, signalVm, anchor) in recordings.items():
        coefficients = model.coefficients(field)
        elementShares, mass, phi, shareSum = groundtruth.shares(model, coefficients, signalVm)
        out[label] = dict(peakToPeakMv=float(np.ptp(phi)), trace=phi, shareSum=shareSum, shareAbsSum=float(np.abs(elementShares).sum()),
                          **massRadii(model, mass, anchor))
    return dict(nodes=int(len(tissue["positions"])), tissueNodes=int(len(tissue["tissueNodes"])), groundNodes=int(len(ground)),
                electrodeNodes=[int(len(s)) for s in sets], solves=solves, recordings=out)


def checkForward(root, gapMm):
    reference = {"tissue.sizeMm": [24.0, 28.0], "bath.thicknessMm": 5.0}
    cases = [("reference", reference, {}, "configuredCorner"),
             ("bath2p5mm", {**reference, "bath.thicknessMm": 2.5}, {}, "configuredCorner"),
             ("bath10mm", {**reference, "bath.thicknessMm": 10.0}, {}, "configuredCorner"),
             ("groundOppositeCorner", reference, {}, "oppositeCorner"),
             ("bloodBelowEpicardium", reference, {"epicardialMedium": "blood"}, "configuredCorner"),
             ("slabPlus4mmEachSide", {**reference, "tissue.sizeMm": [32.0, 36.0]}, {}, "configuredCorner")]
    results = {name: evaluateCase(root, name, changes, sweep, ground, gapMm) for name, changes, sweep, ground in cases}
    base = results["reference"]["recordings"]
    rows = []
    for name, result in results.items():
        for label, values in result["recordings"].items():
            trace, baseTrace = values["trace"], base[label]["trace"]
            rows.append(dict(case=name, recording=label, peakToPeakMv=values["peakToPeakMv"],
                             peakToPeakVersusReferencePercent=float(100 * (values["peakToPeakMv"] / base[label]["peakToPeakMv"] - 1)),
                             traceRelativeL2VersusReference=float(np.linalg.norm(trace - baseTrace) / np.linalg.norm(baseTrace)),
                             # After removing the constant offset: a change of reference shifts a unipole by a common-mode term.
                             traceRelativeL2AfterMeanRemoval=float(np.linalg.norm((trace - trace.mean()) - (baseTrace - baseTrace.mean()))
                                                                   / np.linalg.norm(baseTrace - baseTrace.mean())),
                             shareAbsSum=values["shareAbsSum"],
                             **{k: values[k] for k in values if k.startswith("massRadius")},
                             **{k + "MinusReference": float(values[k] - base[label][k]) for k in values if k.startswith("massRadius")}))
    sizes = {name: {k: result[k] for k in ("nodes", "tissueNodes", "groundNodes", "electrodeNodes", "solves")} for name, result in results.items()}
    return dict(
        supports="5 mm blood bath with no-flux outer walls, 1 mm grounded corner cube, insulated epicardium, 40 x 60 mm slab.",
        kind="numerical",
        method="Reduced model (0.5 / 0.25 mm mesh, 1.0 mm wall, 24 x 28 mm slab, analytic planar wave at 0.9 m/s along +x, "
               f"electrode gap {gapMm:g} mm); production P1 operators, conductivities, fibres, electrode definition, band-pass and "
               "contribution definitions; one setting changed per case. Mass radius = lateral distance from the recording anchor "
               "containing the stated fraction of the |c|-mass. Lateral lattice step 0.5 mm limits the radius resolution.",
        values=dict(gapMm=gapMm, rows=rows, sizes=sizes, waveSpeedMPerS=WAVE_SPEED_M_PER_S))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPOSITORY / "runs" / "verification" / "forward"))
    parser.add_argument("--gaps", type=float, nargs="*", default=[1.0, 0.25], help="electrode gaps (mm)")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    checks = [(f"forwardSensitivity_gap{gap:g}".replace(".", "p"), lambda gap=gap: checkForward(out / "work" / f"gap{gap:g}", gap)) for gap in args.gaps]
    status = runChecks(checks, out)
    return 0 if all(state == "completed" for state in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
