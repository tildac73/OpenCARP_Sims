#!/usr/bin/env python3
"""Methodology verification, part 2: checks that run openCARP (version 19 or later; no lead-field module needed).

    export MPIEXEC_BIN=<mpiexec of the openCARP build>          # as for the production runs
    python3 verification/opencarp_checks.py --out runs/verification/opencarp --np 2 [--only NAME ...]
    python3 verification/opencarp_checks.py --out ... --np 24 --np-list 1 2 4 8 16 24     # workstation scaling

Every simulation goes through the production route (efov.simulation.runCondition -> the carputils
experiment efov/opencarp/slab_experiment.py -> one resolved .par file per launch), so the settings a
check does not deliberately change are exactly those of config/model.par. Each check writes its variant
configuration (verification.common.variantConfig) beside its results. A completed simulation is reused
on a rerun; an interrupted one stops the check with openCARP's own log path.

Checks (each is a small problem; sizes are in the functions)
  conditioning          single-cell limit cycle: tissue action-potential biomarkers and strip CV against the
                        number of single-cell conditioning beats; beat-to-beat drift over 8 tissue beats
  spatialConvergence    strip CV against in-plane and transmural mesh spacing at the tuned beta, along and
                        across fibres (three or four levels, observed order, Richardson extrapolation)
  temporalConvergence   strip CV and APD90 against the solver time step at the production mesh
  stimulusThreshold     smallest 2 ms transmembrane current that captures the strip (bisection), and the
                        safety factor of the configured stimulus
  slabVelocity          first-beat surface CV and transmural activation delay of a small slab for each fibre
                        model and wall thickness in the sweep (the strip CV target is along fibres only)
  parallelEquivalence   the same small slab on 1 and on N MPI processes: largest Vm and activation-time
                        difference, wall time (strong scaling when --np-list is given)

What these checks are: numerical verification of the monodomain solve as configured. They do not
validate the Courtemanche model, the conductivities or the fibre architecture against experiment, and
they do not exercise the lead-field module (openCARP >= 20; checks/container_checks.py W0 does).
Strip and small-slab results do not establish full-size cost.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPOSITORY, cfg, openCarpBuildInfo, relativeDifference, richardsonSpacings, runChecks, variantConfig  # noqa: E402

from efov import physiology, simulation  # noqa: E402
from efov.io import igbHeader, igbTimes, readActivations, readIgb  # noqa: E402
from efov.tune import stripCv  # noqa: E402

ACTIVATION_THRESHOLD_MV = -20.0          # the model's LAT threshold (config/model.par lats[0].threshold)
SMALL_SLAB_MM = [16.0, 16.0]             # x, y; smallest slab that still holds one HD Grid placement (validated by efov.config)
SMALL_SLAB_DURATION_MS = 30.0            # one wave crossing: 16 mm at >= 0.5 m/s plus the 2 ms stimulus


# ----------------------------------------------------------------- shared run helpers
def stripVariant(root, name, fixedChanges=None, sweepOverrides=None, parOverrides=None):
    return variantConfig(Path(root) / name / "config", fixedChanges, sweepOverrides, parOverrides)


def physicsSeconds(log):
    """Seconds openCARP reports under 'Timings of individual physics' (Init + Compute + Output of every physics).

    This is the solve itself. The wall time of a run (efov.simulation.runCondition) also holds the mesher,
    the mesh rewrite with fibres, the carputils start-up and openCARP's set-up, so the two are reported separately.
    """
    import re
    text = Path(log).read_text(errors="replace")
    block = text[text.rfind("Timings of individual physics"):] if "Timings of individual physics" in text else ""
    values = [float(v) for v in re.findall(r"^\s*(?:Init|Compute|Output):\s*([0-9.eE+-]+) sec", block, flags=re.M)]
    return float(sum(values)) if values else float("nan")


def runStrip(config, folder, processCount, state=None):
    """Condition a cell (unless a state is given), run the tuning strip; return (job folder, timing).

    timing: wallSeconds = the whole runCondition call when the run was made (meshing, carputils and openCARP
    set-up included; conditioning excluded), physicsSeconds = openCARP's own physics timing from the run log.
    """
    folder = Path(folder)
    state = state or simulation.conditionCell(config, folder / "conditioning")
    timing = simulation.runCondition(config, "strip", state, folder / "simulations", processCount, kind="strip")
    seconds = dict(wallSeconds=timing.get("wallSeconds"), physicsSeconds=physicsSeconds(folder / "simulations" / "strip.log"))
    return folder / "simulations" / "strip", seconds


def stripBeatTable(config, job):
    """Per beat, on strip nodes inside the CV measurement interval: CV, APD90, resting potential, amplitude.

    Activation = first interpolated upward crossing of -20 mV (as the model's LATs); CV = inverse slope of a
    least-squares line through activation time against x; APD90 as efov.physiology.apd90 (90 % repolarisation
    relative to the pre-beat voltage). Vm is read at the output step (0.2 ms), so upstroke velocity is not
    reported: it cannot be resolved at that sampling.
    """
    t = config["fixed"]["tune"]
    tissue = simulation.tissueModel(job)
    x = tissue["tissuePositions"][:, 0]
    use = (x >= t["measureFromMm"]) & (x <= t["measureToMm"])
    times = igbTimes(Path(job) / "vm.igb")
    cycle = cfg.cycleLengthMs(config)
    beats = int(round((times[-1] - times[0]) / cycle))
    rows = []
    for beat in range(beats):
        start = beat * cycle
        frames = slice(int(np.searchsorted(times, start - 1e-9)), int(np.searchsorted(times, start + cycle - 1e-9)))
        local = times[frames] - start
        voltage = readIgb(Path(job) / "vm.igb", frames)[:, use]
        activation, count = physiology.activationTimes(local, voltage, ACTIVATION_THRESHOLD_MV)
        captured = bool(np.all(count == 1))
        row = dict(beat=beat + 1, captured=captured, restingPotentialMv=float(np.median(voltage[0])),
                   amplitudeMv=float(np.median(np.ptp(voltage, axis=0))))
        if captured:
            slope = np.polyfit(x[use], activation, 1)[0]                     # ms / mm
            apd = physiology.apd90(local, voltage, activation)
            row.update(cvMPerS=float(1.0 / slope), apd90Ms=float(np.nanmedian(apd)),
                       repolarisationComplete=bool(np.isfinite(apd).all()))
        rows.append(row)
    return rows


def slabVariant(root, name, sweepOverrides=None, parOverrides=None):
    """A small slab at the production mesh spacing with one catheter placement (only what validation requires)."""
    changes = {"tissue.sizeMm": SMALL_SLAB_MM, "electrodes.centreYmm": [0.0], "groundTruth.placementsYmm": [0.0]}
    return variantConfig(Path(root) / name / "config", changes, sweepOverrides, parOverrides)


def runSlab(config, folder, processCount, state):
    folder = Path(folder)
    timing = simulation.runCondition(config, "intact", state, folder / "simulations", processCount, durationMs=SMALL_SLAB_DURATION_MS)
    # The recorded timing of the run itself, so a reused run does not report the (near-zero) time of the reuse.
    seconds = dict(wallSeconds=timing.get("wallSeconds"), physicsSeconds=physicsSeconds(folder / "simulations" / "intact.log"))
    return folder / "simulations" / "intact", seconds


def firstActivationMs(job, nodeCount):
    nodes, times = readActivations(Path(job) / "act-thresh.dat")
    first = np.full(nodeCount, np.nan)
    order = np.argsort(times)[::-1]                                          # earliest written last, so it wins
    first[nodes[order]] = times[order]
    return first


# ----------------------------------------------------------------- checks
def checkConditioning(root, processCount, beatCounts=(1, 20, 100, 300, 1000, 2000)):
    root = Path(root)
    rows = []
    for beats in beatCounts:
        config = stripVariant(root, f"beats{beats}", {"conditioning.beats": int(beats), "tune.beats": 4})
        job, _ = runStrip(config, root / f"beats{beats}", processCount)
        table = stripBeatTable(config, job)
        last = table[-1]
        rows.append(dict(conditioningBeats=int(beats), tissueBeats=table, lastBeat=last,
                         apd90ChangeLastTwoBeatsMs=float(abs(table[-1].get("apd90Ms", np.nan) - table[-2].get("apd90Ms", np.nan))),
                         cvChangeLastTwoBeatsPercent=float(100 * abs(relativeDifference(table[-1].get("cvMPerS", np.nan), table[-2].get("cvMPerS", np.nan))))))
    reference = rows[-1]["lastBeat"]
    for row in rows:
        row["apd90MinusLongestConditioningMs"] = float(row["lastBeat"].get("apd90Ms", np.nan) - reference.get("apd90Ms", np.nan))
        row["cvVersusLongestConditioningPercent"] = float(100 * relativeDifference(row["lastBeat"].get("cvMPerS", np.nan), reference.get("cvMPerS", np.nan)))
        row["restingPotentialMinusLongestConditioningMv"] = float(row["lastBeat"]["restingPotentialMv"] - reference["restingPotentialMv"])
    longConfig = stripVariant(root, "tissueBeats8", {"tune.beats": 8})
    longJob, _ = runStrip(longConfig, root / "tissueBeats8", processCount)
    return dict(
        supports="1,000 single-cell conditioning beats at the pacing cycle length, then 4 tissue beats with the last one analysed.",
        kind="simulation",
        method="bench limit cycle of N beats (efov.simulation.conditionCell), state loaded into the production-mesh tuning strip, "
               "4 tissue beats; biomarkers on strip nodes 5-15 mm from the stimulus. Reference: the longest conditioning. "
               "A second strip run of 8 beats after the configured conditioning shows the drift across tissue beats.",
        values=dict(cycleLengthMs=cfg.cycleLengthMs(longConfig), ionicModel=cfg.par(longConfig, "imp_region[0].im", str),
                    rows=rows, eightTissueBeatsAfterConfiguredConditioning=stripBeatTable(longConfig, longJob)))


def convergenceSeries(root, processCount, label, levels, changesFor, parOverrides=None, sweepFor=None):
    rows = []
    for level in levels:
        name = f"{label}{level:g}".replace(".", "p")
        config = stripVariant(root, name, changesFor(level), sweepFor(level) if sweepFor else None, parOverrides)
        job, seconds = runStrip(config, Path(root) / name, processCount)
        nodes = igbHeader(job / "vm.igb")[1]
        rows.append(dict(level=float(level), cvMPerS=stripCv(config, job), stripNodes=int(nodes), **seconds,
                         lastBeat=stripBeatTable(config, job)[-1]))
    finest = rows[-1]["cvMPerS"]
    for row in rows:
        row["cvVersusFinestPercent"] = float(100 * relativeDifference(row["cvMPerS"], finest))
    result = dict(rows=rows)
    if len(rows) >= 3:
        # Spacings are passed because the transmural levels do not have a constant refinement ratio.
        order, extrapolated = richardsonSpacings([r["cvMPerS"] for r in rows[-3:]], [r["level"] for r in rows[-3:]])
        result.update(observedOrderFinestThree=order, richardsonExtrapolatedCvMPerS=extrapolated)
        if np.isfinite(extrapolated):
            for row in rows:
                row["cvVersusExtrapolatedPercent"] = float(100 * relativeDifference(row["cvMPerS"], extrapolated))
    return result


def checkSpatialConvergence(root, processCount, inPlaneMm=(0.5, 0.25, 0.125, 0.0625), transmuralMm=(0.25, 0.1, 0.05)):
    root = Path(root)
    production = cfg.loadConfig()
    gil, git = cfg.par(production, "gregion[0].g_il"), cfg.par(production, "gregion[0].g_it")
    gel, get = cfg.par(production, "gregion[0].g_el"), cfg.par(production, "gregion[0].g_et")
    # A wave along x that travels across fibres: the strip's fibres are along x, so the longitudinal
    # conductivities are set to the transverse values (the tissue becomes isotropic at the transverse values).
    transverse = {"gregion[0].g_il": git, "gregion[0].g_el": get}
    inPlane = lambda dx: {"tissue.spacingInPlaneMm": float(dx)}                               # noqa: E731
    # A 0.25 mm transmural step does not divide the 1.1 mm wall the configuration validates, so that level
    # uses a 1.0 mm wall. The strip itself is 0.5 mm thick at every level (fixed.tune.stripSizeMm).
    transmural = lambda dz: {"tissue.spacingTransmuralMm": float(dz)}                         # noqa: E731
    wallFor = lambda dz: {"wallMm": 1.0} if not np.isclose(1.1 / dz, round(1.1 / dz)) else None  # noqa: E731
    return dict(
        supports="Mesh spacing 0.25 mm in-plane and 0.1 mm through the wall; beta tuned at that spacing.",
        kind="simulation",
        method="Tuning strip (20 x 1 x 0.5 mm, fibres along x, 3 beats, last beat measured) at the production beta, with only the mesh "
               "spacing changed. 'Across fibres' sets the longitudinal conductivities to the transverse values. CV differences are "
               "relative to the finest level and to the Richardson extrapolation of the three finest levels (refinement ratio 2 in-plane).",
        values=dict(productionSpacingMm=cfg.spacingMm(production).tolist(), betaPerUm=cfg.par(production, "imp_region[0].cellSurfVolRatio"),
                    conductivitiesAlongSPerM=dict(g_il=gil, g_el=gel), conductivitiesAcrossSPerM=dict(g_il=git, g_el=get),
                    inPlaneAlongFibres=convergenceSeries(root, processCount, "alongDx", inPlaneMm, inPlane),
                    inPlaneAcrossFibres=convergenceSeries(root, processCount, "acrossDx", inPlaneMm, inPlane, transverse),
                    transmuralAlongFibres=convergenceSeries(root, processCount, "alongDz", transmuralMm, transmural, None, wallFor)))


def checkTemporalConvergence(root, processCount, timeStepsUs=(40.0, 20.0, 10.0, 5.0)):
    rows = []
    for dt in timeStepsUs:
        name = f"dtUs{dt:g}".replace(".", "p")
        config = stripVariant(root, name, None, None, {"dt": float(dt)})
        job, seconds = runStrip(config, Path(root) / name, processCount)
        rows.append(dict(timeStepUs=float(dt), cvMPerS=stripCv(config, job), **seconds, lastBeat=stripBeatTable(config, job)[-1]))
    finest = rows[-1]
    for row in rows:
        row["cvVersusFinestPercent"] = float(100 * relativeDifference(row["cvMPerS"], finest["cvMPerS"]))
        row["apd90MinusFinestMs"] = float(row["lastBeat"].get("apd90Ms", np.nan) - finest["lastBeat"].get("apd90Ms", np.nan))
    return dict(
        supports="Solver time step 20 us (ionic and parabolic), Crank-Nicolson, Vm output every 0.2 ms.",
        kind="simulation",
        method="Production-mesh tuning strip at the production beta with only dt changed (the single-cell conditioning uses the same dt). "
               "CV and APD90 of the last of 3 beats, relative to the smallest time step. wallSeconds = whole run including meshing "
               "and set-up (not the conditioning); physicsSeconds = openCARP's own physics timing.",
        values=dict(rows=rows))


def checkStimulusThreshold(root, processCount, iterations=7):
    root = Path(root)
    production = stripVariant(root, "reference")
    state = simulation.conditionCell(production, root / "reference" / "conditioning")
    configured = cfg.par(production, "stim[0].pulse.strength")

    def captures(strength):
        name = f"strength{strength:.4f}".replace(".", "p")
        config = stripVariant(root, name, None, None, {"stim[0].pulse.strength": float(strength)})
        job = root / name / "simulations" / "strip"
        spec = simulation.runSpec(config, "strip", state, root / name / "meshes", "strip")

        def vmWrittenInFull():
            return (job / "vm.igb").exists() and igbHeader(job / "vm.igb")[2] == simulation.expectedFrames(config, spec)
        # If nothing activates, openCARP may write no activation file, which the production runner reports as a
        # failed run. That is "no capture" only when Vm was written in full; any other failure is raised.
        if not (vmWrittenInFull() and not (job / "act-thresh.dat").exists()):
            try:
                runStrip(config, root / name, processCount, state)
            except RuntimeError:
                if not vmWrittenInFull():
                    raise
                return False
        if not (job / "act-thresh.dat").exists():
            return False
        return bool(stripBeatTable(config, job)[-1]["captured"])
    if not captures(configured):
        raise RuntimeError(f"The configured stimulus ({configured} uA/cm^2) does not capture the strip")
    low, high, trials = 0.0, configured, [dict(strengthUaPerCm2=configured, captured=True)]
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        captured = captures(middle)
        trials.append(dict(strengthUaPerCm2=middle, captured=captured))
        low, high = (low, middle) if captured else (middle, high)
    return dict(
        supports="2 ms, 50 uA/cm^2 transmembrane stimulus on a 0.5 mm band.",
        kind="simulation",
        method="Bisection on the stimulus strength in the production-mesh tuning strip (same band width, duration and conditioning state); "
               "capture = every node 5-15 mm from the band activates once in the last of 3 beats.",
        values=dict(configuredUaPerCm2=configured, durationMs=cfg.stimulusDurationMs(production),
                    bandWidthMm=production["fixed"]["stimulus"]["stripWidthMm"],
                    thresholdBracketUaPerCm2=[low, high], safetyFactorRange=[configured / high, configured / low if low > 0 else float("inf")],
                    trials=trials))


def surfaceWave(job):
    """First-beat plane fit on the catheter-facing surface (|x| <= 4 mm, |y| <= 4 mm) and the transmural delay there."""
    tissue = simulation.tissueModel(job)
    positions = tissue["tissuePositions"]
    first = firstActivationMs(job, len(positions))
    central = (np.abs(positions[:, 0]) <= 4.0) & (np.abs(positions[:, 1]) <= 4.0)
    top = central & np.isclose(positions[:, 2], positions[:, 2].max())
    bottom = central & np.isclose(positions[:, 2], positions[:, 2].min())
    plane = physiology.activationPlane(positions[top], first[top], np.array([1.0, 0.0]))
    bottomPlane = physiology.activationPlane(positions[bottom], first[bottom], np.array([1.0, 0.0]))
    # Same (x, y) lattice on both surfaces, so sorting pairs the nodes.
    orderTop, orderBottom = (np.lexsort((positions[m, 1], positions[m, 0])) for m in (top, bottom))
    delay = first[bottom][orderBottom] - first[top][orderTop]
    return dict(catheterSurfaceSpeedMPerS=plane["speedMPerS"], catheterSurfaceDirectionErrorDeg=plane["directionErrorDeg"],
                catheterSurfacePlaneResidual95Ms=plane["planeResidual95Ms"], farSurfaceSpeedMPerS=bottomPlane["speedMPerS"],
                farMinusCatheterSurfaceActivationMsMedian=float(np.nanmedian(delay)),
                farMinusCatheterSurfaceActivationMsRange=[float(np.nanmin(delay)), float(np.nanmax(delay))],
                allCentralNodesActivated=bool(np.isfinite(first[central]).all()))


def checkSlabVelocity(root, processCount):
    root = Path(root)
    production = cfg.loadConfig()
    bilayer = production["sweep"]["fibres"]["baseline"]
    uniform, rotation = production["sweep"]["fibres"]["planned"]
    cases = [("bilayer_wall1p1", 1.1, bilayer), ("uniform_wall1p1", 1.1, uniform), ("rotation_wall1p1", 1.1, rotation),
             ("uniform_wall1p6_roof", 1.6, uniform), ("bilayer_wall1p9", 1.9, bilayer), ("bilayer_wall3p0", 3.0, bilayer)]
    reference = slabVariant(root, "reference")
    state = simulation.conditionCell(reference, root / "reference" / "conditioning")
    rows = []
    for name, wall, fibres in cases:
        config = slabVariant(root, name, {"wallMm": wall, "fibres": fibres})
        job, seconds = runSlab(config, root / name, processCount, state)
        rows.append(dict(case=name, wallMm=wall, fibreModel=fibres["model"], **seconds, **surfaceWave(job)))
    return dict(
        supports="The CV target (0.90 m/s) is set along fibres on a strip; the slab's surface CV depends on fibre model and wall thickness and is reported.",
        kind="simulation",
        method=f"Intact {SMALL_SLAB_MM[0]:g} x {SMALL_SLAB_MM[1]:g} mm slab at the production mesh spacing and beta, wave along +x, first beat only "
               f"({SMALL_SLAB_DURATION_MS:g} ms). Plane fit to first activation on the central 8 x 8 mm of each surface. "
               "First-beat values from the conditioned state; the production physiology stage reports the last-beat value of the full slab.",
        values=dict(stripTargetMPerS=production["values"]["cvTargetMPerS"], rows=rows))


def checkParallelEquivalence(root, processCounts):
    root = Path(root)
    reference = slabVariant(root, "np1")
    state = simulation.conditionCell(reference, root / "np1" / "conditioning")
    jobs, rows = {}, []
    for count in processCounts:
        config = slabVariant(root, f"np{count}")
        jobs[count], seconds = runSlab(config, root / f"np{count}", count, state)
        rows.append(dict(processes=int(count), **seconds))
    base = jobs[processCounts[0]]
    baseVm = readIgb(base / "vm.igb")
    baseActivation = firstActivationMs(base, baseVm.shape[1])
    for row, count in zip(rows, processCounts):
        vm = readIgb(jobs[count] / "vm.igb")
        activation = firstActivationMs(jobs[count], vm.shape[1])
        row.update(largestVmDifferenceMv=float(np.max(np.abs(vm - baseVm))),
                   largestActivationDifferenceMs=float(np.nanmax(np.abs(activation - baseActivation))),
                   sameNodesActivated=bool(np.array_equal(np.isfinite(activation), np.isfinite(baseActivation))),
                   speedUp=float(rows[0]["physicsSeconds"] / row["physicsSeconds"]))
        row["parallelEfficiency"] = float(row["speedUp"] * processCounts[0] / count)
    return dict(
        supports="Simulations are run in parallel with MPI (openCARP / PETSc domain decomposition); parallel results equal the serial result.",
        kind="simulation",
        method=f"Intact {SMALL_SLAB_MM[0]:g} x {SMALL_SLAB_MM[1]:g} mm slab at the production mesh spacing, {SMALL_SLAB_DURATION_MS:g} ms, identical inputs; "
               "Vm at every tissue node and output frame and first activation times compared with the first process count listed. "
               "wallSeconds is the whole run (mesher, mesh rewrite, carputils and openCARP set-up, solve) and each process count "
               "meshes its own copy; physicsSeconds is openCARP's own 'Timings of individual physics' total. Speed-up and efficiency "
               "use physicsSeconds. On this small problem they are indicative only.",
        values=dict(tissueNodes=int(baseVm.shape[1]), outputFrames=int(baseVm.shape[0]), rows=rows,
                    project_tolerances=dict(activationMs=0.02, vmMv=0.1, source="docs/decision_log.md DL-2026-10-28-05")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPOSITORY / "runs" / "verification" / "opencarp"))
    parser.add_argument("--np", type=int, default=2, help="MPI processes for each run")
    parser.add_argument("--np-list", type=int, nargs="*", help="process counts of parallelEquivalence (default: 1 and --np)")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "openCARP_buildinfo.txt").write_text(openCarpBuildInfo() + "\n")
    counts = args.np_list or sorted({1, args.np})
    checks = [("stimulusThreshold", lambda: checkStimulusThreshold(out / "work" / "stimulusThreshold", args.np)),
              ("temporalConvergence", lambda: checkTemporalConvergence(out / "work" / "temporalConvergence", args.np)),
              ("spatialConvergence", lambda: checkSpatialConvergence(out / "work" / "spatialConvergence", args.np)),
              ("conditioning", lambda: checkConditioning(out / "work" / "conditioning", args.np)),
              ("slabVelocity", lambda: checkSlabVelocity(out / "work" / "slabVelocity", args.np)),
              ("parallelEquivalence", lambda: checkParallelEquivalence(out / "work" / "parallelEquivalence", counts))]
    status = runChecks(checks, out, args.only)
    return 0 if all(state == "completed" for state in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
