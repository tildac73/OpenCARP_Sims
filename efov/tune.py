"""Conduction-velocity tuning of beta at the production mesh (FM-18; method DL-2026-10-28-10).

A strip (fixed.tune.stripSizeMm) is meshed with the production resolution, fibres along x, no bath
and no lesion, and paced in monodomain with model.par (cells, conductivities, stimulus). CV is the
inverse slope of a straight line through the last beat's activation times of all strip nodes with
measureFromMm <= x <= measureToMm. Monodomain Vm depends on sigma / (beta Cm), so CV scales as
beta^(-1/2); each iteration sets beta <- beta (CV / target)^2. Tuning stops within relativeTolerance
of the target or fails after maximumIterations. The CV gate of the model is this stage.
"""
from pathlib import Path

import numpy as np

from .config import cycleLengthMs, par, spacingMm, withParOverrides
from .io import readActivations, saveJson
from .simulation import runCondition, tissueModel

BETA_KEY = "imp_region[0].cellSurfVolRatio"


def stripCv(config, job):
    """CV (m/s) of the last strip beat: least-squares slope of activation time against x."""
    t = config["fixed"]["tune"]
    tissue = tissueModel(job)
    nodes, times = readActivations(Path(job) / "act-thresh.dat")
    beatStart = (t["beats"] - 1) * cycleLengthMs(config)
    position = tissue["tissuePositions"][:, 0]
    first = np.full(len(position), np.nan)
    inBeat = times >= beatStart
    order = np.argsort(times[inBeat])[::-1]                      # earliest event last, so it wins
    first[nodes[inBeat][order]] = times[inBeat][order] - beatStart
    use = (position >= t["measureFromMm"]) & (position <= t["measureToMm"])
    if not np.isfinite(first[use]).all():
        raise RuntimeError("Not every measured strip node activated in the last beat (capture failure?)")
    slope = np.polyfit(position[use], first[use], 1)[0]           # ms / mm
    return float(1.0 / slope)                                     # mm / ms = m / s


def tuneBeta(config, folder, initialState, processCount=1, flavor=None):
    """Iterate beta until the strip CV is within tolerance of values.cvTargetMPerS; returns the record."""
    t, target = config["fixed"]["tune"], float(config["values"]["cvTargetMPerS"])
    beta = par(config, BETA_KEY)
    history = []
    for iteration in range(t["maximumIterations"]):
        trial = withParOverrides(config, {BETA_KEY: beta})
        runCondition(trial, "strip", initialState, Path(folder) / f"iteration{iteration}", processCount, flavor, kind="strip")
        cv = stripCv(trial, Path(folder) / f"iteration{iteration}" / "strip")
        history.append(dict(iteration=iteration, cellSurfVolRatioPerUm=beta, cvMPerS=cv))
        if abs(cv - target) / target <= t["relativeTolerance"]:
            result = dict(cellSurfVolRatioPerUm=beta, cvMPerS=cv, targetMPerS=target, history=history, passed=True,
                          spacingMm=spacingMm(trial).tolist(), method="strip, beta <- beta (CV/target)^2")
            saveJson(Path(folder) / "tuning.json", result)
            return result
        beta = beta * (cv / target) ** 2
    saveJson(Path(folder) / "tuning.json", dict(history=history, passed=False, targetMPerS=target))
    raise RuntimeError(f"CV tuning did not reach {target} m/s within {t['maximumIterations']} iterations: {history}")
