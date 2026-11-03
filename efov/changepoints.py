"""Clinical-style voltage decay-to-plateau change points (reference/EFOV_analysis_pipeline.Rmd), one slab.

For each profile (condition x gap x noise realisation x recording, a distance convention and a
preprocessing variant):
  * keep lesion-side points 0 < d < lesionRoiMm with PP > detectionLimitMv; optionally drop values
    above the trimQuantile (pooled over recording types within the scenario, as the Rmd does);
  * one PP per distance (median), log PP;
  * exponential (x = d) vs power law (x = log d) chosen by AIC of an unbroken line;
  * for each, log PP = a + b min(x, x_cp) over a grid of change points (stepMm), least squares;
  * "supportedDescriptiveChangePoint" if the best fit decays (b < 0), is not at the grid edge, and
    improves AIC over both the line and a constant by >= minimumImprovementAic.

Differences from the clinical analysis: one slab, so ordinary least squares replaces patient random
effects, and the estimator is profile least squares, not the Bayesian mcp posterior
(R/bayesian_change_points.R fits mcp to the same prepared data). Every profile is kept in the
output, including those too short to fit (fitStatus says why).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .io import saveJson


def fitLine(x, y):
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    return coefficients, float(residual @ residual)


def gaussianAic(residualSum, count, parameters):
    """parameters includes the residual variance."""
    return float(count * (np.log(2 * np.pi * max(residualSum / count, np.finfo(float).tiny)) + 1) + 2 * parameters)


def fitProfile(distanceMm, ppMv, settings):
    """Model selection rows, the best fit per model, and its prediction curve."""
    distance, logPp = np.asarray(distanceMm, float), np.log(np.asarray(ppMv, float))
    count, minimum = len(distance), settings["minimumSegmentPoints"]
    if count < settings["minimumFitPoints"] or len(np.unique(distance)) < 2 * minimum + 1:
        return [], [dict(fitStatus="insufficientDetectedDistances", count=count, selected=True)], []
    selection, fits, predictions = [], [], []
    for model in ("Exponential", "Power-law"):
        transform = np.log if model == "Power-law" else np.asarray
        x = transform(distance)
        (intercept, slope), residual = fitLine(x, logPp)
        lineAic = gaussianAic(residual, count, 3)
        constantAic = gaussianAic(np.sum((logPp - logPp.mean()) ** 2), count, 2)
        selection.append(dict(model=model, count=count, AIC=lineAic, intercept=intercept, slope=slope))
        step = settings["stepMm"]
        candidates = []
        for cp in np.arange(np.ceil(distance.min() / step) * step, distance.max(), step):
            left = distance <= cp
            if left.sum() < minimum or (~left).sum() < minimum:
                continue
            (a, b), r = fitLine(np.minimum(x, transform(cp)), logPp)
            candidates.append(dict(changePointMm=float(cp), intercept=float(a), slope=float(b), residualSum=r,
                                   AIC=gaussianAic(r, count, 4), leftCount=int(left.sum()), plateauCount=int((~left).sum())))
        if not candidates:
            fits.append(dict(model=model, count=count, fitStatus="insufficientSegmentCoverage"))
            continue
        best = min(candidates, key=lambda row: row["AIC"])
        improvement = min(lineAic, constantAic) - best["AIC"]
        atEdge = best is candidates[0] or best is candidates[-1]
        supported = bool(best["slope"] < 0 and not atEdge and improvement >= settings["minimumImprovementAic"])
        status = ("supportedDescriptiveChangePoint" if supported else "nonDecayingProfile" if best["slope"] >= 0
                  else "changePointAtSearchEdge" if atEdge else "noSupportedChangePoint")
        fits.append(dict(model=model, **best, count=count, fitStatus=status, descriptiveSupport=supported,
                         improvementOverNoChangeAic=float(improvement),
                         plateauMv=float(np.exp(best["intercept"] + best["slope"] * transform(best["changePointMm"]))),
                         minimumDistanceMm=float(distance.min()), maximumDistanceMm=float(distance.max())))
        grid = np.linspace(distance.min(), distance.max(), 200)
        curve = np.exp(best["intercept"] + best["slope"] * np.minimum(transform(grid), transform(best["changePointMm"])))
        predictions += [dict(model=model, distanceMm=float(d), predictedMv=float(v)) for d, v in zip(grid, curve)]
    chosen = min(selection, key=lambda row: row["AIC"])["model"]
    for row in selection + fits:
        row["selected"] = row.get("model") == chosen
    return selection, fits, predictions


def runChangePoints(config, points, folder, prerequisitesPassed):
    """Fit every profile; write tables. eligibleForInterpretation needs a supported lesion fit, all run
    prerequisites passed, and no supported change point in the matched intact control."""
    s = config["changePoints"]
    folder = Path(folder)
    table = points.assign(PP=points.ppMv)
    selections, fits, predictions, fitData = [], [], [], []
    number = 0
    for (condition, gap, repeat), scenario in table.groupby(["condition", "gapMm", "noiseRepeat"]):
        for convention, column in [("recordingAnchor", "distanceInsideMm"), ("cliqueCentroid", "cliqueDistanceInsideMm")]:
            chosen = scenario if convention == "recordingAnchor" else scenario[scenario.kind == "bi"]
            chosen = chosen.assign(Dist=chosen[column])
            roi = chosen[(chosen.Dist > 0) & (chosen.Dist < s["lesionRoiMm"])]
            for preprocessing in ("clinicalTrimmed", "untrimmed", "cleanNoDetectionCut"):
                if preprocessing == "cleanNoDetectionCut" and repeat != 0:
                    continue
                limit = 0.0 if preprocessing == "cleanNoDetectionCut" else s["detectionLimitMv"]
                detected = roi[np.isfinite(roi.PP) & (roi.PP > limit)]
                cutoff = float(detected.PP.quantile(s["trimQuantile"])) if len(detected) else np.nan
                kept = detected[detected.PP <= cutoff] if preprocessing == "clinicalTrimmed" else detected
                for (kind, label), _ in chosen.groupby(["kind", "label"]):
                    number += 1
                    identity = dict(profileId=f"profile{number:04d}", condition=condition, gapMm=float(gap),
                                    noiseRepeat=int(repeat), kind=kind, label=label,
                                    distanceConvention=convention, preprocessing=preprocessing)
                    profile = (kept[(kept.kind == kind) & (kept.label == label)]
                               .groupby("Dist", as_index=False).PP.median().sort_values("Dist"))
                    fitData += [dict(**identity, distanceMm=r.Dist, ppMv=r.PP, logPp=float(np.log(r.PP)))
                                for r in profile.itertuples()]
                    sel, fit, pred = fitProfile(profile.Dist, profile.PP, s)
                    selections += [dict(**identity, **r) for r in sel]
                    fits += [dict(**identity, **r) for r in fit]
                    predictions += [dict(**identity, **r) for r in pred]
    fits = pd.DataFrame(fits)
    keys = ["gapMm", "noiseRepeat", "kind", "label", "distanceConvention", "preprocessing"]
    support = fits.get("descriptiveSupport", pd.Series(False, index=fits.index)).fillna(False).astype(bool)
    fits["descriptiveSupport"] = support
    control = fits[(fits.condition == "intact") & fits.selected.astype(bool)][keys + ["descriptiveSupport"]]
    control = control.rename(columns={"descriptiveSupport": "intactControlSupportsChangePoint"})
    fits = fits.merge(control, on=keys, how="left", validate="many_to_one")
    fits["eligibleForInterpretation"] = ((fits.condition == "lesion") & fits.selected.astype(bool) & fits.descriptiveSupport
                                         & fits.intactControlSupportsChangePoint.notna()
                                         & ~fits.intactControlSupportsChangePoint.fillna(True).astype(bool)
                                         & bool(prerequisitesPassed))
    for name, rows in [("modelSelection", selections), ("fits", fits), ("predictions", predictions), ("fitData", fitData)]:
        pd.DataFrame(rows).to_csv(folder / f"{name}.csv", index=False)
    saveJson(folder / "method.json", dict(
        source="reference/EFOV_analysis_pipeline.Rmd (make_datasets, select_decay_model, fit_piecewise)",
        settings={k: v for k, v in s.items() if not k.startswith("_")},
        distance="positive into the lesion; zero is the known boundary",
        model="log(PP) = a + b * min(x, x_cp), x = d (exponential) or log d (power law)",
        estimator="profile least squares (Python); Bayesian mcp in R/bayesian_change_points.R",
        meaning="lesion-side decay-to-plateau distance, not boundary localisation or a sensing radius",
        uncertainty="noise repeats measure measurement variation, not biological replication",
        prerequisitesPassed=bool(prerequisitesPassed)))
    return fits


def primaryFits(fits):
    """The headline fits: clean signals, clinical preprocessing, recording-anchor distance, selected model."""
    return fits[(fits.noiseRepeat == 0) & fits.selected.astype(bool) & (fits.preprocessing == "clinicalTrimmed")
                & (fits.distanceConvention == "recordingAnchor")]
