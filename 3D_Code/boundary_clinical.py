"""Clinical-style voltage change points for one controlled simulated slab.

Matches EFOV_analysis_pipeline.Rmd: lesion-positive distance, 0 < d < 15 mm,
PP > 0.01 mV, optional upper 99.9% trimming, log voltage, exponential versus
power-law AIC selection, then a continuous decay followed by a flat plateau.

With one slab, ordinary least squares replaces patient random effects. Python
uses a profile least-squares change point, NOT the Bayesian mcp posterior. The
companion pilot_change_points.R applies mcp to these exact prepared fit data.
Noise realisations and overlapping positions are never treated as patients.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def fitLine(x, y):
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    return coefficients, float(residual @ residual)


def gaussianAic(residualSum, count, parameterCount):
    # The residual variance is a fitted parameter included in parameterCount.
    variance = max(residualSum / count, np.finfo(float).tiny)
    return float(count * (np.log(2 * np.pi * variance) + 1) + 2 * parameterCount)


def analyseProfile(data, settings):
    """Compare both decay forms on identical points, without forcing a breakpoint."""
    distance = data.Dist.to_numpy(float)
    voltageLog = np.log(data.PP.to_numpy(float))
    count = len(data)
    selection, candidates, fits, predictions = [], [], [], []
    minimumSegment = settings["minimumSegmentPoints"]
    if count < settings["minimumFitPoints"] or len(np.unique(distance)) < 2 * minimumSegment + 1:
        return [], [], [dict(fitStatus="insufficientDetectedDistances", count=count)], []
    for model in ["Exponential", "Power-law"]:
        transform = np.log if model == "Power-law" else np.asarray
        x = transform(distance)
        coefficients, residualSum = fitLine(x, voltageLog)
        lineAic = gaussianAic(residualSum, count, 3)
        selection.append(dict(model=model, count=count, AIC=lineAic,
                              intercept=coefficients[0], slope=coefficients[1]))
        constantResidual = np.sum((voltageLog - voltageLog.mean())**2)
        constantAic = gaussianAic(constantResidual, count, 2)
        step = settings["changePointStepMm"]
        grid = np.arange(np.ceil(distance.min() / step) * step, distance.max(), step)
        modelCandidates = []
        for changePoint in grid:
            left = distance <= changePoint
            if left.sum() < minimumSegment or (~left).sum() < minimumSegment:
                continue
            # y = a + b * min(x, cp): continuous, with exactly zero slope after cp.
            changedX = np.minimum(x, transform(changePoint))
            fitted, residual = fitLine(changedX, voltageLog)
            modelCandidates.append(dict(
                model=model, changePointMm=float(changePoint), intercept=float(fitted[0]),
                slope=float(fitted[1]), residualSum=residual,
                AIC=gaussianAic(residual, count, 4), leftCount=int(left.sum()),
                plateauCount=int((~left).sum()),
            ))
        if not modelCandidates:
            fits.append(dict(model=model, count=count, fitStatus="insufficientSegmentCoverage"))
            continue
        best = min(modelCandidates, key=lambda row: row["AIC"])
        improvement = min(lineAic, constantAic) - best["AIC"]
        atEdge = best is modelCandidates[0] or best is modelCandidates[-1]
        supported = bool(best["slope"] < 0 and not atEdge
                         and improvement >= settings.get("minimumImprovementAic", 2.0))
        status = "supportedDescriptiveChangePoint" if supported else (
            "nonDecayingProfile" if best["slope"] >= 0 else
            "changePointAtSearchEdge" if atEdge else "noSupportedChangePoint")
        for candidate in modelCandidates:
            candidate["deltaAicFromBest"] = candidate["AIC"] - best["AIC"]
        candidates.extend(modelCandidates)
        plateau = np.exp(best["intercept"] + best["slope"] * transform(best["changePointMm"]))
        fits.append(dict(
            **best, count=count, fitStatus=status, descriptiveSupport=supported,
            improvementOverNoChangeAic=float(improvement), plateauMv=float(plateau),
            minimumDistanceMm=float(distance.min()), maximumDistanceMm=float(distance.max()),
        ))
        predictionDistance = np.linspace(distance.min(), distance.max(), 200)
        prediction = np.exp(best["intercept"] + best["slope"] * np.minimum(
            transform(predictionDistance), transform(best["changePointMm"])))
        predictions.extend(dict(model=model, Dist=float(d), predictedMv=float(v))
                           for d, v in zip(predictionDistance, prediction))
    selected = min(selection, key=lambda row: row["AIC"])["model"]
    for row in selection:
        row["selected"] = row["model"] == selected
    for row in fits:
        row["selected"] = row.get("model") == selected
    return selection, candidates, fits, predictions


def prepareClinicalPoints(measurements, runFolder, pilotPassed):
    """Export explicit geometry and clinical variables without inventing patient groups."""
    table = measurements.copy()
    table["runId"] = Path(runFolder).name
    table["Electrode"] = table.kind.map({"uni": "Unipolar", "bi": "Bipolar", "omni": "Omnipolar"})
    table["Dist"] = table.distanceInsideMm  # already measured from the known boundary
    table["PP"] = table.ppMv
    table["PP_log"] = np.log(table.PP.where(table.PP > 0))
    table["PFreq"] = np.nan  # no unverified surrogate for the clinical PF measure
    table["pilotAccepted"] = bool(pilotPassed)
    return table


def analyseChangePoints(runFolder, config, measurements, pilotPassed):
    """Always export diagnostic fits, retaining failed prerequisites in every result."""
    settings = config.get("clinical", dict(
        detectionLimitMv=0.01, lesionRoiMm=15.0, healthyRoiMm=10.0,
        trimQuantile=0.999, minimumFitPoints=10, minimumSegmentPoints=3,
        changePointStepMm=0.05, minimumImprovementAic=2.0,
    ))
    folder = Path(runFolder) / "results" / "clinical"
    folder.mkdir(parents=True, exist_ok=True)
    table = prepareClinicalPoints(measurements, runFolder, pilotPassed)
    table.to_csv(folder / "clinicalPoints.csv", index=False)
    selections, candidates, fits, predictions, fitData, flow = [], [], [], [], [], []
    profileNumber = 0
    # Each gap/noise realisation is a separate measurement scenario. Quantile
    # trimming pools recording types within that scenario, as the R notebook does.
    for (condition, gap, repeat), scenario in table.groupby(["condition", "gapMm", "noiseRepeat"]):
        conventions = {"recordingAnchor": "distanceInsideMm", "cliqueCentroid": "cliqueDistanceInsideMm"}
        for convention, distanceColumn in conventions.items():
            selectedScenario = scenario if convention == "recordingAnchor" else scenario[scenario.kind == "bi"]
            if selectedScenario.empty:
                continue
            selectedScenario = selectedScenario.assign(Dist=selectedScenario[distanceColumn])
            roi = selectedScenario[(selectedScenario.Dist > 0) & (selectedScenario.Dist < settings["lesionRoiMm"])]
            for preprocessing in ["clinicalTrimmed", "untrimmed", "cleanNoDetectionCut"]:
                if preprocessing == "cleanNoDetectionCut" and repeat != 0:
                    continue
                detection = 0.0 if preprocessing == "cleanNoDetectionCut" else settings["detectionLimitMv"]
                detected = roi[np.isfinite(roi.PP) & (roi.PP > detection)]
                cutoff = float(detected.PP.quantile(settings["trimQuantile"])) if len(detected) else np.nan
                retained = detected[detected.PP <= cutoff] if preprocessing == "clinicalTrimmed" else detected
                for (kind, label), originalGroup in selectedScenario.groupby(["kind", "label"]):
                    profileNumber += 1
                    identity = dict(profileId=f"profile{profileNumber:04d}", condition=condition,
                                    gapMm=float(gap), noiseRepeat=int(repeat), kind=kind, label=label,
                                    distanceConvention=convention, preprocessing=preprocessing,
                                    pilotAccepted=bool(pilotPassed))
                    group = retained[(retained.kind == kind) & (retained.label == label)]
                    # One value per distance prevents accidental weighting by duplicates.
                    profile = group.groupby("Dist", as_index=False).PP.median().sort_values("Dist")
                    flow.append(dict(**identity, rawCount=len(originalGroup),
                                     roiCount=int(((originalGroup.Dist > 0) & (originalGroup.Dist < settings["lesionRoiMm"])).sum()),
                                     retainedRows=len(group), distinctDistances=len(profile),
                                     detectionLimitMv=detection, trimCutoffMv=cutoff if preprocessing == "clinicalTrimmed" else np.nan))
                    for row in profile.to_dict("records"):
                        fitData.append(dict(**identity, **row, PP_log=float(np.log(row["PP"]))))
                    modelRows, candidateRows, fitRows, predictionRows = analyseProfile(profile, settings)
                    for destination, rows in [(selections, modelRows), (candidates, candidateRows),
                                              (fits, fitRows), (predictions, predictionRows)]:
                        destination.extend(dict(**identity, **row) for row in rows)
    for name, rows in [("changePointModelSelection", selections), ("changePointProfiles", candidates),
                       ("changePointFits", fits), ("changePointPredictions", predictions),
                       ("clinicalFitData", fitData), ("clinicalSampleFlow", flow)]:
        pd.DataFrame(rows).to_csv(folder / f"{name}.csv", index=False)
    fitTable = pd.DataFrame(fits)
    if "selected" in fitTable and "descriptiveSupport" in fitTable:
        controlKeys = ["gapMm", "noiseRepeat", "kind", "label", "distanceConvention", "preprocessing"]
        controls = fitTable[(fitTable.condition == "intact") & (fitTable.selected == True)]
        controls = controls[controlKeys + ["descriptiveSupport"]].rename(
            columns={"descriptiveSupport": "matchedIntactSupportsChangePoint"})
        fitTable = fitTable.merge(controls, on=controlKeys, how="left", validate="many_to_one")
        fitTable["intactControlAvailable"] = fitTable.matchedIntactSupportsChangePoint.notna()
        fitTable["eligibleForInterpretation"] = (
            (fitTable.condition == "lesion") & fitTable.selected.fillna(False)
            & fitTable.descriptiveSupport.fillna(False) & pilotPassed
            & fitTable.intactControlAvailable
            & ~fitTable.matchedIntactSupportsChangePoint.fillna(False))
        fitTable.to_csv(folder / "changePointFits.csv", index=False)
        noisy = fitTable[(fitTable.noiseRepeat > 0) & (fitTable.selected == True)]
        summaryKeys = ["condition", "gapMm", "kind", "label", "distanceConvention", "preprocessing"]
        noiseSummary = noisy.groupby(summaryKeys).agg(
            fitCount=("profileId", "size"), supportedCount=("descriptiveSupport", "sum"),
            meanCandidateCpMm=("changePointMm", "mean"), sdCandidateCpMm=("changePointMm", "std"),
            minimumCandidateCpMm=("changePointMm", "min"), maximumCandidateCpMm=("changePointMm", "max"),
        )
        noiseSummary.to_csv(folder / "changePointNoiseSummary.csv")
        primaryFits = fitTable[(fitTable.noiseRepeat == 0) & (fitTable.selected == True)
                               & (fitTable.preprocessing == "clinicalTrimmed")
                               & (fitTable.distanceConvention == "recordingAnchor")]
        contributionPoints = pd.read_csv(Path(runFolder) / "results" / "points.csv")
        contributionPoints = contributionPoints[(contributionPoints.noiseRepeat == 0)
                                                & (contributionPoints.groundTruthComputed == True)]
        comparison = contributionPoints.merge(
            primaryFits[["condition", "kind", "label", "gapMm", "model", "changePointMm",
                         "fitStatus", "eligibleForInterpretation"]],
            on=["condition", "kind", "label", "gapMm"], how="left", validate="many_to_one",
        )
        comparison.to_csv(folder / "changePointAndContributionComparison.csv", index=False)
        plotChangePoints(folder, fitTable, pd.DataFrame(fitData), pd.DataFrame(predictions))
    metadata = dict(
        sourceMethod="EFOV_analysis_pipeline.Rmd: make_datasets, select_decay_model, fit_piecewise",
        settings=settings, boundaryOffsetMm=0,
        distance="Positive into lesion; zero is the known mesh boundary",
        model="log(PP) = intercept + slope * min(x, cp), x = d or log(d)",
        estimator="Python profile least squares; no Bayesian credible intervals",
        differencesFromClinical="One slab: OLS instead of patient random effects; fixed geometries fitted separately; duplicate distances collapsed",
        changePointMeaning="Lesion-side decay-to-plateau distance, not boundary localisation or a contribution radius",
        uncertainty="Noise repeats measure measurement variation, not biological replication",
        pilotAccepted=bool(pilotPassed),
    )
    (folder / "method.json").write_text(json.dumps(metadata, indent=2) + "\n")


def plotChangePoints(folder, fits, data, predictions):
    import matplotlib.pyplot as plt
    selected = fits[(fits.noiseRepeat == 0) & (fits.selected == True)
                    & (fits.preprocessing == "clinicalTrimmed")
                    & (fits.distanceConvention == "recordingAnchor")]
    if selected.empty:
        return
    combinations = selected[["kind", "label", "gapMm"]].drop_duplicates()
    figure, axes = plt.subplots(len(combinations), 1, figsize=(9, 3 * len(combinations)), squeeze=False)
    for axis, key in zip(axes[:, 0], combinations.to_dict("records")):
        subset = selected[(selected.kind == key["kind"]) & (selected.label == key["label"])
                          & (selected.gapMm == key["gapMm"])]
        for fit in subset.itertuples():
            colour = "tab:orange" if fit.condition == "lesion" else "tab:blue"
            points = data[data.profileId == fit.profileId]
            curve = predictions[(predictions.profileId == fit.profileId) & (predictions.model == fit.model)]
            axis.scatter(points.Dist, points.PP, s=12, color=colour)
            axis.plot(curve.Dist, curve.predictedMv, color=colour,
                      label=f"{fit.condition}: {fit.model}, {fit.fitStatus}")
            if fit.descriptiveSupport:
                axis.axvline(fit.changePointMm, color=colour, ls="--", lw=1)
        axis.set(xlabel="Distance into lesion (mm)", ylabel="Peak-to-peak voltage (mV)",
                 title=f"{key['kind']} {key['label']}, gap {key['gapMm']:g} mm", yscale="log")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "changePointFits.png", dpi=160)
    plt.close(figure)
