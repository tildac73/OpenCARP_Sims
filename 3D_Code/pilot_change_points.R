# Optional Bayesian companion to boundary_clinical.py.
# Usage: Rscript 3D_Code/pilot_change_points.R /path/to/boundaryPilotV2
# Same mcp model/settings as EFOV_analysis_pipeline.Rmd; one slab, no fake patients.
# Primary clean recording-anchor profiles only. Python exports noise and filtering
# sensitivities separately; they must not be pooled into a biological sample.
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1) stop("Supply the pilot run folder")
packages <- c("mcp", "ggdist")
missingPackages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missingPackages)) stop("Install these packages first: ", paste(missingPackages, collapse = ", "),
                                ". mcp also requires JAGS.")
folder <- file.path(normalizePath(arguments[1]), "results", "clinical")
fitData <- read.csv(file.path(folder, "clinicalFitData.csv"), stringsAsFactors = FALSE)
selection <- read.csv(file.path(folder, "changePointModelSelection.csv"), stringsAsFactors = FALSE)
fitStatus <- read.csv(file.path(folder, "changePointFits.csv"), stringsAsFactors = FALSE)
asFlag <- function(x) tolower(as.character(x)) == "true"
selection$selected <- asFlag(selection$selected)
selection$pilotAccepted <- asFlag(selection$pilotAccepted)
fitStatus$selected <- asFlag(fitStatus$selected)
fitStatus$descriptiveSupport <- asFlag(fitStatus$descriptiveSupport)
fitStatus$eligibleForInterpretation <- asFlag(fitStatus$eligibleForInterpretation)
selection <- subset(selection, selected & noiseRepeat == 0 &
                    preprocessing == "clinicalTrimmed" & distanceConvention == "recordingAnchor")
outputFolder <- file.path(folder, "bayesian")
dir.create(outputFolder, showWarnings = FALSE)
chainCount <- 3
iterationCount <- 3000
adaptationCount <- 1500
randomSeed <- 3
summaries <- list()
for (index in seq_len(nrow(selection))) {
    specification <- selection[index, ]
    data <- subset(fitData, profileId == specification$profileId)
    data$x <- if (specification$model == "Power-law") log(data$Dist) else data$Dist
    data <- data.frame(PP_log = data$PP_log, x = data$x)
    message("Bayesian fit: ", specification$profileId, " ", specification$condition,
            " ", specification$label, " gap=", specification$gapMm)
    set.seed(randomSeed)
    fit <- mcp::mcp(list(PP_log ~ 1 + x, ~ 0), data = data, par_x = "x",
                    chains = chainCount, iter = iterationCount, adapt = adaptationCount,
                    cores = chainCount)
    saveRDS(fit, file.path(outputFolder, paste0(specification$profileId, ".rds")))
    parameterSummary <- NULL
    invisible(capture.output(parameterSummary <- summary(fit)))
    write.csv(parameterSummary, file.path(outputFolder, paste0(specification$profileId, "_parameters.csv")),
              row.names = FALSE)
    draws <- do.call(rbind, lapply(fit$mcmc_post, as.matrix))
    cpMm <- if (specification$model == "Power-law") exp(draws[, "cp_1"]) else draws[, "cp_1"]
    plateauMv <- exp(draws[, "int_1"] + draws[, "x_1"] * draws[, "cp_1"])
    cpInterval <- ggdist::mean_hdci(cpMm, .width = 0.95)
    plateauInterval <- ggdist::mean_hdci(plateauMv, .width = 0.95)
    # Preserve both sampler diagnostics and the Python no-change comparison.
    diagnostic <- subset(fitStatus, profileId == specification$profileId & selected)
    finiteDiagnostics <- all(is.finite(parameterSummary$Rhat)) && all(is.finite(parameterSummary$n.eff))
    convergencePassed <- finiteDiagnostics && all(parameterSummary$Rhat <= 1.01) &&
                         all(parameterSummary$n.eff >= 400)
    summaries[[index]] <- data.frame(
        specification[, c("profileId", "condition", "gapMm", "kind", "label", "model", "pilotAccepted")],
        cpMeanMm = cpInterval$y, cpLower95Mm = cpInterval$ymin, cpUpper95Mm = cpInterval$ymax,
        plateauMeanMv = plateauInterval$y, plateauLower95Mv = plateauInterval$ymin,
        plateauUpper95Mv = plateauInterval$ymax,
        probabilityNegativeSlope = mean(draws[, "x_1"] < 0),
        samplerConvergencePassed = convergencePassed,
        pythonDescriptiveStatus = diagnostic$fitStatus,
        eligibleForInterpretation = convergencePassed && isTRUE(diagnostic$eligibleForInterpretation)
    )
}
if (length(summaries)) write.csv(do.call(rbind, summaries), file.path(outputFolder, "bayesianChangePoints.csv"),
                               row.names = FALSE)
writeLines(c(
    "Same mcp linear-decay/zero-slope-plateau specification and defaults as the clinical Rmd.",
    "The Python preparation uses the exact known boundary (offset 0), PP > 0.01 mV, lesion ROI (0,15) mm, and 99.9% trimming.",
    "Model selection uses OLS AIC on one simulated slab, not a patient random-intercept model.",
    "These are conditional model-based credible intervals, not population biological uncertainty.",
    "The Bayesian model can fit a change point even in flat intact controls. Inspect slope, sampler diagnostics and the no-change comparison.",
    "mcp uses its default priors; Python's three-points-per-segment grid restriction is not a Bayesian prior.",
    capture.output(sessionInfo())
), file.path(outputFolder, "methodAndSession.txt"))
