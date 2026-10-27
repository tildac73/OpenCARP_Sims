# Step 4 of the Part 1 refinement (27 Oct 2026): Bayesian refits with longer sampling.
#
# Identical to 3D_Code/pilot_change_points.R except:
#   * sampling: 15000 iterations after 5000 adaptation (was 3000 / 1500)
#   * output goes to a folder you name, so the original run is never modified
#
# Same model (mcp(list(PP_log ~ 1 + x, ~ 0))), default priors, 3 chains, seed 3.
# Profiles: clean (noiseRepeat 0), clinical preprocessing, recording anchor,
# AIC-selected decay form, both conditions and both standoffs.
#
# Run from the Masters_V2 folder on the Mac that has mcp + JAGS:
#   Rscript _refinement_work/bayes_refit.R _refinement_work/run_w25_100 _refinement_work/bayes_w25_100
#   Rscript _refinement_work/bayes_refit.R "boundaryPilotV2 2"        _refinement_work/bayes_original_long
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 2) stop("Usage: Rscript bayes_refit.R <run folder> <output folder>")
packages <- c("mcp", "ggdist")
missingPackages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missingPackages)) stop("Install these packages first: ", paste(missingPackages, collapse = ", "),
                                ". mcp also requires JAGS.")
folder <- file.path(normalizePath(arguments[1]), "results", "clinical")
outputFolder <- arguments[2]
dir.create(outputFolder, showWarnings = FALSE, recursive = TRUE)
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
chainCount <- 3
iterationCount <- 15000
adaptationCount <- 5000
randomSeed <- 3
summaries <- list()
started <- Sys.time()
for (index in seq_len(nrow(selection))) {
    specification <- selection[index, ]
    data <- subset(fitData, profileId == specification$profileId)
    data$x <- if (specification$model == "Power-law") log(data$Dist) else data$Dist
    data <- data.frame(PP_log = data$PP_log, x = data$x)
    message("Bayesian fit ", index, "/", nrow(selection), ": ", specification$profileId, " ",
            specification$condition, " ", specification$label, " gap=", specification$gapMm)
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
        maximumRhat = max(parameterSummary$Rhat), minimumEss = min(parameterSummary$n.eff),
        samplerConvergencePassed = convergencePassed,
        pythonDescriptiveStatus = diagnostic$fitStatus,
        pythonChangePointMm = diagnostic$changePointMm,
        eligibleForInterpretation = convergencePassed && isTRUE(diagnostic$eligibleForInterpretation)
    )
}
if (length(summaries)) write.csv(do.call(rbind, summaries), file.path(outputFolder, "bayesianChangePoints.csv"),
                               row.names = FALSE)
writeLines(c(
    paste("Source run:", normalizePath(arguments[1])),
    paste("Elapsed minutes:", round(as.numeric(difftime(Sys.time(), started, units = "mins")), 1)),
    "Same mcp linear-decay/zero-slope-plateau specification and default priors as the clinical Rmd.",
    "Sampling extended to 15000 iterations after 5000 adaptation (original 3000/1500); 3 chains; seed 3.",
    "Conditional model-based credible intervals for one simulated slab, not population uncertainty.",
    capture.output(sessionInfo())
), file.path(outputFolder, "methodAndSession.txt"))
message("Done: ", file.path(outputFolder, "bayesianChangePoints.csv"))
