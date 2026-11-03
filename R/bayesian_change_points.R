# Bayesian change points (mcp) for the final model's primary profiles.
# Usage: Rscript R/bayesian_change_points.R <run folder>
#
# Same mcp model as reference/EFOV_analysis_pipeline.Rmd: log PP ~ 1 + x, then a flat plateau (~ 0),
# x = distance (exponential) or log distance (power law, as selected in Python). Default mcp priors.
# Sampling: 3 chains, 5,000 adaptation, 15,000 iterations, seed 3 (DL-2026-10-27-03; the clinical
# 3,000 / 1,500 left some fits below Rhat <= 1.01 / ESS >= 400).
# Reads <run>/changePoints/{fitData,modelSelection,fits}.csv written by efov.changepoints.
# Primary profiles only: clean signals, clinical preprocessing, recording-anchor distance.
# These are model-based credible intervals for one simulated slab, not population uncertainty.
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1) stop("Supply the run folder")
for (package in c("mcp", "ggdist")) {
    if (!requireNamespace(package, quietly = TRUE)) stop("Install ", package, " first (mcp also needs JAGS)")
}
folder <- file.path(normalizePath(arguments[1]), "changePoints")
fitData <- read.csv(file.path(folder, "fitData.csv"), stringsAsFactors = FALSE)
selection <- read.csv(file.path(folder, "modelSelection.csv"), stringsAsFactors = FALSE)
fits <- read.csv(file.path(folder, "fits.csv"), stringsAsFactors = FALSE)
flag <- function(x) tolower(as.character(x)) == "true"
selection <- subset(selection, flag(selected) & noiseRepeat == 0 &
                    preprocessing == "clinicalTrimmed" & distanceConvention == "recordingAnchor")
fits$selected <- flag(fits$selected)
fits$eligibleForInterpretation <- flag(fits$eligibleForInterpretation)
output <- file.path(folder, "bayesian")
dir.create(output, showWarnings = FALSE)

summaries <- list()
for (index in seq_len(nrow(selection))) {
    spec <- selection[index, ]
    data <- subset(fitData, profileId == spec$profileId)
    data <- data.frame(PP_log = data$logPp, x = if (spec$model == "Power-law") log(data$distanceMm) else data$distanceMm)
    message("mcp: ", spec$profileId, " ", spec$condition, " ", spec$label, " gap ", spec$gapMm)
    set.seed(3)
    fit <- mcp::mcp(list(PP_log ~ 1 + x, ~ 0), data = data, par_x = "x",
                    chains = 3, adapt = 5000, iter = 15000, cores = 3)
    saveRDS(fit, file.path(output, paste0(spec$profileId, ".rds")))
    parameters <- NULL
    invisible(capture.output(parameters <- summary(fit)))
    write.csv(parameters, file.path(output, paste0(spec$profileId, "_parameters.csv")), row.names = FALSE)
    draws <- do.call(rbind, lapply(fit$mcmc_post, as.matrix))
    cp <- if (spec$model == "Power-law") exp(draws[, "cp_1"]) else draws[, "cp_1"]
    plateau <- exp(draws[, "int_1"] + draws[, "x_1"] * draws[, "cp_1"])
    cpInterval <- ggdist::mean_hdci(cp, .width = 0.95)
    plateauInterval <- ggdist::mean_hdci(plateau, .width = 0.95)
    python <- subset(fits, profileId == spec$profileId & selected)
    converged <- all(is.finite(parameters$Rhat)) && all(parameters$Rhat <= 1.01) && all(parameters$n.eff >= 400)
    summaries[[index]] <- data.frame(
        spec[, c("profileId", "condition", "gapMm", "kind", "label", "model")],
        cpMeanMm = cpInterval$y, cpLower95Mm = cpInterval$ymin, cpUpper95Mm = cpInterval$ymax,
        plateauMeanMv = plateauInterval$y, plateauLower95Mv = plateauInterval$ymin, plateauUpper95Mv = plateauInterval$ymax,
        probabilityNegativeSlope = mean(draws[, "x_1"] < 0), samplerConverged = converged,
        pythonFitStatus = python$fitStatus,
        eligibleForInterpretation = converged && isTRUE(python$eligibleForInterpretation))
}
if (length(summaries)) write.csv(do.call(rbind, summaries), file.path(output, "bayesianChangePoints.csv"), row.names = FALSE)
writeLines(c("mcp linear decay + zero-slope plateau, default priors; 3 chains, adapt 5000, iter 15000, seed 3.",
             "Model family selected by Python OLS AIC on one slab (no patient random effects).",
             "mcp can fit a change point to a flat intact control: read slope, sampler diagnostics and the Python no-change comparison.",
             capture.output(sessionInfo())), file.path(output, "methodAndSession.txt"))
