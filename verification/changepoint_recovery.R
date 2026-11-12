# Methodology verification, part 4: what does the clinical change-point estimator return on profiles
# whose generating process is known?
#
#   Rscript verification/changepoint_recovery.R <output folder> [replicates]
#
# The estimator is Wang et al.'s Bayesian piecewise regression, called exactly as in
# reference/EFOV_analysis_pipeline.Rmd (chunk 'model-functions', fit_piecewise):
#     mcp(list(y ~ 1 + x, ~ 0), data, par_x = "x", chains = 3, iter = 3000, adapt = 1500, cores = 3)
# with set.seed(3) before each fit and the package's default priors. y is the log of the outcome and
# x the distance (exponential form). Nothing here uses the simulated slab; the profiles are synthetic.
#
# Scenarios (x = distance into the scar, 0-15 mm as Wang's region of interest, n points per profile)
#   A  'plateau'    The estimator's own model is true: log y falls linearly at rate k to a change point,
#                   then is flat. Tests bias and 95 % interval coverage at Wang's residual scale.
#   B  'noiseFloor' A physically motivated alternative with NO structural break: an exponentially decaying
#                   signal added in quadrature to a constant floor, y = sqrt((A exp(-k x))^2 + n^2).
#                   The decay meets the floor at x* = ln(A / n) / k. Halving n (or doubling A) moves x* by
#                   ln(2) / k. Tests whether the fitted change point tracks x*, i.e. the signal-to-floor
#                   ratio, as docs/decision_log.md FM-38 predicts.
#   C  'noPlateau'  Pure exponential decay over the whole range. Tests what the estimator reports when
#                   there is no plateau to find.
# Multiplicative log-normal scatter with SD `sigma` (log units) is applied in every scenario. Wang's
# fitted residual SD was 0.83-0.91 (bipolar, omnipolar; Supplemental Table S1); 0.3 is a low-scatter case.
#
# Output: changepoint_recovery.csv (one row per fit) and changepoint_recovery_summary.csv.
# These are properties of the estimator on stated synthetic data. They are not results of the model.

suppressPackageStartupMessages(library(mcp))

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1) stop("Usage: Rscript verification/changepoint_recovery.R <output folder> [replicates]")
outDir <- arguments[1]
replicates <- if (length(arguments) >= 2) as.integer(arguments[2]) else 5L
dir.create(outDir, recursive = TRUE, showWarnings = FALSE)

MCP_CHAINS <- 3; MCP_ITER <- 3000; MCP_ADAPT <- 1500; MCP_SEED <- 3      # Wang's settings (her Rmd, 'setup' chunk)
ROI_SCAR <- 15                                                           # mm
N_POINTS <- 1500                                                         # of the order of one region x configuration in Wang's data
DECAY_RATE <- 0.25                                                       # per mm; Wang's inferior-wall bipolar piecewise slope (Table S1)

fitChangePoint <- function(d) {
  set.seed(MCP_SEED)
  fit <- mcp(list(y ~ 1 + x, ~ 0), data = d, par_x = "x",
             chains = MCP_CHAINS, iter = MCP_ITER, adapt = MCP_ADAPT, cores = MCP_CHAINS)
  s <- summary(fit, width = 0.95)
  row <- function(name) s[s$name == name, ]
  data.frame(cp = row("cp_1")$mean, cpLower = row("cp_1")$lower, cpUpper = row("cp_1")$upper,
             cpRhat = row("cp_1")$Rhat, cpEss = row("cp_1")$n.eff,
             slope = row("x_1")$mean, intercept = row("int_1")$mean, sigmaFit = row("sigma_1")$mean)
}

simulate <- function(scenario, parameter, sigma, seed) {
  set.seed(seed)
  x <- runif(N_POINTS, 0, ROI_SCAR)
  mean_log <- switch(scenario,
    plateau    = -DECAY_RATE * pmin(x, parameter),                                   # parameter = true change point (mm)
    noiseFloor = 0.5 * log(exp(-2 * DECAY_RATE * x) + (1 / parameter)^2),            # parameter = A / n
    noPlateau  = -parameter * x)                                                     # parameter = decay rate (per mm)
  data.frame(x = x, y = mean_log + rnorm(N_POINTS, 0, sigma))
}

design <- rbind(
  expand.grid(scenario = "plateau", parameter = c(6, 10, 13), sigma = c(0.3, 0.9), stringsAsFactors = FALSE),
  expand.grid(scenario = "noiseFloor", parameter = c(5, 10, 20, 40), sigma = 0.9, stringsAsFactors = FALSE),
  expand.grid(scenario = "noPlateau", parameter = c(0.10, 0.25), sigma = 0.9, stringsAsFactors = FALSE))

rows <- list()
for (i in seq_len(nrow(design))) {
  for (r in seq_len(replicates)) {
    case <- design[i, ]
    seed <- 1000 * i + r
    started <- Sys.time()
    fit <- tryCatch(fitChangePoint(simulate(case$scenario, case$parameter, case$sigma, seed)),
                    error = function(e) data.frame(cp = NA, cpLower = NA, cpUpper = NA, cpRhat = NA, cpEss = NA,
                                                   slope = NA, intercept = NA, sigmaFit = NA, error = conditionMessage(e)))
    if (is.null(fit$error)) fit$error <- ""
    expected <- switch(case$scenario, plateau = case$parameter, noiseFloor = log(case$parameter) / DECAY_RATE, noPlateau = NA)
    rows[[length(rows) + 1]] <- cbind(case, replicate = r, dataSeed = seed, expectedMm = expected, fit,
                                      seconds = as.numeric(difftime(Sys.time(), started, units = "secs")))
    cat(sprintf("%s parameter %g sigma %g replicate %d: cp %.2f (expected %s)\n", case$scenario, case$parameter, case$sigma, r,
                fit$cp, ifelse(is.na(expected), "none", sprintf("%.2f", expected))))
    write.csv(do.call(rbind, rows), file.path(outDir, "changepoint_recovery.csv"), row.names = FALSE)   # kept current: a stopped run keeps its fits
  }
}
fits <- do.call(rbind, rows)
summaryRows <- lapply(split(fits, list(fits$scenario, fits$parameter, fits$sigma), drop = TRUE), function(g) {
  data.frame(scenario = g$scenario[1], parameter = g$parameter[1], sigma = g$sigma[1], fits = sum(!is.na(g$cp)),
             expectedMm = g$expectedMm[1], cpMean = mean(g$cp, na.rm = TRUE), cpMin = min(g$cp, na.rm = TRUE), cpMax = max(g$cp, na.rm = TRUE),
             biasMm = mean(g$cp - g$expectedMm, na.rm = TRUE),
             intervalWidthMean = mean(g$cpUpper - g$cpLower, na.rm = TRUE),
             coverage = mean(g$cpLower <= g$expectedMm & g$expectedMm <= g$cpUpper, na.rm = TRUE),
             rhatMax = max(g$cpRhat, na.rm = TRUE), essMin = min(g$cpEss, na.rm = TRUE), slopeMean = mean(g$slope, na.rm = TRUE))
})
write.csv(do.call(rbind, summaryRows), file.path(outDir, "changepoint_recovery_summary.csv"), row.names = FALSE)
writeLines(c(capture.output(sessionInfo()), paste("JAGS", rjags::jags.version())), file.path(outDir, "sessionInfo.txt"))
