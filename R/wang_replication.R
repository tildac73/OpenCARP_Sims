# Wang's EFOV analysis (reference/EFOV_analysis_pipeline.Rmd) run on one simulated dataset (FM-12).
#
# Usage: Rscript R/wang_replication.R <dataset folder> <output folder>
#   <dataset folder> holds merged_rectangle_summary_euclidean.xlsx (lesion run) and
#   merged_rectangle_summary_euclidean_pre.xlsx (intact run), written by efov/export.py.
#
# What is Wang's and what is ours
#   * Her Rmd is read unedited; the run stops if its MD5 differs from the recorded value.
#   * Her chunks are executed in order, as written, in one R session. Her constants (detection limit,
#     ROI, trimming, mcp chains/iterations/adaptation/seed) are hers.
#   * After her 'model-functions' chunk, six functions are replaced, because one simulated slab has
#     one "patient" and a random intercept cannot be estimated:
#       decay_formula        no "+ (1 | patientID)" term
#       select_decay_model   lm (OLS = ML for fixed effects) instead of lmer(REML = FALSE); AIC from lm
#       fit_decay_reml       lm instead of lmer(REML = TRUE) (identical fixed-effect estimates for OLS)
#       tidy_decay           Wald CI from lm; R2m <- R^2 of lm, R2c <- NA (no random effect)
#       predict_decay        coef / vcov of lm
#       drop_unfittable      without the ">= 2 patients" rule
#     Her mcp piecewise fit (fit_piecewise) has no patient term and runs unchanged.
#   * Chunks that cannot apply to one simulated slab are skipped, with the reason recorded:
#       random-slope         needs patients
#       icc-bland-altman     manual vs automated distances do not exist in silico
#       sens-geodesic        flat slab: geodesic = Euclidean distance
#       voltage-comparisons  only when both regions are present (Mann-Whitney inferior vs roof)
#   * chunkStatus.csv records every chunk: ran, skipped (reason) or failed (message). A failure in a
#     core chunk stops the run; others are recorded and the run continues.
# Interpretation: the points of one slab are spatially correlated, so her p-values and credible
# intervals describe this simulated dataset, not population uncertainty.

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 2) stop("Usage: Rscript R/wang_replication.R <dataset folder> <output folder>")
dataDir <- normalizePath(arguments[1], mustWork = TRUE)
outDir <- arguments[2]
dir.create(outDir, recursive = TRUE, showWarnings = FALSE)
outDir <- normalizePath(outDir)

scriptFile <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
repository <- normalizePath(file.path(dirname(scriptFile), ".."))
rmd <- file.path(repository, "reference", "EFOV_analysis_pipeline.Rmd")
EXPECTED_MD5 <- "c739f13e6d07996b81d6f733f04f503d"
if (unname(tools::md5sum(rmd)) != EXPECTED_MD5) stop("reference/EFOV_analysis_pipeline.Rmd differs from Wang's recorded version")

# ---- her chunks, by label -------------------------------------------------------
lines <- readLines(rmd, warn = FALSE)
starts <- grep("^```\\{r", lines)
ends <- grep("^```\\s*$", lines)
chunks <- list()
for (s in starts) {
  e <- min(ends[ends > s])
  label <- sub("^```\\{r ([^,}]+).*$", "\\1", lines[s])
  chunks[[label]] <- if (e > s + 1) lines[(s + 1):(e - 1)] else character()
}

SKIP <- c("random-slope" = "needs patients (random slopes)",
          "icc-bland-altman" = "manual vs automated distances do not exist in silico",
          "sens-geodesic" = "flat slab: geodesic distance equals Euclidean distance")
CORE <- c("packages", "config", "helpers", "model-functions", "data", "primary-run")

# ---- replacements for one slab (applied after 'model-functions') ----------------
oneSlabOverrides <- function() {
  decay_formula <<- function(outcome, model, random = NULL) {
    as.formula(paste(outcome, "~", decay_term(model)))
  }
  select_decay_model <<- function(d, outcome) {
    d <- filter(d, Dist > 0)
    map_dfr(DECAY_MODELS, function(m) {
      fit <- lm(decay_formula(outcome, m), data = d)
      tibble(model = m, n = nobs(fit), AIC = AIC(fit), BIC = BIC(fit))
    }) %>%
      mutate(dAIC = AIC - min(AIC), best = dAIC == 0)
  }
  fit_decay_reml <<- function(d, outcome, model) {
    if (model == "Power-law") d <- filter(d, Dist > 0)
    lm(decay_formula(outcome, model), data = d)
  }
  tidy_decay <<- function(fit, model) {
    term <- decay_term(model)
    ci <- confint.default(fit, parm = c("(Intercept)", term))
    b <- coef(fit)
    tibble(
      model = model, n = nobs(fit),
      b0 = b[[1]], b0_lower = ci[1, 1], b0_upper = ci[1, 2],
      b1 = b[[term]], b1_lower = ci[2, 1], b1_upper = ci[2, 2],
      p_b1 = summary(fit)$coefficients[term, "Pr(>|t|)"],
      A = exp(b[[1]]),
      R2m = summary(fit)$r.squared, R2c = NA_real_,
      RMSE = performance::rmse(fit)
    )
  }
  predict_decay <<- function(fit, model, dist) {
    X <- cbind(1, if (model == "Power-law") log(dist) else dist)
    eta <- drop(X %*% coef(fit))
    se <- sqrt(rowSums((X %*% as.matrix(vcov(fit))) * X))
    tibble(Dist = dist, value = exp(eta), lo = exp(eta - 1.96 * se), hi = exp(eta + 1.96 * se))
  }
  drop_unfittable <<- function(specs) {
    ok <- map_lgl(specs$data, function(d) {
      d <- filter(d, Dist > 0)
      nrow(d) >= MIN_FIT_POINTS && n_distinct(d$Dist) >= 3
    })
    if (any(!ok)) cat("Skipping decay models with too little data:",
                      paste(specs$Location[!ok], specs$Metric[!ok], specs$Electrode[!ok], collapse = "; "), "\n")
    specs[ok, ]
  }
}

# ---- run ---------------------------------------------------------------------------
Sys.setenv(EFOV_DATA_DIR = dataDir, EFOV_OUT_DIR = outDir)
grDevices::pdf(file.path(outDir, "console_plots.pdf"))
status <- data.frame(chunk = character(), status = character(), detail = character(), stringsAsFactors = FALSE)
for (label in names(chunks)) {
  if (label %in% names(SKIP)) {
    status[nrow(status) + 1, ] <- c(label, "skipped", SKIP[[label]]); next
  }
  if (label == "voltage-comparisons" && length(unique(as.character(ds$all$Location))) < 2) {
    status[nrow(status) + 1, ] <- c(label, "skipped", "needs both regions (inferior and roof)"); next
  }
  message("chunk: ", label)
  result <- tryCatch({
    eval(parse(text = chunks[[label]]), envir = globalenv())
    if (label == "model-functions") oneSlabOverrides()
    c(label, "ran", if (label == "model-functions") "six one-slab replacements applied after this chunk" else "")
  }, error = function(e) c(label, "failed", conditionMessage(e)))
  status[nrow(status) + 1, ] <- result
  if (result[2] == "failed" && label %in% CORE) {
    write.csv(status, file.path(outDir, "chunkStatus.csv"), row.names = FALSE)
    stop("Core chunk '", label, "' failed: ", result[3])
  }
}
invisible(grDevices::dev.off())
write.csv(status, file.path(outDir, "chunkStatus.csv"), row.names = FALSE)
writeLines(c(paste("Wang Rmd MD5:", EXPECTED_MD5), paste("Dataset:", dataDir),
             "One simulated slab: lm replaces lmer (six functions, listed in R/wang_replication.R); mcp unchanged.",
             capture.output(sessionInfo())), file.path(outDir, "methodAndSession.txt"))
cat("Done:", sum(status$status == "ran"), "chunks ran,", sum(status$status == "skipped"), "skipped,",
    sum(status$status == "failed"), "failed\n")
