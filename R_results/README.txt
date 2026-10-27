Clinical R figures for boundaryPilotV2 2

Source run: /Users/matildacotton/Downloads/boundaryPilotV2 2
Original R workflow: /Users/matildacotton/Desktop/Masters_V2/EFOV_analysis_pipeline.Rmd
Simulation code: /Users/matildacotton/Desktop/Masters_V2/3D_Code

All 4160 measuredMv traces were read from results/intactTraces.npz and lesionTraces.npz. Peak-to-peak voltage was recalculated, agreeing with exported ppMv to 5.97e-8 mV. Existing signal filtering, final-beat selection and calibration were preserved. R clinical processing begins with these trace-derived voltages; the original Rmd accepts map measurements, not raw waveform files.

Figures 3, 4 and 5 are produced separately for gaps 0.25 and 1 mm, using the clean noiseRepeat=0 recordings. The two omnipolar cliques are separate series. No noise replicates or electrode positions are labelled as patients. The single slab has one anatomical column; inferior/roof labels are not assigned.

The original make_datasets, fit_piecewise, summarise_piecewise and Figure 3 plot helpers are executed directly from the original Rmd. Figure 5 uses the original plotting chunk, substituting coef(lm) for fixef(lmer). The known boundary uses zero offset; applying the clinical 3 mm offset would shift an already-correct boundary. Detection limit >0.01 mV, strict scar ROI 0<distance<15 mm, 99.9th-percentile trimming, exponential/power-law AIC selection, and Bayesian linear-decay/flat-plateau specification are preserved. Bayesian settings: 3 chains, 3000 iterations, 1500 adaptation, seed 3, default priors.

Necessary single-slab adaptation: ordinary least squares replaces patient random-intercept regression for AIC selection, decay estimates and confidence intervals. Model curves and Bayesian intervals are descriptive conditional fits for this slab, not population-level clinical inference. Figure 4 axis labels follow the actually selected model. Figure 3 keeps original voltage limits, which can clip observations just as in the original.

No peak-frequency measurements are available, so peak-frequency panels are omitted rather than estimated with an unverified surrogate. Manual-vs-automated distance validation Figure S2 is unavailable because corresponding validation pairs were not created. No geodesic sensitivity is claimed. Figure 5 is the clinical change-point visualization, not the separate contribution-radius ground truth.

PDF and original 1200 dpi LZW TIFF outputs plus PNG previews are in each gap folder. Tables contain AIC selection and posterior summaries including Rhat and effective sample sizes. Cached Bayesian models and analysis.rds preserve fitted results. Consult sampler diagnostics before interpreting change points.

Reproduce from /Users/matildacotton/MastersResearch/Data_and_code:
python3 boundaryPilotV2_clinical_R/verify_traces.py
Rscript boundaryPilotV2_clinical_R/run_clinical.R

Sampler diagnostic check: maximum Rhat 1.021482; minimum effective sample size 194 across model parameters. Parameters with Rhat>1.01 or ESS<400 are listed in tables/sampler_diagnostic_flags.csv. Original sampling settings were retained for reproduction; these flags limit interpretation of affected posterior summaries.
