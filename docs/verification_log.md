# Verification log

What was checked, how, and the result. Distinguishes code inspection, execution,
numerical verification and physiological validation. Newest last.

---

## VL-2026-10-27-01 - Original analysis reproduces exactly through the refinement driver
- **Type:** Numerical (reproduction). **Result:** Pass.
- **How:** `_refinement_work/run_window.py` with the 0-400 ms window on the raw `boundaryPilotV2 2` data, compared with the stored 20 Oct outputs.
- **Result:** PP for 4,160 recordings identical (max |diff| 0.0 mV); selection weights identical; 24 intact FOV80 radii, correlations and shape errors identical.

## VL-2026-10-27-02 - numpy stand-ins agree with scipy
- **Type:** Numerical (unit). **Result:** Pass.
- **How:** `_refinement_work/test_shim.py` in an environment with scipy.
- **Result:** filter coefficients equal to 1e-15; `sosfilt` max |diff| 0.0 on (7, 16, 3001) data with an impulse; sparse ring sums max |diff| 1.3e-13.

## VL-2026-10-27-03 - Stimulus transient does not set windowed bipolar/omnipolar PP
- **Type:** Numerical. **Result:** Pass (bipolar, omnipolar); caveat (unipolar).
- **Result:** all bipolar/omnipolar extrema at 45-75 ms; PP identical for window starts 25 and 30 ms. Unipolar: extremum in the first 5 ms of the window beyond 5.5-6 mm; PP changes up to 6.5%.

## VL-2026-10-27-04 - Regression test suite for the v2-pilot results
- **Type:** Reproduction (automated). **Result:** Pass.
- **How:** `python3 -m unittest tests.test_v2_reference -v` (+ `EFOV_RUN_SLOW=1` for contributions). Reference built by `tests/reference/make_v2_reference.py` from stored outputs.
- **Result (27 Oct 2026, analysis VM):** OriginalWindow 2/2 (38 s); DeclaredWindow 2/2 (13 s); DeclaredWindowContributions 1/1 (42 s): PP, weights, 180 change-point fits per window, 48 FOV80 radii all reproduce.

## VL-2026-10-27-05 - Existing test suite: 13 pre-existing failures
- **Type:** Code inspection + execution. **Result:** Fail, not caused by the refinement.
- **Finding:** 25 tests run; 12 errors, 1 failure, 1 skipped. Pipeline code is unchanged from commit a8087c3.
  - `tests/test_pipeline.py` (11 errors, ShellIntegration 1): written for the pre-rework API (17 Oct); calls `3D_Code/run.py` and `ground_truth.source_density`, removed in the 19 Oct rework.
  - `test_boundary_pilot.BoundaryIntegrationTests` (2 errors): `ground_truth.sourceDensity` no longer exists; fixture does not create `results/` before `checkExperiment` writes to it.
  - `test_boundary_pilot.testRejectInvalidWindowAndNyquist` (1 failure): `validateConfig` does not check the beat window or the filter band against Nyquist.
- **Action:** Repair or retire stale tests and add the two missing validations in Step 1/2 work. Until then these tests give no protection.

## VL-2026-10-27-06 - Bayesian refit convergence
- **Type:** Numerical (sampler). **Result:** Pass for lesion fits.
- **Result:** all 16 lesion fits (both windows) Rhat <= 1.008, ESS >= 955; intact 7/8 pass (bipole 0.25 mm Rhat 1.07). Original-window refit reproduces earlier R estimates (omnipolar E5-E6-E9 8.66 mm).

## VL-2026-10-27-07 - Step 1 gate: v2-pilot reproduces from a clean checkout
- **Type:** Reproduction (automated, clean worktree). **Result:** Pass.
- **How:** `git worktree add` at tag `v2-pilot`; only the raw Vm/egm/mesh folders linked in (untracked by design); `EFOV_RUN_SLOW=1 python3 -m unittest tests.test_v2_reference -v`.
- **Result:** 5/5 tests pass (77 s). First attempt found `results/calibration.json` and `results/pilotStatus.json` untracked; both now tracked (commit 27fd8d3, tag moved before any push).
