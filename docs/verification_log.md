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

## VL-2026-10-28-01 - compare_runs.py behaves as intended (synthetic)
- **Type:** Unit tests (synthetic run folders). **Result:** Pass, 6/6 (`tests/test_compare_runs.py`).
- **Covers:** identical runs pass; differences inside proposed tolerance pass; Vm offset 0.5 mV, activation shift 0.1 ms, phi_e amplitude +1 % and mismatched inputs each fail.

## VL-2026-10-28-02 - Step 2 gate: electrode signals synthesised from saved Vm reproduce openCARP recovery
- **Type:** Numerical (independent code path vs openCARP). **Result:** Pass (proposed gate: correlation >= 0.999 at every point).
- **How:** `3D_Code/numerics/verify_synthesis.py` on boundaryPilotV2, all 4,304 recovery points x 2 conditions, frames 7500-7600 ms (beat 16 incl. stimulus far-field and local activation). Monodomain source, sigma_b = 1 S/m.
- **Result:** intact corr min 0.99978, rel RMS median 0.49 % / max 2.1 %, amplitude ratio 0.971-1.014 (median 1.005); lesion corr min 0.99947, rel RMS median 0.34 % / max 3.2 %, amplitude ratio 0.956-1.016 (median 1.002).
- **Height dependence:** worst points all 0.25-0.5 mm above tissue (rel RMS <= 3.2 %); above 1 mm rel RMS <= 0.5 %. Consistent with one-point element quadrature error near the surface (target of Step 4). Which code is closer to the exact integral is not established here.
- **Control:** intracellular source gives identical correlation but amplitude ratio 1.24-1.30 (= 1/0.782), so the amplitude test discriminates the source conductivity.
- Results: `_verification/boundaryPilotV2_2/`.

## VL-2026-10-28-03 - electrode_synthesis unit tests
- **Type:** Implementation check. **Result:** Pass, 5/5 (`tests/test_electrode_synthesis.py`).
- **Covers:** series conductivity; single-tetrahedron uniform-gradient potential equals the closed-form dipole expression (implementation check, not discretisation accuracy); source-conductivity scaling; electrode averaging; input validation.

## VL-2026-10-28-04 - Regression after the source-conductivity change
- **Type:** Reproduction. **Result:** Pass, 5/5 incl. FOV80 (`EFOV_RUN_SLOW=1 python3 -m unittest tests.test_v2_reference`).

## Pending - Step 2 MPI equivalence
- Awaiting workstation runs (`3D_Code/numerics/README.md`).
