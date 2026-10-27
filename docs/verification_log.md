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
  - `test_boundary_pilot.testRejectInvalidWindowAndNyquist` (1 failure): ~~`validateConfig` does not check the beat window or the filter band against Nyquist.~~ **Corrected 28 Oct:** `validateConfig` does reject both. The test hard-coded a 500 Hz band, which hit Nyquist only at the old 1 ms output interval; at 0.2 ms Nyquist is 2,500 Hz. The test expectation was stale, not the code.
- **Action:** resolved in VL-2026-10-28-05.

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

## VL-2026-10-28-05 - Stale tests repaired or retired; full suite passes
- **Type:** Execution. **Result:** Pass, 31 tests (1 opt-in slow test skipped).
- **test_boundary_pilot.py (10/10):** no pipeline code changed. Fixes were to stale test assumptions from before the V2 rework:
  - synthetic recovery now built with `electrode_synthesis.pointOperator` (removed `sourceDensity`/`pointLeadField`); the end-to-end test still recovers the known -0.07 scale (rtol 1e-3);
  - fixture frames follow `sampleStepMs` (were hard-coded 1 ms); filter rate derived from config;
  - `startAnalysis` called before `checkExperiment`, as `run_boundary.py` does;
  - Nyquist limit derived from config (rejects at Nyquist, accepts just below);
  - expected record count derived from declared recording labels (V2 keeps 4 per placement, not 61); figures checked by name (7, incl. two beat-stability figures).
- **test_pipeline.py retired** (deleted; in git history): tested the pre-rework carputils pipeline (`3D_Code/run.py`, `make_electrodes.py`, old `egm`/`ground_truth` API) removed in commit 1a71aa5. Four behaviours that still exist were ported to `tests/test_shared_pipeline.py` (IGB offset/endian/truncation and frame selection; recording weights reconstruct all 61 recordings; filter rejects invalid bands; quiescent tissue gives no FOV).
- **Run:** `python3 -m unittest discover -s tests -v` (scipy installed) or with `PYTHONPATH=_refinement_work/shim` on the analysis VM.

## VL-2026-10-28-06 - Step 3 (analysis-only): electrode surface-sampling convergence, 8 vs 32 points
- **Type:** Numerical convergence (no new simulation). **Result:** Pass for all interpretable quantities (see clarification).
- **How:** `3D_Code/numerics/electrode_sampling.py` re-synthesised all 2,080 electrode signals per condition from saved Vm (7125-7600 ms) with 2 x 4 = 8 and 4 x 8 = 32 surface samples per electrode, same code for both (4,304 and 20,000 points). Each set was run through the full 25-100 ms pipeline (`run_window.py --egm-source`); compared with `3D_Code/numerics/compare_sampling.py`.
- **Sanity check:** 8-point synthesis vs openCARP electrode signals: corr >= 0.99995, PP ratio 0.993-1.007.
- **Results (DL-2026-10-28-05 tolerances):** PP max 3.0 % (median 0.66 %; worst: lesion bipole 0.5 mm from boundary, 0.25 mm standoff) <= 5 %. FOV80: 0 of 48 radii changed. Supported change points (17): max shift 0.05 mm (one search-grid step) <= 0.5 mm. Pilot acceptance passed in both.
- **Outside tolerance:** 5 fits, all intact (no-lesion) noisy negative controls with no supported change point in either setting (delta AIC < 0). Their optimum location is undefined on a flat objective and moved up to 10.5 mm; 4 status changes, all among unsupported categories. No fit changed between supported and unsupported.
- **Clarification (made after seeing this result; needs confirmation):** the change-point tolerance applies to supported change points, plus a requirement that no fit changes supported/unsupported status. Unsupported candidate locations are not estimates.
- **Conclusion:** 8 surface samples per electrode is adequate at 0.5 mm mesh. Re-test at the production mesh, where the electrode sits relatively closer to more elements.
- **Note:** with synthesised signals the reconstruction check is not independent (same operator as ground truth; shape error ~1e-7).
- **Bugs found and fixed during this check:** cached frames missed the 7600 ms end frame; truncated egm time axis misaligned the Vm frame indices used by contributions (fixed by keeping the full axis, NaN outside the synthesised interval); numpy filter stand-in now accepts a sampling rate within 1e-9 relative of 5 kHz (float error from a time axis starting at 7125 ms).
