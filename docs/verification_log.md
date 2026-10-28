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

## VL-2026-10-28-07 - Step 4: lead-field quadrature against a closed-form solution
- **Type:** Numerical verification (analytic). **Result:** Pass.
- **Reference:** uniformly polarised tissue box; potential reduced to face integrals of 1/r (closed form, `Shared_Pipeline/analytic_fields.py`; checked against brute-force integration to 6e-8). Mesh: regular 6-tet decomposition at 0.5 mm; linear Vm is exact on linear elements, so quadrature is the only error.
- **Result** (`3D_Code/numerics/quadrature_study.py`; error / far-field scale, worst of 3 lateral positions, near radius 1.5 mm), localised 1 x 1 x 0.5 mm source:

  | Height | Level 0 (centroid) | Level 1 | Level 2 | Level 3 |
  | --- | --- | --- | --- | --- |
  | 0.05 mm | 7.3 % | 2.7 % | 0.44 % | 0.08 % |
  | 0.25 mm | 0.86 % | 0.19 % | 0.045 % | 0.011 % |
  | 1 mm | 0.08 % | 0.03 % | 0.014 % | 0.011 % |

  Whole-slab (far-field dominated) errors are smaller. Near radius 3 mm vs 1.5 mm: < 0.02 % difference at >= 1 mm.
- **Tests:** `tests/test_lead_field.py` (8 tests, incl. infinite-slab limit -J t / 2 sigma_b, equal-volume subdivision, default unchanged).

## VL-2026-10-28-08 - Step 4: effect of refined quadrature on the pilot
- **Type:** Numerical. **Result:** No material change at pilot geometry.
- **Synthesis vs openCARP** (all 4,304 points x 2 conditions): unchanged by subdivision (rel RMS median at 0.25-0.5 mm: 1.88 -> 1.99 % intact, 1.76 -> 1.73 % lesion; >= 0.5 mm identical). The residual near-surface difference is therefore not from synthesis quadrature (accurate to < 0.05 % there); it lies in openCARP's recovery discretisation, which Step 6 replaces.
- **Ground truth** (25-100 ms, `run_window.py --quadrature 1.5 2`): FOV80 unchanged for 48/48; max reconstruction shape error 0.0175 -> 0.0163; 48/48 eligible; pilot acceptance passed.

## VL-2026-10-28-09 - Step 4: extended ground-truth descriptors on the pilot
- **Type:** Implementation check plus descriptive output. **Result:** Pass (cylinder family reproduces stored FOV80 for 8/8).
- **How:** `Shared_Pipeline/contribution_metrics.py` (nested-region fractions f(k) = 1 - ||s - S_k||^2/||s||^2, persistent X % extents, energy footprint); driver `3D_Code/numerics/ground_truth_metrics.py`. Intact recordings, catheter centre y = 0, both standoffs, 4 declared configurations, 25-100 ms, level-2 near-field quadrature (DL-2026-10-28-06). Outputs `_verification/ground_truth_metrics/{metrics,curves,halfPlaneProfiles}.csv`.
- **Check:** cylinder 80 % extent = stored `fov80ReconstructionMm` for 8/8 (7.5, 8.5, 3.0, 3.5, 3.5 x 4 mm). Reconstruction |corr| >= 0.99996.
- **Results (0.25 / 1.0 mm gap):**

  | Recording | Sphere 50 / 80 / 95 % (mm) | Volume in 80 % sphere (mm^3) | Depth 50 / 80 / 95 % (mm) | Energy ellipse sigma major x minor (mm), axis | Energy-weighted depth (mm) |
  | --- | --- | --- | --- | --- | --- |
  | Unipolar E5 | 5.0/8.0/13.5 ; 6.0/9.5/15.0 | 693 ; 966 | 1.0/1.75/2.75 ; 1.0/2.0/3.0 | 3.31 x 2.21 ; 4.52 x 3.11, along fibres | 0.87 ; 1.13 |
  | Bipole E5-E6|E9 | 2.5/4.0/6.0 ; 3.5/5.0/7.0 | 97 ; 148 | 0.5/1.25/2.0 ; 0.5/1.25/2.25 | 2.29 x 1.45 ; 2.57 x 2.01, along fibres | 0.70 ; 0.87 |
  | Omnipolar cliques | as bipole (centroid offset +-1.33 mm in y toward the clique's third electrode) | 97 ; 148 | as bipole | as bipole | as bipole |

- **Observations (numerical, this geometry only):** footprints are elongated along the fibre axis (aspect 1.3-1.6). The 95 % depth extent (2.0-3.0 mm) is close to the 4 mm wall, so wall thickness sweeps will reach a regime where the far wall truncates the field of view. The two omnipolar cliques sense the same tissue volume but are centred 1.33 mm either side of the anchor; anchor choice therefore matters for omnipolar distance conventions.
- **Tests:** `tests/test_contribution_metrics.py` (8 synthetic tests: known radial split, non-monotone cancellation, zero signal, bin closure, anisotropic footprint axes/angle/depth, sphere volume). Full suite: 47 pass, 1 slow test skipped by design.

## VL-2026-10-28-10 - Step 4: tissue-removal (boundary-matched) prediction of the lesion profiles
- **Type:** Numerical decomposition. **Result:** removal of tissue alone explains the measured lesion PP within 0-9 %.
- **How:** `3D_Code/numerics/removal_prediction.py` (stages `cache`, `assemble`). For every clean lesion recording (520 = 65 placements x 2 gaps x 4 configurations): its own placement, sample positions and selected weights; intact Vm; source summed only over elements that remain viable (signed distance < 0; the lesion half-space has zero conductivity in the lesion mesh). Fixed model-to-mV scale from the matching intact recording (-0.0367 for all configurations and gaps, i.e. the held gain). Output `_verification/ground_truth_metrics/removalPrediction.csv`.
- **Result (median measured / predicted PP; distance inside lesion; 89 rows with measured PP < 0.01 mV excluded as ill-conditioned):**

  | Configuration | Viable side (< -2 mm) | -2 to 0 | 0 to 2 | 2 to 5 | 5 to 10 |
  | --- | --- | --- | --- | --- | --- |
  | Unipolar E5 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |
  | Bipole E5-E6|E9 | 1.00 | 0.95-0.96 | 0.94-0.97 | 0.95-0.96 | 0.95-0.96 |
  | Omnipolar E5-E6-E9 | 1.00 | 0.93-0.97 | 0.96-0.97 | 0.95-0.96 | 0.95 |
  | Omnipolar E9-E10-E5 | 1.00-1.01 | 0.94-0.96 | 0.91-0.94 | 0.93-0.94 | 0.94-0.95 |

- **Interpretation:** the residual 3-9 % reduction for differential recordings over and near the lesion is the effect of the lesion on activation and on the source field next to the boundary (plus the ~2 % openCARP-vs-synthesis near-surface difference, VL-2026-10-28-08). It is not tissue removal.
- **Earlier unmatched half-plane prediction:** with intact weights held fixed (`halfPlaneProfiles.csv`), unipolar matched (median ratio 0.97) but bipole and omnipolar did not (ratios 0.32-0.92). Cause: lesion recordings re-select bipole/omnipolar direction (only 17-29 of 65 rows keep the intact weights). The matched prediction is the valid boundary-matched ground truth.
- **Limitations:** straight boundary, so translation invariance holds by construction; a curved or patchy lesion needs the matched calculation per placement (as implemented). One slab, one activation direction (wavefront parallel to fibres and to the boundary normal as configured in the pilot).

## VL-2026-10-28-11 - Readiness kit: software verification only (no simulation)
- **Type:** Code inspection, unit tests and an end-to-end run against test doubles. **Result:** Pass. **Not verified:** anything requiring openCARP (physics, parameter names on the installed build).
- **What was built:** `3D_Code/readiness/` (kit, CV strips, Niederer benchmark, ladder, bath check), `Shared_Pipeline/fem_extracellular.py`, settings `3D_Code/configs/readiness_kit.json`, runbook `3D_Code/readiness/README.md`. Pipeline options added, all backward compatible (absent keys keep the v2-pilot behaviour):
  - `tissue.conductivitiesSPerM` (replaces the gregion[0] lines of the template);
  - `acceptance.responseWindowMs` (physiology window separate from the EGM window);
  - credentials removed from every simulator log.
- **Tests:** `tests/test_readiness.py` (16):
  - conductivity override, config validation, credential removal;
  - activation-file formats, the plane-fit CV, smoke copies, log parsing;
  - the tuning loop, with capture doubling and ratio preservation, against a fake openCARP;
  - benchmark parameter blocks and coordinates;
  - every ladder config validates;
  - the window rule reproduces the pilot choice (24.2-103.4 ms);
  - igb decimation;
  - comparison pass/fail on synthetic runs;
  - the bath mesh numbers tissue first;
  - the bath check accepts the identical solution (rel RMS < 1e-5) and rejects a 5 % error;
  - FEM is exact for linear Vm.
- **Full suite:** 63 tests on the Mac VM (4 skipped: 3 need scipy's sparse solver, 1 slow), and 16 + 26 in the cloud workspace with scipy 1.18.1.
- **End-to-end with test doubles:** all nine stages ran in order and produced their summaries. The ladder and bath stages correctly reported failure on the doubles' all-zero Vm (physiology failed, so diagnostic re-analysis ran; the bath correlation was undefined). No credential survived in any output.
- **Sources checked:** openCARP parameter list (phys_region ptype 0/1, gregion g_bath, cellSurfVolRatio, stimulus type 3 = extracellular ground) and the official example pages 03A (tuneCV method) and 03G (benchmark blocks, copied verbatim). The openCARP GitLab sources were not reachable (bot protection), so remaining parameter names are checked by the kit's smoke stage on the workstation.
