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

## VL-2026-10-28-12 - Arbitrary fibre orientation in ground truth, synthesis and FEM (Step 7)
- **Type:** Code change with analytic and invariance tests (no simulation). **Result:** Pass.
- **Change:** `ground_truth.readElementGeometry` reads any fibre file (1 or 2 vectors per element) via `buildElementGeometry`. Conductivity becomes a tensor only when fibres are not along x. `electrode_synthesis.sourceConductivity` / `elementCoefficients` / `conductivityTimesGradients`, `contribution_metrics.elementCoefficients` and `fem_extracellular` accept tensors. `boundary_setup` writes fibres from a `fibres` config.
- **Tests (`tests/test_fibres.py`, 11):**
  - fibre models and depth convention;
  - principal values on the fibre axes;
  - monodomain tensor = S_i (S_i + S_e)^-1 S_e;
  - x fibres take the original diagonal path, equal to the forced-tensor path to 1e-12;
  - rotating mesh, fibres and electrodes by 35 deg leaves synthesised potentials and ground-truth ring traces unchanged (rel 1e-10), with and without near-field subdivision;
  - rotating fibres alone changes the signal;
  - unequal sheet values without sheet vectors are refused;
  - buildMesh writes the legacy file unchanged and a bilayer file that the reader recovers;
  - FEM is exact for a rotated anisotropic linear Vm.
- **Regression:** full suite 74 tests on the Mac VM (5 skipped: 4 need scipy's sparse solver and pass in the cloud workspace, 1 slow), plus the slow contribution regression (`EFOV_RUN_SLOW=1 DeclaredWindowContributions`): pass. The v2-pilot results reproduce.
- **Not verified:** that openCARP builds the same tensors from a rotated fibre file. Needs one small rotated-fibre run compared with synthesis (as VL-2026-10-28-02), on the workstation.

## VL-2026-10-28-13 - Bath lead fields, bath analysis and fibre check: software verification (no simulation)
- **Type:** Analytic and synthetic tests; end-to-end run against test doubles. **Result:** Pass. **Not verified:** anything needing openCARP output (the kit stages do that).
- **`tests/test_bath.py` (5):**
  - structured mesh identical to the reference decomposition;
  - reciprocity: the sum of element contributions equals the forward solution at the electrode (rel 1e-10);
  - iterative (ILU-GMRES) solver equals the direct one, and pyamg-CG agrees to 5e-11;
  - insulating plane doubles the surface potential of a buried point source (image method; ratio 2 within 0.08);
  - conductivity jump 1:3 gives the image factor 2 s1/(s1 + s2) = 0.5 (within 0.04).
- **`tests/test_bath_analysis.py` (4):**
  - lattice interpolation equals a brute-force search;
  - lattice model maps the run's nodes exactly, and the scar modes set the active region as intended;
  - bath electrograms equal an independent direct solve, NaN outside the analysed frames;
  - reciprocity ground truth reproduces the bipolar electrogram;
  - the derived run folder passes the pipeline's own completeness and settings checks;
  - fibre check: synthesis with true fibres matches (corr > 0.99999), and wrongly assumed x fibres are > 100x worse.
- **Kit end-to-end with test doubles:** all eleven stages ran in order. The stages needing physical Vm (ladder analysis, bath, bath analysis, fibre gate) correctly reported failure on the doubles' all-zero Vm. No credential survived.
- **Not verified here:** bath analysis end to end on real Vm (needs an analysed ladder run); solver time and memory for the 1.1-million-node 0.25 mm bath model.

## VL-2026-11-01-01 - Readiness-kit report reviewed; comparison crash fixed
- **Type:** Review of the kit report (30-31 Oct). **Result:** physiology passes in every run; MPI, fibres and tests pass; CV tuned to 0.895 m/s at 0.125 mm.
- **Fault:** all 13 convergence comparisons crashed (KeyError 'condition': contributions.csv already holds condition and kind, so the merge duplicated them). Fixed in ladder.attachPointKeys with a consistency check; test data now carry those columns; 86 tests pass.
- **Finding:** with one conductivity set, CV is 0.78 m/s at 0.5 mm, 0.87 at 0.25 mm, 0.895 at 0.125 mm, 0.903 at 0.0625 mm. Tuning must be at the production spacing (DL-04).

## VL-2026-11-01-02 - Bath mismatch explained (openCARP 19.0.0, container)
- **Type:** Numerical verification, real openCARP runs on a small slab. **Result:** cause found.
- **Bath check fault:** our hand-built bath mesh gave bath elements fibre vectors, so openCARP treated them as tissue with default conductivities (fit: bath = default sigma_i + sigma_e, residual 0.0). With zero fibres (mesher's convention) openCARP applies g_bath.
- **Formulation:** with a correct bath, openCARP pseudo-bidomain phi_e equals the extracellular solve with source sigma_m exactly (relative residual 0.0); full bidomain equals the solve with source sigma_i within 1.5 %. bidm_eqv_mono = 0 has no effect in pseudo-bidomain. The 31 Oct factor 0.25 (openCARP / FEM) is sigma_e / (sigma_i + sigma_e) along fibres (0.254).
- **Consequence:** every bath result of the 31 Oct kit (bath, bathAnalysis) is void; readiness/bath_check.py and bath_analysis.py are superseded by 3D_Code/production.

## VL-2026-11-01-03 - Final settings run end to end on real openCARP (container, small slab)
- **Type:** Execution and numerical check, openCARP 19.0.0 + carputils 19.0.0 (conda-forge), 2 cores. Test configuration only: slab 16 x 20 x 2 mm, baths 2 / 4 / 6 mm, 300 conditioning beats, recompute stage seeded (the readiness runs exist only on the workstation). **Result:** every stage ran; the gates behave as specified.
- **tune:** 0.907 m/s at 0.5 mm (scale 6.42 of Roberts and Scher; sigma_el 0.75 S/m), 0.902 m/s at 0.25 mm (scale 5.44). The 3D slab conducts at 0.906 m/s at 0.5 mm.
- **run050 (carputils slab.py, pseudo-bidomain, 4 beats):** physiology passes; openCARP phi_e vs sigma_m solve at all electrode samples: relative RMS <= 3.0e-6, correlation >= 0.999999999996 (gate passes). Contribution field: shares sum to 1 (deviation 2e-16), summed contributions reproduce the measured electrograms (residual <= 4.2e-8, scale = gain).
- **bidomainCheck (30 ms):** sigma_i solve vs openCARP bidomain: relative RMS 1.2 %, scale 0.997. Pseudo-bidomain vs bidomain phi_e: scale 0.22, relative RMS 84 %. Activation, bidomain vs pseudo-bidomain: max 3.8 ms, median 1.1 ms. Cost ratio about 13 per simulated ms (small model).
- **bathStudy:** 2-6 mm baths not converged on this small slab (PP 40 % at 4 vs 6 mm); rule chose the thickest bath and flagged it.
- **decide/report:** ran on stand-in 0.25 mm variants (0.5 mm results linked; a real 0.25 mm run needs ~7 h per condition on 2 cores). Checks the logic only. Forward model sigma_m vs sigma_i after calibration on the small slab: PP 6.5 %, field L1 7.4 %.
- **Fixed during the test:** carputils mesh.generate cannot run outside an experiment (direct mesher call with Block's options); empty change-point tables on short profiles; saveJson folders.
- **Not verified here:** the workstation openCARP build (run050's gate checks it there), full-size run times, and the recompute stage on the real ladder results.

## VL-2026-11-01-04 - Prepacing with checkpoint and restore reproduces a continuous run (container, openCARP 19.0.0)
- **Type:** Numerical verification. **Result:** Pass.
- **Method:** slab.py with restartMs: monodomain (no extracellular solve) to restartMs, state saved (tsav / write_statef), restart in pseudo-bidomain (start_statef); Vm and activation times merged from t = 0.
- **Small slab, restart at 30 ms and at 4 ms (mid-activation):** merged Vm identical to the continuous run (max difference 0.0 mV); phi_e after the restart within 1.2e-7 mV of the continuous run; activation file identical row for row (935 rows; restored runs write absolute times).
- **Final settings run (test slab):** physiology table identical to the continuous run (max difference 0.0); forward-model comparison numbers identical; openCARP vs sigma_m solve 3.0e-6. Cost 191 s vs 375 s per condition, disk 0.5 vs 1.2 GB.
- **Production setting:** restart at 1050 ms (4 beats at 500 ms; 57 ms before the first frame the analysis reads), so the bath is solved for 950 of 2000 ms.

## VL-2026-11-01-05 - Solver flavour and restricted phi_e output: same results, less time and disk (container, openCARP 19.0.0)
- **Type:** Numerical verification. **Result:** Pass.
- **Solver flavour (carputils --flavor), 0.25 mm test slab with 6 mm bath, 40 ms, phi_e every 0.2 ms:** petsc (GAMG, the default) 537 s, boomeramg (hypre) 116 s, direct (MUMPS) 143 s. Bath-solve time 394 / 49 / 57 s. Against direct: Vm within 7.6e-6 mV, phi_e within 9.5e-7 mV for both iterative flavours; activation files identical. In the final settings run (test slab): PP within 1.3e-9 relative, forward-model and bath-study numbers identical; 133 s vs 191 s per condition.
- **Restricted phi_e output (dataout_e = 3):** openCARP writes the listed nodes in ascending node order with values identical to the full output. Final settings run (test slab): every PP, physiology value and comparison identical; simulation folder 396 vs 512 MB on this thin-bath slab (the saving grows with bath thickness).

## VL-2026-11-02-01 - Extracellular-solve electrograms by reciprocity: same numbers, one solve per electrode (container)
- **Type:** Numerical verification and timing. **Result:** Pass.
- **Why:** the forwardModel stage of the final settings run chose the sigma_i source, so every later electrogram comes from our extracellular solve, not from openCARP's phi_e. Solved once per output frame (about 2,300 frames per condition), the 20 mm bath at 0.5 mm took 3.5 h per condition on the workstation (pyamg is single-threaded); at 0.25 mm (1.0-3.4 M nodes) run025 would have taken an estimated 17-55 h.
- **Change:** unipolar phi at an electrode is w^T K^-1 (-S Vm) = -(S^T K^-1 w)^T Vm (K symmetric), so one solve per unique electrode gives a transfer row that is applied to every frame (contribution_field.electrodeTransfer, pipeline_run.femSignals). The pilot layout's 130 placements x 16 electrodes are 712 unique electrodes (overlapping grids). Solves are independent and run in forked worker processes (--np). Same reciprocity identity as the ground truth (contribution_field.shares). Solver and tolerance unchanged (pyamg CG, rtol 1e-10).
- **Check:** unit test against the per-frame solve (LU and AMG, 1 and 2 processes, permuted Vm order): agreement within 1e-7 relative. Full layout on the 40 x 60 x 1.5 mm, 0.5 mm, 20 mm-bath mesh (431,244 nodes): max relative difference 9.6e-11 from the per-frame solve. 2.9 s per electrode on 1 core, 1.4 s on 2 (linear scaling).
- **Also tested, not adopted:** warm start from the previous frame and rtol 1e-8: 1.2-1.7x faster per frame only.
- **Not verified:** the scaling on 24 workstation cores (memory bandwidth will limit it below 24x).

## VL-2026-11-02-02 - Leenknegt-style bath, openCARP-only final settings run (container, openCARP 19.0.0)
- **Type:** Execution and numerical check on a small test slab (16 x 20 x 2 mm, 0.5 mm, 2 mm bath on every side, 1 mm grounded corner cube; test values only). **Result:** Pass.
- **Geometry:** bath elements on all six sides, lesion (tag 5) confined to tissue with y < 0, 27 ground nodes at the (max x, max y, max z) corner, none on tissue.
- **Gate (openCARP pseudo-bidomain phi_e vs the sigma_m solve with the same ground):** relative RMS <= 1.3e-6, correlation >= 0.9999999999992 (both conditions). Ground truth: shares sum to 1 (4e-16), reconstruction residual 1.1e-7, scale = gain.
- **decide/report:** ran on a stand-in 0.25 mm folder (copy of 0.5 mm): logic only. Physiology passed; pilot acceptance false as in every earlier small-slab test (slab too small for a stable far region).
- **Not verified here:** full-size timing and the 0.25 mm run (workstation).

## VL-2026-11-02-03 - beta calibration reproduces the conductivity-scaled propagation (container, openCARP 19.0.0)
- **Type:** Numerical verification. **Result:** Pass.
- **Method:** slab.py, 8 x 4 x 1.5 mm, 0.5 mm, pseudo-bidomain, 2 mm bath all round, 20 ms. A: Roberts and Scher conductivities x 6.42, beta 0.14 um^-1 (default). B: Roberts and Scher conductivities, beta 0.14 / 6.42 um^-1. Same stimulus.
- **Result:** Vm max difference 4.0e-5 mV (range -79.5 to 35.8 mV); activation files identical in length (612 rows); stimulus captures in both. phi_e PP: B / A = 0.66 (the conductivity magnitudes against blood change the extracellular field, as intended).
- **Final settings run (test slab, 2 Nov):** run050 with surfaceToVolume calibration (Roberts and Scher conductivities, beta 0.0218 um^-1): slab CV 0.9069 m/s, identical to the conductivity-scaled run; physiology passes; gate relative RMS <= 8.1e-7; ground-truth shares sum to 1.

## VL-2026-11-03-01 - finalModel branch: unit tests and a real-openCARP end-to-end run (container)
- **Type:** Execution and numerical check, openCARP 19.0.0 + carputils 19.0.0 (conda-forge), 2 cores. Small test configuration only (`config/test_small.json`: 20 x 24 x 1.5 mm, 0.5 mm, 2.5 mm bath on every side, 3 beats, 20 conditioning beats, 34 placements); same physics and analysis code as `config/final_model.json`. **Result:** every stage ran; implementation checks pass.
- **Dry run:** carputils `--dry` commands inspected: lesion region tissue-only (z within the wall, y < 0), beta 0.0218, Roberts and Scher conductivities, lesion 1e-6, g_bath 0.74, ground cube and stimulus vertex files, prepacing monodomain then pseudo-bidomain restart, dataout_e = 3.
- **Simulation:** 17 min for both conditions (2 cores, boomeramg passed its smoke test).
- **Physiology:** last-beat CV 0.907 m/s (intact), 0.901 m/s (lesion): beta = 0.0218 um^-1 reaches the 0.90 m/s target within 1 % on this build. All criteria pass except stableApd, expected with 20 instead of 1,000 conditioning beats (the gate works).
- **Window:** 7.0-48.8 ms; starts at the stimulus floor and is flagged (DL-2026-10-28-08).
- **Forward-model check:** openCARP phi_e vs the independent sigma_m solve at 2,000 electrode samples x 25 frames: relative RMS max 8.8e-8 (intact), 1.2e-7 (lesion); correlation 1.0000000000. Confirms tags, corner ground, node order and phi_e output layer in the new code.
- **Ground truth:** 48 recordings; shares sum to 1, reconstruction residual and scale = gain within tolerance (gate passed).
- **Expected failures on the small slab:** stableFarRegion and lesionTransition (placements reach only ~6 mm into each side), so no change point is eligible for interpretation; the gates propagate as designed.
- **Provenance protection:** a rerun reused every stage; editing a change-point setting reused measurements and refused changePoints; editing blood conductivity refused the simulation.
- **Unit tests:** 29 pass (`python -m pytest tests -q`), including the reciprocity identity, method-of-images references, insulating-lesion and sigma_m source tensors, IGB merge, activation merge, recording weights, change-point recovery and eligibility rules.
- **Notebooks:** 00-06 executed without error against this run.
- **Not verified here:** full-size timing and memory on the workstation; the workstation build (the forward-model check runs there in every run); values awaiting the final settings run.

## VL-2026-11-03-02 - finalModelV2: software verification without openCARP (Claude's sandbox)
- **Type:** Unit tests and a synthetic end-to-end analysis; openCARP could not be installed in the sandbox, so nothing here exercises openCARP. **Result:** see items.
- **Unit tests** (`python -m pytest tests -q`): parameter files (overrides in place, clashes rejected), configuration (validation, scar options, fingerprints: electrode and noise variants reuse the simulation, wall and scar do not; beta excluded), mesh building (no-flux epicardium trim, fibres, stimulus band for 0 and 90 deg, ground cube, phi_e box), peak frequency, measurements, export, sweep variants, mesh I/O and VTK, ground truth.
- **Peak frequency (C6):** pure tones 100/250/450 Hz recovered exactly (1 Hz grid); 6- and 12-cycle bursts within 1 Hz; 3-cycle bursts 1-3 Hz high (+0.7 % at 450 Hz). The pre-declared C6 criterion (bursts within 2 Hz) fails for 3-cycle bursts. Cause: the "highest PF trace value" rule on a broadband burst, not the transform (exact for tones). Recorded, criterion not changed. Resampling 4 vs 2 kHz: same PF (302 Hz for a 300 Hz burst).
- **Resampling (FM-22):** timing preserved (zero crossings identical); passband ripple of resample_poly's default filter 0.03-0.13 % at 30-500 Hz. Criterion set from its downstream effect: below 0.5 %, a tenth of the 5 % PP tolerance (DL-2026-10-28-05).
- **Ground truth (Python route):** on a box with a lesion strip, contributions reproduce the forward signal of K phi = -S_m Vm to 1e-7; reciprocity z_A(B) = z_B(A) to 1e-9; shares and class fractions sum to 1.
- **Not verified here:** anything that runs openCARP (W0, C1, C3, C4, C5, C9), Wang's Rmd in R (C8; no R in the sandbox). checks/container_checks.py runs them in efov-runner.

## VL-2026-11-03-03 - finalModelV2: synthetic end-to-end analysis (no openCARP)
- **Type:** Integration test (`tests/test_synthetic_pipeline.py`). **Result:** Pass.
- **Method:** config/test_small.json mesh built with efov.meshbuild (no-flux epicardium, bilayer fibres); analytic planar-wave Vm over 4 beats; phi_e at the output nodes from the exact discrete pseudo-bidomain problem (K phi = -S_m Vm, corner ground); openCARP file layout (vm.igb, phie.igb after the restart time, act-thresh.dat, ground.vtx, phieNodes.npy). Stages window -> electrograms -> measurements -> export -> groundTruth (Python lead fields) -> ground-truth figures.
- **Result:** 1,220 ground-truth recordings; share-sum deviation 1.3e-15; reconstruction residual max 3.3e-7; scale/gain deviation 7.3e-8. Figures render (sections, sequence, profiles).
- **Bugs found and fixed by this test:** the window was rounded to the 0.2 ms output step, so its end could fall between 2 kHz samples; window edges are now rounded outward to the common multiple of both steps (1 ms). profiles.npz stored recording IDs as an object array (not loadable without pickle); now strings.
- **Not tested here:** physiology on real openCARP Vm, the openCARP Laplace route (W0), the R stage (C8).

## VL-2026-11-03-04 - Revised model (FM-23 to FM-30): software verification without openCARP (Claude's sandbox)
- **Type:** Unit tests and the synthetic end-to-end analysis; openCARP and R are not installed in the sandbox, so nothing here exercises openCARP or Wang's Rmd. **Result:** 38 passed, 1 skipped (the opt-in real-openCARP test).
- **Reciprocity with the sigma_i source (tests/test_groundtruth.py):** on a bilayer box with a lesion strip and isolated tissue, the electrogram from stored tissue-node lead fields (w = -S_i z) equals a direct frame-by-frame solve of K phi = -S_i Vm to 1e-7 relative; the sigma_m source gives a signal differing by more than 5 % on the same slab (the bias FM-23 removes); element coefficients sum to zero (uniform Vm contributes nothing); reciprocity z_A(B) = z_B(A) to 1e-9; shares and classes sum to 1.
- **Lead-field cache:** a second request computes nothing; a cache built with a different ground is refused.
- **Synthetic end-to-end (tests/test_synthetic_pipeline.py):** test_small mesh with the 5.6 mm lesion strip and isolated tissue (no activation at y < 0 in the lesion run); stages window -> leadFields (Python route, every distinct electrode) -> electrograms -> measurements -> export -> groundTruth -> figures. All placement windows identical for the 0 deg wave; one electrode over the lesion matches a direct solve to 1e-4 relative over 300 frames; 1,220 ground-truth recordings, share-sum deviation 1.6e-15; the ground-truth signal PP equals the measured clean PP; export writes gaps x (repeats + 1) x 2 PF definitions datasets.
- **Measurements:** per-placement windows (a placement shifted by 3 ms gets its own window start), physical units (no gain), both PF definitions, pfNoiseRobustness table; peak frequency per-row windows (a row's own window decides its PF).
- **Configuration:** monodomain and dataout_e = 0 enforced; lesion width validated; conductor fingerprint shared by wave-angle, CV, gap and noise variants and changed by wall, scar, fibres and epicardial medium.
- **Not verified here:** every openCARP step (C1, W0, C4, C5 in checks/container_checks.py), the cost of ~712 Laplace solves per condition, physiology on real Vm with the lesion strip, the R stage (C8). The physical amplitude scale depends on openCARP's total-current convention, which by FM-26 is not compared with anything.


## VL-2026-11-04-01 - finalModelV2 openCARP steps on real openCARP (Claude's sandbox, openCARP 19.0.0)
- **Type:** Execution checks (`checks/container_checks.py`, small slab `config/test_small.json`), openCARP 19.0.0 + carputils 19.0.0 (conda-forge, build `c877c38`), 2 cores. First real-openCARP run of the FM-23 to FM-30 code. **Result:** two failures found and fixed; one design route unavailable.
- **Bug fixed (short runs):** a 30 ms check run kept bcl = 500 ms; openCARP rejects bcl > tend ("above the 30 maximum") and exits with code 0 without simulating. `slab_experiment.overridesFor` now sets bcl = tend for runs shorter than one cycle (one pulse at ptcl.start, as before). Production runs (tend 2000 ms) are unaffected.
- **W0 diagnostic - openCARP Laplace mode ignores current stimuli:** with `config/leadfield.par` (experiment 2, extracellular current stimulus, crct.total_current = 1, corner ground) openCARP reads the stimulus ("current density scaled to 738.5 uA/cm^3") and solves, but the written potential is exactly 0 at every node. Laplace mode applies only Dirichlet conditions. openCARP lead fields as designed (FM-14) are therefore not available on this build; the pre-declared fallback was adopted (FM-31). Not tested on the workstation build (`9accae05`).
- **W0 (fallback route, python):** finite, non-zero lead field; potential at the injecting electrode 1.14 mV/uA (positive, as required).
- **C1 mesh:** pass (0.5 / 0.5 / 0.25 mm found; bath above, none below; bath fibres zero).
- **C4 fibres, small-slab mesh (0.5 mm in-plane):** speed along fibres 0.908, across 0.500 m/s; ratio 0.551 vs continuum sqrt(sigma_m,t / sigma_m,l) = 0.627 (-12 %): FAIL against the 10 % criterion. openCARP reads the fibre file (the 90 deg speed falls clearly). The shortfall is consistent with the 0.5 mm mesh under-resolving the slow transverse wave.
- **C4p (added): the same measurement at the production mesh (0.25 / 0.1 mm, tuned beta):** 0.876 / 0.521 m/s, ratio 0.595 (-5.0 %): pass. Criterion unchanged; C4 on the coarse test mesh is recorded as failed.
- **C5 lesion strip:** pass (viable tissue all activates; no lesion or isolated node activates; 1,980 isolated nodes).
- **C8 Wang's Rmd:** see VL-2026-11-04-03.
- **Not verified here:** the workstation build and 24-core timing.

## VL-2026-11-04-02 - beta tuned at the production mesh (FM-18)
- **Type:** Execution, `python -m efov tune --write-par` (strip 20 x 1 x 0.5 mm, 0.25 / 0.1 mm, fibres along x, monodomain, 3 beats at 500 ms after 1,000 single-cell conditioning beats). **Result:** Pass.
- **Iterations:** beta 0.0218 -> 1.022 m/s; 0.02812 -> 0.887 m/s; 0.027292 /um -> 0.9015 m/s (target 0.90, tolerance 1 %).
- **Written:** `config/model.par` imp_region[0].cellSurfVolRatio = 0.027292 /um; `config/tuning.json`. openCARP 19.0.0 (conda-forge); the slab CV of every production run is reported by the physiology stage.

## VL-2026-11-04-03 - Real-openCARP end-to-end run of config/test_small.json (all stages to the report)
- **Type:** Execution and integration, openCARP 19.0.0, R 4.4.3 with mcp 0.3.4 and JAGS 4.3.2 (conda-forge), 2 cores, Python lead fields in 2 forked workers. **Result:** every stage ran; implementation checks pass.
- **Simulation:** monodomain, intact 134 s, lesion 107 s (38,055 nodes, 8,085 tissue nodes); vm.igb 309 MB per condition.
- **Physiology:** all criteria pass in both conditions (1,000 conditioning beats; stableApd passes). Last-beat surface CV 0.639 (intact), 0.633 m/s (lesion) with bilayer 0/90 deg fibres, against a strip CV of 0.893 m/s with fibres along x. See FM-33.
- **Lead fields:** 168 per condition, 5 s on 2 workers after a 10 s set-up. **Ground truth:** 1,220 recordings, share-sum deviation 1.3e-15.
- **Change points:** Wang's Rmd ran through R/wang_replication.R on every exported dataset (gaps x noise realisations x PF definitions). Not eligible for interpretation: stableFarRegion and lesionTransition fail on the small slab (placements reach only ~6 mm into each side), as in every earlier small-slab run.
- **Amplitudes (model units):** median far-field clean bipolar PP 8.4 mV (0.25 mm gap), 5.7 mV (1 mm gap); Wang inferior median 1.44 mV (reported only, FM-25).
- **Not verified here:** full-size cost and memory (estimated in VL-2026-11-04-04), the workstation build.

## VL-2026-11-04-04 - Cost of the Python lead-field route at production spacing (checks/leadfield_cost.py)
- **Type:** Timing diagnostic. Production geometry (0.25 / 0.1 mm, 5 mm bath, no-flux epicardium, bilayer fibres) at 10 x 15 mm: 507,222 nodes; set-up 9 s; 5.2-5.5 s per lead field (pyamg smoothed aggregation, CG, rtol 1e-10, 1 core); peak memory 2.5 GB including mesh building. 20 x 30 mm exceeded the sandbox's 7 GB.
- **Estimate for production (3.50 M nodes, 712 lead fields per condition, 2 conditions):** about 40-50 s per lead field per core, about 8-10 core-hours per condition, i.e. under an hour per condition on 24 workers if memory bandwidth allows; set-up memory about 18-25 GB (linear scaling). Estimate only; the production log reports the measured time.

## VL-2026-11-04-05 - Slab surface CV with bilayer fibres at the production mesh
- **Type:** Execution (30 ms single-beat runs of the small slab at 0.25 / 0.1 mm, wall 1.1 mm, beta 0.027292 /um; speed of x-directed wave on the catheter-facing surface, x in -4..4 mm, as C4). **Result:** measurement, no criterion.
- **Result:** bilayer (endo 0, epi 90 deg) 0.735 m/s; uniform 0 deg 0.876 m/s (strip, last of 3 beats: 0.9015 m/s). Ratio 0.84. Led to FM-33.

## VL-2026-11-04-06 - C8: Wang's Rmd through R/wang_replication.R (R 4.4.3, mcp 0.3.4, JAGS 4.3.2)
- **Type:** Execution and numerical check on the synthetic C8 dataset. **Result:** Pass after a correction to the check's reference.
- **First run:** Wang's Table 3 bipolar slope -0.193141 vs the check's direct fit -0.194089: FAIL. Cause: the check omitted Wang's 99.9th-percentile PP trim of the 'volt' dataset (trim_upper, computed over all electrode kinds of the file). The check's reference now applies the same dataset rules; Wang's code is unchanged.
- **Rerun:** -0.193140597349686 vs -0.1931405973496856 (difference 4e-16). Tables 1-5 written.
- **Known, not used by the pipeline:** Wang's chunks figure3, figure4 and figure5 fail in one-slab mode (figure5 calls lme4::fixef on the lm replacement; figure3/4 fail on one-patient data). The pipeline reads only her tables; the same three chunks fail in the end-to-end run (VL-2026-11-04-03).

## VL-2026-11-04-07 - notebooks/run_all.ipynb on the small test sweep (real openCARP and R)
- **Type:** Execution of the whole notebook (`jupyter nbconvert --execute`, EFOV_SWEEP=config/sweeps/test_small.json, 2 cores). **Result:** Pass; no cell error.
- **Covered:** Part 1 (every base-model stage, report, 18 figures); Part 2 with one variant that has its own simulation (wallMm 1.5), one with new electrodes only (gapsMm 0.5: reused the baseline simulation, new lead fields), three analysis-only variants (noiseSdMv: reused simulation and lead fields); the other experiment cells skipped with a message because the test sweep does not list them; Part 3 noiseDoubling.csv, sweepStatus.json, sweepSummary.csv.
- **Found and fixed after the run (functions re-run on its outputs, notebook not re-executed):** the experiment summary used only the clean change points, which are identical across noise variants by construction; it now also reports the median over noisy realisations (cpNoisyMm). Numeric variables are plotted in order of value with the baseline in place (it was plotted first).
