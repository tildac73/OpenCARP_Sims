# Decision log

One entry per consequential choice. Status: Open, Decided, Superseded.
Irreversible = cannot be changed without discarding production results.
If an earlier decision log exists (D1, D2, ...), merge these entries into it and keep both ID schemes.
Open decisions R1-R13 for the reference model are listed in the "Model readiness plan" doc.

---

## DL-2026-10-27-01 - EGM measurement window 25-100 ms
- **Status:** Decided (provisional until R5, Step 5 of the readiness plan). **Irreversible:** No.
- **Context:** In the 0-400 ms window, far-field from the synchronously firing pacing strip (0-2 ms, ~18 mm away) dominated bipolar PP beyond ~8 mm into the lesion and created a false plateau.
- **Decision:** Measure PP, bipole/omnipolar selection and contributions over 25-100 ms after the beat-16 stimulus. Physiology checks keep 0-400 ms.
- **Basis:** Intact activation timing only (beat 16): strip 0.9 ms, grid x = -6..+6 mm 39.3-73.4 ms, far edge 112.2 ms. Declared before any windowed result was computed; the time-binned diagnostic had been seen.
- **Alternatives:** keep 0-400 ms (rejected: artefact); per-recording activation-referenced window (planned, R5).
- **Consequence:** Unipolar PP depends on the window start (up to 6.5%; change point +0.4 mm at 30 ms). Bipolar/omnipolar insensitive.

## DL-2026-10-27-02 - Gain held at the original calibration for windowed reprocessing
- **Status:** Decided. **Irreversible:** No.
- **Decision:** Keep gain 0.461584 (from the 0-400 ms calibration) rather than recalibrating in the new window, so only one factor changes.
- **Open:** Source of the 1.44 mV target to be stated (R7).

## DL-2026-10-27-03 - Bayesian sampling extended to 15,000 iterations after 5,000 adaptation
- **Status:** Decided. **Irreversible:** No.
- **Context:** The original clinical settings (3,000 / 1,500) left some fits below Rhat <= 1.01 / ESS >= 400.
- **Decision:** Same mcp model, default priors, 3 chains, seed 3; sampling extended. All lesion fits now pass; one intact fit (bipole, 0.25 mm) has Rhat 1.07.

## DL-2026-10-27-04 - numpy stand-ins for scipy on the analysis VM only
- **Status:** Decided. **Irreversible:** No.
- **Decision:** `_refinement_work/shim/scipy` replaces `signal.butter` (declared filter only), `signal.sosfilt` and `sparse.coo_matrix(...).tocsr()` where scipy cannot be installed. Restricted to the exact calls used; anything else raises.
- **Verification:** see VL-2026-10-27-02. Not used on the simulation workstation.

## DL-2026-10-27-05 - Volume conductor route: openCARP bath and openCARP lead fields (R1)
- **Status:** Decided. **Irreversible:** Yes, once production runs start.
- **Decision:** Electrograms from openCARP pseudo-bidomain with a blood bath; per-electrode lead fields from static openCARP extracellular solves with current injection. Ground truth uses those lead fields.
- **Revisits:** the earlier choice of a bespoke Stage A structured-grid lead-field solver (D2). That solver becomes the fallback if openCARP lead fields cannot be produced cleanly.
- **Rationale:** established, citable code; less new code to verify; two-route agreement (bath EGM vs lead-field integral) becomes a verification check.
- **Next:** feasibility test on a small slab, one electrode, verified against the openCARP documentation and examples for the installed version.

## DL-2026-10-27-06 - Adopt the model readiness plan
- **Status:** Decided. **Irreversible:** No.
- **Decision:** Steps 1-9 of the "Model readiness plan" doc are completed, gate by gate, before any parameter sweep. Tolerances marked "proposed" await sign-off.

## DL-2026-10-28-01 - Monodomain source conductivity for synthesis and ground truth
- **Status:** Decided. **Irreversible:** No.
- **Context:** Part 1 ground truth used the intracellular conductivity G_i and absorbed the scale with one fitted factor (-0.0621). That factor matches (1/4 pi sigma_b) x G_e/(G_i+G_e) = 0.0622 for sigma_b = 1 S/m, i.e. openCARP recovery behaves as a monodomain source G_m = G_i G_e/(G_i+G_e).
- **Decision:** `electrode_synthesis` and `ground_truth.ringOperator` use G_m (option `source="intracellular"` kept for comparison).
- **Evidence:** VL-2026-10-28-02 (amplitude ratio median 1.002-1.005 with G_m; 1.24-1.30 with G_i).
- **Consequence:** Pilot FOV80 unchanged (equal anisotropy ratios make G_m/G_i constant; regression test passes). Required before conductivities with unequal anisotropy (e.g. Roberts and Scher 1982) are introduced, where a single fitted scale could not absorb the difference.
- **Open:** confirm sigma_b and the recovery formula against the openCARP manual or source for build 9accae05 (manual PDF not readable here); the numerical agreement is the current evidence.

## DL-2026-10-28-02 - Step 2 equivalence test on a reduced slab
- **Status:** Decided. **Irreversible:** No.
- **Decision:** serial vs MPI equivalence tested with `3D_Code/configs/mpi_equivalence.json` (20 x 30 x 4 mm, 3 beats; all physics, solver and output settings as boundary_pilot.json). Equivalence of the parallel solver does not depend on domain size; a smaller run keeps the test short.

## DL-2026-10-28-03 - Conduction velocity target: 0.90 m/s along fibres, sensitivity 0.70 and 1.25 m/s
- **Status:** Decided (provisional until Step 3 tuning shows it is reachable at the production mesh). **Irreversible:** Yes, once production runs start.
- **Evidence:** human sinus-rhythm CV 89.6 cm/s LA median, 412 patients (van Schie 2021); 90-92 cm/s biatrial (Heida 2021); 88 +/- 9 cm/s RA (Hansson 1998); 1.25 +/- 0.38 m/s endocardial EAM normal-voltage (Heil 2023); 70 +/- 10 cm/s LA controls (Zheng, via Heida). No fibre-resolved human atrial-body value found; Hansson 1998 found no group-level direction dependence (directions not fibre-referenced).
- **Decision:** tune longitudinal planar CV to 0.90 m/s; transverse follows the Roberts and Scher anisotropy (DL-2026-10-28-04). Sensitivity runs at 0.70 and 1.25 m/s.
- **Limitation (state in thesis):** clinical values are direction-mixed local estimates, so the longitudinal target is conservative; the pilot's 0.35 m/s is outside every reported value.

## DL-2026-10-28-04 - Conductivity strategy (R4): Roberts and Scher anisotropy, magnitudes scaled to the CV target
- **Status:** Decided. **Irreversible:** Yes, once production runs start.
- **Decision:** keep the intracellular and extracellular along/across ratios of Roberts and Scher 1982 (canine LV in situ; 5.8 intracellular, 1.5 extracellular); scale magnitudes to reach DL-2026-10-28-03 at the production mesh using openCARP CV tuning. Record the scale factors as a calibration, not a measurement.
- **Limitation:** ventricular, canine anisotropy applied to human atrium; no atrial bidomain conductivities were found.

## DL-2026-10-28-05 - Convergence tolerances adopted (Step 3)
- **Status:** Decided. **Irreversible:** No.
- **Decision:** between the two finest settings: CV <= 2 %, PP <= 5 %, FOV80 <= one ring (0.5 mm), change point <= 0.5 mm. Step 2 MPI tolerances also adopted (activation <= 0.02 ms, Vm <= 0.1 mV, phi_e peak-to-peak <= 0.1 %).

## DL-2026-10-28-06 - Near-field lead-field quadrature for the reference model
- **Status:** Decided. **Irreversible:** No.
- **Decision:** ground truth and electrode synthesis use recursive midpoint subdivision (level 2) for elements within 3 mesh spacings of each field point (1.5 mm at 0.5 mm mesh); set via `analysis.leadFieldQuadrature`. The v2-pilot results keep the centroid rule (option absent) so they still reproduce exactly.
- **Evidence:** VL-2026-10-28-07/08. Analytic box: level 2 error <= 0.44 % of the potential scale at 0.05 mm, <= 0.05 % at >= 0.25 mm; centroid rule 7.3 % at 0.05 mm.
- **Why adopt although the pilot is unaffected:** the production mesh, thinner walls and smaller standoffs move field points closer to elements in mesh units; the cost is small (~340 subdivided elements per point).

## DL-2026-10-28-07 - Ground-truth descriptor set for the reference model (proposed)
- **Status:** Proposed; needs confirmation. **Irreversible:** No.
- **Proposal:** primary ground truth stays FOV80 (cylinder, persistent 80 %). Reported alongside: sphere radius and volume at 50/80/95 %, depth extents at 50/80/95 %, energy footprint ellipse (sigma major/minor, axis angle, centroid offset, energy-weighted depth). For lesion experiments, the boundary-matched ground truth is the matched tissue-removal prediction (VL-2026-10-28-10), and the change point is compared with the distance at which that predicted profile reaches its plateau (definition to be fixed before sweeps).
- **Why:** FOV80 alone is a lateral radius; wall-thickness and standoff sweeps need depth and volume, fibre-orientation sweeps need the ellipse, and the change-point validation needs a ground truth defined in the same geometry as the change point.
- **Open:** the plateau definition for the removal-predicted profile (e.g. distance at which predicted PP falls to within X % of its far-lesion value); choose X before any sweep result is seen.

## DL-2026-10-28-08 - Measurement window for tuned-model runs: derived from activation (proposed)
- **Status:** Proposed; needs confirmation before ladder results are interpreted. **Irreversible:** No.
- **Context:** DL-2026-10-27-01 (25-100 ms) was set from the pilot's timing at 0.35 m/s (grid activation 39.3-73.4 ms). At the 0.90 m/s target the grid activates at roughly 15-30 ms, so a fixed 25-100 ms window would cut through local activation.
- **Proposal:** for every new run, from that run's intact last-beat activation of top-surface nodes under the grid (|x| <= 6.5 mm): start = max(first - 15, stimulus end + 5) ms; end = last + 30 ms after the beat stimulus; rounded outward to the output step. Computed before any electrogram is measured (`3D_Code/readiness/ladder.py: deriveWindow`); written to `windowDerivation.json`.
- **Basis:** the margins reproduce DL-2026-10-27-01 on the pilot (24.2-103.4 ms vs the chosen 25-100 ms; unit test). The 5 ms floor after the 2 ms stimulus is an ASSUMPTION.
- **Consequence to watch:** with the pilot's 40 mm length, the grid is ~15 ms of travel from the pacing strip at 0.90 m/s, so the window starts at the floor (~7 ms). Early far-field from the strip may enter the window (the DL-2026-10-27-01 artefact). This is the same at every ladder level, so convergence comparisons remain valid. For the reference model it argues for a longer slab along propagation or a per-recording window (R5, Step 5). The 60 x 90 mm domain run shows its size.

## DL-2026-10-28-09 - Convergence ladder design (Step 3)
- **Status:** Proposed (implemented in `3D_Code/configs/readiness_kit.json`). **Irreversible:** No.
- **Design:** physical parameters fixed across levels (tuned conductivities and stimulus from the CV stage), so differences are discretisation or domain effects. Discretisation runs on a 40 x 30 x 4 mm slab (pilot length along propagation, half width across the boundary; 33 placements, -8..8 mm). Levels: mesh 0.5 / 0.25 mm (0.125 mm optional, tier B); dt 20 / 10 us at 0.25 mm; output 0.2 / 0.1 ms (one 0.1 ms run decimated for analysis only, since output sampling does not change the solution); 4 / 8 tissue beats; domain 40 x 30 / 40 x 60 / 60 x 90 mm at 0.5 mm.
- **ASSUMPTIONS:** 4 tissue beats after 1000 single-cell beats (pilot 16), tested by the 4-vs-8 comparison and the beat-stability checks. Discretisation error on the reduced slab transfers to the full slab.
- **Comparison rules (DL-2026-10-28-05):** PP compared before calibration (PP / gain), because each run calibrates its own gain. FOV80 is gain-independent. Change points use each run's calibrated analysis, as the clinical method would. Strict and clarified change-point rules are both reported (the clarification of VL-2026-10-28-06 is still to be confirmed).
- **Not automatic:** the production spacing, time step, domain and beat count are chosen afterwards, from the comparison table.

## DL-2026-10-28-10 - CV tuning implementation (implements DL-2026-10-28-04)
- **Status:** Decided for method; tolerance PROPOSED. **Irreversible:** No.
- **Method:** the documented tuneCV update rule (all sigma_i and sigma_e multiplied by (CV_target/CV)^2, ratios fixed; openCARP example 03A_study_prep_tuneCV), applied on a 20 mm tetrahedral strip from the same mesher call and with the same solver settings, ionic state, pacing (3 beats, 500 ms) and LAT threshold (-20 mV) as the slab. The official tuneCV tool was not used, because its hexahedral cable does not reproduce CV on the slab's tetrahedra at the same spacing.
- **Settings:** tune at 0.125 mm; CV from a least-squares plane fit to last-beat activation over x in [-5, 5] mm. Tolerance 1 % (PROPOSED: half the 2 % convergence tolerance). Starting multiplier 10 (estimate from the pilot, affects only the number of iterations).
- **Capture rule:** if a run does not capture, the stimulus is doubled, at most twice per tuning. The capturing strength is recorded and reused.
- **Also measured:** CV at 0.5 / 0.25 / 0.0625 mm and dt 10 us with the tuned values (CV resolution); transverse CV (the anisotropy follows from Roberts and Scher, not tuned); pilot conductivities at 0.5 / 0.25 / 0.125 mm (how much of the pilot's 0.35 m/s is mesh under-resolution); tuned sets for 0.70 and 1.25 m/s.

## DL-2026-10-28-11 - Step 6 feasibility gate (proposed)
- **Status:** Proposed. **Irreversible:** No.
- **Test:** pseudo-bidomain openCARP run (bidomain = 2) with a 10 mm blood layer (0.74 S/m) on the catheter side, grounded on the far face. phi_e at electrode samples is compared with an independent P1 finite-element solve of the same discrete problem from the same Vm (`Shared_Pipeline/fem_extracellular.py`, exact on analytic cases).
- **Proposed gate:** relative RMS <= 1 % and correlation >= 0.9999 per sample, plus confirmation (from the activation map) that openCARP's Vm ordering matches the mesh. This tests the configuration (tags, conductivities, ground), not the physiology. Bath-size convergence and the analytic half-space check remain for Step 6 proper.

## DL-2026-10-28-12 - Fibre-field convention for the reference model (Step 7)
- **Status:** Decided for the convention; no fibre angles chosen yet. **Irreversible:** No.
- **Convention (`Shared_Pipeline/fibres.py`):**
  - Fibres lie in planes parallel to the surface.
  - The angle is measured in the x-y plane from +x, counter-clockwise seen from the catheter.
  - Depth is measured from the catheter-facing (endocardial) surface.
- **Models:**
  - `uniform`;
  - `linearRotation` (endocardial to epicardial angle);
  - `bilayer` (abrupt or ramped transition at a stated depth). The bilayer follows the qualitative description in Pashakhanloo et al. 2016; no angles are taken from it.
  - No `fibres` key: every fibre along +x, the v2-pilot file byte for byte.
- **Tensors:** sigma = sum_k s_k a_k a_k^T on the axes (fibre, sheet, normal). Intracellular and extracellular tensors share axes, so the monodomain tensor is the per-axis series combination, rotated.
- **Sheet vectors:** with one fibre vector per element, the tissue must be transversely isotropic (sheet value = normal value); otherwise the reader refuses.
- **Open:** fibre angles for the reference model (R11); whether openCARP builds its tensors with the same convention (verification on the workstation, below).

## DL-2026-10-28-13 - Wall thickness as a parameter
- **Status:** Decided. **Irreversible:** No.
- **Decision:** wall thickness = `slabSizeMm[2]`. Electrodes, depth metrics, fibre depth and the window derivation all reference the catheter-facing surface z = +thickness/2, so no other setting changes with it.
- **Limitation:** mesh convergence is being tested at 4 mm only. Sourced walls are 1.5-2.7 mm (parameter_sources.md), i.e. 3-5 elements at 0.5 mm, so convergence must be rechecked at the thinnest wall in the sweep.

## DL-2026-10-28-14 - Step 6 route: bath electrograms and ground truth by post-processing pseudo-bidomain Vm (proposed)
- **Status:** Proposed. **Irreversible:** No.
- **Basis:** in pseudo-bidomain (bidomain = 2) Vm is the monodomain solution, so the bath changes phi_e only. phi_e with a bath is therefore one elliptic solve per frame on Vm that openCARP already produced.
- **Proposal:** the kit's `bath` stage checks, on this build, that openCARP's pseudo-bidomain phi_e equals `fem_extracellular`'s solve of the same problem (DL-2026-10-28-11). Once it does, bath electrograms and bath ground truth for any run come from that solver (`3D_Code/readiness/bath_analysis.py`):
  - electrograms are written into a derived run folder and analysed by the unchanged pipeline;
  - ground truth uses a reciprocity lead field per recording.
- **Relation to DL-2026-10-27-05:** that decision chose openCARP lead fields by current injection. The FEM lead field solves the same discrete problem and is checked against openCARP, so it is equivalent and avoids one openCARP run per electrode. If the bath check fails, DL-27-05's route stands.
- **Models in the kit:**
  - bath 5 / 10 / 20 mm on the 0.5 mm ladder run;
  - bath 10 mm on the 0.25 mm run;
  - insulating vs conducting scar (conducting = passive with the healthy extracellular tensor: ASSUMPTION).
- **Comparisons:** bath size and bath mesh are judged by DL-2026-10-28-05. Bath vs unbounded and scar type are effects to report.
- **Kept as limitations:**
  - bath on the catheter side only, with no epicardial medium (no sourced conductivity);
  - lateral bath = tissue extent;
  - point electrode samples;
  - unipolar reference = the grounded far face;
  - pseudo-bidomain, not full bidomain.

## DL-2026-10-28-15 - Additional readiness checks (Step 7 and wall thickness)
- **Status:** Decided (part of the kit). **Irreversible:** No.
- **Fibre check:** openCARP runs with uniform 30 deg and bilayer 0/90 deg fibres, using the tuned (unequal-anisotropy) conductivities, compared with electrode synthesis. Gate PROPOSED: corr >= 0.999, as Step 2. Control: the same comparison with fibres wrongly taken as x must be clearly worse.
- **Thin wall:** 1.5 mm wall (CT posterior-wall mean, Li et al. 2024), 0.5 vs 0.25 mm mesh, DL-05 tolerances (addresses DL-2026-10-28-13).

## DL-2026-11-01-01 - Pending readiness decisions confirmed
- **Status:** Decided (confirmed by Matilda, 1 Nov 2026). **Irreversible:** No.
- **Confirmed:** VL-2026-10-28-06 clarification (change-point tolerance on supported change points only, plus no support-status change); DL-2026-10-28-08 (window rule); DL-2026-10-28-10 (1 % tuning tolerance), with tuning at the production spacing as DL-2026-10-28-04 states; DL-2026-10-28-11 (forward-model match: relative RMS <= 1 %, correlation >= 0.9999).
- **DL-2026-10-28-09 confirmed with an amendment:** change-point convergence is judged only on runs that pass pilot acceptance (the 40 x 60 and 60 x 90 mm domains). On the reduced 40 x 30 mm slab, every run fails stableFarRegion and lesionTransition (16 contributions vs 48), so only CV, PP and the contribution field are judged there. Made from the acceptance flags, before any comparison result was seen.
- **Rejected:** DL-2026-10-28-14 (bath electrograms by post-processing). Electrograms come from openCARP's built-in bath; the extracellular solver stays as the independent check (DL-2026-10-27-05 route).

## DL-2026-11-01-02 - Ground truth: the exact contributing-tissue field (supersedes DL-2026-10-28-07)
- **Status:** Decided (Matilda, 1 Nov 2026); comparison tolerance PROPOSED. **Irreversible:** No.
- **Decision:** the ground truth is the full field of element contributions, no radius: for each recording, s_e = sum_t c_e(t) phi(t) / sum_t phi(t)^2 with c_e from the reciprocity lead field (Shared_Pipeline/contribution_field.py). Shares sum to 1 exactly; cancellation is kept; nothing is cut off. FOV80 is no longer the ground truth.
- **Comparison between runs:** L1 distance of share fields on 0.5 mm voxels, sum|a - b| / sum|b|, maximum over ground-truth recordings <= 0.10 (PROPOSED; to confirm before the final settings run).
- **Check in every run:** summed contributions reproduce the electrogram the pipeline measured (container test: residual 3e-8, scale = gain).

## DL-2026-11-01-03 - One bath for all production runs
- **Status:** Decided (Matilda, 1 Nov 2026). **Irreversible:** Yes, once production runs start.
- **Decision:** bath thickness and mesh are chosen once by the final settings run (thinnest of 5 / 10 / 20 mm passing against the next thicker, both forward models) and fixed for every sweep run.
- **Note:** the standoff sweep must stay well inside the chosen bath; bath convergence is shown at the pilot standoffs (0.25 and 1 mm).

## DL-2026-11-01-04 - Production model as a carputils experiment; openCARP defaults where they exist
- **Status:** Decided (Matilda, 1 Nov 2026). **Irreversible:** No.
- **Decision:** the production simulation is 3D_Code/production/slab.py (carputils standard parser, carpexample, mesh.Block with set_bath, carp_cmd, job.carp). Bath from mesher (tag 0, zero fibres). Electrograms from openCARP phi_e. Kept as our code, with reasons: CV tuning on tetrahedral strips (tuneCV uses hexahedra, DL-10); electrogram measures and change points (the pipeline); contribution field (openCARP outputs only the summed signal; its lead-field module is not in openCARP 19.0.0); the extracellular solver as an independent check.

## DL-2026-11-01-05 - Forward model (pseudo-bidomain vs bidomain source): decided by the final settings run
- **Status:** Open; rule fixed. **Irreversible:** Yes, once production runs start.
- **Finding (VL-2026-11-01-02):** openCARP pseudo-bidomain uses the monodomain conductivity as the extracellular source; full bidomain uses sigma_i. With the tuned conductivities (sigma_i > sigma_e) the two differ by ~4x in amplitude and in shape.
- **Rule:** decided first, on the 0.5 mm Vm with the 10 mm run bath: if sigma_m vs sigma_i passes (calibrated PP, supported change points, contribution field), openCARP pseudo-bidomain is the production model; otherwise the sigma_i source is required, and the route (openCARP full bidomain, or the verified sigma_i solve on openCARP Vm) is decided with the measured costs (optional bidomainCheck stage). The bath study and the 0.25 mm run then use only the chosen model (simplification, 1 Nov 2026, before any production result).

## DL-2026-11-01-06 - Prepacing: bath switched on only from 1050 ms
- **Status:** Decided (Matilda, 1 Nov 2026). **Irreversible:** No.
- **Decision:** production runs pace beats 1-3 in monodomain (Vm and activation times written as before; no extracellular solve), checkpoint at 1050 ms, and restart with the bath. Standard openCARP checkpoint/restore; equal to a continuous run (VL-2026-11-01-04).
- **Why:** the bath solve is a whole-domain linear system at every output step; about three-quarters of those solves fell before the frames the analysis reads. Halves the bath cost and phi_e storage.

## DL-2026-11-01-07 - boomeramg solver flavour; phi_e written only on the electrode layer
- **Status:** Decided (1 Nov 2026, under Matilda's instruction to optimise without losing anything required). **Irreversible:** No.
- **Solver:** carputils flavour boomeramg (hypre AMG; standard carputils option) for openCARP; same tolerance as the default, results equal to float precision, about 8x faster bath solves (VL-2026-11-01-05). preflight smoke-tests it and falls back to the default (petsc) if this openCARP build lacks hypre.
- **Output:** phi_e written only for bath nodes within 3 mm of the tissue surface (openCARP dataout_e = 3). The whole domain is still solved. Nothing downstream reads phi_e elsewhere; an electrode sample outside the layer stops the analysis. Sweeps with larger standoffs must raise phieOutputHeightMm.

## DL-2026-11-02-01 - Forward model: openCARP pseudo-bidomain (sigma_m), overriding the rule outcome
- **Status:** Decided (Matilda, 2 Nov 2026). **Irreversible:** No. Recorded in finalise.json (`forwardModelDecision`).
- **Decision:** every electrogram is openCARP's own phi_e from pseudo-bidomain runs (source sigma_m). The ground truth uses the same source (contribution_field with "sigmaM"; openCARP pseudo-bidomain phi_e equals that solve exactly, checked in every run).
- **Rule outcome it overrides (forwardModel stage, 1 Nov 2026, 0.5 mm, 10 mm bath):** the rule chose sigma_i because one criterion failed: contribution-field L1 max 0.109 vs a PROPOSED tolerance of 0.10 that was never confirmed (median 0.003, 48 recordings). The confirmed criteria passed: calibrated PP max 4.2 % (tolerance 5 %; median 0.06 %), supported change-point shift 0.1 mm (0.5 mm), no support-status change. Raw amplitude differs about 4x (gain 0.158 vs 0.040); the pipeline's gain calibration to 1.44 mV removes it.
- **Consequence:** the production model carries a flag that this choice overrides the rule. The thesis should state the override and the numbers above.
- **Scar (R10):** the effect of a conducting vs insulating scar, measured on the sigma_i solve (10 mm grounded-plate bath, 1 Nov run), was PP max 23 %, supported change-point shift 3.8 mm, 6 support changes: the scar choice matters. Still your decision.

## DL-2026-11-02-02 - Bath: modelled as Leenknegt et al. (2023), stated as an assumption; bath study removed
- **Status:** Decided (Matilda, 2 Nov 2026). **Irreversible:** No. finalise.json `bath`.
- **Decision:** 5 mm of blood (0.74 S/m) on every side of the tissue slab, no-flux outer walls (openCARP default), extracellular ground on the nodes of a 1 mm cube at one bath corner (max x, max y, max z). Unipolar potentials are referenced to that corner. Same in the FEM check and the ground truth (ground taken from the run's ground.vtx).
- **Source:** Leenknegt, Panfilov and Dierckx (2023), Front. Physiol. 14:1213218: openCARP slab "surrounded by a bath of 5 mm in all three directions", Neumann outer boundary, "a grounded region of 1 mm3" at a bath corner. They did not test bath size; neither do we. Stated as a modelling assumption.
- **Why:** the 1 Nov bath study (grounded top face, 5 / 10 / 20 mm, sigma_i solve) did not converge (10 vs 20 mm: PP max 63 %, median 0.4 %; change points 3.2 mm). No published electrogram study found reports a bath-size convergence test; most use an unbounded volume conductor, and those with a bath use small baths with no-flux walls and a small ground (literature check 2 Nov 2026). The grounded top face was our own choice and made the reference move with the bath.
- **Consequence:** the bath study and forward-model stages are removed from the final settings run; the 1 Nov results stay in 3D_Code/runs/final as history. The final settings run restarts in 3D_Code/runs/final_v2 (all simulations use the new bath).

## DL-2026-11-02-03 - CV calibration through beta; conductivities kept at measured values
- **Status:** Decided (2 Nov 2026, delegated by Matilda: "make an informed decision"). **Irreversible:** No. finalise.json `tissue.cvCalibration = "surfaceToVolume"`.
- **Problem:** reaching 0.90 m/s (DL-2026-10-28-03) by scaling the Roberts and Scher 1982 conductivities needed s = 6.42 (0.5 mm) and 5.44 (0.25 mm): sigma_il 1.9-2.2 S/m and sigma_el 0.64-0.75 S/m, i.e. tissue about as conductive as blood (0.74 S/m). Measured values are several times lower (Roberts and Scher 1982: sigma_il 0.344, sigma_el 0.117 S/m; review values for canine atria and ventricles: intracellular longitudinal about 0.06-0.42 S/m). With a bath, the tissue-to-blood conductivity ratio shapes phi_e, which is the quantity this project measures.
- **Decision:** keep the measured Roberts and Scher conductivities (ratios and magnitudes) and reach the CV target with the surface-to-volume ratio: beta = 0.14 um^-1 / s (0.0218 um^-1 at 0.5 mm, 0.0257 um^-1 at 0.25 mm; openCARP imp_region[].cellSurfVolRatio, default 0.14 um^-1). The tune stage is unchanged; its scale s converts exactly.
- **Why this placement:** monodomain Vm depends only on sigma / (beta Cm), so propagation is identical either way (checked: Vm within 4e-5 mV, VL-2026-11-02-03). beta does not appear in the pseudo-bidomain extracellular equation or in the contribution field; the conductivity magnitudes do. Putting the calibration factor in beta keeps every quantity that shapes the electrograms at a measured value.
- **Limitation (state in thesis):** beta = 0.022-0.026 um^-1 is below physiological estimates (openCARP default 0.14 um^-1); it is a calibration parameter absorbing the gap between measured conductivities and the clinical CV with the Courtemanche model at these mesh spacings, not a measurement. Most atrial modelling studies instead tune conductivities to the CV target (e.g. Ferrer et al. 2015; openCARP tuneCV); carputils' CVtuning exposes beta as an input. Roberts and Scher are canine ventricular values (sigma_il > sigma_el); canine atrial measurements (Le Guyader et al. 1995, 1997, seen only in a 2024 review) have sigma_el > sigma_il; a switch would need the primary papers.
- **Effect on earlier results:** the 1 Nov forward-model comparison used scaled conductivities (same ratios, so sigma_m vs sigma_i is unchanged in relative terms; only the tissue/blood ratio differs). Electrogram amplitudes change (gain calibration absorbs a common factor).

## DL-2026-11-03-01 - Final model branch (finalModel): one configuration, one package, one pipeline
- **Status:** Decided (Matilda, 3 Nov 2026). **Irreversible:** No.
- **Decision:** the production model and its analysis are rebuilt as the `efov` package with every scientific value in `config/final_model.json` (with provenance) and one resumable pipeline (`python -m efov run`, notebooks 00-06). Development code (2D sheet, pilot, readiness kit, finalise.py) stays on `pilot_v2` and in the history.
- **Model content unchanged from v3-reference (b473499):** same carputils slab, bath, forward model, conductivities, beta calibration, electrodes, window rule, recordings, calibration, noise, change points and contribution field.
- **R10 scar:** insulating (all six conductivities 1e-6 S/m), decided 3 Nov 2026.
- **Values not yet confirmed from the workstation final settings run** (domain, mesh, time step, output step, beats, beta, stimulus strength): the config holds the expected rule outcomes, flagged `awaitingWorkstation`; `python -m efov import-production-model` replaces them from productionModel.json. `acceptance.json` reports `productionReady = false` until they are confirmed.
- **Changes in behaviour (review issues I1-I5 of the 2 Nov code review):**
  - every profile is kept in the change-point tables, including those too short to fit (I1);
  - physiology now includes a CV check against the target; physiology and profile prerequisites decide `eligibleForInterpretation` (I2, I3);
  - each stage stores a fingerprint of the settings it used and its parents'; stale outputs are refused, not reused (I4);
  - ground-truth checks (share sum, reconstruction residual, scale = gain) stop the run if they fail (I5);
  - the measurement window records whether it starts at the stimulus floor (DL-2026-10-28-08 consequence).
- **Removed (not needed by the final model):** the unbounded-medium lead-field ground truth and FOV80, the electrode-transfer extracellular electrograms (sigma_i route), the comparison and decision stages of finalise.py, the pilot-style run-folder adapter.

## DL-2026-11-03-02 - Final model v4 (finalModelV2): decisions FM-01 to FM-22
- **Status:** Decided 3 Nov 2026. Matilda gave the goals (peak frequency, transmural depth, openCARP lead fields, all simulator settings in a .par, endo/epi/far-field contributions, fibre models, an ablation-like lesion, Wang's change-point code) and delegated the open choices ("what most other researchers would do"). Full reasoning and sources: project document `claude/final_model_plan_2026-11-03.md`. **Irreversible:** none until production runs start.
- **FM-01** Every openCARP setting in `config/model.par`; each launch reads one resolved file (`<job>/<step>.par`: model.par with overrides in place, then run paths). carputils adds only solver-flavour options (recorded in runRecord.json).
- **FM-02** `config/model.json`: `sweep` (baseline, planned values, what each acts on) and `fixed` (settings openCARP cannot read). openCARP rejects unknown parameters, so mesher and analysis settings cannot live in the .par.
- **FM-03** Domain 40 x 60 mm, 4 beats at 500 ms, dt 20 us fixed now (the 1 Nov rule outcomes); the new mesh invalidates the workstation final-settings run. 500 ms cycle length: assumption [VERIFY].
- **FM-04** Mesh 0.25 mm in-plane, 0.1 mm through the wall (mesher -resolution[2]; AnisotropicBlock). 0.1 mm isotropic rejected (40.7 M nodes).
- **FM-05** Bath on the catheter side and laterally; no-flux epicardium (bath below the tissue removed after meshing). Blood all round (Leenknegt et al. 2023) kept as sweep variant `epicardialMedium = blood`.
- **FM-06** Fibre models uniform / bilayer / linearRotation (ported from Shared_Pipeline/fibres.py, DL-2026-10-28-12); .lon written per run from element depth. Angles are assumptions.
- **FM-07** Lesion `ablation`: no intracellular coupling (1e-6 S/m), healthy extracellular tensor; brackets `ablationPorated` (sigma_i + sigma_e) and `insulating` (earlier reference).
- **FM-08** Regions: inferior (1.1 mm, bilayer) and roof (1.6 mm, uniform), from Whitaker et al. 2021 (CCTA, n = 193) and Pashakhanloo et al. 2016.
- **FM-09** Acquisition at 2 kHz (EnSite X; Europace 2024;26:euae154); 30-500 Hz band-pass for differential signals (Wang supplement; replaces the unsourced 450 Hz).
- **FM-10** Peak frequency: complex Morlet (omega0 = 6, L1), PF trace, 10 % magnitude gate, highest trace value; undetectable when PP < 0.04 mV. Global scalogram peak reported as an alternative.
- **FM-11** Bipolar/omnipolar PP within t_PF +- 30 ms inside the window (EnSite voltage annotation).
- **FM-12** Change points by Wang's Rmd itself (R/wang_replication.R: MD5-checked, chunk by chunk, six one-slab replacements: lm for lmer). Python profile least squares removed. All 16 unipoles, 9 bipoles and 36 omnipoles per placement are exported.
- **FM-13** Independent FEM forward-model check removed.
- **FM-14** Lead fields from openCARP Laplace solves (experiment 2, 1 uA total-current extracellular stimulus on the electrode's node set, extracellular conductivities g_i + g_e). Electrodes = mesh nodes inside each cylinder (replaces 8 surface samples). Pre-declared fallback: `leadFieldMethod = python` (same discrete problem) if W0 fails.
- **FM-15** Contribution classes: near/far (lateral 4 mm, Schlageter et al. 2024) x endocardial/epicardial half; signed share and |c|-mass; continuous curves alongside.
- **FM-16** Ground truth unchanged in definition (DL-2026-11-01-02); every recording at 17 placements (every 2 mm), both gaps.
- **FM-17** Sweep runner with a simulation cache keyed by the simulation fingerprint; wave angle as a sweep variable (stimulus band at the trailing edge of the wave direction).
- **FM-18** beta tuned on a strip at the production mesh (method DL-2026-10-28-10); `efov tune --write-par` writes it into model.par and config/tuning.json. The CV gate is the tune stage; slab CV is reported.
- **FM-19** Gain computed once in the inferior reference run, fixed for all other runs. R7 closed: 1.44 mV = Wang Table 2 inferior healthy bipolar median. Noise 5 uV: assumption [VERIFY].
- **FM-20** Storage: no output restriction on Vm; a variant whose simulation no other variant reuses is compacted after its report (Vm kept for the analysis frames; vm.igb and phie.igb deleted). The baseline simulation is never compacted.
- **FM-21** Ground-truth outputs: element shares (npz, legacy VTK for ParaView), sections, depth and lateral profiles, classes, boundary sequence, descriptive shape metrics.
- **FM-22** (new, found during implementation) A 0.25 ms output step is not a multiple of dt = 20 us. Kept the earlier verified dt 20 us and 0.2 ms output step (5 kHz) and resample to 2 kHz with scipy.signal.resample_poly (2/5, Kaiser-window FIR as the anti-alias filter) instead of integer decimation. Changing dt to 25 us was rejected: unverified numerics.

## DL-2026-11-03-03 - Final model v4 revised after the independent review: decisions FM-23 to FM-30
- **Status:** Decided 3 Nov 2026 by Matilda after the independent methodology review (reviewer subagent; findings relayed the same day). Her instructions: no bath convergence study; fix the lesion as proposed; no scaling to Wang's values, keep them separate; no comparison of openCARP values at all; the reviewer's recommendations for the rest. **Supersedes:** DL-2026-11-02-01 (sigma_m forward model), FM-07 geometry, FM-10 (PF primary), FM-13/FM-14 checks, FM-15 wording, FM-19 (calibration), FM-20 (phi_e output). **Irreversible:** none until production runs start.
- **FM-23 Forward model: monodomain Vm with the sigma_i source, by reciprocity (supersedes DL-2026-11-02-01).** openCARP's pseudo-bidomain drives phi_e with sigma_m (VL-2026-11-01-02). With sigma_i,l/sigma_i,t = 5.7 and sigma_e,l/sigma_e,t = 1.5 (model.par), sigma_m under-weights fibre-parallel sources relative to cross-fibre ones by about 2.3 x; with bilayer fibres this biases the endocardial/epicardial split, the core deliverable. The bidomain extracellular equation has the sigma_i source; recovering phi_e from a monodomain Vm with a forward model is established practice: Potse et al. 2006 (IEEE TBME, "A comparison of monodomain and bidomain reaction-diffusion models for action potential propagation in the human heart"; abstract read 3 Nov 2026, volume and pages [VERIFY]) computed extracellular potentials of the monodomain model with a high-resolution forward model and report electrograms visually indistinguishable from the bidomain's. Implementation: tissue runs monodomain throughout (bidomain = 0, no phi_e output, no prepacing restart); every distinct electrode gets an openCARP Laplace lead field (config/leadfield.par, sigma_i + sigma_e, 1 uA over its node set); electrogram phi(t) = -z^T S_i Vm(t), exact for the discrete problem. Lead fields cached per conductor fingerprint and node set (wave-angle, CV, electrode, noise and analysis variants reuse them). Our own evidence for the sigma_i solve: full openCARP bidomain vs the sigma_i solve on bidomain Vm differed by 1.2 % (VL-2026-11-01-02). **Cost:** one Laplace solve per distinct electrode (about 712 per condition at baseline, `efov plan` prints the number); W0 measures the time per solve. **Remaining limitation:** monodomain Vm has no bath loading (activation bidomain vs pseudo-bidomain: max 3.8 ms, median 1.1 ms, VL-2026-11-01-02).
- **FM-24 Lesion: ablation strip plus isolated viable tissue (supersedes the half-plane of FM-07).** Single-ring isolation leaves a ring of ablated tissue and viable but isolated tissue inside it. Lesion condition: transmural strip boundary - 5.6 mm <= y < boundary (tag 5; conductivities of FM-07); tissue beyond it stays viable (tag 1, coupled, with its own Vm) but is never stimulated (stimulus restricted to y >= boundary). 5.6 mm = mean lesion width in Wang's supplement (5.6 +- 1.2 mm, Friedman et al. 2021 as cited there). This matches the 2D specification of October ("ablation line plus the viable but isolated posterior wall"). Contribution coefficients of each element sum to zero, so resting isolated tissue contributes nothing unless its Vm varies in space.
- **FM-25 Amplitudes in physical model units; no calibration (supersedes FM-19).** The gain to Wang's 1.44 mV fed a clinical voltage into the model the validation compares against, and with noise in absolute mV it also set the signal-to-noise ratio and hence possibly the change point. Wang's medians (inferior 1.44, roof 0.97 mV, Table 2) are reported beside the model's in amplitudes.json and the report, never applied. **Pre-registered primary test (before any result):** the change in each of Wang's change points per doubling of noise (noiseSdMv 0.0025, 0.005, 0.01, 0.02 mV; `noiseDoubling.csv` from the sweep). A change point that moves with the noise floor measures detectability, not sensing volume.
- **FM-26 No comparison of openCARP outputs (supersedes the gates of FM-13/FM-14 and checks C3 and W0 b-c).** By Matilda's instruction no check compares openCARP values with other openCARP values or with our own solver. The ground-truth reconstruction gate is removed (it is now an identity: the electrograms are computed from the same lead fields); the ground truth keeps the share-sum check. W0 now only confirms the Laplace mode runs and measures its cost. The Python solver stays for the unit tests (reciprocity against a direct solve of our own code) and as FM-14's fallback. **Consequence, stated as a limitation:** the physical amplitude scale rests on openCARP's total-current convention for the Laplace stimulus (crct.total_current, 1 uA), which no check in the pipeline confirms; shares, shapes and change-point positions in the clean signal do not depend on that scale, the signal-to-noise ratio does.
- **FM-27 Peak frequency: two co-primary definitions.** The reviewer showed that with 5 uV white noise the 'highest trace value' PF rises toward 460-480 Hz as signal falls, opposite to Wang's fall into scar. Both definitions (EnSite-style highest trace value; global scalogram maximum) are exported as separate Wang datasets and analysed by her code; `pfNoiseRobustness.csv` reports clean against noisy PF per distance bin (descriptive). A measured clinical noise model remains open (Junction 1).
- **FM-28 Measurement window per placement (refines DL-2026-10-28-08).** One window from activation under the whole y range breaks for oblique waves (it would span activation across 60 mm). Each placement gets its window from activation under its own footprint (same rule, same length for all placements); for the 0 deg reference wave all windows are identical.
- **FM-29 Code fixes.** Vm is read resampleMarginMs = 10 ms beyond the latest window end, so the resampling FIR end effect (half-length 5 ms at 2/5) stays outside every window. Wording: the R wrapper checks Wang's Rmd by MD5 and parses its chunks itself (not SHA-256 or knitr::purl, as the plan said); the output rate is 5 kHz.
- **FM-30 Near/far cut-off presented as a convention.** Schlageter et al. 2024's 4 mm is an electrode-to-tissue distance, not a lateral distance in tissue; citing it for the lateral cut-off was wrong. nearFieldRadiusMm = 4 mm is one HD Grid electrode spacing, a stated convention; the continuous lateral and depth curves are the primary ground-truth result.
- **Not acted on (recorded):** bath convergence (Matilda: not needed; the 5 mm Leenknegt bath stays an assumption, DL-2026-11-02-02); the roof prediction (a thicker roof predicts higher voltage, Wang measured lower: to be pre-registered as a falsifiable regional prediction when the roof run is analysed); canine-LV conductivities; a 3 x 3 wall x standoff block.

## DL-2026-11-04-01 - Final model made runnable on real openCARP: decisions FM-31 to FM-33
- **Status:** 4 Nov 2026, Matilda's instruction: finalise the model fully this session; the next run is a production run. Evidence: VL-2026-11-04-01 to -04. **Irreversible:** none until the production run starts.
- **FM-31 Lead fields from the Python route (pre-declared fallback of FM-14, triggered by W0).** openCARP 19.0.0's Laplace mode reads the 1 uA electrode current but writes a potential of exactly zero: it applies only Dirichlet conditions. `fixed.groundTruth.leadFieldMethod = "python"`: the same discrete problem (P1 on the run mesh, sigma_i + sigma_e in tissue and lesion, g_bath in the bath, corner ground, 1 uA spread equally over the electrode's nodes), assembled once per condition and solved with AMG-preconditioned CG (rtol 1e-10). Solves now run in forked worker processes (`--np`), each writing its own file atomically, so an interrupted stage resumes. Consequences: (1) the electrograms and the ground truth still come from one lead field per electrode, exactly as FM-23; (2) the physical amplitude scale now rests on our own units derivation (efov/leadfield.py docstring), not on openCARP's total-current convention (the FM-26 limitation changes accordingly); (3) the thesis states that openCARP computes Vm and the forward model is our P1 solve on openCARP's mesh. The openCARP route is kept for a build whose Laplace mode accepts current stimuli; W0 re-tests it as a diagnostic.
- **FM-32 beta tuned at the production mesh:** 0.027292 /um (strip CV 0.9015 m/s, target 0.90; VL-2026-11-04-02), written to model.par and config/tuning.json. This replaces the 0.5 mm value 0.0218 and closes the `awaitingWorkstation` item for beta. Tuned with conda-forge openCARP 19.0.0; the production physiology stage reports the slab CV on the workstation build.
- **FM-33 CV target stays along fibres; the slab surface CV is reported (confirms DL-2026-10-28-03; decided by Matilda 4 Nov 2026).** With bilayer 0/90 deg fibres the wave along x is longitudinal in the endocardial half and transverse in the epicardial half, so the coupled slab conducts more slowly than the strip: first-beat surface CV 0.735 m/s (bilayer) vs 0.876 m/s (uniform 0 deg) on the small slab at the production mesh, wall 1.1 mm, tuned beta (VL-2026-11-04-05). Expected last-beat production surface CV about 0.75 m/s, inside the reported clinical central values (0.70-1.25 m/s) but below the 0.90 m/s direction-mixed median the target came from. Options weighed: tune the slab surface CV instead (matches the clinically measured quantity, but every fibre variant would need its own beta, so the fibre sweep would also change conduction). Chosen: keep one tissue (beta tuned along fibres) for every fibre variant, so the fibre sweep changes only fibre architecture; the physiology stage reports the slab CV of every run; the thesis states the gap as a limitation.

## DL-2026-11-07-01 - FM-34 Lead fields from openCARP Laplace solves; the Python lead-field solver removed
- **Status:** Decided by Matilda, 7 Nov 2026 ("use the Laplace openCARP solver and not our own"). Supersedes FM-31; refines FM-14. **Irreversible:** none until production runs start. Evidence: VL-2026-11-07-01 to -03.
- **What openCARP can and cannot do (openCARP 19.0.0):** extracellular current stimuli (crct.type 1; with total_current, with a balancing electrode, or as a density) are ignored in Laplace (experiment 2) and in bidomain runs: the potential stays zero (Laplace) or at round-off, about 1e-11 mV (bidomain). An extracellular potential stimulus (crct.type 2, closed loop) is applied as a Dirichlet condition, and the Laplace operator is g_i + g_e in tissue and g_bath in the bath.
- **Method:** each distinct electrode gets one openCARP Laplace run (config/leadfield.par): its mesh nodes held at 1 mV, the 1 mm corner cube at 0 mV, the model's own g_i, g_e and g_bath. The electrode current is I = u^T K u / V (K = sigma_i + sigma_e in tissue and lesion, g_bath in the bath, assembled once per condition from openCARP's mesh), and the lead field is z = u / I (mV/uA). Reciprocity for a floating equipotential electrode gives phi_S(t) = -z^T S_i Vm(t), exact for the discrete problem; electrograms and ground truth use z exactly as before (FM-23).
- **Electrode model changes:** from the mean potential over the electrode's bath nodes (passive averaging) to a conductor (one shared potential, no net current), which is closer to a metal electrode. On the test slab the lead fields differ by 0.07 % (median) on the tissue nodes, but the electrogram weights -S_i z, which depend on the field gradient at the tissue surface 0.25-1 mm below the electrode, differ by 1.8 % (median; up to 11 %). End to end: unipolar signals 1.0 % RMS (largest sample 4.6 %), clean bipolar and omnipolar PP 1.1 % and 1.0 % (median), ground-truth class fractions up to 0.024, Wang's change points up to 0.18 mm (VL-2026-11-07-03). A real but small change; the thesis describes the electrode as an equipotential conductor.
- **Solver tolerance:** cg_norm_ellip = 2 (relative residual, 1e-10). openCARP's default (0) is an absolute preconditioned residual; on a 0.25 mm test box the initial residual already met 1e-10 and openCARP returned u = 0 away from the electrode without iterating. leadFieldFromPotential now refuses a potential that is zero on most of the mesh, or outside [0, V].
- **Removed:** PythonLeadFieldSolver, the forked-worker solve, fixed.groundTruth.leadFieldMethod. Kept: the P1 assembly for the sigma_i source term and contributions (ContributionModel, unchanged) and for I. The tests keep an independent direct solve (tests/helpers.py) as their oracle; tests/test_opencarp_laplace.py compares openCARP with it.
- **Cost (VL-2026-11-07-02):** one openCARP run per electrode, about 82 s and 11-16 GB at the production mesh (one process; estimate from 0.31 and 0.51 M-node runs). Runs are scheduled by available memory (laplaceSchedule; on a 128 GB workstation about 5 at a time x 4 MPI ranks); about 16 core-hours per condition, roughly 2.5-4.5 h of wall time per condition. Every variant with a new conductor or new electrode positions needs its own lead fields.
- **Not verified:** the workstation openCARP build (W0 re-checks the boundary values, the maximum principle, the electrode current and whether current stimuli are applied).

## DL-2026-11-07-02 - FM-35 Lead fields and electrograms from openCARP's lead-field module (openCARP >= 20)
- **Status:** Decided by Matilda, 7 Nov 2026 (option 1 of the method comparison; "use the v20 openCARP"). Supersedes FM-34 (the openCARP 19.0.0 Laplace workaround, kept in git at 89acf3a). **Irreversible:** none until production runs start. **Not yet run on openCARP 20** (not available in Claude's sandbox: conda-forge has 19.0.0, git.opencarp.org and Docker Hub are blocked); W0 is the first test.
- **Method (openCARP manual 4.2.6, physics/leadfield.cc, example 02_EP_tissue/09_leadfield):** one post-processing run per condition (experiment 4, post_processing_opts 64, config/leadfield.par). Every distinct electrode is a lead-field stimulus (crct.type 13, strength 1 = 1 uA spread over its node set by volume: strength x 1e12 / volume, then the mass matrix), the corner cube is the ground (crct.type 3). openCARP solves the bulk system once per electrode (LF_<name>.igb, tissue nodes) and streams the monodomain vm.igb into V(t) = Z . (K_i Vm(t)) per electrode (ECG_<name>.dat) at the Vm output step.
- **bidomain = 1 in the lead-field run:** openCARP's K_i is the parabolic solver's rhs_parab, which the bidomain elliptic step uses as its sigma_i source; in monodomain and pseudo-bidomain mode that matrix is built with sigma_m (consistent with VL-2026-11-01-02), which would bias the endocardial/epicardial weighting (FM-23). W0 runs bidomain = 0 as a diagnostic.
- **Electrode model:** back to an averaged electrode (as FM-14 intended), now volume-weighted: the reading is b^T phi_e with b = M chi_S / (sum of the lumped masses of S). The conductor model of FM-34 changed amplitudes by about 1 % on the test slab.
- **What efov still computes:** the element contributions for the ground truth (ContributionModel, from openCARP's lead fields and the sigma_i tensors). efov.electrograms checks openCARP's traces against -Z^T S_i Vm for 8 electrodes per condition (relative L2 < 1e-3; float32 files) and stops otherwise, so electrograms and ground truth stay consistent.
- **Removed:** the Laplace route (laplacePotential, current normalisation, run scheduling). Tests use a direct-solve oracle that reproduces openCARP's electrode current; tests/test_opencarp_leadfield.py (skipped below openCARP 20) compares openCARP with it.
- **Cache:** lead fields are cached per conductor and node set as openCARP's LF igb files. A run whose lead fields are all cached loads them (lf_dir) and only streams Vm; otherwise it constructs all of its run's lead fields in one openCARP run.
- **To verify on openCARP 20 (W0):** lead fields vs direct solve < 1e-6; electrograms vs -Z^T S_i Vm < 1e-4; the bidomain = 0 diagnostic differs; parameter names (crct.type 13, post_processing_opts, lf_vmfile, lf_dir, ecg_timedt) accepted; run time (checks/leadfield_cost.py).
