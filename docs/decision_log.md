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
