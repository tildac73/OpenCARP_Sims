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
