# The final model (v3-reference): what it is and why

Every value below is in `config/final_model.json`. Decision IDs refer to `docs/decision_log.md`,
verification IDs to `docs/verification_log.md`, sources to `docs/parameter_sources.md`.

## Purpose

A controlled, idealised preparation in which the true contribution of every piece of tissue to each
recording is computable. The clinical estimator (a decay-to-plateau change point of voltage against
distance into a lesion) is applied to simulated recordings and compared with that ground truth, in the
same model. The claim the model supports is about this preparation, not a universal sensing radius.

## Model

| Part | Choice | Why / source | Status |
| --- | --- | --- | --- |
| Geometry | 40 x 60 mm slab, 1.5 mm wall, 0.5 mm tetrahedral mesh | wall: posterior LA CT mean (Li et al. 2024); domain and mesh: final settings run rules | wall decided; domain and mesh await the workstation result |
| Fibres | uniform along x, transversely isotropic | controlled abstraction (lateral wall/roof per Pashakhanloo et al. 2016) | decided |
| Cells | Courtemanche (healthy), 1,000 bench beats at 500 ms | human atrial model | decided |
| Conductivities | sigma_i 0.344 / 0.060, sigma_e 0.117 / 0.080 S/m (l / t = n) | Roberts and Scher 1982, measured, unscaled (DL-2026-11-02-03) | decided |
| CV 0.90 m/s along fibres | beta = 0.0218 um^-1 (0.14 / 6.42) | van Schie et al. 2021 target (DL-2026-10-28-03); beta carries the calibration so every quantity shaping phi_e stays measured | target decided; beta to confirm; checked in every run |
| Lesion | y < 0, transmural, insulating (all six conductivities 1e-6 S/m) | idealised electrically silent region (R10, 3 Nov 2026) | decided |
| Bath | 5 mm blood (0.74 S/m) on every side, no-flux walls, 1 mm^3 ground at the (max x, max y, max z) corner | Leenknegt et al. 2023, assumption (DL-2026-11-02-02); Traugott et al. 1981 | decided, untested |
| Forward model | openCARP pseudo-bidomain; electrograms = openCARP phi_e (source sigma_m) | DL-2026-11-02-01 (overrides the 1 Nov rule outcome) | decided |
| Pacing | 2 ms transmembrane strip (0.5 mm) at x min, viable nodes only, 500 ms, 4 tissue beats | planar wave parallel to the boundary | beats await the workstation result |
| Numerics | dt 20 us, output 0.2 ms, Crank-Nicolson, mass lumping, tolerances 1e-8; boomeramg; monodomain prepacing to 1050 ms | final settings run rules; identical results with prepacing and boomeramg (VL-2026-11-01-04/05) | dt and output await the workstation result |
| Catheter | 4 x 4, 4 mm pitch, 1 x 1 mm cylinders along x, gaps 0.25 and 1 mm, 8 samples per electrode; 65 positions y = -16..16 mm | HD-Grid-like geometry | decided |

## Analysis

| Step | Definition |
| --- | --- |
| Window | intact last-beat activation of the top surface under the grid: start = max(first - 15, stimulus end + 5) ms, end = last + 30 ms |
| Recordings | unipole E5 (unfiltered); bipole E5-E6\|E9 (larger of the two orthogonal pairs, per placement); omnipoles E5-E6-E9 and E9-E10-E5 (2 deg direction search); differential signals band-passed 30-450 Hz, causal Butterworth order 2 (4th-order band-pass) |
| Calibration | gain so the median of the nine bipoles at the intact y = 0, 0.25 mm placement is 1.44 mV (imposed; source open, R7) |
| Noise | 3 independent realisations of 5 uV Gaussian noise per electrode (measurement repeats, not biological replicates) |
| Change points | 0 < d < 15 mm, PP > 0.01 mV, 99.9 % trim, exponential vs power law by AIC, zero-slope plateau, profile least squares; supported if decaying, not at the grid edge, AIC gain >= 2 |
| Ground truth | signed element shares of each recording's energy, reciprocity lead field with the same sigma_m source and ground; 0.5 mm voxels; no radius |

## Checks in every run (`acceptance.json`)

* Implementation (stop the run if they fail): openCARP phi_e equals the independent sigma_m solve
  (relative RMS <= 1 %, correlation >= 0.9999); ground-truth shares sum to 1, reproduce the measured
  electrogram (residual <= 1e-3) with scale = gain.
* Physiology (recorded): capture, repolarisation, quiet lesion core, planar wave in +x, beat stability,
  CV within 2 % of 0.90 m/s.
* Profile prerequisites (recorded): filter history, stable far plateau, lesion transition.
* Change points are eligible for interpretation only if all of these pass and the matched intact
  control has no supported change point.

## Known limitations to state in the thesis

* Bath geometry and reference copy Leenknegt et al. (2023) without a convergence test; blood also lies
  against the epicardium; the ground corner sits on the viable side (max y), so unipolar signals are
  not referenced symmetrically about the boundary (differential signals cancel the reference).
* Pseudo-bidomain: no bath loading of propagation; sigma_m rather than sigma_i source (DL-2026-11-02-01;
  the 1 Nov comparison found calibrated PP within 4.2 % and change points within 0.1 mm).
* beta = 0.0218 um^-1 is a calibration value below physiological estimates; conductivities are canine
  ventricular (Roberts and Scher 1982).
* At 0.90 m/s on the 40 mm slab the derived window starts at the 5 ms stimulus floor, so far-field
  from the pacing strip can enter it (DL-2026-10-28-08; flagged in window.json).
* Convergence evidence for output step, time step, beats and domain comes from 4 mm-wall,
  unbounded-medium ladder runs; electrode quadrature (8 samples) was checked on the 4 mm pilot only.
* Healthy Courtemanche kinetics, one planar activation direction, a straight insulating boundary.
