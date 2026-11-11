# Final model: edit plan and decision register (3 Nov 2026)

Branch: `finalModel` at `7cee884` (tildac73/OpenCARP_Sims). **Implemented 3 Nov 2026 on branch `finalModelV2`; revised the same day after the independent review (Section 0, FM-23 to FM-30)** (decision log DL-2026-11-03-02; verification VL-2026-11-03-02/03). Changes made during implementation are marked **[implementation]** below.
Decision IDs FM-01 to FM-22 go into `docs/decision_log.md` when each is implemented.
Status labels:
- **Decided**: follows Matilda's instruction or a sourced value.
- **Assumption**: no primary source; stated as such in the thesis.
- **Confirm**: a choice Claude made that Matilda should check before implementation.
- **[VERIFY]**: a value or claim not yet checked against a primary source.

Literature access this session: Scite lapsed, PMC behind CAPTCHA, several publishers blocked. Every source below was actually read (abstract or page) unless marked [VERIFY].

## 0. Revision after the independent review (3 Nov 2026, evening): FM-23 to FM-30

An independent reviewer (subagent with no access to the reasoning) checked the implementation against the Working Methodology, the Proposed Project, Wang's manuscript and supplement, and the literature. Matilda decided: no bath convergence study; fix the lesion as proposed; no scaling to Wang's values (keep them separate); no comparison of openCARP values at all; the reviewer's recommendations for the rest. Decision log DL-2026-11-03-03; verification VL-2026-11-03-04 (sandbox, no openCARP).

| ID | Decision | Supersedes |
|---|---|---|
| FM-23 | **Forward model: monodomain Vm + sigma_i source, by reciprocity.** openCARP's pseudo-bidomain drives phi_e with sigma_m; with the model's unequal anisotropy (intracellular 5.7, extracellular 1.5) that under-weights fibre-parallel sources by ~2.3x and biases the endo/epi split. Tissue runs are now monodomain (no phi_e output, no prepacing restart); every distinct electrode (712 per condition at baseline) gets an openCARP Laplace lead field; phi = -z^T S_i Vm. Lead fields cached per conductor fingerprint and node set. Precedent: monodomain Vm with a forward model for phi_e (Potse et al. 2006, IEEE TBME; abstract read). Limitation kept: no bath loading in Vm. | DL-2026-11-02-01, FM-13, FM-14 source, FM-20 phi_e |
| FM-24 | **Lesion = 5.6 mm ablation strip + viable isolated tissue beyond it** (never stimulated), as after single-ring isolation and as in the October 2D spec. Width: Wang supplement (5.6 +- 1.2 mm, citing Friedman et al. 2021). | FM-07 geometry (half-plane) |
| FM-25 | **Physical model units; no calibration.** Wang's medians (1.44 / 0.97 mV) reported beside the model, never applied. Pre-registered primary test: change-point shift per doubling of noise (0.0025-0.02 mV), `noiseDoubling.csv`. | FM-19 |
| FM-26 | **No openCARP-vs-openCARP or openCARP-vs-own-solver comparisons.** Reconstruction gate removed (now an identity); share-sum check kept; W0 = Laplace runs + cost; C3 removed; C5 = lesion-strip activation check. Limitation: the physical amplitude scale rests on openCARP's total-current convention, unchecked. | FM-13/14 gates, C3, W0 (b-c) |
| FM-27 | **PF: EnSite-style and global maximum co-primary**, each exported as its own Wang dataset; `pfNoiseRobustness.csv` (clean vs noisy PF by distance). | FM-10 primary/alternative |
| FM-28 | **Measurement window per placement** (same rule, same length), so oblique waves are measured locally. | DL-2026-10-28-08 (single window) |
| FM-29 | Code fixes: 10 ms of Vm beyond the latest window for the resampling FIR; wording: the R wrapper uses MD5 and its own chunk parser (not SHA-256 or purl); output rate 5 kHz. | FM-12 wording |
| FM-30 | **Near/far 4 mm is a convention** (one HD Grid spacing); Schlageter's 4 mm is electrode-to-tissue distance. Continuous curves are primary. | FM-15 rationale |

Not acted on (recorded): bath convergence (Matilda); roof prediction to be pre-registered when analysed; canine-LV conductivities; 3x3 wall x standoff block.

Container checks now: C1 (mesh), W0 (Laplace accepted; cost of 712 solves per condition), C4 (fibres), C5 (lesion strip quiet, isolated tissue quiet, viable side activates), C8 (Wang R).

---

## 1. What this plan delivers

| Goal (Matilda, 3 Nov) | Where in this plan |
|---|---|
| Peak frequency recorded as Wang/EnSite do | FM-09 to FM-11 |
| Resolved transmural depth profile | FM-04, FM-05, FM-15 |
| Remove our FEM validation; use openCARP | FM-13, FM-14 |
| Clear 3D shape of the contributing tissue | FM-16, FM-21 |
| Every simulator setting in a `.par`; sweep variables in JSON | FM-01, FM-02 |
| Endocardial / epicardial / far-field contributions resolved | FM-15, FM-16 |
| Recommended fibre models | FM-06 |
| A scar type closer to an ablation lesion | FM-07 |
| Wang's change-point code exactly, without cohorts | FM-12 |
| No more open-ended workstation runs | Section 4 (container checks first; three pre-declared workstation runs) |

---

## 2. Decision register

### Configuration

**FM-01 - Every openCARP setting lives in `config/model.par`.** Decided.
- One commented openCARP parameter file. It holds:
  - solver settings, ionic model, β, all conductivity regions (tissue, lesion variants, bath);
  - stimulus, timing (dt, spacedt, tend), output and LATs.
- Each line has units and its source or decision ID.
- `slab_experiment.py` passes `+F model.par`. Only sweep values and run-specific paths (mesh, vertex files, simID, restart files) are added on the command line.
- Every run writes `simulations/<condition>/resolved.par`: model.par plus that run's overrides, as one file. The resolved file alone reproduces the openCARP run.
- Why: Matilda's instruction. It also restores a single readable simulator file (the old `3d_pilot.par` was split up on `finalModel`).

**FM-02 - Sweep variables, and settings openCARP cannot read, live in `config/model.json`.** Decided (3 Nov 2026; Matilda delegated the choice: "what most other researchers would do").
- `model.json` has two sections:
  - `sweep`: each variable has a baseline value, the planned values (Methodology), and its target. The target is either an openCARP parameter name (e.g. `gregion[1].g_el`) or a geometry/analysis key.
  - `fixed`: values that are neither openCARP parameters nor sweep variables: mesh spacing, bath geometry, electrode geometry, measurement window rule, PF algorithm settings.
- openCARP rejects unknown parameters, so mesher and analysis settings cannot go in the `.par`. This is the smallest split that keeps Matilda's rule: simulator settings in `.par`, sweep variables in JSON.
- Wang's analysis constants (detection limit, ROI, trim, mcp settings) stay in her Rmd and are not copied (FM-12).

### Mesh, domain and bath

**FM-03 - Domain, beats and time step fixed now, not by a further workstation rule run.** Decided.
- Domain 40 × 60 mm, 4 beats at 500 ms, dt 20 µs: the expected outcomes of the 1 Nov final-settings rules.
- The new mesh (FM-04) invalidates the workstation final-settings run, so its rule outcomes are not awaited.
- Domain and beat count are not re-tested. The physiology gate (stable last three beats) and the profile prerequisites still run in every run and stop interpretation if they fail.
- 500 ms cycle length: Assumption [VERIFY]. Wang mapped during distal CS pacing but reports no cycle length.

**FM-04 - Anisotropic mesh: 0.25 mm in-plane, 0.1 mm through the wall.** Decided.
- The openCARP mesher accepts a separate resolution per axis (`-resolution[0..2]`; carputils 19.0.0 `Block.mesher_opts` writes the three entries but fills them with one value). We set `resolution[2]` = 100 µm.
- Node counts (40 × 60 mm, 5 mm bath, from the block formula):

| Mesh | Bath | Total nodes | Tissue nodes |
|---|---|---|---|
| 0.5 mm isotropic (current) | all sides | 0.34 M | 0.04 M |
| 0.25 mm isotropic | all sides | 2.65 M | 0.27 M |
| 0.1 mm isotropic | all sides | 40.7 M | 3.86 M |
| 0.25 / 0.1 mm | all sides | 6.55 M | 0.62 M |
| **0.25 / 0.1 mm** | **catheter side + lateral (FM-05)** | **3.73 M** | **0.62 M** |

- 0.1 mm isotropic is rejected: 41 M nodes, with an elliptic solve at every output step.
- The chosen mesh gives 15 elements (16 node layers) through a 1.5 mm wall. The Methodology asked for 10-30 transmural elements; 0.25 mm in-plane is the common modelling resolution.
- Remaining discretisation error is measured once, by W3 (Section 4). It is not argued away.
- Wall thicknesses must be multiples of 0.1 mm (FM-08).

**FM-05 - Blood on the catheter side and the lateral sides; no-flux epicardium.** Decided (3 Nov 2026; Matilda delegated the choice: "what most other researchers would do"). Changes DL-2026-11-02-02.
- Reason for choosing it: the boundary follows the in vivo anatomy, and the uncertain alternative is kept as a sweep variant rather than discarded. That is the conventional way to handle an unsourced boundary condition: choose the anatomically closer case, bracket it, and report both.
- The tissue surface away from the catheter faces pericardium, oesophagus and lung in vivo, not blood.
- A conducting layer there lets current return around the wall. That changes the transmural lead field, which is the quantity the depth profile reports.
- No sourced conductivity exists for the epicardial medium (parameter_sources.md, "Open"). The two bounds are an insulator (no-flux) and blood (Leenknegt et al. 2023, all sides).
- Primary: no-flux. Sweep variable `epicardialMedium` = `none` | `blood` brackets it.
- carputils `Block.set_bath(thickness, both_sides=(True, True, False))` places bath on one z side. [VERIFY] in the container (C1) that the bath goes on +z, the catheter side.
- Unchanged:
  - 5 mm bath thickness;
  - 0.74 S/m blood (Traugott et al. 1981);
  - 1 mm³ ground at the far bath corner.
- Stated limitation: unipolar signals are referenced to that corner inside a closed box. The 1 Nov bath study did not converge (PP up to 63 %), so unipolar amplitudes and change points are model-specific. There is no remote far-field source, so Wang's unipolar plateau (~0.56 mV, far-field from the whole heart) is not reproduced by construction. The bipolar and omnipolar conclusions do not depend on this.

### Tissue

**FM-06 - Three fibre models, selected by the sweep variable `fibres`.** Decided; angles Assumption.
- `uniform`: all fibres along +x. Pashakhanloo et al. (2016) report constant fibres through the wall on the lateral wall and roof.
- `bilayer`: endocardial layer 0°, epicardial layer 90°, abrupt transition at mid-wall. Pashakhanloo et al. describe near-perpendicular layers with an abrupt mid-wall change in the posterior-inferior and anterior LA. They give no angles, so 0° / 90° is an assumption, and the transition depth is a parameter.
- `linearRotation`: 0° → 90° linear through the wall. Methodology sensitivity case; also the mesher's built-in rule (`Block.set_fibres(endo, epi)`).
- Implementation: port `Shared_Pipeline/fibres.py` from `pilot_v2` (convention DL-2026-10-28-12, verified VL-2026-10-28-12). The `.lon` file is written from element-centroid depth after meshing.
- Tissue stays transversely isotropic (one fibre vector per element), as now.
- Angles are measured from +x, which is also the reference wave direction and parallel to the lesion line. The thesis must state that fibre angle and wave angle are separate sweep variables (FM-17).

**[geometry superseded by FM-24]** **FM-07 - Primary lesion: inexcitable, intracellularly uncoupled, extracellularly conducting.** Decided (3 Nov 2026; Matilda delegated the choice: "what most other researchers would do").
- Reason for choosing it: a passive, non-excitable region that keeps a conducting extracellular space is the usual simulation representation of scar or lesion. It asserts no conductivity change without a measurement; the porated and insulating cases are reported as bounds.
- Basis:
  - RF lesions are coagulation necrosis with sarcolemmal damage, and irreversible injury above ~50 °C (Thoracic Key chapter on RF lesion biophysics; book identifier 9780323529921, authors not shown in the excerpt).
  - Dead cells generate no current and do not couple, so intracellular conductivity → ~0 (1e-6 S/m, as now).
  - The tissue remains an electrolyte-filled conductor. An insulating extracellular lesion (the current R10 choice) removes it from the volume conductor and pushes current into the bath over the lesion, which is exactly where the change point is fitted.
- No measured conductivity of acute RF-ablated myocardium was retrieved; the conductivity-imaging papers found were not accessible. So the lesion conductivity is bracketed, not asserted:
  - `ablation` (primary): extracellular tensor = healthy σe (lower bound for a conducting lesion: no gain from membrane disruption).
  - `ablationPorated` (upper bound): extracellular tensor = σi + σe. This is the bidomain limit for fully porated membranes, where both domains carry current in parallel. It is a theoretical bound, not a measurement.
  - `insulating`: kept as the earlier reference case.
- The sweep variable `scar` selects among the three; their conductivities are named blocks in `model.par` comments, applied as overrides.
- Geometry unchanged: transmural half-plane y < 0. The excitable edge is Wang's d = 0, 3 mm from the lesion centre (Wang supplement; lesion width 5.6 ± 1.2 mm, Friedman et al. 2021 as cited by Wang).
- Not modelled: a border zone of reversibly injured tissue. There are no sourced parameters for it, and it is a stated limitation.
- Earlier evidence that this choice matters: conducting vs insulating shifted supported change points by 3.8 mm (DL-2026-11-02-01).

**FM-08 - Regions are two parameter sets: inferior (reference) and roof.** Decided; values Assumption-rounded.
- Wall thickness (Whitaker et al. 2020, CCTA, n = 193):
  - inferior posterior wall median 1.14 mm (5-95 %: 0.72-2.71) → **1.1 mm**;
  - roof median 1.55 mm (0.70-3.72) → **1.6 mm**.
  - Rounded to the 0.1 mm transmural mesh.
- Fibres: inferior `bilayer`, roof `uniform` (Pashakhanloo et al. 2016).
- Same standoffs in both regions. No sourced regional contact data. Wang's respiratory-motion argument (Koch et al. 2013) is qualitative.
- This replaces the single 1.5 mm wall (Li et al. 2024 posterior wall 1.48 mm).
- It makes validation target V-1 (regional dissociation: bipolar differs, omnipolar does not) testable.
- Finding to state in the thesis: in vivo CT has the inferior posterior wall *thinner* than the roof. Wang's discussion argues it is thicker (citing Sánchez-Quintana et al.). Imaging and post-mortem series disagree; the model uses imaging.

### Measurement

**FM-09 - Acquisition emulated at 2 kHz.** Decided.
- **[implementation, FM-22]** openCARP writes φe every 0.2 ms (5 kHz; the earlier verified output step), because 0.25 ms is not a multiple of dt = 20 µs. The analysis resamples to 2 kHz with `scipy.signal.resample_poly` (ratio 2/5; its Kaiser-window FIR is the anti-alias filter; measured passband ripple 0.03-0.13 % at 30-500 Hz), the EnSite X acquisition rate reported by the Europace 2024 OT Near Field study (euae154; a VT study using EnSite X).
- Differential signals: 30-500 Hz band-pass (Wang supplement), replacing the unsourced 450 Hz. Order 2, causal Butterworth kept (DL record; EnSite filter type unknown, Junction 1).
- Unipolar signals: the resampling filter only. The clinical unipolar filter is unknown: [VERIFY], Junction 1.
- Peak-to-peak, PF and everything downstream run on the 2 kHz signals.

**[primary definition superseded by FM-27]** **FM-10 - Peak frequency: an EnSite OT Near Field approximation.** Decided; wavelet details Assumption.
- Published description (Europace 2024;26:euae154):
  - the algorithm is proprietary and wavelet-derived;
  - it gives a peak-frequency trace over time; each point is assigned the highest PF and its time;
  - it applies only to bipolar and omnipolar signals;
  - it runs only when PP ≥ 0.04 mV;
  - voltage is taken within ±30 ms of the PF annotation.
- Not published: the mother wavelet, the scales, and the frequency bounds.
- Implementation, one function `peakFrequency(signal, fs)`:
  1. Complex Morlet continuous wavelet transform (ω0 = 6, numpy only), frequencies 30-500 Hz in 1 Hz steps (the analysed band; Wang reports PF to 1 Hz).
  2. PF trace: at each sample, the frequency of the largest wavelet magnitude.
  3. Samples with magnitude below 10 % of the window maximum are ignored (gate; assumption).
  4. PF = highest remaining trace value; t_PF = its time.
  5. If PP < 0.04 mV, PF is undetectable (NA), which Wang's `freq_ok` excludes.
- Pre-declared alternative, computed and reported but not primary: PF = frequency of the global scalogram maximum.
- Unipolar signals get no PF (as EnSite and Wang).

**FM-11 - Bipolar and omnipolar PP taken within ±30 ms of t_PF, inside the measurement window.** Decided.
- Matches the EnSite voltage annotation (Europace 2024) and therefore Wang's exported PP.
- Unipolar PP is unchanged: the whole measurement window.
- If PF is undetectable, PP falls back to the whole window and the row is flagged.

**FM-12 - Change points by Wang's own Rmd code, with the patient terms removed.** Decided.
- Wang's `fit_piecewise` (mcp linear decay + flat plateau, 3 chains, 3000 iterations, 1500 adaptation, seed 3) has no patient term. The patient random intercept appears only in `lmer` model selection, REML estimates and the random-slope check.
- New `R/wang_replication.R`:
  1. Records the SHA-256 of `reference/EFOV_analysis_pipeline.Rmd` (ab443252…) and refuses to run if the file changed.
  2. Extracts her chunks with `knitr::purl` and runs them by label, unedited.
  3. Replaces exactly these functions, each in one short block:
     - `decay_formula` (no random term);
     - `select_decay_model` and `fit_decay_reml` (`lm`; OLS = ML for fixed effects);
     - `tidy_decay` (Wald CI from `lm`; R² reported in place of R²m; R²c = NA);
     - `predict_decay` (`coef`/`vcov` from `lm`);
     - `drop_unfittable` (no ≥ 2-patients rule).
  4. Skips only the chunks that cannot apply to one slab: random-slope check, manual-vs-automated distance validation, and geodesic sensitivity (flat slab: geodesic = Euclidean).
- Input: an export in Wang's EnSite column format (`patientID` = 1, `inf0_roof1`, `uni1_bi2_omni3`, `DistAcrossPath` = distance − 3 mm, `PP`, `PFreq`), one file per run.
- mcp settings are hers. Convergence is reported against her criterion (R̂ < 1.1); settings are not changed to make fits converge.
- The Python profile-least-squares change point (`efov/changepoints.py`) is removed.
- Points per placement, so the data resemble a clinical export:
  - all 16 unipoles;
  - all 9 orientation-selected bipoles;
  - all 36 omnipolar cliques;
  - each at its own anchor distance.
- One gap and one noise realisation per analysis.
- Overlapping placements are not independent: Wang's credible intervals here describe one simulated slab, not population uncertainty.
- The intact run (virtual boundary) goes through the same code as a control and is reported beside the lesion run.

### Electrograms, ground truth and contributions

**[source superseded by FM-23]** **FM-13 - Remove the independent FEM forward-model check.** Decided (Matilda).
- Remove the electrograms stage's `forwardModelCheck`, its acceptance thresholds, and the method-of-images / insulating-plane unit tests.
- Electrograms are openCARP pseudo-bidomain φe, as before (DL-2026-11-02-01).

**[source and checks superseded by FM-23, FM-26]** **FM-14 - Lead fields computed by openCARP: Laplace solves with a total-current electrode, by reciprocity.** Decided (3 Nov 2026; Matilda delegated the choice: "what most other researchers would do"). Feasibility checked first (W0).
- Reason for choosing it: the lead field comes from the same simulator, mesh, conductivities and ground as the electrograms, so attribution and signal share one discrete problem. The reconstruction check proves it on every run.
- Corrects the earlier draft of this plan, which said openCARP cannot provide lead fields. Precisely:
  - openCARP's dedicated lead-field stimulus (`LF_I`) needs `ENABLE_LEADFIELD` at build time. The conda-forge openCARP 19.0.0 binary contains no lead-field code (no `LF_I` or `leadfield` strings; inspected 3 Nov 2026).
  - The same binary has a **Laplace experiment** ("Laplace solve"; it converts the extracellular-electrics region to the Laplace region) and an **extracellular current stimulus** ("extracellular current (in uA/cm^3)") with `crct.total_current` ("Treat strengths as total current (uA) and not density").
  - Together these give a lead field by reciprocity: inject 1 µA at the electrode, ground the corner cube, solve once, read the potential at every node.
- Making the openCARP lead field match the openCARP electrograms exactly:
  - pseudo-bidomain φe solves ∇·((σi+σe)∇φ) = −∇·(σm∇Vm) over tissue + bath (VL-2026-11-01-02);
  - the Laplace run uses the extracellular conductivities, so its tissue and lesion extracellular values are set to σi + σe (healthy and lesion blocks; bath 0.74 S/m; same mesh, tags, fibres and ground). Its operator is then the one openCARP uses for φe. These values sit in a named Laplace block in `model.par`.
- Electrode = the set of mesh nodes inside the electrode cylinder. This replaces the 8 interpolated surface samples. The same node set is used for the injected current and for reading the electrode potential (mean φe over the set). Reciprocity is then exact for the discrete problem with no interpolation weights to reproduce. At 0.25 / 0.1 mm a 1 mm electrode holds roughly 150 nodes. Both versions are passive-sensing approximations; the change is recorded.
- Python keeps only element gradients and the reciprocity sum (c_e = −∇z·σm∇Vm). **No linear solver remains in our code.**
- The reconstruction check stays and becomes a test of openCARP against openCARP: summed contributions from the openCARP lead field must reproduce openCARP's φe for every ground-truth recording (relative residual ≤ 1e-3; scale = gain within 1e-3). If it fails, the run stops.
- Pre-declared fallback (decided now, so W0 cannot turn into a loop): if W0 shows that the Laplace mode rejects a current stimulus, or that the reconstruction fails, the lead field reverts to the earlier Python reciprocity solve (verified VL-2026-10-28-07, VL-2026-11-02-01). The ground-truth definition and outputs are identical either way. No third route is explored.

**[cut-off rationale superseded by FM-30]** **FM-15 - Contribution classes: endocardial, epicardial, far-field.** Decided (3 Nov 2026; Matilda delegated the choice: "what most other researchers would do").
- Reason for the 4 mm cut-off: it is the published near-field cut-off for human atrial electrograms (Schlageter et al. 2024), so results can be compared with it. The continuous curves are reported alongside, so no conclusion depends on the cut-off value.
- For each recording, every tissue element is assigned to exactly one class:
  - **near-field endocardial**: lateral distance from the recording anchor ≤ R_nf, depth from the catheter-facing surface < T/2;
  - **near-field epicardial**: lateral ≤ R_nf, depth ≥ T/2 (for `bilayer` fibres this is the fibre-layer boundary);
  - **far-field**: lateral > R_nf, all depths, also split endo/epi.
- R_nf = 4 mm, the near-field cut-off proposed by Schlageter et al. (2024), as cited in Wang's manuscript. Equal to the HD Grid centre spacing.
- Reported per class: signed share (sums to 1 across classes) and |c|-mass fraction (Σ|c_e|, no cancellation).
- Continuous curves are also kept so the cut-off is not hidden:
  - cumulative share against lateral distance (0.25 mm steps);
  - against depth (0.1 mm, one bin per node layer);
  - the full voxel field at 0.25 × 0.25 × 0.1 mm.
- Far-field from tissue outside the slab (rest of the atria, ventricles) does not exist in this model and cannot be attributed. "Far-field" here means slab tissue beyond R_nf. Stated limitation (see FM-05).

**FM-16 - Ground truth: the same method as before, all recordings, every 2 mm.** Decided.
- The ground-truth quantity is unchanged from DL-2026-11-01-02, which Matilda confirmed as the method to keep:
  - c_e(t) from the reciprocity lead field and the σm source, per element;
  - s_e = Σ_t c_e(t)φ(t) / Σ_t φ(t)²;
  - signed, nothing cut off, cancellation kept, Σ s_e = 1 exactly;
  - band-passed Vm for differential recordings (the filter is linear and commutes with the field).
- Placements y = −16 … 16 mm every 2 mm (17), both gaps, every recording of FM-12 (16 unipoles, 9 bipoles, 36 omnipoles per placement).
- One openCARP Laplace run per unique electrode position: about 184 at 3.7 M nodes. Time per run measured in W0.
- Each lead field is stored on tissue and lesion nodes (~2.5 MB each, ~0.5 GB per run). Any later recording combination, including the E5 recombination search, needs no new solve.

**FM-21 - Ground-truth outputs: the full 3D shape of the contributing tissue.** Decided.
- Primary output, per recording: s_e on every tissue element of the openCARP mesh, written as a `.vtu` (cell data: s_e, |c|-mass, contribution class, depth, lateral distance). Opens in ParaView with no resampling.
- Voxel field at 0.25 × 0.25 × 0.1 mm, which equals the mesh spacing, so voxelising loses nothing. Saved as npz for between-run comparison (L1 distance, DL-2026-11-01-02).
- Figures, all from s_e with no cut-off:
  - orthogonal sections through the recording anchor: x–y at the endocardial surface, mid-wall and epicardial surface; x–z and y–z planes;
  - a symmetric-log colour scale (signed), so negative (cancelling) tissue stays visible;
  - the transmural depth profile, one value per node layer (0.1 mm);
  - cumulative share against lateral distance;
  - contribution classes (FM-15);
  - for the lesion run, the same sections across the 17 placements, showing how the field changes as the catheter crosses the boundary.
- Shape descriptors, secondary and labelled descriptive, never used as the ground truth:
  - for p = 50, 80, 90 %: the smallest set of elements holding p % of Σ|s_e|;
  - its volume, x and y extent, maximum depth, centroid offset from the anchor, and principal-axis eccentricity and angle against fibres and wave.
  - These test the hemisphere assumption directly. They are not a radius.

### Sweeps, tuning and storage

**FM-17 - Sweep runner with a shared simulation cache.** Decided.
- `python -m efov sweep config/sweeps/<name>.json` runs the baseline plus one-at-a-time variants.
- Each variant's simulation fingerprint decides whether openCARP must run. Variants that change only electrodes, noise, filters or recordings reuse the cached Vm and φe. Today every variant needs a new folder and new openCARP runs.
- Variables wired now:
  - wall thickness;
  - fibres model and angles;
  - epicardial medium;
  - scar type;
  - standoff;
  - electrode radius and length;
  - 4 × 4 grid spacing;
  - wave angle;
  - noise SD;
  - CV target (via FM-18);
  - ionic model (hook only; the Koivumäki openCARP model name is [VERIFY]).
- Wave angle α is measured from +x. α = 0 is the current wave parallel to the boundary; α = 90° travels toward the lesion. The stimulus is the viable-node band with the smallest projection on the wave direction (corner band for oblique angles, as the 2D specification's 45° case). The physiology check measures planarity and direction error in the analysis region and gates the run.
- Deferred, by the agreed drop order: Latin hypercube sampling, curvature, the dense virtual electrode array.

**FM-18 - β tuned by a tune stage at the production mesh.** Decided.
- The new mesh invalidates β = 0.0218 µm⁻¹.
- The tune stage pre-paces a 20 mm strip from the same mesher call and resolution (DL-2026-10-28-10 method), with fibres along the strip.
- Update rule: β ← β·(CV/CV_target)², because CV ∝ (σ/β)^½; stop within 1 %. The 0.90 m/s target and its source are unchanged (DL-2026-10-28-03).
- The tuned β is written into `model.par` with its provenance (`python -m efov tune --write-par`). Sweep variants with a different CV target or ionic model retune into their own resolved .par.
- β is tuned once on uniform fibres and held fixed across fibre models. The effective slab CV then changes with fibre architecture, as it physically should, and is reported. The CV gate applies to the strip, not to bilayer slabs.

**[superseded by FM-25]** **FM-19 - Calibration gain fixed from the inferior reference run.** Decided.
- The gain is computed once: median bipolar PP at the intact reference placement = 1.44 mV.
- R7 resolved: 1.44 mV is Wang's Table 2 median healthy-tissue bipolar voltage on the inferior wall (IQR 0.70-2.29).
- That gain is applied unchanged to the roof run and every sweep variant. Recalibrating each run would erase regional and sweep amplitude differences, which are predictions.
- Noise stays 5 µV SD. Assumption [VERIFY]: no source recorded. It is a sweep variable.
- Circularity, stated in the thesis: the absolute amplitude is calibrated to a Wang value and noise is absolute, so the inferior wall's SNR is partly set by calibration. Validation target V-5 (the 0.05 mV crossing) is therefore partly circular for the inferior wall. The roof prediction is not.

**[phi_e output removed by FM-23]** **FM-20 - Storage: compaction of unshared simulations.** Decided; **[implementation]** revised.
- Vm output is not restricted (the planned `dataout_i = 3` was dropped: it would have needed an unverified parameter and a split physiology read).
- φe is written on the electrode envelope box (about 0.7 M nodes, ~13 GB per condition at 5 kHz). Without compaction the one-at-a-time sweep would need ~1.1 TB.
- A sweep variant whose simulation no other variant reuses is compacted after its report: Vm kept for its analysis frames (`vm_analysis.npz`), `vm.igb` and `phie.igb` deleted, electrode signals already extracted. The baseline simulation, shared by the electrode, noise and analysis variants, is never compacted. Estimated sweep total ~200 GB. Raw output is regenerated from the resolved .par files.

**FM-22 - Output step and resampling [implementation].** Decided.
- A 0.25 ms output step is not a whole multiple of dt = 20 µs. Kept dt 20 µs and 0.2 ms output (both verified in earlier runs); resample 5 → 2 kHz (FM-09). Changing dt to 25 µs was rejected as unverified numerics.
- Measurement-window edges are rounded outward to 1 ms, the common multiple of the 0.2 ms and 0.5 ms grids (found by the synthetic end-to-end test).

---

## 3. File-by-file edits

| File | Change |
|---|---|
| `config/model.par` (new) | Every openCARP setting, commented with units and source/decision ID (FM-01). Replaces `efov/opencarp/solver.par`. |
| `config/model.json` (replaces `final_model.json`) | `sweep` and `fixed` sections (FM-02); provenance kept. |
| `config/sweeps/oat_baseline.json` (new) | Methodology one-at-a-time plan (FM-17). |
| `efov/opencarp/slab_experiment.py` | `+F model.par` plus overrides; anisotropic resolution; one-sided bath; fibres `.lon`; lesion variants; wave-angle stimulus band; writes `resolved.par`; restricted Vm output (FM-01, 04-07, 17, 20). |
| `efov/fibres.py` (port) | uniform / bilayer / linearRotation from `pilot_v2` (FM-06). |
| `efov/tune.py` (new) | Strip CV tuning of β (FM-18). |
| `efov/simulation.py` | Remove values now in .par; tune stage hook; compaction (FM-20). |
| `efov/peakfrequency.py` (new) | Morlet CWT, PF trace, PF, t_PF, gates (FM-10). |
| `efov/measurements.py` | 2 kHz emulation, 30-500 Hz, PP at ±30 ms of t_PF, PF columns, fixed gain (FM-09, 11, 19). |
| `efov/recordings.py` | Labels for all 16 / 9 / 36 recordings; no algorithm change (FM-12). |
| `efov/export.py` (new) | Wang-format xlsx per run (FM-12). |
| `R/wang_replication.R` (replaces `bayesian_change_points.R`) | Wang's Rmd via purl with the listed overrides (FM-12). |
| `efov/changepoints.py` | Removed (FM-12). |
| `efov/forward.py` | Removed. Replaced by `efov/leadfield.py`: writes and runs the openCARP Laplace experiment per electrode (I_ex, total current 1 µA, Laplace conductivity block), reads the potential, stores lead fields; element gradients for c_e (FM-13, 14). |
| `efov/geometry.py` | Electrodes as mesh node sets inside each cylinder (FM-14). |
| `efov/electrograms.py` | `forwardModelCheck` removed (FM-13). |
| `efov/groundtruth.py` | Same s_e definition; classes, |c|-mass, 0.1 mm depth profile, stored lead fields, 2 mm placements (FM-15, 16). |
| `efov/groundtruth_figures.py` (new) | `.vtu` export, sections, depth and lateral profiles, boundary sequence, descriptive shape metrics (FM-21). |
| `efov/physiology.py` | CV on catheter-facing surface; direction check against the intended α (FM-06, 17). |
| `efov/pipeline.py` | Stages: preflight, tune, conditioning, simulation, physiology, window, electrograms, measurements, export, changePoints (R), groundTruth, report, compact. |
| `efov/sweep.py` (new) | Runner and simulation cache (FM-17). |
| `tests/` | Remove FEM/image tests; add PF synthetic tests, fibre-field tests, .par round-trip test, R override test, class-partition test. |
| `notebooks/` | Regenerated for the new stages. |
| `docs/` | FM-01 to FM-20 into the decision log; parameter_sources.md updated (Whitaker 2020, Europace 2024); verification log entries for C1-C10. |

Code style for every module: plain functions, units in names, no scientific constant in code, each function a one-line docstring with units.

---

## 4. Verification plan: container first, then four pre-declared workstation runs

### Container checks (openCARP 19.0.0 from conda-forge; small slab; before anything reaches the workstation)

| # | Check | Pass criterion |
|---|---|---|
| C1 | Mesher: anisotropic resolution and one-sided bath | Node spacing 0.25 / 0.1 mm; bath on +z and lateral sides only; tags and zero bath fibres |
| C3 | `resolved.par` round trip | **[implementation]** A replay from `run.par` plus the recorded carputils solver-flavour options equals the run (Vm max difference 0); `run.par` alone is also replayed and its difference reported |
| C4 | Fibre `.lon` read by openCARP | Bilayer and linear: CV along x on the endocardial surface within 2 % of the fibre-direction strip prediction for uniform; `.lon` order matches elements |
| C5 | Lesion variants | Lesion stays inactive; conducting variants change φe over the lesion; reconstruction passes |
| C6 | Peak frequency on synthetic signals | Morlet bursts at 100-450 Hz recovered within 2 Hz; PP < 0.04 mV gives NA; decimation does not move PF by > 2 Hz below 500 Hz. **Done:** tones exact; 6- and 12-cycle bursts within 1 Hz; 3-cycle bursts 1-3 Hz high (criterion fails there; recorded, not changed) |
| C7 | Node-set electrodes (**done** in the synthetic test) | Every electrode set non-empty and inside the bath; PP from node-set vs 8-sample electrodes reported (difference is a model change, not a pass/fail) |
| C8 | Wang replication script (needs R: container) | On simulated test data: her unmodified chunks run; overrides give `lm` estimates equal to a direct `lm` fit; mcp output columns identical to her Table 5 format |
| C9 | Ground truth (**done** on the synthetic run, Python route: residual 3e-7) | Classes partition all tissue elements; shares sum to 1; reconstruction ≤ 1e-3; stored and reused lead fields give identical shares; `.vtu` cell order matches the openCARP element order |
| C10 | Sweep cache (**done** as fingerprint unit tests) | An electrode-only variant reuses the simulation; a wall variant refuses it |

### Workstation runs (decided now, with outcomes fixed in advance)

- **W0 - openCARP lead-field feasibility (about 10 minutes).** Small slab (16 × 20 mm, 1.1 mm wall, 2.5 mm bath, 0.25 / 0.1 mm), one pseudo-bidomain run of 30 ms, three electrodes.
  - (a) The Laplace experiment accepts an extracellular current stimulus with `total_current = 1` and writes the potential at all nodes.
  - (b) Reciprocity: z_A read at electrode B equals z_B read at A (relative difference ≤ 1e-6).
  - (c) Reconstruction: Σ c_e(t) with the openCARP lead field reproduces openCARP's pseudo-bidomain φe at each electrode (relative residual ≤ 1e-3, scale 1 ± 1e-3).
  - (d) Time per Laplace run at full size, extrapolated.
  - Pass: openCARP lead fields (FM-14). Fail: the pre-declared fallback. Either way no further feasibility runs.
- **W1 - Tune and smoke test.** Tune β at 0.25 / 0.1 mm (minutes), then a 5 ms solver-flavour smoke test.
  - Pass: CV within 1 % of 0.90 m/s.
  - Fail: stop and report; no parameter is changed to pass.
- **W2 - Reference runs.** Inferior and roof, intact and lesion, full pipeline including R and ground truth.
  - These are the reference results.
  - If physiology or profile prerequisites fail, the run is reported as ineligible and nothing is reinterpreted.
- **W3 - Discretisation error, once.** On a reduced 40 × 30 mm slab:
  - in-plane 0.25 vs 0.125 mm;
  - transmural 0.1 vs 0.05 mm.
  - Tolerances already confirmed (DL-2026-10-28-05; DL-2026-11-01-01): CV within 1 %; PP within 5 %; share-class fractions within 0.02 absolute; change-point shift < 0.5 mm on supported fits only.
  - Also reported, without a tolerance (FM-46, 11 Nov 2026): clean peak frequency (both definitions) per 1 mm distance bin, and the boundary-matched |c|-mass crossing distances (FM-37, FM-43). Peak frequency depends on how well the depolarisation front is resolved (a few elements at 0.25 mm in-plane), and neither quantity was covered by the tolerances above. No tolerance is set after the fact: the differences are reported as the discretisation error band of these quantities.
  - Pre-declared outcome: whatever W3 shows is reported as the model's discretisation error band. Failing W3 does not trigger a finer production mesh: a 0.125 mm in-plane production mesh (≈ 15 M nodes) is outside the compute budget.

After W2 passes, sweeps start from `config/sweeps/oat_baseline.json`. No other workstation run is needed to finalise the model.

---

## 5. Open items

Every modelling choice in this plan is now decided. Matilda delegated the open choices (FM-02, FM-05, FM-07, FM-14, FM-15) on 3 Nov 2026, and each records its reason.

Still external (Junction 1). None blocks implementation; each is marked [VERIFY] where it is used:
- EnSite unipolar filter;
- pacing cycle length;
- noise level;
- OT Near Field wavelet settings (ask Abbott via Fadhly Amri).

Where W0 and the container checks run: in Matilda's `efov-runner` Docker container, from one script delivered with the implementation. openCARP could not be installed in Claude's sandbox. The checks are software and feasibility runs on a small slab, not model runs.

---

## 6. Sources read for this plan

- Whitaker J, Karády J, Karim R, et al. Standardised computed tomographic assessment of left atrial morphology and tissue thickness in humans. *Int J Cardiol Heart Vasc* 2021;32:100694. doi:10.1016/j.ijcha.2020.100694 (regional wall thickness, read in full).
- Near-field detection and peak frequency metric for substrate and activation mapping of ventricular tachycardias in two- and three-dimensional circuits. *Europace* 2024;26(7):euae154 (OT Near Field description; read). Primary algorithm paper: Payne et al., *Heart Rhythm* 2024 (cited there; not read) [VERIFY].
- Abbott, EnSite OT Near Field product page (concept only; read).
- openCARP doxygen: `stimulate.h` (stimulus enum incl. `LF_I`), `stimulate.cc` (`LF_I` requires `ENABLE_LEADFIELD`; `I_ex` in µA/cm³; `vtx_fcn` only for Dirichlet), read.
- openCARP 19.0.0 conda-forge binary (`opencarp-19.0.0-h79d7346_0`), parameter help strings: Laplace experiment, extracellular current in µA/cm³, `crct.total_current`; no lead-field strings. Inspected, not executed (running it in this session's sandbox was not permitted).
- openCARP / CARPentry Laplace tutorial (02_EP_tissue/13_laplace): experiment = 2, extracellular voltage and ground stimuli, read.
- carputils 19.0.0 source (`mesh/block.py`, `mesh/fibres.py`) from the conda-forge package: per-axis mesher resolution, one-sided bath, fibre rotation, read.
- Thoracic Key, Biophysics and pathophysiology of radiofrequency lesion formation (excerpt; book identifier 9780323529921): membrane damage, ~50 °C irreversible isotherm, read.
- From earlier sessions and project documents (not re-read today): Pashakhanloo et al. 2016; Schlageter et al. 2024; Leenknegt et al. 2023; Roberts and Scher 1982; Traugott et al. 1981; van Schie et al. 2021; Li et al. 2024; Friedman et al. 2021; Wang et al. manuscript and supplement.
