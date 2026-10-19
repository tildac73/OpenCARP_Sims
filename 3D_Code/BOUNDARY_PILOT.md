# Controlled active-boundary experiment: pilot V2

## Purpose and evidence behind the revision

The October 19 serial run contained complete, matching 256,000-element meshes and one launch per condition. Its active wave was planar and repeatably captured, with speed about 0.35 m/s. The failures were specific:

| Finding in V1 | Targeted V2 change | Intended effect and remaining limitation |
|---|---|---|
| APD90 still changed by roughly 7–8 ms; plateaux differed between beats | Precondition one Courtemanche cell for 1000 beats, then run 16 tissue beats | Reduce the initial-state transient before expensive propagation; still require tissue stability |
| Vm peaks poorly resolved at 1 ms output | Record every 0.2 ms, retaining the 20 microsecond solver step | Better peak/APD measurement; this is not temporal convergence |
| Bipolar far region contained only 10 and 12 mm positions | Denser, wider sweep with four fixed geometries | Several far-region distances per geometry, plus enough lesion-side points for fitting |
| Wider sampling would approach the old y edges | Extend slab y from 40 to 60 mm | 7.5 mm clearance from the most peripheral electrode samples; domain convergence remains unverified |
| Mirrored omnipolar triangles were stitched into one jagged curve | Fit and plot each declared triangle separately | Preserve geometry-dependent boundary overlap |
| Lesion-side unipolar reconstruction had up to 69% shape error | Use each tetrahedron's gradient, conductivity and volume | Respect the conductivity discontinuity; six diagnostic reconstructions of V1 improved strongly, but full V2 validation is outstanding |
| Physiology failure was labelled as inactive/negligible signal | Separate physiology, signal and recovery failure reasons | Explain why a radius is unavailable without implying it is zero |

The settings are prespecified. We do not pick a favourable beat, loosen thresholds, force a change point, or manufacture biological replicates from noise.

## Default protocol

- Slab: **40 × 60 × 4 mm**, regular 0.5 mm mesh, fibres along x.
- Intact and lesion conditions share coordinates and tetrahedral connectivity.
- Straight boundary at y = 0. Tissue at y < 0 has near-insulating conductivities; the same Courtemanche model is retained. This is an electrically isolated region, not a validated ionic model of scar.
- Left-edge, full-thickness stimulation strip: 0.5 mm; 50 stimulus strength, 2 ms duration, BCL 500 ms, 16 tissue beats.
- One cell is first paced for **1000 beats** at the same BCL, strength and duration. The state at the end of the final cycle is used for both slabs.
- Integration: **20 microseconds**. Output: **0.2 ms**. Tissue duration: 8000 ms.
- Prespecified measurement: beat 16, 0–400 ms after its stimulus. Filter history: 200 ms; compare against 400 ms history.
- Physiological comparison: beats 14–16, using half-open complete cycle windows. Viable validation ROI: x in [-8,8] mm, y in [1,24] mm, all depths.
- Sixteen finite cylindrical electrodes: radius 0.5 mm, axial length 1 mm, two axial by four angular surface samples. Gaps: 0.25 and 1 mm from the tissue to the nearest cylinder surface.
- Catheter centres: y = -16 to +16 mm in 0.5 mm steps, x = 0. Largest lateral electrode sample extent is y = ±22.5 mm, inside y = ±30 mm tissue.
- Measured configurations: E5 unipolar; E5–E6 or E5–E9 selected bipolar; E5–E6–E9 and E9–E10–E5 omnipolar triangles.
- Differential filter: causal second-order Butterworth, 30–450 Hz. Unipolar signals remain unfiltered.
- Gain: one fixed scale making the intact reference placement's median of nine selected bipoles 1.44 mV at y = 0, gap = 0.25 mm. This is imposed calibration, not validation of absolute voltage.
- Noise: zero-noise primary and three Gaussian electrode-noise realisations, SD 0.005 mV **per raw sample**, with reproducible seeds. Differential filtering changes the resulting noise floor. Because V2 samples faster, its filtered noise floor need not equal V1's; compare clean and noisy analyses explicitly. No noise repeats are biological replicates.
- Contribution subset: the four declared recording geometries at catheter centres -8, 0 and +8 mm, both gaps, both conditions. Other measurements have no borrowed contribution radius.

## Why condition the cell first?

openCARP's [single-cell limit-cycle tutorial](https://opencarp.org/documentation/examples/01_ep_single_cell/01_basic_bench) demonstrates pacing a cell, saving its state, and reusing it. `conditionCells()` implements this directly with `bench`, rather than making the 3D slabs perform all initial ionic conditioning.

The command explicitly sets the unmodified Courtemanche model, stimulus protocol and timestep. It requests exactly 1000 pulses and saves at 500,000 ms, one cycle after the last pulse, before another stimulus. Bench uses milliseconds for its integration timestep; the tissue simulator uses microseconds. The conversion is explicit.

Both conditions reference `conditioning/Courtemanche.sv` through `imp_region[0].im_sv_init`. This parameter is documented in the [openCARP parameter reference](https://opencarp.org/parameters/master/).

A saved state is evidence of completed conditioning, not proof that every ionic variable is at a strict limit cycle. Tissue coupling changes the response. The original final three-beat acceptance criteria therefore remain mandatory. If adaptation still fails, inspect the exported traces before extending the declared protocol in a new run.

## Running on the simulation computer

Copy the complete updated `3D_Code` and `Shared_Pipeline` folders together. Do not mix V1 and V2 modules. The original clinical Rmd is not needed to execute the Python pilot.

Activate the Python/openCARP environment, then work from the directory containing those two folders:

```bash
python3 3D_Code/run_boundary.py --check-config
python3 3D_Code/run_boundary.py --np 1 --output 3D_Code/runs/boundaryPilotV2
```

Use a new output folder. Simulation settings are compared against saved settings before reuse. Old prepared meshes or old state files are not silently substituted.

Optional explicit stages:

```bash
python3 3D_Code/run_boundary.py --prepare-only --output 3D_Code/runs/boundaryPilotV2
python3 3D_Code/run_boundary.py --condition-only --output 3D_Code/runs/boundaryPilotV2
python3 3D_Code/run_boundary.py --simulate-only --np 1 --output 3D_Code/runs/boundaryPilotV2
python3 3D_Code/run_boundary.py --process-only --output 3D_Code/runs/boundaryPilotV2
```

`--prepare-only` runs mesher, but does not run bench or tissue simulations. `--condition-only` prepares if necessary, then runs bench only. `--simulate-only` includes conditioning if needed. Completed conditions can be reused; incomplete tissue output requires a new folder or deliberate recovery by the researcher.

If commands are not available on PATH, set their actual installed paths:

```bash
export OPENCARP_BIN=/absolute/path/to/openCARP
export MESHER_BIN=/absolute/path/to/mesher
export BENCH_BIN=/absolute/path/to/bench
```

Only use `--np 8` after selecting the MPI launcher compatible with that openCARP installation:

```bash
export MPIEXEC_BIN=/absolute/path/to/compatible/mpiexec
```

Do not move MPI binaries. Repeated-launch detection is a post-run safeguard, not a full MPI compatibility test. The serial path avoids that launcher dependency.

If using a changed configuration, pass `--config path/to/config.json` to every stage. The notebook exposes the same workflow with one cell per research step and no stale V1 default overrides.

### Resources

The expected regular mesh has 88,209 nodes and 384,000 tetrahedra. At 40,001 frames, one float32 Vm file is about 14.1 GB; the pair is about 28.2 GB. Allow at least 40 GB free disk, more for retained reruns. Prefer 16 GB RAM. Actual solver memory and runtime depend on the installation.

Finite-electrode export gathers recovery data in short time blocks. Contribution reconstruction uses sparse maps from nodal Vm to tissue rings, avoiding a dense three-component source array for every element and time. It still reads the measurement/history interval and can take appreciable time.

## Physiological acceptance

The existing limits remain:

- Exactly one activation per beat throughout the viable ROI; no lesion-core threshold crossings.
- Complete APD90 and the response inside the 0–400 ms measurement window.
- Activation-plane 95th-percentile residual ≤2 ms and direction error ≤5 degrees.
- Maximum beat-to-beat activation change ≤2 ms, amplitude change ≤5%, APD90 change ≤5 ms.

The CSV adds median/95th-percentile changes, failing fractions and worst-node IDs, without replacing the maxima used for acceptance. `results/*BeatTraces.csv` and `figures/06_*_beat_stability.png` show relevant Vm traces.

The normal runner and notebook stop when physiology fails. To produce clearly diagnostic signal/contribution/change-point exports from that same failed run:

```bash
python3 3D_Code/run_boundary.py --process-only --diagnostic --output 3D_Code/runs/boundaryPilotV2
```

Diagnostic mode does not make radii eligible or change the failed pilot status. Analysis reruns archive previous derived results; raw files remain unchanged.

## Contribution reconstruction

For each linear tetrahedron, calculate the three components of grad(Vm), retain its material conductivity tensor and volume, and project onto the finite-electrode lead field at its centroid. Do not average conductivity between viable and lesion elements.

The element coefficients form a sparse linear operator. Elements are assigned to concentric rings by their lateral centroid distance from the recording anchor. Each ring includes the full tissue thickness. The sum over all rings reconstructs the full trace. The persistent 80% radius is the first radius after which residual energy stays ≤20% at every larger radius.

This is an element-centroid quadrature approximation. It still needs mesh, electrode-quadrature and domain convergence. One signed scale from valid intact unipolar shapes is applied to all configurations; correlation magnitude must be ≥0.95, shape error ≤0.2 and relative scale error ≤0.3. A negligible clean signal remains ineligible. Negative correlation is consistent with the sign convention and is not itself a failure.

Raw diagnostic traces/radii are saved even when physiology fails, but accepted `fov80ReconstructionMm` remains missing. Reasons distinguish physiology failure, negligible/invalid signal, missing intact scale and recovery mismatch. Outer-edge reach is flagged. The old cumulative node-energy radius is removed; it is not equivalent to the primary reconstruction radius or to an element-energy calculation.

## Clinical-style change-point analysis

### What is being estimated?

The physical boundary is known at zero. The clinical change point is the **lesion-side distance at which voltage decay approaches a plateau**. A contribution radius is a full-thickness spatial reconstruction quantity. They can be compared at matching configurations but are not equal by definition. The code does not treat their difference as boundary-localisation error.

### Mapping to EFOV_analysis_pipeline.Rmd

| Clinical workflow | Pilot V2 |
|---|---|
| `Dist = DistAcrossPath + 3`, positive into scar | `Dist = distanceInsideMm`; exact mesh boundary, offset **0** |
| Voltage model on `0 < Dist < 15` and `PP > 0.01` | Same primary lesion-side ROI and detection exclusion |
| Upper 99.9th-percentile trimming | Same quantile, within each condition/gap/noise scenario across declared recordings; untrimmed sensitivity retained |
| Exponential versus power-law decay by ML AIC on identical observations | OLS Gaussian AIC on identical profile points; no patient random effect exists in one slab |
| Continuous log-voltage decay then zero-slope plateau | Same equation; Python profiles candidate change points, optional R uses `mcp` |
| Patient-level biological sampling | One slab; gaps, triangles, positions and noise repeats remain separate measurement factors |
| Peak-frequency analysis | Not calculated; exported PFreq is missing rather than inventing a clinical frequency estimator |

Four configurations are fitted separately at each height. Duplicate anchor distances within a configuration are collapsed to median voltage, with sample flow recorded. No fake patient IDs or roof/inferior labels are created.

Python fits `log(PP) = a + b * min(x, cp)`, with x = distance for exponential or log(distance) for power-law. It searches on a 0.05 mm candidate grid with at least three observed distances on either side. This grid spacing is numerical search resolution, not 0.05 mm experimental accuracy.

Both forms are exported; the primary form is selected from the unsegmented decay fits as in the clinical Rmd. A candidate is labelled descriptively supported only if the slope is negative, the optimum is away from the search edges, and its AIC improves by at least 2 over both the corresponding no-break decay and a constant model. This is a declared descriptive rule, not a formal change-point significance test. Python exports no Bayesian intervals.

### Sensitivities and controls

- Fit intact profiles as negative controls. A flat control can still produce a numerical candidate; it must not automatically be interpreted as a change point.
- Fit clean and each noisy realisation separately; report candidate variation and the number of supported fits.
- Compare clinical-trimmed with untrimmed profiles.
- For clean signals, additionally remove the detection cutoff to reveal exclusion-induced behaviour.
- For bipolar signals, compare actual selected-pair midpoint distance with the original three-electrode clique-centroid convention.
- Keep the two omnipolar geometries separate throughout.

The Gaussian noise SD is defined per raw sample, not as a fixed continuous spectral density. Changing output sampling changes the filtered noise floor. The clean analysis and exported measured noise traces allow this effect to be examined rather than silently interpreted as tissue EFOV.

### Optional Bayesian R analysis

After Python processing, install `mcp`, `ggdist` and JAGS on the analysis computer, then:

```bash
Rscript 3D_Code/pilot_change_points.R 3D_Code/runs/boundaryPilotV2
```

This reads the exact clean, clinical-trimmed, recording-anchor profiles and AIC-selected forms from Python. It uses the original Rmd specification `mcp(list(PP_log ~ 1 + x, ~ 0))`, 3 chains, 3000 iterations, 1500 adaptation iterations, seed 3 and default priors. It exports fit objects, parameter summaries, cp in mm and its 95% highest-density interval, plateau voltage, negative-slope probability and sampler diagnostics.

Rhat ≤1.01 and effective sample size ≥400 are declared sampler criteria. The Python no-change comparison and physical pilot acceptance are retained; sampler convergence alone does not establish an interpretable change point. These are conditional model-based credible intervals, not biological population uncertainty. Spatial observations remain correlated. A lesion change point is marked eligible for interpretation only when the physical pilot passes, its descriptive fit is supported, and the matched intact control has an available fit without supported change-point evidence. The R result additionally requires sampler convergence.

Python's three-observations-per-segment restriction is not imposed as an mcp prior, so differences between estimators should be investigated. The full original clinical Rmd is unchanged: its mixed model requires multiple real patients and must not be supplied artificial patient IDs to make it run.

## Output reading order

1. `conditioning/bench.log`, `conditioning/complete.json`: conditioning provenance.
2. `physiologyStatus.json`, `physiologyChecks.csv`, Vm overlays: active experiment.
3. `results/boundaryChecks.csv`: each label/gap's far coverage and lesion transition.
4. `results/contributions.csv`, `*ContributionCurves.npz`: recovery agreement, eligibility, ring curves and diagnostic full traces.
5. `results/pilotStatus.json`: physical pilot decision. Change-point support is reported separately; the code never forces a plateau to make the physical pilot pass.
6. `results/clinical/clinicalPoints.csv`: signal, signed distances, exact weights and geometry.
7. `clinicalSampleFlow.csv`, `clinicalFitData.csv`: exclusions and actual fit inputs.
8. `changePointModelSelection.csv`, `changePointFits.csv`, `changePointProfiles.csv`, `changePointFits.png`: candidate forms, no-change comparisons and fitted curves. Unsupported candidates remain visible with status labels.
9. `changePointNoiseSummary.csv`: measurement variation; check supportedCount alongside candidate means/SDs.
10. `changePointAndContributionComparison.csv`: matching configuration-level quantities, with their different meanings retained.
11. Optional `results/clinical/bayesian/`: original mcp model results and session information.

## Before selecting final thesis parameters

First resolve any failed capture/adaptation or reconstruction check. Then compare spatial resolution, output interval, electrode surface sampling and domain extent while holding the physiological and measurement protocol fixed. Inspect negative controls and noise/detection sensitivities before claiming that a plateau distance is a stable sensing property. Circular lesions and fibre/bundle complications should follow this straight-boundary calibration.

## Revision verification status

The code and notebook were reviewed statically. No new bench/openCARP simulation, Python numerical workflow, R fit or test suite was run locally, following the request to execute on the separate simulation computer. The earlier six-recording element reconstruction was diagnostic analysis of existing V1 data, not execution or validation of this V2 implementation.
