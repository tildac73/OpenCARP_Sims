# 3D EFOV pipeline

The default shell command runs **four sequential simulation stages**, followed by EGM measurements, lead-field ground-truth estimates and contribution maps. `common.par` supplies shared settings; `scar.par` is an overlay, not a fifth simulation.

```bash
bash /path/to/Masters_V2/3D_Code/efov_3d.sh
```

Use an activated Python environment containing `carputils`, NumPy, SciPy, pandas and Matplotlib, with openCARP and mesher configured in carputils. Alternatively select it explicitly:

```bash
PYTHON=/path/to/environment/bin/python bash /path/to/Masters_V2/3D_Code/efov_3d.sh
```

The command works from any directory. For an unattended run:

```bash
nohup bash /path/to/Masters_V2/3D_Code/efov_3d.sh > log_3d.txt 2>&1 < /dev/null &
```

## Simulation sequence

| Stage | Parameter layers | Restart | End | Purpose |
|---|---|---:|---:|---|
| `s1_pace` | common + s1 | fresh | 1795 ms | Initial pacing |
| `s2_block` | common + s2 | s1 at 1795 ms | `T_LIFT`, default 1950 ms | Final stimulus with exit blocked |
| `s3_reentry` | common + s3 | s2 at `T_LIFT` | 4000 ms | Restore conduction; observe activity |
| `s4_ablate_terminate` | common + scar + s4 | **s3 at 3500 ms** | 6000 ms | Apply the lesion and observe activity |

The fourth stage branches from the 3500 ms checkpoint, not the end of s3. Its interval overlaps s3 from 3500 to 4000 ms; do not concatenate these as one uninterrupted trace. Stage names describe intended experiments, not verified physiological outcomes.

`SCENARIO=pre` runs s1–s3; `SCENARIO=ablate` runs s4 using an existing s3 checkpoint. `SCENARIO=reinduce` runs three additional post-ablation stages from fresh initial conditions. `SCENARIO=all` runs all seven stages. Reinduction is not part of the default four-stage sequence.

The default job folder is `3D_Code/runs/reentry_tlift1950_np1/`. `T_LIFT` and `NP` change the folder name. Existing nonempty stage folders are rejected rather than overwritten. To repeat a simulation, first archive or rename the old job folder. To recalculate the outputs from completed simulations:

```bash
SKIP_SIMS=1 bash /path/to/Masters_V2/3D_Code/efov_3d.sh
```

Retain the matching `T_LIFT`, `NP` and `SCENARIO` when reprocessing. The runner needs immediate execution: on a cluster, run inside your allocated compute job using a local carputils platform. Deferred carputils batch submission and `--dry` are rejected because later stages need files from the previous stage immediately.

## Outputs

Results go to `results/slab3d_default__reentry_tlift1950_np1__protocol/` for the default settings. Scenario-specific result names prevent post-only processing from replacing the four-stage results. `CONFIG` accepts a config filename under `Shared_Pipeline/configs`, or an absolute JSON path.

- Each stage's `egm.npz`: 242 catheter placements (121 centres at each of two electrode gaps), 16 unipoles per placement, and the full electrode lattice. Eight cylindrical-surface samples are averaged per electrode. Time comes from the IGB header and must match Vm.
- `points.csv`: unipolar, bipolar and omnipolar peak-to-peak voltage and differential peak frequency; 61 recordings per placement with the default settings. It also records the analysis interval and the exact electrode weights selected for each recording.
- `run_summary.json`: the applied configuration, gain, noise level and frequency checks.
- `groundtruth.csv`: two 80% field-of-view radius estimates, tissue-region shares and unipolar agreement with openCARP recovery. By default it covers three selected catheter centres at both gaps, **not every placement**.
- `points_with_gt.csv`: all EGM measurements, with ground truth joined where calculated. `groundtruth_computed` distinguishes an uncomputed placement from a computed quiescent signal whose field of view is undefined.
- `maps/`: top and depth PNGs, map arrays, CSV summaries, and VTK volumes/electrode positions. Use `MAPS=0` to skip this step.

Both EGM and ground-truth calculations use the final 400 samples at 1 ms spacing, or all samples for shorter stages, plus up to 100 ms of preceding data to initialise filters. Change `analysis_window_last_ms` and `filter_warmup_ms` in the configuration to adjust this consistently. A separate `groundtruth.window_last_ms` must not disagree with the EGM window. The default band is 30–450 Hz at 1000 Hz sampling; an upper cutoff at 500 Hz would equal Nyquist and is rejected.

Auto gain calibrates using the supplied pre-ablation runs. Post-only scenarios use the matching s3 recording for calibration, so that recording must already exist. Select fixed gain explicitly if this is not desired. Different scenarios can therefore have different calibrated gains; use a common fixed gain for strict cross-scenario amplitude comparisons. Noise is reproducible from the configured seed.

## Scientific limits to check on real simulations

The parameters currently use **Courtemanche with selected cAF modifications**, an approximation to the cited paper's Koivumäki model. They do not implement its full calcium-handling remodelling. The numerical defaults and lesion geometry were preserved. Verify capture, activation paths, the block-lift window, sustained reentry, and the response to ablation from Vm/LAT outputs. The pipeline does not infer those outcomes merely because a run finishes.

`TUNE_CV` / `--tune-cv` are disabled: the previous tuner used an unremodelled ionic model, so its target velocity could not be assumed to apply to these simulations. Conductivities remain in `common.par`; direct `run.py` use also supports a validated pair of `--gi` / `--ge` values, with distinct output-folder names.

Ground truth uses **finite differences on a regular Cartesian grid**, nodal conductivity averages and volume weights. It approximates openCARP's finite-element recovery; successful execution is not proof of quantitative equivalence. Check `r_vs_recovery` and `scale_vs_recovery` on actual active recordings. Correlation below 0.95 in absolute value produces a warning. A flat signal has undefined correlation and field of view. Gain and additive measurement noise are excluded from tissue contributions, but the saved recording orientation is reused exactly. The radii are lateral x–y distances, not spherical 3D radii.

At the full mesh size (about 249,000 nodes), Vm files and ground-truth source arrays are large. Allow several GB of RAM and roughly 7 GB for raw Vm output across the default four stages, plus recovered potentials and other files. The map calculation repeats the source calculation for each stage. Ground truth is restricted to selected placements to limit cost.

## Relationship to the R Markdown analysis

`EFOV_analysis_pipeline.Rmd` expects clinical-style Excel inputs, patient IDs, Inferior/Roof labels, and distance-validation inputs. The 3D config intentionally disables that export. The shell pipeline ends at the 3D CSVs and maps; it does **not** run the clinical R report. Mapping stages or catheter gaps to fictitious patients or anatomical locations would invalidate that analysis. A simulation-specific EFOV estimation report needs its own statistical design before those outputs can be compared to ground truth.

## Verification

From the repository root, using a Python environment with NumPy, SciPy, pandas and Matplotlib:

```bash
python -m unittest discover -s tests -v
```

The regression tests verify restart selection, stage dispatch through a mocked carputils interface, IGB offsets/endianness/truncation, electrode export, recording weights, invalid windows, and a shell-driven synthetic fixture through CSVs, PNGs and VTKs. They also check that missing output fails rather than printing success. These are software checks, not a live openCARP validation or evidence of reentry.

Implementation references: [openCARP checkpoints](https://opencarp.org/documentation/examples/02_ep_tissue/23_checkpoint), [openCARP parameters](https://opencarp.org/parameters/master/), [Courtemanche parameters](https://opencarp.org/doxygen-carputils/master/classcarputils_1_1model_1_1ionic_1_1CourtemancheIonicModel.html), and [carputils wrapper source](https://git.opencarp.org/openCARP/carputils/-/blob/master/carputils/tools.py).
