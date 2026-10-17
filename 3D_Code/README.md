# 3D EFOV pipeline

The default shell command runs **four sequential simulation stages**, followed by EGM measurements, lead-field ground-truth estimates, contribution maps and a set of thesis figures. `common.par` supplies shared settings; `scar.par` is an overlay, not a fifth simulation.

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

## Running on a remote machine

1. Install openCARP, `mesher` and carputils ([installation guide](https://opencarp.org/download/installation)) and configure carputils for a **local** (non-batch) platform. Point `settings.execs.CARP` / `MESHER` in carputils' `settings.yaml` at the installed binaries.
2. Install the Python packages: `pip install -r 3D_Code/requirements.txt` into the same environment.
3. Copy the repository (no `runs/` or `results/` needed) and start the run unattended:

```bash
cd /path/to/Masters_V2
NP=16 nohup bash 3D_Code/efov_3d.sh > efov3d.out 2>&1 < /dev/null &
```

On a SLURM cluster, adapt and `sbatch 3D_Code/slurm_example.sh`.

The launcher checks everything it can before simulating: Python version and packages, that the openCARP and mesher executables exist, that carputils is not in batch mode, the config, and free disk space (it warns below 20 GB). Each run writes `runs/<job>/logs/pipeline_*.log` and `environment_*.txt` (host, git commit, package versions, simulator paths); the log is copied to the results folder at the end. `NP` sets the MPI process count. The default `NP=1` is very slow at about 249,000 nodes, so use the cores you have. `NP` is part of the job-folder name.

**Resuming.** `RESUME=1` (the default) makes a rerun of the same command continue where it stopped. Stages whose `meta.json` is marked complete are reused. An incomplete stage folder is renamed `<stage>.incomplete-<time>` and rerun; anything restarting from a rerun stage is renamed `.stale-<time>` and rerun too. `RESUME=0` restores the strict behaviour of refusing existing output. Resuming requires the same `T_LIFT` and `NP`.

**Restart time headers.** If an openCARP build writes restarted IGB files without the absolute start time, `run.py` infers the axis from the restart checkpoint and frame count. It prints a `NOTE` and records the times in `meta.json` and `egm.npz`. A frame count that matches neither interpretation is an error.

## Simulation sequence

| Stage | Parameter layers | Restart | End | Purpose |
|---|---|---:|---:|---|
| `s1_pace` | common + s1 | fresh | 1795 ms | Initial pacing |
| `s2_block` | common + s2 | s1 at 1795 ms | `T_LIFT`, default 1950 ms | Final stimulus with exit blocked |
| `s3_reentry` | common + s3 | s2 at `T_LIFT` | 4000 ms | Restore conduction; observe activity |
| `s4_ablate_terminate` | common + scar + s4 | **s3 at 3500 ms** | 6000 ms | Apply the lesion and observe activity |

The fourth stage branches from the 3500 ms checkpoint, not the end of s3. Its interval overlaps s3 from 3500 to 4000 ms; do not concatenate these as one uninterrupted trace. Stage names describe intended experiments, not verified physiological outcomes.

`SCENARIO=pre` runs s1–s3; `SCENARIO=ablate` runs s4 using an existing s3 checkpoint. `SCENARIO=reinduce` runs three additional post-ablation stages from fresh initial conditions. `SCENARIO=all` runs all seven stages. Reinduction is not part of the default four-stage sequence.

The default job folder is `3D_Code/runs/reentry_tlift1950_np1/`. `T_LIFT` and `NP` change the folder name. Completed stages are reused (see Resuming); with `RESUME=0`, existing nonempty stage folders are rejected rather than overwritten. To repeat a simulation from scratch, first archive or rename the old job folder. To recalculate the outputs from completed simulations:

```bash
SKIP_SIMS=1 bash /path/to/Masters_V2/3D_Code/efov_3d.sh
```

Retain the matching `T_LIFT`, `NP` and `SCENARIO` when reprocessing. The runner needs immediate execution: on a cluster, run inside your allocated compute job using a local carputils platform. Deferred carputils batch submission and `--dry` are rejected because later stages need files from the previous stage immediately.

## Outputs

Results go to `results/slab3d_default__reentry_tlift1950_np1__protocol/` for the default settings. Scenario-specific result names prevent post-only processing from replacing the four-stage results. `CONFIG` accepts a config filename under `Shared_Pipeline/configs`, or an absolute JSON path.

- Each stage's `egm.npz`: 242 catheter placements (121 centres at each of two electrode gaps), 16 unipoles per placement, and the full electrode lattice. Eight cylindrical-surface samples are averaged per electrode. Time comes from the IGB header and must match Vm.
- `points.csv`: unipolar, bipolar and omnipolar peak-to-peak voltage and differential peak frequency; 61 recordings per placement with the default settings. It also records the analysis interval and the exact electrode weights selected for each recording.
- `run_summary.json`: the applied configuration, gain, noise level and frequency checks.
- `groundtruth.csv`: two 80% field-of-view radius estimates, tissue-region shares and unipolar agreement with openCARP recovery. By default it covers 15 catheter centres (x = -5…5 mm along the bundle axis, plus y = ±3, ±5 mm through the lesion) at both gaps, **not every placement**.
- `groundtruth_curves.npz`: for each ground-truth recording, the share of signal energy explained by tissue within each lateral radius (the field-of-view curve).
- `points_with_gt.csv`: all EGM measurements, with ground truth joined where calculated. `groundtruth_computed` distinguishes an uncomputed placement from a computed quiescent signal whose field of view is undefined.
- `maps/`: top and depth PNGs, map arrays, CSV summaries, and VTK volumes/electrode positions. Use `MAPS=0` to skip this step.
- `figures/`: thesis figures (see below). Use `FIGURES=0` to skip this step.

## Thesis figures

`figures.py` runs last and writes about 33 figures to `results/<name>/figures/`, each as PNG (300 dpi) and PDF (vector, for LaTeX). `FIGURES.md` gives every figure with a caption and the numbers measured from that run. `figure_stats.json` holds those numbers, and `tables/` holds CSV and LaTeX summary tables.

| Group | Figures |
|---|---|
| Model and protocol | mesh regions and probe sites; HD Grid layout and electrode gaps; stage timeline with analysis windows |
| Electrophysiology (from Vm) | probe Vm traces over 0–6000 ms; the T_LIFT window check (APD90 beside the exit and arrival at the interface); surface snapshots per stage; transmural snapshots; time–space plots; activation maps; activation raster, cycle length and termination time; conduction velocity along and across fibres |
| Electrograms | 4×4 unipolar grid before/after ablation; uni/bi/omni examples at both gaps; voltage maps per gap; log2 post/pre ratio map; PP against distance from the lesion; gap effect; PP distributions; peak frequency |
| Ground truth | fov80 by kind/stage/gap; field-of-view curves; region shares; validation against openCARP recovery; fov against position; fov against PP |

Each figure is independent: one that cannot be made is logged in `figures/failures.log` and the rest continue, and the final message says how many failed. Use `STRICT_FIGURES=1` to make any figure failure fail the run. Vm planes and probe traces are extracted once into `figures/cache/` (about 0.6 GB at full resolution). Regenerate figures without rerunning anything else:

```bash
python3 3D_Code/figures.py --results results/<name> --config 3D_Code/runs/<job>/configs/slab3d_default.json \
    3D_Code/runs/<job>/s1_pace 3D_Code/runs/<job>/s2_block 3D_Code/runs/<job>/s3_reentry 3D_Code/runs/<job>/s4_ablate_terminate
```

Add `--only 05,17` to remake selected figures. Vm figures describe what the simulation did. Activation is an upward -20 mV crossing; CV is a gradient estimate from the first paced beat. They do not decide whether reentry or termination was physiologically correct.

Both EGM and ground-truth calculations use the final 400 samples at 1 ms spacing, or all samples for shorter stages, plus up to 100 ms of preceding data to initialise filters. Change `analysis_window_last_ms` and `filter_warmup_ms` in the configuration to adjust this consistently. A separate `groundtruth.window_last_ms` must not disagree with the EGM window. The default band is 30–450 Hz at 1000 Hz sampling; an upper cutoff at 500 Hz would equal Nyquist and is rejected.

Auto gain calibrates using the supplied pre-ablation runs. Post-only scenarios use the matching s3 recording for calibration, so that recording must already exist. Select fixed gain explicitly if this is not desired. Different scenarios can therefore have different calibrated gains; use a common fixed gain for strict cross-scenario amplitude comparisons. Noise is reproducible from the configured seed.

## Scientific limits to check on real simulations

The parameters currently use **Courtemanche with selected cAF modifications**, an approximation to the cited paper's Koivumäki model. They do not implement its full calcium-handling remodelling. The numerical defaults and lesion geometry were preserved. Verify capture, activation paths, the block-lift window, sustained reentry, and the response to ablation from Vm/LAT outputs. The pipeline does not infer those outcomes merely because a run finishes.

`TUNE_CV` / `--tune-cv` are disabled: the previous tuner used an unremodelled ionic model, so its target velocity could not be assumed to apply to these simulations. Conductivities remain in `common.par`; direct `run.py` use also supports a validated pair of `--gi` / `--ge` values, with distinct output-folder names.

Ground truth uses **finite differences on a regular Cartesian grid**, nodal conductivity averages and volume weights. It approximates openCARP's finite-element recovery; successful execution is not proof of quantitative equivalence. Check `r_vs_recovery` and `scale_vs_recovery` on actual active recordings. Correlation below 0.95 in absolute value produces a warning. A flat signal has undefined correlation and field of view. Gain and additive measurement noise are excluded from tissue contributions, but the saved recording orientation is reused exactly. The radii are lateral x–y distances, not spherical 3D radii.

At the full mesh size (about 249,000 nodes), Vm files and ground-truth source arrays are large. Allow at least 8 GB of RAM (16 GB is comfortable) and roughly 7 GB for raw Vm output across the default four stages, plus recovered potentials, the figure cache and other files: plan for 15–20 GB of disk. The map calculation repeats the source calculation for each stage. Ground truth is restricted to selected placements to limit cost: roughly 30 s per placement per stage, so about an hour in total for the default 30 placements × 4 stages. Fewer `placements_xy_mm_*` in the config make it faster.

## Relationship to the R Markdown analysis

`EFOV_analysis_pipeline.Rmd` expects clinical-style Excel inputs, patient IDs, Inferior/Roof labels, and distance-validation inputs. The 3D config intentionally disables that export. The shell pipeline ends at the 3D CSVs and maps; it does **not** run the clinical R report. Mapping stages or catheter gaps to fictitious patients or anatomical locations would invalidate that analysis. A simulation-specific EFOV estimation report needs its own statistical design before those outputs can be compared to ground truth.

## Verification

From the repository root, using a Python environment with NumPy, SciPy, pandas and Matplotlib:

```bash
python -m unittest discover -s tests -v
```

The regression tests verify restart selection, resume after an interrupted stage (including time-axis recovery for headers without a start time), stage dispatch through a mocked carputils interface, IGB offsets/endianness/truncation, electrode export, recording weights, invalid windows, and a shell-driven synthetic fixture through CSVs, PNGs, VTKs and thesis figures. They also check that missing output fails rather than printing success. These are software checks, not a live openCARP validation or evidence of reentry.

Implementation references: [openCARP checkpoints](https://opencarp.org/documentation/examples/02_ep_tissue/23_checkpoint), [openCARP parameters](https://opencarp.org/parameters/master/), [Courtemanche parameters](https://opencarp.org/doxygen-carputils/master/classcarputils_1_1model_1_1ionic_1_1CourtemancheIonicModel.html), and [carputils wrapper source](https://git.opencarp.org/openCARP/carputils/-/blob/master/carputils/tools.py).
