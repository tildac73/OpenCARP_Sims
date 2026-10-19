# Controlled boundary pilot

Start with **run.ipynb** for the readable, step-by-step workflow, or use **efov_boundary.sh** for an unattended run.

This pilot uses the latest notebook's homogeneous 40 × 40 × 4 mm slab and **unmodified Courtemanche (CRN)**. It does not use the cAF modifications or the bundle from the older reentry experiment. Both simulations are fresh runs under the same pacing schedule, with identical node coordinates, fibres, catheter positions and recording times.

## First experiment

- Wave travels along +x; fibres point along x.
- Straight transmural lesion occupies y < 0. The boundary is parallel to propagation.
- Stimulate a full-thickness strip at the left x edge. Exclude every node touching a lesion element from the lesion stimulus. The same viable-tissue stimulus rule is applied to both conditions, so their node sets differ.
- Retain the virtual y = 0 boundary in the intact reference.
- Translate the 4 × 4 grid across y at x = 0. Grid pitch is 4 mm. Both 0.25 and 1 mm standoffs are included.
- Average eight lateral-cylinder samples per electrode, as in the notebook. Recovery points shared by placements are calculated once.

The lesion has near-zero conductivity (1e-6 S/m), not a passive ionic model. Its core is explicitly checked for activation. The forward solution remains openCARP extracellular recovery in an unbounded homogeneous medium; this is **not** a finite blood-bath or pseudo-bidomain solve.

## Defaults are pilot choices

The initial configuration is `configs/boundary_pilot.json`:

| Setting | Pilot value |
|---|---|
| Slab | 40 × 40 × 4 mm |
| Mesh spacing | **0.5 mm**, deliberately coarse for the first pilot |
| Integration step | 20 µs |
| Recording interval | 1 ms |
| Pacing | 8 beats, 500 ms cycle length |
| Stimulus | 50 µA/cm² for 2 ms, 0.5 mm edge strip |
| Measurement | Final beat, 0–400 ms after its stimulus; end excluded |
| Filter history | 200 ms, checked against twice that history |
| Unipolar filter | None |
| Differential filter | Causal Butterworth, order parameter 2, 30–450 Hz |
| Calibration | Intact centre y = 0, gap = 0.25 mm; selected bipolar median scaled to 1.44 mV |
| Noise | Zero-noise reference plus three independent repeats at SD 0.005 mV |
| Contribution placements | Centres y = -8, 0, 8 mm, both gaps |

Pacing interval, stimulus strength, beat count and acceptance tolerances are starting choices, not already validated physiology. The final three beats are checked for one-to-one capture, complete APD90, adaptation, propagation direction and planarity. If eight beats are insufficient, increase `beatCount` in a copied configuration and use a new run folder. Do not relax tolerances merely to obtain a passing status.

The 0.5 mm mesh has 59,049 nodes and about 307,200 tetrahedra. Two complete 4-second Vm records at 1 ms spacing require roughly 1.9 GB before other outputs. Keep at least 8 GB free and preferably 8–16 GB RAM. Source reconstruction temporarily holds several large arrays. The later 0.25 mm comparison is substantially more expensive. Runtime depends on your machine; no live openCARP timing has been measured for this pilot.

## Set up on the simulation computer

Copy **3D_Code and Shared_Pipeline**, keeping them as sibling folders under your project directory. For optional software checks, also copy `tests/test_boundary_pilot.py` into a sibling `tests` directory. Activate your usual openCARP environment. The pilot runner invokes the simulator directly and does not require carputils.

Required commands:

```bash
python3 --version
command -v openCARP
command -v mesher
command -v mpiexec
```

Use Python 3.9 or newer with NumPy, SciPy, pandas and Matplotlib. Install the existing requirements if needed:

```bash
python3 -m pip install -r 3D_Code/requirements.txt
```

If your executable names or locations differ, set explicit paths:

```bash
export OPENCARP_BIN=/absolute/path/to/openCARP
export MESHER_BIN=/absolute/path/to/mesher
export MPIEXEC_BIN=/absolute/path/to/mpiexec
```

`MPIEXEC_BIN` is needed only for more than one process. Set `PYTHON` if your Python command is not `python3`. On a cluster, run inside an allocated compute job with the appropriate MPI launcher. The supplied launcher uses `mpiexec -n`; adapt that short command in `simulateExperiment()` if your cluster requires another form. Do not run a multi-hour simulation on a login node.

The parameter syntax follows the current stimulus interface used by your existing project (`stim[0].crct`, `pulse`, `ptcl`, `elec`). Vertex files use input-mesh indices and the `extra` header, as specified by the [official electrode documentation](https://opencarp.org/documentation/examples/02_ep_tissue/24_electrode_definition). Use an openCARP version supporting that interface.

## Run from the terminal

From the project root, check the configuration without running a simulation:

```bash
python3 3D_Code/run_boundary.py --check-config
```

Prepare and inspect inputs before the first full run:

```bash
python3 3D_Code/run_boundary.py --prepare-only --output 3D_Code/runs/boundaryPilot
```

Then run both conditions and the complete analysis:

```bash
NP=8 bash 3D_Code/efov_boundary.sh --output 3D_Code/runs/boundaryPilot
```

Choose the process count for your allocated resources. The pilot writes the exact commands and simulator output to `intact.log` and `lesion.log` in the run folder. While a simulation is running, follow its log with:

```bash
tail -f 3D_Code/runs/boundaryPilot/intact.log
```

For an unattended run:

```bash
NP=8 nohup bash 3D_Code/efov_boundary.sh --output 3D_Code/runs/boundaryPilot > boundaryPilot.out 2>&1 < /dev/null &
```

To simulate now and analyse later:

```bash
NP=8 bash 3D_Code/efov_boundary.sh --simulate-only --output 3D_Code/runs/boundaryPilot
python3 3D_Code/run_boundary.py --process-only --output 3D_Code/runs/boundaryPilot
```

A configuration supplied with `--config` may have an absolute path. The launcher works from any directory; explicitly relative output paths remain relative to your current directory.

## Run from Jupyter

Open `3D_Code/run.ipynb` with the same Python environment. Its ten steps expose the scientific sequence:

1. Import the experiment functions.
2. Choose the run folder and settings.
3. Inspect finite-electrode positions and surface samples.
4. Prepare the matched meshes, stimuli and parameter files.
5. Run both simulations and average the recovery samples.
6. Check capture, adaptation and planarity.
7. Calibrate one gain from the intact active reference.
8. Export clean and noisy recordings for both conditions.
9. Reconstruct tissue contributions at the selected placements.
10. Apply acceptance criteria and display the figures.

The notebook and terminal runner use the same functions. Only the notebook exposes the intermediate tables step by step. Do not change settings between the analysis steps; restart from step 6 if you change analysis choices. The normal notebook path stops when physiology fails.

## Resume and reprocess safely

Rerun the same command to reuse completed intact/lesion simulations. A complete condition must contain its simulator outputs, EGM export, and `meta.json`. An interrupted condition is **not** silently treated as complete or overwritten. Move that incomplete condition directory aside, then rerun. Its corresponding log will be replaced; copy the log too if you want to retain it.

Changed mesh, pacing, electrodes, solver settings, or simulation-generation code require a new output folder. The runner directly compares the saved simulation settings and parameter template before reuse. Source snapshots record the implementation, but code changes are no longer detected by hashes; choose a new folder yourself after changing simulation-generation code.

Analysis settings and acceptance tolerances can change without rerunning the simulations:

```bash
python3 3D_Code/run_boundary.py --process-only --config /path/to/modifiedAnalysis.json --output 3D_Code/runs/boundaryPilot
```

The simulation settings in that file must still match. Previous derived results, figures and acceptance tables are moved to `analysisArchive/<UTC timestamp>/`. Raw simulation outputs remain in place. Each new analysis saves its own configuration and source snapshot. Declare why a criterion changed; do not present retrospective tuning as prespecified validation.

## Read the result status first

Exit codes:

- **0:** requested preparation/simulation-only action completed, or the complete analysis passed the provisional pilot criteria.
- **1:** configuration, missing dependency/file, or execution error.
- **2:** processing reached an acceptance check and the physiological or pilot criteria did not pass.

Simulation completion alone is not acceptance. Start with:

- `physiologyStatus.json` and `physiologyChecks.csv`. The JSON lists each physiological criterion by name for each condition.
- `results/pilotStatus.json`, including the named overall acceptance criteria.
- `results/boundaryChecks.csv` and the filter-history JSON files.
- `figures/01_activation.png` and `figures/02_clean_voltage_distance.png`.

If physiology fails, normal execution writes its diagnostic activation figure and stops before calibration and contribution processing. To inspect signals anyway:

```bash
python3 3D_Code/run_boundary.py --process-only --diagnostic --output 3D_Code/runs/boundaryPilot
```

This does not override acceptance: physiological eligibility remains false and no accepted FOVs are assigned. An entirely flat intact reference cannot be calibrated, so diagnostic amplitude processing will also stop; use its Vm/activation diagnostics instead.

Passing the pilot criteria still does **not** establish mesh/time-step/domain convergence. Those fields remain explicitly false. A larger slab and a finer mesh are follow-up comparisons, especially for the long-range unipolar signal.

## Output definitions

- `config.json`, `manifest.json`, `sourceSnapshot/`: simulation provenance.
- `mesh/`: identical topology with intact or lesion element tags.
- `electrodes/layout.npz`: finite electrode samples, unique recovery points and gather indices.
- `intact.par`, `lesion.par`, `*Stimulus.vtx`: resolved simulator inputs.
- `intact/`, `lesion/`: raw Vm, recovered extracellular potentials, original unscaled EGM exports and completion metadata.
- `results/calibration.json`: the single fixed gain and its reference.
- `results/points.csv`: every clean/noisy recording, selected weights, locations, distances, amplitudes and joined eligible contribution results.
- `results/*Traces.npz`: measured traces, noise-free traces at the same selected weights, and their noise component. Join using condition and `traceIndex`.
- `results/*Electrodes.csv`: individual electrode coordinates and tissue classification.
- `results/noiseSummary.csv`: within-recording amplitude variability across noise repetitions; these are not biological replicates.
- `results/contributions.csv`: raw and eligible 80% reconstruction/node-energy radii, signed correlations and scale/shape checks.
- `results/*ContributionCurves.npz`: curves keyed by `recordingId`.
- `results/analysisConfig.json`, `analysisManifest.json`, `analysisSource/`: processing provenance.

Signed distance is positive inside the lesion. `distanceViableMm` has the opposite sign. Bipolar anchors use the **selected pair midpoint**; omnipolar anchors use the three-electrode clique centroid. The mean electrode distance and clique distance remain separate columns. This also matters when using the optional circular boundary.

Contribution estimates are computed for the **clean primary recordings only**, at the configured subset of catheter centres. Noise repetitions retain their own selected weights and signal traces, but do not borrow a clean recording's FOV. `groundTruthComputed` distinguishes an uncomputed row from a computed but ineligible row. The radii are lateral full-thickness cylinder radii, not spherical radii or sensed volumes.

The source calculation reuses `ground_truth.py`: regular-grid finite differences, nodal conductivity averaging and finite-electrode lead fields. Its omitted physical prefactor and opposite polarity are handled by reporting a single signed scale estimated from intact unipoles; reconstruction checks then also cover bipolar and omnipolar recordings. Raw radii are retained for diagnosis, while eligible radii are blanked for failed signal or waveform checks.

## What to change after inspecting the first run

- **Capture failure:** inspect stimulus nodes and Vm first; review strength, duration and pacing interval.
- **Adaptation failure:** increase conditioning beats while keeping the final-beat rule.
- **Window failure:** extend the declared response window within the cycle, or increase the cycle length.
- **Planarity failure:** inspect the fitted region, depth behaviour and stimulus geometry; do not assume the boundary-adjacent wave is planar.
- **Filter-history failure:** increase the history in analysis settings and reprocess.
- **No stable far region:** enlarge the slab and extend the sampling range. Do not infer a plateau from a fitted model alone.
- **Reconstruction mismatch:** inspect signed correlation, shape error and global scale error before reporting FOV.
- **Finer pilot:** copy the configuration, change `meshSpacingMm` to 0.25 and `timeStepUs` if needed, then use a new run folder.
- **Circular lesion:** replace the boundary object with `{"shape":"circle","centreMm":[0.0,0.0],"radiusMm":2.5}`. This is an optional extension; a wave can curve around it, so the straight-boundary acceptance assumptions require review. It is not the default validated case.

## Software verification and validation status

The existing optional unit and synthetic end-to-end checks are intended to cover geometry, selected-pair midpoints, finite-electrode gathering, fixed weighting, signal export, contribution reconstruction, noise seeds, and rejection of flat tissue. Synthetic fixture waves are not CRN simulation results. The readability rewrite has not been run through these tests or openCARP, as requested. Only static source, notebook syntax and dependency references were reviewed. Numerical equivalence and simulator compatibility remain unverified.

On your simulation computer, the optional software suite is:

```bash
python3 -m unittest discover -s tests -p 'test_boundary_pilot.py' -v
```

These folders now contain only pilot files and their shared dependencies. The older reentry scripts, parameter overlays, unused map generator and legacy configurations have been removed. The older `tests/test_pipeline.py` outside these folders is not a pilot test and relies on the removed legacy files; use the pilot-specific command above.
