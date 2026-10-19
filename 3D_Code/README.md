# Active controlled-boundary pilot

Start with **[run.ipynb](run.ipynb)**. It follows the experiment from settings and electrode layout to matched simulations, physiological checks, signal measurements and contribution estimates. It is also the recommended reading order for colleagues.

The scientific defaults remain in [configs/boundary_pilot.json](configs/boundary_pilot.json), with ionic model and conductivity settings in [3d_pilot.par](3d_pilot.par). This rewrite has not been executed or tested. Only static source and notebook checks were performed.

## Read the code in this order

| File | Question it answers |
|---|---|
| `boundary_setup.py` | How are the two meshes, stimulus nodes, recovery points and openCARP commands constructed? Start at `prepareExperiment`, then `simulateExperiment`. |
| `boundary_checks.py` | Did we capture repeatable beats and measure the intended planar wave? |
| `../Shared_Pipeline/egm.py` | How are unipolar, selected bipolar and omnipolar signals calculated? |
| `boundary_analysis.py` | How do gain, filtering, noise, distances and contribution acceptance fit together? Start at `analyseExperiment`. |
| `../Shared_Pipeline/ground_truth.py` | How is the tissue contribution approximation calculated? |
| `../Shared_Pipeline/boundary_geometry.py` | How are finite electrodes and signed boundary distances defined? |
| `boundary_figures.py` | How is each diagnostic figure drawn? |
| `../Shared_Pipeline/igb_io.py` | How are selected frames read from the simulator files? |
| `run_boundary.py` | How are the same steps run unattended from the terminal? |

Functions and local variables use descriptive camelCase names. The `Recording` dataclass groups only four related values: kind, label, traces and weights. The specialised array operations are confined to the signal and contribution calculations, with dimensions and purposes explained alongside them. Historical NPZ keys retain their original spelling so saved raw exports remain readable.

## Run on your simulation computer

Keep `3D_Code` and `Shared_Pipeline` beside each other. Activate your openCARP/Python environment, then run from the parent project folder:

```bash
python3 3D_Code/run_boundary.py --check-config
python3 3D_Code/run_boundary.py --prepare-only --output 3D_Code/runs/boundaryPilot
NP=8 bash 3D_Code/efov_boundary.sh --output 3D_Code/runs/boundaryPilot
```

Choose the process count to match your computer. For the interactive workflow, open `run.ipynb` instead. Full installation instructions, reprocessing commands, outputs and failure interpretation are in **[BOUNDARY_PILOT.md](BOUNDARY_PILOT.md)**.

## Scope of the readability rewrite

The pilot still uses the same settings, model, finite-electrode construction, recording definitions and acceptance thresholds. Unused legacy sweeps, clinical exports and general-purpose configuration layers have been removed from the shared modules. Those modules now implement this 3D pilot, not the old 2D/reentry command-line workflows.

Saved simulation settings and the parameter template are compared directly before reuse; source snapshots are retained. Source-code hashes are no longer used. Use a new run folder after editing simulation-generation code. Analysis reruns archive previous derived outputs. Physiological and overall pilot status files now include individually named acceptance criteria.

The contribution reader requires explicit conductivity values for every tissue tag and x-aligned fibres. This prevents an incomplete parameter file from silently supplying an assumed conductivity. The filter-history check also requires the full doubled history to be available. Configuration checks reject invalid integer counts and missing calibration placements before running. Numerical equivalence and remote execution remain unverified until you run them.
