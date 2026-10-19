# Controlled active-boundary pilot V2

Start with [run.ipynb](run.ipynb). The notebook and terminal runner use the same ordinary Python functions. The revision is based on the October 19 serial pilot: capture and planarity passed, but adaptation, spatial coverage and the nodal contribution approximation needed correction.

## Run on the simulation computer

Copy **both** `3D_Code` and `Shared_Pipeline`, keeping them beside each other. Activate the environment containing openCARP, mesher, bench, NumPy, SciPy, pandas and Matplotlib. From their parent directory:

```bash
python3 3D_Code/run_boundary.py --check-config
python3 3D_Code/run_boundary.py --np 1 --output 3D_Code/runs/boundaryPilotV2
```

Use a **new** output folder. The default run conditions one cell, simulates both slabs, checks the last three beats, reconstructs contributions, and exports clinical-style change-point results. Allow at least **40 GB free disk**, preferably **16 GB RAM**. The two uncompressed Vm files alone are approximately 28 GB.

If executables are not on PATH, set `OPENCARP_BIN`, `MESHER_BIN`, and `BENCH_BIN` to their full paths. Parallel runs additionally require an explicit, compatible `MPIEXEC_BIN`; start serially until that installation is verified.

## What changed

- 1000 single-cell conditioning beats, followed by 16 matched tissue beats at BCL 500 ms.
- Vm/recovery saved every 0.2 ms; solver timestep remains 20 microseconds.
- Slab extended only across y to 60 mm; catheter centres span -16 to +16 mm at 0.5 mm spacing.
- Four fixed recording geometries, with mirrored omnipolar triangles analysed separately.
- Contributions calculated per tetrahedron, with sparse ring sums instead of conductivity averaging across the interface.
- Automatic clinical-style voltage change-point analysis, with intact controls, noise-repeat fits, detection/trimming sensitivities and a bipolar distance-convention comparison.

Read [BOUNDARY_PILOT.md](BOUNDARY_PILOT.md) for the reasoning and detailed output definitions. The acceptance thresholds are unchanged. A passing next run is not guaranteed; this revision has had **static review only, no local execution or numerical tests**.

## Reading order

| File | Role |
|---|---|
| `configs/boundary_pilot.json` | All scientific settings |
| `boundary_setup.py` | Mesh, geometry, bench conditioning and matched simulations |
| `boundary_checks.py` | Capture, plane fit and final three-beat stability |
| `../Shared_Pipeline/egm.py` | Existing unipolar, selected bipolar and omnipolar definitions |
| `boundary_analysis.py` | One gain, filtering/noise, contribution eligibility and pilot decision |
| `../Shared_Pipeline/ground_truth.py` | Tetrahedron gradients, finite lead fields and persistent reconstruction radius |
| `boundary_clinical.py` | Automatic representative Python change-point workflow |
| `pilot_change_points.R` | Optional original Bayesian `mcp` specification |
| `boundary_figures.py` | Geometry-specific profiles and diagnostic figures |

## Clinical analysis

After the run, inspect `results/clinical/changePointFits.csv` and its figure. Python preserves the clinical voltage transformations and model family but uses OLS/profile least squares for a single slab. It does not invent patient random effects or Bayesian intervals.

For the Bayesian companion, install JAGS and R packages `mcp` and `ggdist`, then run:

```bash
Rscript 3D_Code/pilot_change_points.R 3D_Code/runs/boundaryPilotV2
```

The original `../EFOV_analysis_pipeline.Rmd` remains unchanged. Its multi-patient models cannot be applied unchanged to one simulated slab. The companion reads exactly the prepared pilot profiles and uses the same `mcp` equation, chains, iterations, adaptation and seed.
