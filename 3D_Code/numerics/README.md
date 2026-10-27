# Step 2: serial vs MPI equivalence (run on the workstation)

Purpose: show that a parallel openCARP run reproduces the serial run before any large
simulation is launched in parallel, and measure the speed-up for planning Step 3.
Config: `3D_Code/configs/mpi_equivalence.json` (20 x 30 x 4 mm slab, 3 beats; every physics,
solver and output setting identical to `boundary_pilot.json`). Not used for scientific results.

## 1. Find the MPI launcher that matches openCARP (inside the efov-runner container)

```bash
nproc                                   # CPUs visible to the container (Docker Desktop may limit this)
which openCARP mpirun mpiexec
ldd "$(which openCARP)" | grep -i mpi   # the MPI library openCARP is linked against
mpirun --version                        # must belong to that same MPI implementation
```

Use the launcher that belongs to the MPI library `openCARP` links against. If unsure, send me
the output of these four commands before running step 3.

## 2. Serial reference (about an hour; estimate only)

```bash
cd <repository root>
time python3 3D_Code/run_boundary.py --config 3D_Code/configs/mpi_equivalence.json \
     --simulate-only --np 1 --output 3D_Code/runs/mpiCheck_np1 2>&1 | tee 3D_Code/runs/mpiCheck_np1.time.log
```

## 3. Parallel run, same config, new folder

```bash
export MPIEXEC_BIN=/full/path/to/the/matching/mpirun
time python3 3D_Code/run_boundary.py --config 3D_Code/configs/mpi_equivalence.json \
     --simulate-only --np 8 --output 3D_Code/runs/mpiCheck_np8 2>&1 | tee 3D_Code/runs/mpiCheck_np8.time.log
```

Optional, for planning only: repeat with `--np 16` into `mpiCheck_np16` to see how the speed-up scales.

## 4. Compare

```bash
python3 3D_Code/numerics/compare_runs.py 3D_Code/runs/mpiCheck_np1 3D_Code/runs/mpiCheck_np8
```

Prints PASS/FAIL per condition and writes `mpiCheck_np8/mpiEquivalence.json`.
It first checks the two runs had identical inputs (mesh, stimulus, recovery points,
conditioning state), then compares every activation event, every Vm value and the recovered
extracellular signals. Tolerances are proposed (activation <= 0.02 ms, Vm <= 0.1 mV,
extracellular peak-to-peak <= 0.1 %) and are reported as-is, never loosened to pass.

## 5. Send back

Copy into the Masters_V2 folder on the Mac: `mpiEquivalence.json`, both `*.time.log` files and
both run folders' `intact.log` / `lesion.log`. The igb files are not needed for review.
If the comparison fails, keep both run folders; do not rerun into the same folder.
