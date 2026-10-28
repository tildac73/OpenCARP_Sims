# Readiness kit: run Steps 2, 3 and the Step 6 feasibility test in one go

One command runs everything that needs the workstation before the reference model can be built:
the MPI equivalence gate (Step 2), the Niederer benchmark, conduction-velocity tuning and the CV
resolution study, the 3D convergence ladder including a 1.5 mm wall (Step 3), the openCARP fibre
check (Step 7), and the blood-bath work (Step 6): an openCARP pseudo-bidomain check, then bath
electrograms and ground truth for bath thickness, bath mesh, scar type and bath vs unbounded.
Every stage writes its own summary. Blocking stages stop the run if their gate fails. Re-running the
same command resumes and reuses every completed simulation.

All scientific settings are in `3D_Code/configs/readiness_kit.json`, each with its source or an
explicit ASSUMPTION/PROPOSED label. Nothing in the kit changes them.

## 1. Before you start (inside the efov-runner container)

1. Bring the workstation copy of the repository up to the commit that contains this file (copy the
   Masters_V2 folder across, or pull branch `pilot_v2`). The kit records the commit it ran from.
2. Find the MPI launcher that matches openCARP (unchanged from `3D_Code/numerics/README.md`, step 1):
   ```bash
   nproc
   which openCARP mesher bench mpirun mpiexec
   ldd "$(which openCARP)" | grep -i mpi
   mpirun --version
   export MPIEXEC_BIN=/full/path/to/the/matching/mpirun
   ```
   If unsure which launcher matches, send me that output before step 3.
3. Choose the process count: about the number of physical cores you can spare (`--np 16` below).

## 2. Check the plan (seconds, no simulator)

```bash
cd <repository root>
python3 3D_Code/readiness/kit.py plan --np 16            # tier A (default)
python3 3D_Code/readiness/kit.py plan --np 16 --tier B   # adds the 0.125 mm ladder run
```
It lists every simulation with node count, node-steps and disk. Tier A needs about 120 GB, tier B
about 320 GB. For the bath analysis install pyamg in the container first (`pip install pyamg`);
without it the largest bath model (0.25 mm, 1.1 million nodes) falls back to a much slower solver. Hours appear once the mpi stage has measured this machine's speed (re-run `plan`
after the mpi stage).

## 3. Run

```bash
cd <repository root>
export MPIEXEC_BIN=/full/path/to/the/matching/mpirun
nohup python3 3D_Code/readiness/kit.py run --np 16 > 3D_Code/runs/readiness_console.log 2>&1 &
```
Check progress at any time:
```bash
python3 3D_Code/readiness/kit.py status
```
`status` shows every stage's state and the last line of the most recent logs, including openCARP's
own ETA.

If a blocking stage fails, read `3D_Code/runs/readiness/<stage>/stage.json` (it has the error and
traceback), fix the cause, and run the same `run` command again. Completed stages and simulations
are skipped. If an openCARP run was interrupted half way, the kit refuses to reuse its folder and
says which one: move that folder aside (do not delete raw output you may want to inspect), then rerun.

## 4. Stages and gates

| Stage | What it does | Gate | If it fails |
| --- | --- | --- | --- |
| preflight | Records versions, executables, MPI launcher, CPU, memory, disk, repository commit, carputils location; lists ionic models | All executables present; scipy importable; enough disk | Stops |
| tests | Runs the repository's unit tests | All pass | Stops |
| mpi | Serial and `--np` runs of `mpi_equivalence.json`, then `compare_runs.py`; measures speed-up | Activation <= 0.02 ms, Vm <= 0.1 mV, phi_e PP <= 0.1 % | Stops. Keep both folders |
| smoke | 1 ms openCARP run of every kind of parameter file used later (benchmark, strips, ladder intact and lesion, bath) | openCARP starts once and writes its outputs | Stops: a parameter name or file is wrong |
| niederer | 3 spacings x 3 time steps of the Niederer et al. 2011 benchmark | Reported | Continues |
| cv | Tunes all conductivities (Roberts and Scher ratios fixed) to 0.90 m/s along fibres at 0.125 mm; then CV at 0.5, 0.25, 0.0625 mm and at dt 10 us; transverse CV; pilot conductivities at three spacings; sensitivity targets 0.70 and 1.25 m/s | Tuning within 1 % of target | Stops: ladder and bath need the tuned values |
| fibres | Two small runs with rotated (30 deg) and bilayer (0/90 deg) fibres; openCARP's recovered phi_e vs electrode synthesis; control with fibres wrongly taken as x | Reported (proposed: corr >= 0.999; control clearly worse) | Continues |
| ladder | Full pilot pipeline on the tuned model: mesh 0.5 / 0.25 (/ 0.125 tier B) mm, dt 20 / 10 us, output 0.2 / 0.1 ms, 4 / 8 beats, domain 40x30 / 40x60 / 60x90 mm, 1.5 mm wall at 0.5 / 0.25 mm; then DL-2026-10-28-05 comparisons | Reported per comparison (strict and clarified change-point rule) | Continues; failed runs listed |
| bath | Pseudo-bidomain with a 10 mm blood layer; phi_e compared with an independent finite-element solve of the same problem | Reported (proposed: rel RMS <= 1 %, corr >= 0.9999) | Continues |
| bathAnalysis | From ladder Vm (no openCARP): bath electrograms re-analysed by the pipeline, bath-aware ground truth; bath 5 / 10 / 20 mm, bath mesh 0.5 / 0.25 mm, insulating / conducting scar, bath vs unbounded | Convergence rows judged by DL-05; effect rows reported | Continues |
| report | Copies every summary and the first and last 40 lines of every log, with credentials removed | - | - |

## 5. Send back

Copy `3D_Code/runs/readiness/report/` into `Masters_V2/_verification/readiness/` on the Mac. It holds
summaries and log excerpts only (no igb files), with any credentials removed. Keep the full
`3D_Code/runs/readiness/` folder on the workstation until the results have been reviewed.

## 6. What the kit decides, and what it does not

The kit only applies rules fixed in advance. It does not choose the production mesh or time step,
and it does not change any tolerance. Those decisions are taken afterwards, from the reports:

- the production spacing and time step (coarsest level whose comparison with the next finer level
  is within DL-2026-10-28-05);
- whether the tier B (0.125 mm) run is needed (if 0.5 vs 0.25 mm fails);
- whether the 40 x 60 mm domain is large enough (domain comparison);
- whether 4 tissue beats suffice (4-vs-8-beat comparison);
- whether 0.5 mm is adequate for a 1.5 mm wall (thin-wall comparison);
- the bath thickness for the reference model (bath-size comparisons), and whether the bath model needs
  the finer mesh (bath-mesh comparison);
- whether the fibre sweeps can rely on openCARP's fibre handling (fibre check).

Rules applied automatically, all recorded in the outputs:

- **Stimulus capture:** if a strip run does not capture, the stimulus is doubled (at most twice
  per tuning). The capturing strength is used for all later runs. Capture is part of the protocol,
  not a fitted parameter.
- **Measurement window:** derived per ladder run from that run's intact activation under the grid,
  before any electrogram is measured (PROPOSED DL-2026-10-28-08).
- **Physiology failures:** if a ladder run fails its physiology checks, the kit re-analyses it in
  diagnostic mode so the failure can be inspected. Its comparisons carry the physiology flags.

## 7. Known limitations of this kit

- openCARP's source repository and full parameter manual could not be read from the development
  environment. Parameter names not already used by the pilot are taken from the openCARP parameter
  list and the official examples (named in each module). The smoke stage exists to catch any
  mismatch with the installed build within minutes.
- The bath analysis assumes pseudo-bidomain (phi_e does not feed back on Vm). A full bidomain
  bath would differ slightly; that comparison is not part of this kit.
- Bath only on the catheter side; no epicardial medium (no sourced conductivity); the conducting
  scar uses the healthy extracellular tensor (an assumption).
- The bath check relies on openCARP numbering the intracellular submesh in global-node order
  (tissue nodes are written first). The check verifies this from the activation map rather than
  assuming it.
- Tests use test doubles for mesher, bench and openCARP (`tests/fakes`). They verify the kit's logic
  and file handling, not physics. The physics checks are the stages themselves.
