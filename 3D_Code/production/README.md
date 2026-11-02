# Production model and the final settings run

Goal: one validated production model (`v3-reference`) for the parameter sweeps. Every remaining
setting is chosen by a rule fixed in `3D_Code/configs/finalise.json` before the run.

## What is openCARP and what is ours

| Part | Tool |
| --- | --- |
| Mesh with blood bath | carputils `mesh.Block` + `set_bath` (mesher). Bath elements get tag 0 and zero fibres, which is how openCARP recognises bath |
| Cell model, conditioning | openCARP Courtemanche; `bench` limit cycle |
| Propagation and extracellular field | openCARP pseudo-bidomain (`bidomain = 2`); full bidomain (`bidomain = 1`) in the check |
| Run orchestration | carputils experiment (`slab.py`: standard parser, `@tools.carpexample`, `tools.carp_cmd`, `job.carp`) |
| Prepacing | openCARP checkpoint and restore (`tsav` / `write_statef`, `start_statef`): beats before the analysis window run monodomain without the bath solve; the bath is switched on from 1050 ms |
| Elliptic solver | carputils flavour `boomeramg` (hypre), falls back to the default if unavailable |
| phi_e output | openCARP `dataout_e = 3`: only the bath layer within 3 mm of the tissue, where electrodes are |
| Activation times | openCARP LATs |
| CV tuning | our tetrahedral strip with the tuneCV update rule (openCARP's tuneCV uses a hexahedral cable; DL-10) |
| Electrogram measures, change points | our pipeline (`run_boundary.py --process-only`), unchanged |
| Ground truth | `Shared_Pipeline/contribution_field.py`: exact per-element share of each recording (reciprocity) |
| Independent check of openCARP's extracellular field | `Shared_Pipeline/fem_extracellular.py` |
| Electrograms when the sigma_i source is chosen | `pipeline_run.femSignals`: our extracellular solve, one solve per unique electrode (reciprocity, VL-2026-11-02-01), in `--np` worker processes |

## Fixed before the run (2 Nov 2026)

- **Forward model:** openCARP pseudo-bidomain; every electrogram is openCARP's phi_e (your decision,
  DL-2026-11-02-01; it overrides the 1 Nov rule outcome, which rested on an unconfirmed tolerance).
- **Bath:** 5 mm on every side, no-flux walls, 1 mm^3 grounded corner, as Leenknegt et al. 2023
  (assumption, DL-2026-11-02-02).

Background (1 Nov, openCARP 19.0.0): bath elements must have zero fibres for openCARP to apply
g_bath (mesher does this); pseudo-bidomain phi_e equals our extracellular solve with source sigma_m
exactly, which every run checks (gate).

## Run it (workstation, inside efov-runner)

```bash
cd <repository root>                              # pull the commit with this file first
export MPIEXEC_BIN=/usr/local/lib/opencarp/lib/petsc/bin/mpiexec
python3 3D_Code/production/finalise.py plan --np 24
nohup python3 3D_Code/production/finalise.py run --np 24 > 3D_Code/runs/final_v2_console.log 2>&1 &
python3 3D_Code/production/finalise.py status     # any time
```

Expected cost (24 cores, estimated from the 1 Nov workstation timings): about 7-8 h and about 60 GB.
run025 (2.65 M nodes) is most of both: openCARP about 2.5-3 h per condition (the 3.4 M-node run took
3.0 and 3.6 h), then the forward-model gate, pipeline and contribution field. Optimisations, each with
results identical to the unoptimised run: prepacing (bath solved only from 1050 ms, VL-2026-11-01-04),
the boomeramg solver flavour and phi_e written only on the 3 mm electrode layer (VL-2026-11-01-05).

Re-running the same command resumes: finished stages and simulations are reused. If a stage
fails, `3D_Code/runs/final_v2/<stage>/stage.json` has the error; an interrupted openCARP run must
be moved aside (the script says which).

carputils needs a settings file. If `CARPUTILS_SETTINGS` is not set, the run writes
`3D_Code/runs/final_v2/carputils_settings.yaml` with `MPIEXEC = $MPIEXEC_BIN`.

## What each stage contributes to the final model

| Stage | Simulation | Final-model parameter it decides |
| --- | --- | --- |
| preflight | none | none: records versions so the result is reproducible |
| recompute | none (existing readiness runs) | output step, time step, beats, domain |
| tune | strips, minutes | conductivities at 0.5 and 0.25 mm |
| run050 | 0.5 mm slab, intact + lesion | none directly; gate: openCARP phi_e = the verified solve |
| run025 | 0.25 mm slab, intact + lesion | none directly: the finer level of the mesh decision |
| decide | none | mesh spacing; writes productionModel.json |

Send back `3D_Code/runs/final_v2/report.md`, `productionModel.json` and every `*/stage.json`
(a few MB). Raw output stays on the workstation.

## Production sweeps afterwards

`slab.py --spec <spec>.json --np 24` is the production simulation. The spec holds every
scientific value; `finalise.py` writes the frozen one into `productionModel.json`.
