# Methodology verification harness

Produces the numbers the thesis Methodology cites for each modelling and numerical choice. It reads the
production configuration (`config/model.json`, `config/model.par`) and the production code (`efov/`) and
never edits either: a check that needs a different mesh spacing, time step or protocol writes a variant
configuration into its own output folder (`<out>/work/<check>/<variant>/config/`, with `changes.json`
listing exactly what differs). Results are JSON, one file per check, each with the statement it supports,
the method, the values, the git revision and the software versions. `runs/` is not tracked by git.

| Part | Command | Needs | Run time |
| --- | --- | --- | --- |
| 1 analysis chain | `python3 verification/local_checks.py --out runs/verification/local` | numpy, scipy | under a minute |
| 2 monodomain solve | `python3 verification/opencarp_checks.py --out runs/verification/opencarp --np 2` | openCARP >= 19, carputils | hours on 2 cores (small strips and 16 x 16 mm slabs) |
| 3 volume conductor | `python3 verification/forward_sensitivity.py --out runs/verification/forward` | numpy, scipy | minutes |
| 4 change-point estimator | `Rscript verification/changepoint_recovery.R runs/verification/changepoint 5` | R, mcp, JAGS | about 15 s per fit |
| summary | `python3 verification/summarise.py runs/verification` | - | writes `verification_numbers.md` |

Part 2 on the workstation, with strong scaling of the parallel solve:

```bash
export MPIEXEC_BIN=/usr/local/lib/opencarp/lib/petsc/bin/mpiexec
python3 verification/opencarp_checks.py --out runs/verification/opencarp --np 24 --np-list 1 2 4 8 16 24
```

`--only NAME ...` runs selected checks. A completed simulation is reused on a rerun.

## What each part establishes, and what it does not

- **Part 1** (numerical properties of the analysis chain): conductivity arithmetic, mesh and catheter
  geometry, electrode resolution, the resampler, the band-pass, the noise level against the absolute
  thresholds, wavelet peak-frequency accuracy and the trailing-signal margin.
- **Part 2** (numerical verification of the monodomain solve as configured): convergence of conduction
  velocity in space and time, stimulus safety factor, limit-cycle conditioning, slab conduction velocity
  per fibre model and wall thickness, serial against parallel results.
- **Part 3** (sensitivity, reduced model): bath thickness, ground position, epicardial medium and slab
  margin, each varied alone, on the recorded signal and on the lateral extent of its contributing tissue.
- **Part 4** (estimator behaviour): bias and interval coverage of Wang's piecewise fit when its model is
  true; what it returns for a decaying signal over a constant floor, and with no plateau at all.

Not covered here, and still required before production results are interpreted:
- openCARP's lead-field module (openCARP >= 20): `checks/container_checks.py` W0 (lead fields against a
  direct solve, electrograms against -Z^T S_i Vm).
- Convergence of electrogram measures, peak frequency and change points on the full slab (plan item W3):
  needs the lead-field module.
- Monodomain against full bidomain: branch `bidomainReference` (`docs/bidomain_reference.md`).
- Any physiological validation. Every number here is a property of the numerical method or of the
  analysis code, not evidence that the model represents human atrial tissue.
