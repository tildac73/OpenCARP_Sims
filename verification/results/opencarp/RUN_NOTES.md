# Run notes: `verification/opencarp_checks.py` on openCARP 19.0.0 (cloud container, 12 Nov 2026)

All six checks completed. Nothing was dropped: every level and case of the harness ran (including the
0.0625 mm in-plane level, the 2000-beat conditioning case and the 3.0 mm wall), except the roof case, which
was removed deliberately (see "Configuration" below). No tolerance or expectation was changed. No file in
`efov/`, `config/` or `tests/` was edited.

## Environment

| Component | Version / build |
| --- | --- |
| openCARP | 19.0.0, conda-forge `opencarp-19.0.0-h79d7346_0`; `openCARP -buildinfo`: tag and hash `c877c38dd2851ae04db17de93be6997ba1c42dc6` (conda-forge/opencarp-feedstock) |
| carputils | 19.0.0, conda-forge `np2py312h1a77e3e_0` |
| PETSc / MPI | petsc 3.25.6 `real_h1bf4665_0`; mpich 5.0.1 `hfbb40bb_4` |
| Python | 3.12.16 `h5f976f7_0_cpython` |
| numpy / scipy / pandas | 2.5.4 `py312he827f4e_0` / 1.18.1 `py312h106c528_2` / 3.1.1 `np2py312h91ec553_0` |
| pytest | 9.2.0 |
| Installer | micromamba 2.10.0, channel conda-forge only |
| Machine | Linux container, 4 cores, 15 GB RAM |

`MPIEXEC_BIN=/opt/mamba/envs/carp/bin/mpiexec` (the MPICH launcher of the same environment), `OMP_NUM_THREADS=1`.
Command: `python3 verification/opencarp_checks.py --out runs/verification/opencarp --np 2 [--only NAME]`
(`--np 2` on the 4-core container; parallel equivalence compared 1 and 2 processes).

Unit tests before the runs (`python3 -m pytest tests -q`, at revision 6b4686c): 48 passed, 2 skipped
(the lead-field tests that need openCARP >= 20), 21 min 32 s.

## Configuration the runs used

- `stimulusThreshold`, `temporalConvergence`, `spatialConvergence` were simulated on the checkout at
  6b4686c (baseline wall 1.1 mm). After the branch head (e0abb58: baseline wall 1.9 mm, FM-48; roof removed,
  FM-49) was merged, the three checks were re-evaluated on the merged checkout. The production runner
  refuses to reuse a simulation whose run spec differs, and it reused all 23 strip simulations, so these
  results are the same under the 1.9 mm configuration (the strip is 20 x 1 x 0.5 mm whatever the wall).
- `slabVelocity`, `parallelEquivalence` and `conditioning` were simulated after the merge, on the 1.9 mm
  baseline. `parallelEquivalence` therefore uses the 1.9 mm bilayer slab (84,500 tissue nodes).
- `efov/` and `config/model.par` are identical before and after the merge.
- The JSON files record revision `6ef5795` (local commit: merge of e0abb58 plus the harness fixes below).
  `git push` was refused in this container (HTTP 403: the Claude GitHub App has no write access), so the
  files were uploaded through the GitHub connector and the local commit hashes do not exist on GitHub.
  The uploaded `verification/common.py` and `verification/opencarp_checks.py` are the exact files that ran.

## Code fixes (all in `verification/`), and why

1. `common.richardsonSpacings` (new) and its use in `convergenceSeries`. `richardson()` assumes a constant
   refinement ratio of 2; the transmural levels 0.25 / 0.1 / 0.05 mm have ratios 2.5 and 2, so the observed
   order and extrapolated value of that series would have been wrong. The new function solves
   f = f* + C h^p for the three given spacings; for ratio 2 it returns the same numbers as `richardson()`.
2. Wall time (item e). `wallSeconds` of a strip run is the whole `runCondition` call: mesher, mesh rewrite,
   carputils and openCARP set-up, and the solve (conditioning is excluded). For the slabs it was
   `time.time()` around `runCondition`, which on a reused run would have reported the near-zero reuse time.
   Now every row carries `wallSeconds` (the recorded timing of the run itself) and `physicsSeconds` (the
   total under "Timings of individual physics" in openCARP's log). Parallel speed-up uses `physicsSeconds`.
   The method text said meshing is included "on the first run"; in fact each process count meshes its own
   copy, and the text now says so. Meshing and set-up dominate the small slabs (34-43 s of 46-77 s).
3. `temporalConvergence` now also reports the observed order in dt and the dt -> 0 extrapolation (an added
   output; the existing columns are unchanged).
4. `convergenceSeries` adds `extrapolationNote` when the observed order lies outside 0.5-3, so a
   meaningless extrapolation is not read as a converged value (it applies to the transmural series).
5. `checkSlabVelocity` cases follow the configured baseline wall: bilayer, uniform and linear rotation at
   1.9 mm; bilayer and uniform at 1.1 mm (where the earlier verification-log values were measured);
   bilayer at 3.0 mm. Removed: `uniform_wall1p6_roof` (FM-49) and `rotation_wall1p1`.
6. Removed an unused import.

No bug was found in the simulation route itself: the harness ran on openCARP at the first attempt.

## Things the harness author could not test

- (a) `act-thresh.dat` with no activation: openCARP 19.0.0 writes the file with zero rows (5 sub-threshold
  runs, 0 bytes each) and exits normally with a full `vm.igb`. The production runner accepts the run and
  `stripBeatTable` returns `captured = False`. The bisection's "no activation file" branch is never taken
  on this build but is harmless.
- (b) `stripBeatTable` / `readIgb`: `vm.igb` is (7501 frames, 2430 nodes) for the 1500 ms strip, 0.2 ms
  step, all nodes at -79.37 mV in frame 0. Activation times computed from Vm per beat agree with
  openCARP's own LATs at every node within 0.018 ms (mean 0.0005 ms) in each of the 3 beats, so the frame
  and node axes and the per-beat slicing are right. `stripBeatTable` CV of the last beat 0.90219 m/s
  against `efov.tune.stripCv` 0.90150 m/s (0.08 % apart: Vm sampled at 0.2 ms against openCARP's LATs).
- (c) `firstActivationMs` indexing, on the 1.9 mm bilayer slab with bath (771,750 mesh nodes, 84,500 tissue
  nodes, tissue node numbers 2120-218379, not contiguous): LAT node numbers run 0-84,499, i.e. tissue-node
  order. With that reading Vm-derived and LAT activation agree within 0.019 ms at all 84,500 nodes; reading
  them as global mesh node numbers fails (51,680 tissue nodes without a time, differences up to 20 ms).
- (d) `surfaceWave` pairing: 1089 nodes on each surface of the central 8 x 8 mm; after sorting, the (x, y)
  of each top/bottom pair are identical (largest mismatch 0.0 mm).
- (e) Wall time: see fix 2.

## Results against the earlier anchors (`docs/verification_log.md`)

| Quantity | Earlier | This run |
| --- | --- | --- |
| Production-mesh strip CV, beta 0.027292 /um (VL-2026-11-04-02) | 0.9015 m/s | 0.90150 m/s |
| Bilayer small slab, first beat, wall 1.1 mm (VL-2026-11-04-05) | 0.735 m/s | 0.7334 m/s |
| Uniform small slab, first beat, wall 1.1 mm | 0.876 m/s | 0.8745 m/s |
| Across / along speed ratio (VL-2026-11-04-01, C4p) | 0.595 | 0.588 (strips, 0.25 mm) |

The ratio differs by 1.2 % from C4p; C4p was a different measurement (small slab), so they are not expected
to be identical. No material disagreement.

## Wall times (2 processes unless stated)

Sum of the tissue runs per check (whole-run wall time): stimulusThreshold 8 runs 152 s; temporalConvergence
4 runs 136 s; spatialConvergence 11 runs 940 s (the two 0.0625 mm strips 335 s and 319 s);
conditioning 7 runs 202 s; slabVelocity 6 runs 349 s; parallelEquivalence 2 runs 126 s. Each strip variant
also runs its own single-cell conditioning with `bench` (about 2 minutes per 1000 beats, serial); that is
most of the elapsed time of the strip checks and is not in any `wallSeconds`. The unit tests were running
on a third core during stimulusThreshold, temporalConvergence and the first spatialConvergence pass, so
those wall times may be slightly inflated. Timings are of small problems and say nothing about the full slab.

## Points that need a decision or a careful sentence in the Methodology

1. **The production mesh is not converged in space; beta compensates.** Along fibres the strip CV is
   0.836 / 0.9015 / 0.9244 / 0.9302 m/s at 0.5 / 0.25 / 0.125 / 0.0625 mm, observed order 2.00, extrapolated
   0.9321 m/s: the 0.25 mm mesh is 3.3 % slow. Across fibres 0.444 / 0.530 / 0.563 / 0.574 m/s, order 1.68,
   extrapolated 0.578 m/s: the 0.25 mm mesh is 8.3 % slow. Because beta is tuned for 0.90 m/s along fibres at
   this mesh, the along-fibre speed is right by construction, but the anisotropy ratio is mesh dependent:
   0.588 at the production mesh against 0.620 extrapolated (0.627 from the conductivities,
   sqrt of the ratio of the harmonic-mean conductivities). Transverse conduction in the production model is
   about 5 % slower, relative to longitudinal, than the configured conductivities imply.
2. **The time step is first order and 20 us is 1.3 % slow.** CV 0.8901 / 0.9015 / 0.9074 / 0.9104 m/s at
   40 / 20 / 10 / 5 us; observed order 0.97 (not 2, although the parabolic solve is Crank-Nicolson), dt -> 0
   value 0.9135 m/s. APD90 changes by 1.4 ms between 20 and 5 us. Again absorbed by the beta tuning for the
   along-fibre speed.
3. **The transmural series does not converge cleanly.** CV 0.8988 / 0.9015 / 0.9034 m/s at 0.25 / 0.1 /
   0.05 mm: a 0.5 % spread, but observed order 0.10, so the extrapolated value in the JSON (0.930 m/s) is
   not a converged estimate and must not be quoted (the JSON carries a note). What can be said: halving the
   transmural spacing from 0.1 to 0.05 mm changes the strip CV by 0.21 %. This series is a plane wave
   along fibres in a 0.5 mm strip, so it does not test transmural conduction across the bilayer transition.
4. **No single-cell limit cycle after 1000 or 2000 beats.** Resting potential and APD90 of the last tissue
   beat are not monotonic in the number of conditioning beats (resting potential -78.81, -78.74, -78.80,
   -79.00, -79.29 mV after 20, 100, 300, 1000, 2000 beats; APD90 299.8, 301.4, 301.2, 300.5, 299.4 ms), so
   the 2000-beat case is a longer conditioning, not a converged reference. The differences are small:
   between 20 and 2000 beats APD90 stays within 2.0 ms, resting potential within 0.55 mV and CV within
   0.22 %. One beat of conditioning gives APD90 6.3 ms shorter than 2000 beats.
5. **Tissue beats still drift.** After the configured 1000 beats, over 8 tissue beats APD90 goes 308.4,
   301.7, 300.9, 300.5, 300.0, 299.6, 299.3, 298.9 ms (still -0.3 to -0.4 ms per beat at beat 8), resting
   potential -79.37 -> -78.86 -> -79.09 mV, CV within 0.10 % throughout. The analysed 4th beat is not a
   steady state in APD90 at the 0.4 ms-per-beat level; CV is steady.
6. **Uniform slab 3 % slower than the strip.** With uniform fibres along the wave the 16 x 16 mm slab gives
   0.8745 m/s (both walls) against 0.9015 m/s on the tuning strip at the same mesh spacing and beta. Not
   investigated here; the earlier log has the same value (0.876), so it is reproducible. The 0.90 m/s
   target therefore holds on the strip, not on the slab surface.
7. Stimulus: the threshold bracket (23.83-24.22 uA/cm^2) is from a 7-step bisection in the strip with the
   0.5 mm band; capture was all-or-none (sub-threshold runs produced no -20 mV crossing at any node).

None of these looked like a defect in `efov/` or `config/`.
