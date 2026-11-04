# Software environment and provenance

Record of the software that produced each set of results. Update when anything changes.

## Simulation workstation (Dell 7960, Docker image `efov-runner`)

| Component | Version | Notes |
| --- | --- | --- |
| openCARP | git `9accae0522f4108774ed8eb1df90b8ef0afd4c8e` | Built from tag `latest`, which moves; the hash is the pin. |
| carputils | 0.0.0 (installed from source) | Commit hash still to record (`direct_url.json` in the dist-info folder). |
| Python (container) | 3.13 | From the carputils install path. |

Used for: `boundaryPilotV2 2` simulations (19-20 Oct 2026) and the original 0-400 ms analysis.

## Analysis machine (Linux VM on the Mac, used 27 Oct 2026)

| Component | Version |
| --- | --- |
| Python | 3.10.12 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scipy | not installable; replaced by validated numpy stand-ins (`_refinement_work/shim`, see decision DL-2026-10-27-04) |

Used for: windowed reprocessing, noise sweep, regression tests (`_refinement_work/run_*`).

## Bayesian analysis (Mac)

| Component | Version |
| --- | --- |
| R | 4.4.1 |
| mcp | 0.3.4 |
| ggdist | 3.3.3 |
| JAGS | to record (`jags` banner or `rjags::jags.version()`) |

Full session information: `_refinement_work/bayes_*/methodAndSession.txt`, `R_results/sessionInfo.txt`.

## Final-model verification environment (cloud container, 3 Nov 2026)

| Component | Version |
| --- | --- |
| openCARP | 19.0.0 (conda-forge `opencarp-19.0.0-h79d7346_0`) |
| carputils | 19.0.0 (conda-forge) |
| Python | 3.12 |
| numpy / scipy / pandas / pyamg | 2.5.4 / 1.18.1 / 3.1.0 / 5.4.1 |

Used for: the `finalModel` unit tests and the real-openCARP end-to-end run of `config/test_small.json`
(VL-2026-11-03-01). Not the workstation build (`9accae05`); the forward-model check in every run
verifies the configuration on whichever build runs it.
