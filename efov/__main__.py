"""Command line for the final model.

    python -m efov plan   [--config C]                    mesh size, phi_e box, values and assumptions in force
    python -m efov tune   [--config C] --out OUT --np 24 [--write-par]
                                                          tune beta at the production mesh (FM-18); --write-par
                                                          writes it into config/model.par and config/tuning.json
    python -m efov dry    [--config C] --out OUT          print the carputils/openCARP setup (no simulation)
    python -m efov run    [--config C] --out OUT --np 24  every stage (resumes; refuses stale outputs)
    python -m efov run    ... --until measurements        stop after a stage
    python -m efov status --out OUT
    python -m efov sweep  config/sweeps/oat_baseline.json --out OUT --np 24 [--only NAME ...]
    python -m efov wang-combine RUN_A RUN_B --out OUT      pool the inferior and roof exports for Wang's code
"""
import argparse
import sys

from . import config as cfg
from . import export, pipeline, simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "tune", "dry", "run", "status", "sweep", "wang-combine"])
    parser.add_argument("sources", nargs="*", help="sweep file (sweep) or run folders (wang-combine)")
    parser.add_argument("--config", default=str(cfg.DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(cfg.REPOSITORY / "runs" / "v4_reference"))
    parser.add_argument("--np", type=int, default=1, help="MPI processes for openCARP")
    parser.add_argument("--until", choices=pipeline.STAGES)
    parser.add_argument("--only", nargs="*", help="sweep: run only these variant names")
    parser.add_argument("--write-par", action="store_true", help="tune: write the tuned beta into model.par")
    args = parser.parse_args()

    if args.command == "sweep":
        if len(args.sources) != 1:
            parser.error("give one sweep file")
        from .sweep import runSweep
        status = runSweep(args.sources[0], args.out, args.np, args.only)
        for name, item in status.items():
            print(f"{name:40s} {item['status']}")
        return 0
    if args.command == "wang-combine":
        for path in export.combineExports(args.sources, args.out):
            print(path)
        return 0
    config = cfg.loadConfig(args.config)
    if args.command == "plan":
        size, spacing = cfg.slabSizeMm(config), cfg.spacingMm(config)
        print(f"{config['modelName']}: slab {size[0]:g} x {size[1]:g} x {size[2]:g} mm at {spacing[0]:g} / {spacing[2]:g} mm, "
              f"epicardial medium {config['values']['epicardialMedium']} -> {cfg.meshNodeCount(config):,} nodes")
        print(f"{cfg.beatCount(config)} beats at {cfg.cycleLengthMs(config):g} ms; dt {cfg.timeStepUs(config):g} us; "
              f"output {cfg.sampleStepMs(config):g} ms; monodomain")
        from .geometry import electrodeLayout
        print(f"{len(electrodeLayout(config)['uniqueCentres'])} distinct electrodes per condition: one Laplace lead field each (FM-23)")
        print(f"{len(cfg.centreYmm(config)) * len(config['values']['gapsMm'])} catheter placements; "
              f"ground truth at {len(cfg.groundTruthCentresYmm(config))} centres")
        beta = cfg.tunedBetaFor(config)
        print("beta: " + (f"tuned {beta:.6g} /um for this mesh and target" if beta is not None else "NOT tuned for this mesh: run `python -m efov tune --write-par`"))
        for key, value in config["provenance"].items():
            if value.get("status") == "assumption":
                print(f"ASSUMPTION {key}: {value['source']}")
        return 0
    run = pipeline.Run(config, args.out, args.np)
    if args.command == "status":
        for name, state in run.status().items():
            print(f"{name:14s} {state}")
        return 0
    if args.command == "tune":
        run.runStage("preflight"); run.runStage("conditioning")
        result = run.runStage("tune")
        if args.write_par:
            from .parfile import setValue
            from .tune import BETA_KEY
            setValue(config["modelPar"], BETA_KEY, result["cellSurfVolRatioPerUm"],
                     f"1/um; [tuned] {result.get('cvMPerS', float('nan')):.4f} m/s strip CV at {cfg.spacingMm(config).tolist()} mm (FM-18)")
            cfg.recordTuning(config, dict(result, modelPar=config["modelPar"]))
            print(f"Wrote beta {result['cellSurfVolRatioPerUm']:.6g} /um to {config['modelPar']} and {cfg.TUNING_RECORD}")
        return 0
    if args.command == "dry":
        state = run.runStage("conditioning")["state"]
        for condition in simulation.CONDITIONS:
            print(simulation.runCondition(config, condition, state, run.out / "dry" / "simulations", args.np, dry=True)["dryRun"])
        return 0
    run.all(until=args.until)
    return 0


if __name__ == "__main__":
    sys.exit(main())
