"""Command line for the final model.

    python -m efov plan   [--config C]                       what will run, mesh size, unconfirmed values
    python -m efov dry    [--config C] --out OUT             print the openCARP commands (carputils --dry)
    python -m efov run    [--config C] --out OUT --np 24     run every stage (resumes; refuses stale outputs)
    python -m efov run    ... --until measurements           stop after a stage
    python -m efov status --out OUT
    python -m efov import-production-model productionModel.json --write config/final_model.json
"""
import argparse
import sys

from . import config as cfg
from . import pipeline, simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "dry", "run", "status", "import-production-model"])
    parser.add_argument("source", nargs="?", help="productionModel.json (import-production-model only)")
    parser.add_argument("--config", default=str(cfg.DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(cfg.REPOSITORY / "runs" / "v3_reference"))
    parser.add_argument("--np", type=int, default=1, help="MPI processes for openCARP")
    parser.add_argument("--until", choices=pipeline.STAGES)
    parser.add_argument("--write", help="import-production-model: config file to write")
    args = parser.parse_args()

    if args.command == "import-production-model":
        if not args.source or not args.write:
            parser.error("give productionModel.json and --write <config>")
        cfg.importProductionModel(args.source, args.config, args.write)
        print(f"Wrote {args.write}")
        return 0
    config = cfg.loadConfig(args.config)
    if args.command == "plan":
        size = cfg.slabSizeMm(config)
        print(f"{config['modelName']}: slab {size[0]:g} x {size[1]:g} x {size[2]:g} mm at {config['tissue']['spacingMm']} mm, "
              f"bath {config['bath']['thicknessMm']:g} mm all round -> {cfg.meshNodeCount(config):,} nodes")
        print(f"{config['pacing']['beatCount']} beats at {config['pacing']['cycleLengthMs']:g} ms; dt {config['numerics']['timeStepUs']:g} us; "
              f"output {config['numerics']['sampleStepMs']:g} ms; prepacing restart {cfg.restartTimeMs(config)} ms")
        print(f"{len(cfg.centreYmm(config)) * len(config['electrodes']['gapsMm'])} catheter placements")
        for key, value in cfg.unconfirmed(config).items():
            print(f"UNCONFIRMED {key}: {value['source']}")
        return 0
    if args.command == "status":
        for name, state in pipeline.Run(config, args.out).status().items():
            print(f"{name:14s} {state}")
        return 0
    if args.command == "dry":
        run = pipeline.Run(config, args.out, args.np)
        state = run.runStage("conditioning")["state"]
        for condition in simulation.CONDITIONS:
            print(simulation.runCondition(config, condition, state, run.out / "dry", args.np, dry=True)["dryRun"])
        return 0
    pipeline.Run(config, args.out, args.np).all(until=args.until)
    return 0


if __name__ == "__main__":
    sys.exit(main())
