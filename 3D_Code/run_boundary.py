#!/usr/bin/env python3
"""Run the controlled-boundary pilot; see run.ipynb for the step-by-step version."""
import argparse
import subprocess
from pathlib import Path
import sys
from boundary_setup import (
    loadConfig,
    prepareExperiment,
    simulateExperiment,
    checkSimulationSettings,
    findExecutable,
    startAnalysis,
    checkCompletedSimulations,
    conditionCells,
)


def main():
    arguments = readArguments()
    # 1. Load the experimental settings.
    config = loadConfig(arguments.config)
    if arguments.np < 1:
        raise ValueError("MPI process count must be positive")
    if arguments.checkConfig:
        print(
            "Configuration valid. Pilot mesh spacing:",
            config["meshSpacingMm"],
            "mm; convergence unverified.",
        )
        return 0
    runFolder = arguments.output.resolve()
    # 2. Prepare and run, or reuse the saved raw simulations.
    if arguments.processOnly:
        checkSimulationSettings(runFolder, config)
    else:
        if not arguments.prepareOnly and not arguments.conditionOnly:
            findExecutable("openCARP", "OPENCARP_BIN")
            if arguments.np > 1:
                findExecutable("mpiexec", "MPIEXEC_BIN")
        prepareExperiment(runFolder, config)
        if arguments.prepareOnly:
            return 0
        if arguments.conditionOnly:
            conditionCells(runFolder, config)
            return 0
        simulateExperiment(runFolder, config, arguments.np)
        if arguments.simulateOnly:
            return 0
    checkCompletedSimulations(runFolder)
    from boundary_checks import checkExperiment
    from boundary_analysis import analyseExperiment
    from boundary_figures import makeFigures

    # 3. Check the physiological experiment before measuring contributions.
    startAnalysis(runFolder, config)
    physiologyPassed = checkExperiment(runFolder, config)
    if not physiologyPassed and not arguments.diagnostic:
        makeFigures(runFolder, config)
        print(
            "Physiological checks failed. Inspect physiologyChecks.csv and figures/01_activation.png.\n"
            "No accepted EFOV estimates were produced. Use --process-only --diagnostic for diagnostic signal exports."
        )
        return 2
    # 4. Measure signals, estimate contributions, and draw the results.
    _, pilotPassed = analyseExperiment(runFolder, config, physiologyPassed)
    makeFigures(runFolder, config)
    print(f"Outputs saved to {runFolder}; pilot acceptance passed: {pilotPassed}")
    return 0 if pilotPassed else 2


def readArguments():
    """Command-line options; the scientific workflow is in main above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/boundary_pilot.json"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs/boundaryPilotV2")
    parser.add_argument("--np", type=int, default=1, help="MPI processes for openCARP")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--check-config",
        dest="checkConfig",
        action="store_true",
        help="Validate settings without a simulator",
    )
    actions.add_argument(
        "--prepare-only",
        dest="prepareOnly",
        action="store_true",
        help="Write mesh, electrodes and parameter files",
    )
    actions.add_argument(
        "--condition-only", dest="conditionOnly", action="store_true",
        help="Prepare the experiment and pace the single cell, then stop",
    )
    actions.add_argument(
        "--simulate-only",
        dest="simulateOnly",
        action="store_true",
        help="Prepare and simulate, then stop",
    )
    actions.add_argument(
        "--process-only",
        dest="processOnly",
        action="store_true",
        help="Analyse existing complete simulation outputs",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Export diagnostic signals even if physiological checks fail; FOV remains ineligible",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Pilot stopped: {error}", file=sys.stderr)
        sys.exit(1)
