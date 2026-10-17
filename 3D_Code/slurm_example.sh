#!/usr/bin/env bash
# Example SLURM job for the full 3D pipeline; adjust partition, modules and paths for your cluster.
# Submit with: sbatch 3D_Code/slurm_example.sh   (resubmitting resumes from the last finished stage)
#SBATCH --job-name=efov3d
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=efov3d_%j.out
set -euo pipefail
# module load opencarp            # or activate the environment that provides openCARP + carputils
# source ~/venvs/opencarp/bin/activate
REPO="${REPO:-$HOME/Masters_V2}"
NP="${SLURM_NTASKS:-16}" PYTHON="${PYTHON:-python3}" bash "$REPO/3D_Code/efov_3d.sh"
