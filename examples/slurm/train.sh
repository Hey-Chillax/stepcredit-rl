#!/bin/bash
#SBATCH --job-name=stepcredit-rl
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm_logs/slurm-%j.out
#
# Slurm wrapper around scripts/train.sh: one agentic RL run on one node. Submit from the repository root (the log
# directory must exist first) with the training environment active, or activate it below.
#
#   mkdir -p outputs/slurm_logs
#   sbatch -p <partition> examples/slurm/train.sh <task> <algo> [Hydra overrides...]
#
# Adjust --gres / --time on the command line, e.g. `sbatch --gres=gpu:4 --time=12:00:00 ...`. NGPUS follows the
# GPUs Slurm assigns. Always set a --time: on many clusters jobs without a limit are never backfilled.
# Every environment variable of scripts/train.sh passes through, e.g.
#
#   EXP_NAME=alfworld_steplevel_200 sbatch -p <partition> examples/slurm/train.sh alfworld gigpo

set -euo pipefail
# Slurm runs a spooled copy of this file, so locate the repository from the submit directory
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f scripts/train.sh ]; then
    echo "ERROR: submit from the repository root (got $(pwd))" >&2
    exit 1
fi
# source .venv/bin/activate   # uncomment if the environment is not inherited from the submitting shell
exec bash scripts/train.sh "$@"
