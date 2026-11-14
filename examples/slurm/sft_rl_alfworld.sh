#!/bin/bash
#SBATCH --job-name=stepcredit-sft-rl
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm_logs/slurm-%j.out
#
# Slurm wrapper around scripts/sft_rl_alfworld.sh: expert data generation, SFT and RL on ALFWorld in a single
# allocation, so the RL stage does not queue again. Submit from the repository root:
#
#   mkdir -p outputs/slurm_logs
#   EXP_NAME=alfworld_sft_rl sbatch -p <partition> examples/slurm/sft_rl_alfworld.sh gigpo [RL Hydra overrides...]
#
# With 2 x A100 80GB the defaults (100 RL steps) take about 17 hours: about 40 minutes of expert data generation,
# about 1.4 hours of SFT and about 15 hours of RL.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f scripts/sft_rl_alfworld.sh ]; then
    echo "ERROR: submit from the repository root (got $(pwd))" >&2
    exit 1
fi
# source .venv/bin/activate   # uncomment if the environment is not inherited from the submitting shell
exec bash scripts/sft_rl_alfworld.sh "$@"
