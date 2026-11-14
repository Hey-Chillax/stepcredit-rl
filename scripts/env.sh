#!/bin/bash
# Runtime environment shared by scripts/train.sh and scripts/sft_rl_alfworld.sh (sourced, not executed).
# Sets NGPUS and CUDA_VISIBLE_DEVICES, offline Hugging Face settings, a run-private Ray temp dir, optional CUDA
# forward-compat libraries, and the require_path helper. Expects DRY_RUN to be set (0 or 1).

# Fail on a missing input, or only warn in a dry run. Usage: require_path <path> <description>
require_path() {
    [ -e "$1" ] && return 0
    if [ "$DRY_RUN" = 1 ]; then
        echo "WARNING (dry run): missing $2: $1" >&2
    else
        echo "ERROR: missing $2: $1" >&2
        exit 1
    fi
}

# The CUDA 12.8 wheels (torch 2.8 / vLLM 0.11) need driver >= 570; older drivers can use the forward-compat libcuda
if [ -n "${CUDA_COMPAT_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$CUDA_COMPAT_PATH:${LD_LIBRARY_PATH:-}"
fi
# Slurm may export ROCR_VISIBLE_DEVICES (the AMD equivalent); verl refuses to start when it is set next to
# CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_USAGE_STATS_ENABLED=0
# Models and data are local paths, so training never needs the Hub
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Run-private Ray temp dir, so ray.init() never attaches to another run's Ray cluster on a shared node
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-$$}}"

if [ -z "${NGPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NGPUS="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
    elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        NGPUS="$SLURM_GPUS_ON_NODE"
    else
        NGPUS="$(nvidia-smi -L 2> /dev/null | wc -l || true)"
        [ "${NGPUS:-0}" -gt 0 ] || NGPUS=2
    fi
fi
export NGPUS
# The example launchers default to GPUs 0-3; pin them to the GPUs this run uses
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPUS - 1)))}"
