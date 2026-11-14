#!/bin/bash
# Expert SFT followed by agentic RL on ALFWorld, end to end:
#   1. recipes.alfworld.data_preprocess.generate_expert_sft plays the training games with ALFWorld's hand-coded
#      (PDDL) expert and writes one SFT row per step. Each row's prompt is built exactly like the RL rollout's, and
#      generation fails unless a sample of 200 rows tokenizes identically to the rollout prompt.
#   2. verl's fsdp_sft_trainer trains on the expert tool calls (loss on the assistant turn only) and saves a
#      Hugging Face checkpoint.
#   3. scripts/train.sh runs RL from that checkpoint; its initial validation scores the SFT model.
#
#   bash scripts/sft_rl_alfworld.sh <algo> [RL Hydra overrides...]
#
# Environment variables (plus those of scripts/train.sh; outputs go to outputs/$EXP_NAME, SFT under its sft/):
#   SFT_EPOCHS        default 1
#   SFT_LR            default 1e-5
#   SFT_BATCH         global batch size in rows, default 64
#   SFT_MAX_LENGTH    tokens per row, default 2048 (RL prompts reach about 1,350)
#   SFT_DATA_WORKERS  processes playing expert episodes, default 16
#   STEPS / TEST_FREQ RL steps and validation interval, default 100 / 10
#   DRY_RUN=1         print the SFT config and the RL config, no data generation and no training

set -euo pipefail

ALGO="${1:-gigpo}"
if [ $# -gt 0 ]; then shift; fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
DRY_RUN="${DRY_RUN:-0}"
# shellcheck source=scripts/env.sh
source "$PROJECT_DIR/scripts/env.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen3-4B-Instruct-2507}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
ALFWORLD_ROOT="${ALFWORLD_ROOT:-$DATA_ROOT/alfworld}"
export EXP_NAME="${EXP_NAME:-alfworld_sft_${ALGO}_$(date +%m%d_%H%M%S)}"
export STEPS="${STEPS:-100}"
export TEST_FREQ="${TEST_FREQ:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-stepcredit-rl}"
export WANDB_MODE="${WANDB_MODE:-offline}"

SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_LR="${SFT_LR:-1e-5}"
SFT_BATCH="${SFT_BATCH:-64}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-2048}"
SFT_DATA_WORKERS="${SFT_DATA_WORKERS:-16}"
if [ "$DRY_RUN" = 1 ]; then
    SFT_DIR="$(mktemp -d)/sft"
else
    SFT_DIR="$PROJECT_DIR/outputs/$EXP_NAME/sft"
    mkdir -p "$SFT_DIR" "$RAY_TMPDIR"
fi
export WANDB_DIR="$SFT_DIR"
require_path "$MODEL_PATH/config.json" "model"
require_path "$ALFWORLD_ROOT/train.parquet" "ALFWorld train split"
require_path "$ALFWORLD_ROOT/games" "ALFWorld game files"

SFT_ARGS=(
    data.train_files="$SFT_DIR/data/train.parquet"
    data.val_files="$SFT_DIR/data/val.parquet"
    data.multiturn.enable=true
    data.multiturn.messages_key=messages
    data.max_length="$SFT_MAX_LENGTH"
    data.truncation=error
    data.train_batch_size="$SFT_BATCH"
    data.micro_batch_size_per_gpu=4
    model.partial_pretrain="$MODEL_PATH"
    model.enable_gradient_checkpointing=True
    use_remove_padding=True
    optim.lr="$SFT_LR"
    trainer.default_local_dir="$SFT_DIR/ckpt"
    trainer.project_name="$WANDB_PROJECT"
    trainer.experiment_name="${EXP_NAME}_sft"
    trainer.total_epochs="$SFT_EPOCHS"
    trainer.n_gpus_per_node="$NGPUS"
    trainer.nnodes=1
    trainer.resume_mode=disable
    'trainer.logger=["console","wandb"]'
    'trainer.checkpoint.save_contents=["hf_model"]'
    hydra.run.dir="$SFT_DIR/hydra"
)

echo "[$(date +%H:%M:%S)] ALFWorld expert SFT, then $ALGO RL$([ "$DRY_RUN" = 1 ] && echo ' (dry run)')"
echo "  exp_name = $EXP_NAME"
echo "  base     = $MODEL_PATH"
echo "  sft      = lr $SFT_LR, batch $SFT_BATCH, $SFT_EPOCHS epoch(s), outputs in $SFT_DIR"

if [ "$DRY_RUN" = 1 ]; then
    python3 -m verl.trainer.fsdp_sft_trainer "${SFT_ARGS[@]}" --cfg job
    MODEL_PATH="$MODEL_PATH" bash "$PROJECT_DIR/scripts/train.sh" alfworld "$ALGO" "$@"
    exit 0
fi

python3 -m recipes.alfworld.data_preprocess.generate_expert_sft \
    --train_parquet "$ALFWORLD_ROOT/train.parquet" \
    --data_root "$ALFWORLD_ROOT" \
    --output_dir "$SFT_DIR/data" \
    --tokenizer_path "$MODEL_PATH" \
    --num_workers "$SFT_DATA_WORKERS"

torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" -m verl.trainer.fsdp_sft_trainer "${SFT_ARGS[@]}"

SFT_MODEL="$(find "$SFT_DIR/ckpt" -maxdepth 2 -type d -name huggingface | sort -V | tail -n 1)"
if [ -z "$SFT_MODEL" ] || [ ! -f "$SFT_MODEL/config.json" ]; then
    echo "ERROR: no Hugging Face checkpoint under $SFT_DIR/ckpt" >&2
    exit 1
fi
echo "[$(date +%H:%M:%S)] SFT finished, starting RL from $SFT_MODEL"

MODEL_PATH="$SFT_MODEL" bash "$PROJECT_DIR/scripts/train.sh" alfworld "$ALGO" "$@"
