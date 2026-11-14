#!/bin/bash
# Launch one agentic RL run on a single node.
#
#   bash scripts/train.sh <task> <algo> [Hydra overrides...]
#
#   task  alfworld | hotpotqa | webshop | gsm8k_tool
#   algo  examples/<task>/run_<algo>.sh: grpo, gigpo, rloo, reinforce, gspo (critic-free: TASKS prompts x N_ROLLOUTS
#         rollouts per step) or ppo, steppo (with a critic: TASKS x 1). gsm8k_tool takes gae or grpo.
#
# Run from the repository root inside the training environment (README: Setup). Trailing arguments are Hydra
# overrides and win over everything set here. Under Slurm, submit examples/slurm/train.sh, which calls this script.
#
# Environment variables (defaults reproduce the README experiments):
#   MODEL_PATH        local Hugging Face model dir       default models/Qwen3-4B-Instruct-2507
#   DATA_ROOT         dataset root (README: Data)         default data
#   NGPUS             GPUs on this node                   default: CUDA_VISIBLE_DEVICES, Slurm, nvidia-smi, else 2
#   EXP_NAME          run name, outputs in outputs/$EXP_NAME   default <task>_<algo>_<timestamp>
#   STEPS             training steps                      default 200 (gsm8k_tool 10)
#   TEST_FREQ         validation interval in steps        default 10 (hotpotqa 20, gsm8k_tool STEPS/2)
#   TASKS             prompts per step                    default 16 (128 for ppo / steppo)
#   N_ROLLOUTS        rollouts per prompt                 default 8 (1 for ppo / steppo)
#   VAL_SAMPLES       cap on validation items             default: all (alfworld), 1000 (hotpotqa), 100 (webshop),
#                                                         128 (gsm8k_tool)
#   AGENT_WORKERS     rollout agent-flow workers          default 8 for alfworld, recipe default otherwise
#   SEED              data shuffling seed                 default 42
#   SAVE_FREQ         checkpoint interval, -1 = never     default -1
#   WANDB_MODE        W&B mode                            default offline (runs land in outputs/$EXP_NAME/wandb)
#   WANDB_PROJECT / WANDB_ENTITY / WANDB_TAGS             default stepcredit-rl / unset / <task>,<algo>
#   CUDA_COMPAT_PATH  CUDA forward-compat libs, prepended to LD_LIBRARY_PATH (for drivers older than CUDA 12.8)
#   DRY_RUN=1         print the composed Hydra config and exit; no GPU needed, missing data only warns
# Task-specific paths (HOTPOTQA_*, WEBSHOP_*, JAVA_HOME, ...) are listed in the per-task blocks below.
#
# Verl resumes from checkpoints/<project>/<experiment> when one exists (trainer.resume_mode=auto), so give every new
# run its own EXP_NAME.

set -euo pipefail

TASK="${1:?usage: scripts/train.sh <task> <algo> [Hydra overrides...]}"
ALGO="${2:?usage: scripts/train.sh <task> <algo> [Hydra overrides...]}"
shift 2

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
DRY_RUN="${DRY_RUN:-0}"
# shellcheck source=scripts/env.sh
source "$PROJECT_DIR/scripts/env.sh"

# === Paths and run identity ===
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen3-4B-Instruct-2507}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
EXP_NAME="${EXP_NAME:-${TASK}_${ALGO}_$(date +%m%d_%H%M%S)}"
SEED="${SEED:-42}"
SAVE_FREQ="${SAVE_FREQ:--1}"
if [ "$DRY_RUN" = 1 ]; then
    OUT_DIR="$(mktemp -d)"
else
    OUT_DIR="$PROJECT_DIR/outputs/$EXP_NAME"
    mkdir -p "$OUT_DIR" "$RAY_TMPDIR"
fi
# The example launchers tee their output into $LOG_FILE
export LOG_DIR="$OUT_DIR" LOG_FILE="$OUT_DIR/train.log"
require_path "$MODEL_PATH/config.json" "model"

# W&B: verl passes trainer.project_name to wandb.init and reads the entity and tags from the environment
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$OUT_DIR"
WANDB_PROJECT="${WANDB_PROJECT:-stepcredit-rl}"
export WANDB_TAGS="${WANDB_TAGS:-$TASK,$ALGO}"

case "$ALGO" in
    ppo | steppo)
        DEFAULT_TASKS=128
        N_ROLLOUTS=1
        ;;
    *)
        DEFAULT_TASKS=16
        N_ROLLOUTS="${N_ROLLOUTS:-8}"
        ;;
esac
TASKS="${TASKS:-$DEFAULT_TASKS}"
# One PPO mini-batch per step unless ppo_mini_batch_size is overridden (e.g. 4 for four updates per step)
BATCH_ARGS=(
    data.train_batch_size="$TASKS"
    actor_rollout_ref.rollout.n="$N_ROLLOUTS"
    actor_rollout_ref.actor.ppo_mini_batch_size="$TASKS"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    critic.ppo_micro_batch_size_per_gpu=4
)

# === Task setup: sets SCRIPT and TASK_ARGS ===
TASK_ARGS=()
case "$TASK" in
    alfworld)
        STEPS="${STEPS:-200}"
        TEST_FREQ="${TEST_FREQ:-10}"
        # ALFWORLD_ROOT: train/valid_seen/valid_unseen parquet files and games/ (unzipped games.zip)
        ALFWORLD_ROOT="${ALFWORLD_ROOT:-$DATA_ROOT/alfworld}"
        require_path "$ALFWORLD_ROOT/games" "ALFWorld game files (unzip games.zip)"
        for split in train valid_seen valid_unseen; do
            require_path "$ALFWORLD_ROOT/$split.parquet" "ALFWorld $split split"
        done
        # Read by examples/alfworld/run_*.sh and by the env wrapper (games/ is resolved under ALFWORLD_DATA_ROOT)
        export ALFWORLD_DATA_ROOT="$ALFWORLD_ROOT"
        export ALFWORLD_MODEL_PATH="$MODEL_PATH"
        export ALFWORLD_TRAIN_PATH="$ALFWORLD_ROOT/train.parquet"
        export ALFWORLD_VAL_SEEN_PATH="$ALFWORLD_ROOT/valid_seen.parquet"
        export ALFWORLD_VAL_UNSEEN_PATH="$ALFWORLD_ROOT/valid_unseen.parquet"
        export ALFWORLD_VAL_DUMP_DIR="$OUT_DIR/validation"
        SCRIPT="examples/alfworld/run_${ALGO}.sh"
        TASK_ARGS=("${BATCH_ARGS[@]}" actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS:-8}")
        if [ -n "${VAL_SAMPLES:-}" ]; then
            TASK_ARGS+=(data.val_max_samples="$VAL_SAMPLES")
        fi
        ;;
    hotpotqa)
        STEPS="${STEPS:-200}"
        TEST_FREQ="${TEST_FREQ:-20}"
        # HOTPOTQA_ROOT: train/validation parquet files; HOTPOTQA_CORPUS_DATA_ROOT: index.bin and hpqa_corpus.jsonl
        HOTPOTQA_ROOT="${HOTPOTQA_ROOT:-$DATA_ROOT/hotpotqa}"
        export HOTPOTQA_CORPUS_DATA_ROOT="${HOTPOTQA_CORPUS_DATA_ROOT:-$DATA_ROOT/hotpotqa_corpus}"
        export HOTPOTQA_EMBEDDING_MODEL="${HOTPOTQA_EMBEDDING_MODEL:-$PROJECT_DIR/models/bge-large-en-v1.5}"
        # Query embedding on CPU, so it does not compete with vLLM for GPU memory. Each agent-flow worker loads its
        # own copy of the index and the embedding model (about 4 GB of RAM each).
        export HOTPOTQA_EMBEDDING_DEVICE="${HOTPOTQA_EMBEDDING_DEVICE:-cpu}"
        export HOTPOTQA_EMBEDDING_PER_WORKER_GPU="${HOTPOTQA_EMBEDDING_PER_WORKER_GPU:-0}"
        export HOTPOTQA_MODEL_PATH="$MODEL_PATH"
        export HOTPOTQA_ENABLE_CROSS_EVAL="${HOTPOTQA_ENABLE_CROSS_EVAL:-0}"
        require_path "$HOTPOTQA_ROOT/train.parquet" "HotpotQA train split"
        require_path "$HOTPOTQA_ROOT/validation.parquet" "HotpotQA validation split"
        require_path "$HOTPOTQA_CORPUS_DATA_ROOT/index.bin" "HotpotQA FAISS index"
        require_path "$HOTPOTQA_CORPUS_DATA_ROOT/hpqa_corpus.jsonl" "HotpotQA passage corpus"
        require_path "$HOTPOTQA_EMBEDDING_MODEL/config.json" "BGE embedding model"
        SCRIPT="examples/hotpotqa/run_${ALGO}.sh"
        TASK_ARGS=(
            data.train_files="$HOTPOTQA_ROOT/train.parquet"
            data.val_files="$HOTPOTQA_ROOT/validation.parquet"
            data.val_max_samples="${VAL_SAMPLES:-1000}"
            "${BATCH_ARGS[@]}"
        )
        ;;
    webshop)
        STEPS="${STEPS:-200}"
        TEST_FREQ="${TEST_FREQ:-10}"
        # WEBSHOP_ROOT: train/test parquet, goals.json, products.sqlite, lucene_index/; the env server needs a JDK 21
        WEBSHOP_ROOT="${WEBSHOP_ROOT:-$DATA_ROOT/webshop_full}"
        for file in train.parquet test.parquet goals.json products.sqlite lucene_index; do
            require_path "$WEBSHOP_ROOT/$file" "WebShop $file"
        done
        if [ -z "${JAVA_HOME:-}" ]; then
            [ "$DRY_RUN" = 1 ] || { echo "ERROR: set JAVA_HOME to a JDK 21 (needed by pyserini)" >&2; exit 1; }
        else
            require_path "$JAVA_HOME/lib/server/libjvm.so" "JDK 21"
            export PATH="$JAVA_HOME/bin:$PATH"
            export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"  # pyjnius loads the JVM from here
        fi
        # Read by recipes/webshop/env/run_env_server.sh (server side) and examples/webshop/run_*.sh (client side)
        export WEBSHOP_DATA_ROOT="$WEBSHOP_ROOT"
        export WEBSHOP_DATASET_MODE=full
        export WEBSHOP_INDEX_DIR="$WEBSHOP_ROOT"
        export WEBSHOP_DATA_DIR="$WEBSHOP_ROOT"
        export WEBSHOP_ENV_HOST=127.0.0.1
        export WEBSHOP_ENV_PORT="${WEBSHOP_ENV_PORT:-$((14000 + ${SLURM_JOB_ID:-$$} % 1000))}"
        export WEBSHOP_ENV_WORKERS="${WEBSHOP_ENV_WORKERS:-4}"
        export WEBSHOP_ENV_BASE_URL="http://127.0.0.1:$WEBSHOP_ENV_PORT"
        SCRIPT="examples/webshop/run_${ALGO}.sh"
        TASK_ARGS=(data.val_max_samples="${VAL_SAMPLES:-100}" "${BATCH_ARGS[@]}")
        ;;
    gsm8k_tool)
        STEPS="${STEPS:-10}"
        TEST_FREQ="${TEST_FREQ:-$((STEPS / 2 > 0 ? STEPS / 2 : 1))}"
        # GSM8K_DIR: train/test parquet from python3 -m recipes.gsm8k.data_preprocess.process_gsm8k_tool
        GSM8K_DIR="${GSM8K_DIR:-$DATA_ROOT/gsm8k_tool}"
        require_path "$GSM8K_DIR/train.parquet" "GSM8K tool train split"
        require_path "$GSM8K_DIR/test.parquet" "GSM8K tool test split"
        SCRIPT="examples/gsm8k/run_steppo_tool.sh"
        # Both variants keep 32 trajectories per step in two PPO mini-batches
        case "$ALGO" in
            gae) TASK_ARGS=(algorithm.adv_estimator=gae data.train_batch_size=32 actor_rollout_ref.rollout.n=1
                actor_rollout_ref.actor.ppo_mini_batch_size=16) ;;
            grpo) TASK_ARGS=(algorithm.adv_estimator=grpo critic.enable=False data.train_batch_size=8
                actor_rollout_ref.rollout.n=4 actor_rollout_ref.actor.ppo_mini_batch_size=4) ;;
            *)
                echo "ERROR: gsm8k_tool takes gae or grpo, got '$ALGO'" >&2
                exit 1
                ;;
        esac
        TASK_ARGS+=(
            data.train_files="$GSM8K_DIR/train.parquet"
            data.val_files="$GSM8K_DIR/test.parquet"
            data.val_max_samples="${VAL_SAMPLES:-128}"
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
            critic.ppo_micro_batch_size_per_gpu=4
        )
        ;;
    *)
        echo "ERROR: unknown task '$TASK' (alfworld, hotpotqa, webshop or gsm8k_tool)" >&2
        exit 1
        ;;
esac
require_path "$PROJECT_DIR/$SCRIPT" "launch script for algo '$ALGO'"

# Overrides shared by every task, placed after the task's own so they win; trailing "$@" wins over both
COMMON_ARGS=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    critic.model.path="$MODEL_PATH"
    data.seed="$SEED"
    trainer.n_gpus_per_node="$NGPUS"
    trainer.nnodes=1
    trainer.total_training_steps="$STEPS"
    trainer.test_freq="$TEST_FREQ"
    trainer.val_before_train=True
    trainer.save_freq="$SAVE_FREQ"
    'trainer.logger=["console","wandb"]'
    trainer.project_name="$WANDB_PROJECT"
    trainer.experiment_name="$EXP_NAME"
    trainer.log_val_generations=8
    trainer.rollout_data_dir="$OUT_DIR/rollout"
    trainer.validation_data_dir="$OUT_DIR/validation"
    actor_rollout_ref.rollout.trace.backend=null
    # verl 0.7.0 sets vLLM's max_model_len to max_position_embeddings (262144 for Qwen3-4B-Instruct-2507), whose KV
    # cache does not fit the recipes' 0.5 x 80 GB; vLLM frees this memory while the actor trains
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    hydra.run.dir="$OUT_DIR/hydra"
)

echo "[$(date +%H:%M:%S)] $TASK / $ALGO$([ "$DRY_RUN" = 1 ] && echo ' (dry run)')"
echo "  exp_name = $EXP_NAME"
echo "  model    = $MODEL_PATH"
echo "  data     = $DATA_ROOT ($TASKS prompts x $N_ROLLOUTS rollouts per step)"
echo "  gpus     = $NGPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  steps    = $STEPS (validation every $TEST_FREQ), seed $SEED"
echo "  outputs  = $OUT_DIR"
echo "  extra    = $*"

if [ "$DRY_RUN" = 1 ]; then
    exec bash "$SCRIPT" "${TASK_ARGS[@]}" "${COMMON_ARGS[@]}" "$@" --cfg job
fi

# WebShop: the env server runs inside this job on 127.0.0.1. Start it, wait until it answers, and stop it on exit.
# The first request each gunicorn worker serves loads the product DB and the Lucene index, so a few extra /health
# calls warm up the other workers.
if [ "$TASK" = webshop ]; then
    env_log="$OUT_DIR/webshop_env.log"
    echo "[$(date +%H:%M:%S)] starting WebShop env server at $WEBSHOP_ENV_BASE_URL ($WEBSHOP_ENV_WORKERS workers)"
    bash recipes/webshop/env/run_env_server.sh > "$env_log" 2>&1 &
    ENV_SERVER_PID=$!
    trap 'kill "$ENV_SERVER_PID" 2> /dev/null || true' EXIT
    ready=0
    for i in $(seq 1 120); do
        if ! kill -0 "$ENV_SERVER_PID" 2> /dev/null; then
            echo "ERROR: WebShop env server exited early; last lines of $env_log:" >&2
            tail -n 30 "$env_log" >&2
            exit 1
        fi
        if curl -sf --noproxy '*' -m 300 "$WEBSHOP_ENV_BASE_URL/health" > "$OUT_DIR/webshop_env_health.json"; then
            echo "  env server ready after ~$((i * 5))s"
            for _ in $(seq 1 $((WEBSHOP_ENV_WORKERS * 2))); do
                curl -sf --noproxy '*' -m 300 "$WEBSHOP_ENV_BASE_URL/health" > /dev/null || true
            done
            ready=1
            break
        fi
        sleep 5
    done
    if [ "$ready" != 1 ]; then
        echo "ERROR: WebShop env server not healthy after 10 minutes; last lines of $env_log:" >&2
        tail -n 30 "$env_log" >&2
        exit 1
    fi
fi

bash "$SCRIPT" "${TASK_ARGS[@]}" "${COMMON_ARGS[@]}" "$@"
