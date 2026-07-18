#!/bin/bash
# ============================================================
# MultihopSpatial - GRPO training for Qwen3-VL
# ============================================================
# Reproduces the recipe behind the etri-vilab/MultiHopSpatial-Qwen3-VL-*
# checkpoints. The dataset and base model are downloaded automatically on the
# first run, so this is all you need:
#
#     bash train_grpo.sh                # 4B (default)
#     bash train_grpo.sh --model 8b
#     bash train_grpo.sh --model 32b
#
# Usage:
#     bash train_grpo.sh --epochs 10 --lr 5e-5
#     bash train_grpo.sh --gpus 0,1,2,3 --wandb
#     bash train_grpo.sh --data_dir /path/to/dataset/cache
#
# All three sizes use the same recipe; only the base model and the per-device
# batch size differ (32B needs a smaller one, made up for with accumulation).
#
# Reward: format + alpha * mcq + beta * bbox + gamma * truncation
# ============================================================

set -eo pipefail

export PYTHONPATH=src:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

# ---- Where things get downloaded ----
#
# Dataset (~1 GB: 6,791 samples + 6,493 images)
#   Pulled from the HF Hub on the first run and written under $DATA_DIR:
#       $DATA_DIR/data/multihop_train_6791.json      raw split, as published
#       $DATA_DIR/data/images/                       images
#       $DATA_DIR/multihop_train_6791_grpo.json      converted training file
#   Re-runs reuse whatever is already there, so this happens exactly once.
#   Point --data_dir at shared storage to share it across users or machines.
#   The Hub rate-limits downloads this size; that is handled by waiting out the
#   window and resuming, so a pause partway through is expected, not a failure.
#
# Base model (8-65 GB depending on size)
#   Downloaded by transformers into the standard Hub cache, NOT into $DATA_DIR:
#       ${HF_HOME:-~/.cache/huggingface}/hub/
#   Set HF_HOME to move it (e.g. off a small home partition).
#
# Outputs
#   Checkpoints, the merged model, and logs all land in $OUTPUT_DIR (see below).
#   Nothing is written outside $DATA_DIR, $OUTPUT_DIR, and the Hub cache.

# ---- Defaults ----
MODEL_SIZE="4b"   # 4b | 8b | 32b - selects the base model and batch size below
MODEL_ID=""       # derived from MODEL_SIZE unless given explicitly
DATA_DIR="data/multihopspatial"
DATA_PATH=""      # defaults to $DATA_DIR/multihop_train_6791_grpo.json
IMAGE_FOLDER=""   # defaults to $DATA_DIR/data/images
OUTPUT_DIR=""
GPUS="0,1,2,3,4,5,6,7"

NUM_TRAIN_EPOCHS=10
LEARNING_RATE=5e-5
VISION_LEARNING_RATE=5e-6

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=""   # derived from MODEL_SIZE unless given explicitly

NUM_GENERATIONS=4
MAX_COMPLETION_LENGTH=2048
MAX_PROMPT_LENGTH=1024
LOSS_TYPE="grpo"

# Reward coefficients: format + alpha * mcq + beta * bbox + gamma * truncation
MSR_REWARD_ALPHA=1.0
MSR_REWARD_BETA=1.0
TRUNCATION_PENALTY_WEIGHT=1.0

LORA_RANK=64
LORA_ALPHA=64
LORA_EXCLUDE="['visual', 'lm_head', 'embed_tokens']"

USE_WANDB=0
RUN_MERGE=1

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_SIZE="$2"; shift 2 ;;
        --model_id) MODEL_ID="$2"; shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --data_path) DATA_PATH="$2"; shift 2 ;;
        --image_folder) IMAGE_FOLDER="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --epochs) NUM_TRAIN_EPOCHS="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --vision_lr) VISION_LEARNING_RATE="$2"; shift 2 ;;
        --batch_per_device) BATCH_PER_DEVICE="$2"; shift 2 ;;
        --global_batch_size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --num_generations) NUM_GENERATIONS="$2"; shift 2 ;;
        --alpha) MSR_REWARD_ALPHA="$2"; shift 2 ;;
        --beta) MSR_REWARD_BETA="$2"; shift 2 ;;
        --gamma) TRUNCATION_PENALTY_WEIGHT="$2"; shift 2 ;;
        --max_completion_length) MAX_COMPLETION_LENGTH="$2"; shift 2 ;;
        --wandb) USE_WANDB=1; shift ;;
        --no_merge) RUN_MERGE=0; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---- Per-size configuration ----
# Only the base model and the per-device batch size vary. 32B holds far less
# per GPU, so it trades batch size for accumulation to keep the same global
# batch of 128, and needs a less fragmentation-prone allocator.
MODEL_SIZE="$(tr '[:upper:]' '[:lower:]' <<< "$MODEL_SIZE")"
case "$MODEL_SIZE" in
    4b)  SIZE_MODEL_ID="Qwen/Qwen3-VL-4B-Instruct";  SIZE_BATCH=16 ;;
    8b)  SIZE_MODEL_ID="Qwen/Qwen3-VL-8B-Instruct";  SIZE_BATCH=16 ;;
    32b) SIZE_MODEL_ID="Qwen/Qwen3-VL-32B-Instruct"; SIZE_BATCH=1
         # Qwen3-VL-32B has a large vocab and long sequences; without this the
         # allocator fragments and OOMs even when total free memory is fine.
         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ;;
    *)   echo "Error: unknown --model '$MODEL_SIZE' (expected 4b, 8b or 32b)"; exit 1 ;;
esac

MODEL_ID="${MODEL_ID:-$SIZE_MODEL_ID}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-$SIZE_BATCH}"

# ---- Derived configuration ----
DATA_PATH="${DATA_PATH:-${DATA_DIR}/multihop_train_6791_grpo.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_DIR}/data/images}"

NUM_DEVICES=$(awk -F',' '{print NF}' <<< "$GPUS")
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

if (( GRAD_ACCUM_STEPS < 1 )); then
    echo "Error: global batch size ($GLOBAL_BATCH_SIZE) is smaller than"
    echo "       per-device batch ($BATCH_PER_DEVICE) x GPUs ($NUM_DEVICES)."
    echo "       Lower --batch_per_device or raise --global_batch_size."
    exit 1
fi

RUN_NAME="GRPO-Qwen3-VL-$(tr '[:lower:]' '[:upper:]' <<< "$MODEL_SIZE")-Instruct-MultihopSpatial-e${NUM_TRAIN_EPOCHS}-lr${LEARNING_RATE}-alpha${MSR_REWARD_ALPHA}-beta${MSR_REWARD_BETA}-gamma${TRUNCATION_PENALTY_WEIGHT}"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="output/${RUN_NAME}"
fi

if (( USE_WANDB )); then
    REPORT_TO="wandb"
    export WANDB_PROJECT="${WANDB_PROJECT:-MultihopSpatial-GRPO}"
    export WANDB_NAME="$RUN_NAME"
else
    REPORT_TO="none"
fi

# DeepSpeed rendezvous port: honour MASTER_PORT if set, else pick a free one.
if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT=$(python - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
)
fi

# ---- Preflight: fetch the dataset on first run ----
if [[ ! -f "$DATA_PATH" || ! -d "$IMAGE_FOLDER" ]]; then
    echo "Training data not found - downloading it now (~1 GB, one time only)."
    echo "The Hub rate-limits large downloads, so this may pause and resume."
    echo ""
    python prepare_data.py --data_dir "$DATA_DIR" --output "$DATA_PATH"

    if [[ ! -f "$DATA_PATH" || ! -d "$IMAGE_FOLDER" ]]; then
        echo "Error: data preparation did not produce the expected files:"
        echo "  $DATA_PATH"
        echo "  $IMAGE_FOLDER"
        exit 1
    fi
    echo ""
fi

echo "=========================================="
echo "MultihopSpatial GRPO training"
echo "=========================================="
echo "Base model:  $MODEL_ID"
echo "Data:        $DATA_PATH"
echo "Images:      $IMAGE_FOLDER"
echo "Output:      $OUTPUT_DIR"
echo "Epochs:      $NUM_TRAIN_EPOCHS"
echo "LR:          LLM=$LEARNING_RATE, vision/merger=$VISION_LEARNING_RATE"
echo "Batch:       ${NUM_DEVICES} GPUs x ${BATCH_PER_DEVICE} x ${GRAD_ACCUM_STEPS} accum = ${GLOBAL_BATCH_SIZE}"
echo "Generations: $NUM_GENERATIONS  |  max completion: $MAX_COMPLETION_LENGTH"
echo "Reward:      format + ${MSR_REWARD_ALPHA} * mcq + ${MSR_REWARD_BETA} * bbox + ${TRUNCATION_PENALTY_WEIGHT} * truncation"
echo "LoRA:        r=$LORA_RANK alpha=$LORA_ALPHA"
echo "Logging:     $REPORT_TO"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

# ---- Phase 1: Training ----
echo "[Phase 1/2] Training..."

deepspeed --include "localhost:${GPUS}" --master_port "$MASTER_PORT" \
    src/train/train_grpo.py \
    --deepspeed zero2_no_offload.json \
    --model_id "$MODEL_ID" \
    --num_generations $NUM_GENERATIONS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --loss_type $LOSS_TYPE \
    --use_liger_kernel False \
    --lora_enable True \
    --lora_namespan_exclude "$LORA_EXCLUDE" \
    --lora_r $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --use_dora False \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 32 * 32)) \
    --image_max_pixels $((1280 * 32 * 32)) \
    --learning_rate $LEARNING_RATE \
    --merger_lr $VISION_LEARNING_RATE \
    --vision_lr $VISION_LEARNING_RATE \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --dataloader_num_workers 4 \
    --msr_reward_alpha $MSR_REWARD_ALPHA \
    --msr_reward_beta $MSR_REWARD_BETA \
    --truncation_penalty_weight $TRUNCATION_PENALTY_WEIGHT \
    --report_to $REPORT_TO 2>&1 | tee -a "$LOG_FILE"

# ---- Phase 2: Merge LoRA weights ----
if (( RUN_MERGE )); then
    echo "[Phase 2/2] Merging LoRA weights..."
    bash merge_lora.sh "$OUTPUT_DIR" --base_model "$MODEL_ID" \
        --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA
else
    echo "[Phase 2/2] Skipped (--no_merge)."
fi

echo "=========================================="
echo "Training complete."
echo "  Output:  $OUTPUT_DIR"
if (( RUN_MERGE )); then
    echo "  Merged:  $OUTPUT_DIR/merged"
    echo ""
    echo "Evaluate with:"
    echo "  cd ../eval && python benchmark_qwen_vllm.py --model_path ../train/$OUTPUT_DIR/merged"
fi
echo "  Log:     $LOG_FILE"
echo "=========================================="
