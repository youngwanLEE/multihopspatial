#!/bin/bash
# ============================================================
# Merge GRPO LoRA weights into the base model
# ============================================================
# GRPO training saves LoRA adapters plus a non-LoRA state dict holding the
# tuned vision tower and merger. This folds both back into the base model,
# producing a standalone checkpoint that vLLM / transformers can load directly.
#
# Usage:
#     bash merge_lora.sh <CHECKPOINT_DIR> --base_model <BASE_MODEL>
#     bash merge_lora.sh output/my-run --base_model Qwen/Qwen3-VL-4B-Instruct
#
# The merged model is written to <CHECKPOINT_DIR>/merged.
# LoRA rank/alpha must match what training used (defaults: 64/64).
# ============================================================

set -eo pipefail

export PYTHONPATH=src:$PYTHONPATH

if [ -z "$1" ]; then
    echo "Error: CHECKPOINT_DIR is required"
    echo "Usage: bash merge_lora.sh <CHECKPOINT_DIR> --base_model <BASE_MODEL>"
    exit 1
fi

MODEL_PATH="$1"
shift

BASE_MODEL=""
LORA_RANK=64
LORA_ALPHA=64

while [[ $# -gt 0 ]]; do
    case $1 in
        --base_model) BASE_MODEL="$2"; shift 2 ;;
        --lora_rank) LORA_RANK="$2"; shift 2 ;;
        --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$BASE_MODEL" ]]; then
    echo "Error: --base_model is required (e.g. Qwen/Qwen3-VL-4B-Instruct)"
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: checkpoint directory not found: $MODEL_PATH"
    exit 1
fi

SAVE_PATH="${MODEL_PATH}/merged"

echo "============================================"
echo "Merging LoRA weights"
echo "============================================"
echo "Base model:  $BASE_MODEL"
echo "Checkpoint:  $MODEL_PATH"
echo "Save to:     $SAVE_PATH"
echo "LoRA:        rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "============================================"

python src/merge_lora_weights_grpo.py \
    --model-path "$MODEL_PATH" \
    --model-base "$BASE_MODEL" \
    --save-model-path "$SAVE_PATH" \
    --lora-alpha $LORA_ALPHA \
    --lora-rank $LORA_RANK \
    --safe-serialization

echo "============================================"
echo "Done. Merged model saved to: $SAVE_PATH"
echo "============================================"
