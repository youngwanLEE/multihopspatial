#!/bin/bash
# GRPO training for Qwen3-VL-32B-Instruct.
#
# Thin wrapper around train_grpo.sh - identical to running:
#     bash train_grpo.sh --model 32b
#
# Every option train_grpo.sh accepts works here too, e.g.:
#     bash train_grpo_qwen3vl_32b.sh --epochs 10 --lr 5e-5 --wandb
#
# Note: 32B trains with per-device batch 1 and 16 accumulation steps (same
# global batch of 128 as the smaller models), and sets an allocator flag to
# avoid fragmentation OOMs. Both are handled by train_grpo.sh.
#
# See train_grpo.sh for the full option list and for where the dataset and
# base model are downloaded and cached.

exec bash "$(dirname "$0")/train_grpo.sh" --model 32b "$@"
