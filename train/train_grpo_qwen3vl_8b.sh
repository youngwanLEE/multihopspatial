#!/bin/bash
# GRPO training for Qwen3-VL-8B-Instruct.
#
# Thin wrapper around train_grpo.sh - identical to running:
#     bash train_grpo.sh --model 8b
#
# Every option train_grpo.sh accepts works here too, e.g.:
#     bash train_grpo_qwen3vl_8b.sh --epochs 10 --lr 5e-5 --wandb
#
# See train_grpo.sh for the full option list and for where the dataset and
# base model are downloaded and cached.

exec bash "$(dirname "$0")/train_grpo.sh" --model 8b "$@"
