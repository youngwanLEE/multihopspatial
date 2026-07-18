# GRPO Training

Training code for the MultihopSpatial models — GRPO post-training of Qwen3-VL on
the MultihopSpatial train split (6,791 samples), which is what produced the
released [`etri-vilab/MultiHopSpatial-Qwen3-VL-*`](https://huggingface.co/etri-vilab)
checkpoints.

## Setup

```bash
pip install -r requirements.txt
# flash-attn builds against the installed torch, so install it last:
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

Verified on 8x A100 80GB with CUDA 12.8.

## Quick start

```bash
cd train

# 1. Download the train split from the HF Hub and convert it to the training format
python prepare_data.py

# 2. Train (defaults: 8 GPUs, 10 epochs, lr 5e-5)
bash train_grpo_qwen3vl_4b.sh
```

`prepare_data.py` fetches the dataset (JSON + 6,493 images, ~1 GB) into
`data/multihopspatial/` and writes the converted training JSON alongside it. The
download is cached, so re-running is cheap. The base model
(`Qwen/Qwen3-VL-4B-Instruct`) is fetched automatically on the first training run.

That file count is enough to hit the Hub's rate limit (5,000 requests per 5
minutes) on a free account. The script handles this: it keeps what it already
downloaded, waits out the window, and resumes. If you see a rate-limit message,
just let it run — no need to restart it.

Training writes to `output/<run-name>/` and, unless you pass `--no_merge`, folds
the LoRA adapters back into the base model at `output/<run-name>/merged` — a
standalone checkpoint you can evaluate directly:

```bash
cd ../eval
python benchmark_qwen_vllm.py --model_path ../train/output/<run-name>/merged
```

## Options

```bash
bash train_grpo_qwen3vl_4b.sh --epochs 10 --lr 5e-5     # training length / LR
bash train_grpo_qwen3vl_4b.sh --gpus 0,1,2,3            # subset of GPUs
bash train_grpo_qwen3vl_4b.sh --alpha 1.0 --beta 1.0    # reward coefficients
bash train_grpo_qwen3vl_4b.sh --wandb                   # enable W&B logging
bash train_grpo_qwen3vl_4b.sh --no_merge                # skip the LoRA merge
bash train_grpo_qwen3vl_4b.sh --model_id Qwen/Qwen3-VL-8B-Instruct
```

Gradient accumulation is derived from `--global_batch_size` (default 128),
`--batch_per_device` (16) and the GPU count, so the effective batch size stays
fixed when you change the number of GPUs. Lower `--batch_per_device` if you hit
OOM.

## Reward

GRPO optimizes a weighted sum of four rewards, computed in
`src/train/reward_funcs.py` and combined in `src/trainer/grpo_trainer.py`:

```
reward = format + alpha * mcq + beta * bbox + gamma * truncation
```

| Term | Coefficient | What it scores |
|---|---|---|
| `format` | — | response follows the `Answer:` / `Bounding Box:` template |
| `mcq` | `alpha` (1.0) | multiple-choice answer is correct |
| `bbox` | `beta` (1.0) | IoU of the predicted box against ground truth |
| `truncation` | `gamma` (1.0) | penalizes generations cut off at the length limit |

## Training configuration

Defaults match the released checkpoints:

| | |
|---|---|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` |
| Epochs / LR | 10 / 5e-5 (vision + merger: 5e-6) |
| Batch | 8 GPUs x 16 prompts x 1 accum = 128 |
| Generations per prompt | 4 |
| Max prompt / completion | 1024 / 2048 tokens |
| LoRA | r=64, alpha=64, dropout 0.05 |
| Trainable | vision tower + merger + LoRA; LLM frozen |
| Precision | bf16, gradient checkpointing, DeepSpeed ZeRO-2 |

The vision tower and merger use a 10x smaller learning rate than the LoRA
parameters, which is what the `--vision_lr` / `merger_lr` defaults encode.

## A note on prompt tags

The dataset ships two question fields: `question` (plain) and `question_tag`,
which marks the spatial relation with `<ATT>`/`<POS>`/`<REL>` tags.

**Training uses `question_tag`.** The tags were present in the prompts the
released checkpoints were trained on, so keeping them is what reproduces those
results. Evaluation, however, strips them (`eval/benchmark_qwen.py:remove_tags`),
meaning the models are trained on tagged prompts and evaluated on untagged ones.

This asymmetry is inherited from the original setup and preserved deliberately.
If you want a train/eval-consistent recipe instead, switch `to_grpo_format()` in
`prepare_data.py` to use `record["question"]` — but expect results to differ from
the published numbers, since that is a different training condition.

## Layout

```
train/
  prepare_data.py              downloads + converts the training data
  train_grpo_qwen3vl_4b.sh     training entry point
  merge_lora.sh                folds LoRA adapters back into the base model
  zero2_no_offload.json        DeepSpeed ZeRO-2 config
  requirements.txt
  src/
    train/train_grpo.py        training entry point (launched by deepspeed)
    train/reward_funcs.py      the four reward functions
    trainer/grpo_trainer.py    GRPO trainer, reward combination
    dataset/grpo_dataset.py    prompt construction, bbox normalization
    params.py                  all training arguments
    merge_lora_weights_grpo.py LoRA merge implementation
```
