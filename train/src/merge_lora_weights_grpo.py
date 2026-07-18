"""
Merge LoRA weights into base model (supports both GRPO and SFT checkpoints).

Works with two different checkpoint formats produced by this codebase:

1. GRPO checkpoint (train_grpo.py → QwenGRPOTrainer):
   - adapter_model.safetensors: LoRA weights
   - non_lora_state_dict.bin: Vision tower, merger, embed_tokens, lm_head (~17GB)
   - Key format: model.language_model.layers.0.mlp.down_proj.lora_A.default.weight
     → peft_config를 TRL trainer에 전달하면 trainer가 내부적으로 LoRA를 적용.
       ".default" adapter 이름이 키에 포함됨.

2. SFT checkpoint (train_sft.py → HF Trainer + get_peft_model):
   - adapter_model.safetensors: LoRA weights
   - non_lora_state_dict.bin: 거의 비어있음 (vision/merger frozen이면)
   - Key format: base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight
     → get_peft_model()이 모델을 PeftModel로 wrapping하면서
       "base_model.model." prefix가 추가되고, ".default"는 없음.

두 형식의 차이는 같은 PEFT 라이브러리를 사용하지만
LoRA 적용 방식이 다르기 때문 (모델 자체와는 무관):
  - SFT: get_peft_model(model, config) → PeftModel wrapper가 prefix 추가
  - GRPO: trainer(peft_config=config) → TRL이 내부적으로 적용, .default adapter 이름 포함

Usage:
    # GRPO checkpoint merge
    python src/merge_lora_weights_grpo.py \\
        --model-path output/GRPO-Qwen3-VL-8B-... \\
        --model-base checkpoints/Qwen3-VL-8B-Thinking \\
        --save-model-path output/GRPO-.../merged

    # SFT checkpoint merge
    python src/merge_lora_weights_grpo.py \\
        --model-path output/SFT-LoRA-Qwen3-VL-8B-... \\
        --model-base checkpoints/Qwen3-VL-8B-Instruct \\
        --save-model-path output/SFT-.../merged
"""

import argparse
import os
import re
import time
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoProcessor,
    AutoConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except ImportError:
    Qwen3VLMoeForConditionalGeneration = None


MODEL_CLASS_MAP = {
    "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration,
    "qwen3_vl": Qwen3VLForConditionalGeneration,
    "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
    "qwen2_vl": Qwen2VLForConditionalGeneration,
}

# Order matters: check longer prefixes first to avoid "qwen3_vl" matching "qwen3_vl_moe"
_MODEL_TYPE_MATCH_ORDER = ["qwen3_vl_moe", "qwen3_vl", "qwen2_5_vl", "qwen2_vl"]


def resolve_model_class(model_type: str):
    """Resolve model class from model_type string.

    Handles variant suffixes from newer transformers (e.g. "qwen2_5_vl_text")
    by using substring matching with longest-prefix-first ordering.
    """
    # Exact match first
    if model_type in MODEL_CLASS_MAP:
        return MODEL_CLASS_MAP[model_type]
    # Substring match (longest prefix first)
    for key in _MODEL_TYPE_MATCH_ORDER:
        if key in model_type:
            return MODEL_CLASS_MAP[key]
    return None


def load_lora_weights(model_path: str) -> dict:
    """Load LoRA weights from adapter_model.safetensors or model.safetensors."""
    # Check standard PEFT format first, then legacy format
    lora_path = os.path.join(model_path, "adapter_model.safetensors")
    if not os.path.exists(lora_path):
        lora_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(
            f"LoRA weights not found in {model_path} "
            f"(checked adapter_model.safetensors and model.safetensors)"
        )

    lora_weights = {}
    with safe_open(lora_path, framework="pt") as f:
        for key in f.keys():
            lora_weights[key] = f.get_tensor(key)

    return lora_weights


def _ensure_safetensors(model_path: str) -> str | None:
    """Convert non_lora_state_dict.bin to safetensors if needed.

    Returns path to the safetensors file, or None if not found/corrupted.
    First run does a one-time conversion (~18s); subsequent runs are instant.
    """
    st_path = os.path.join(model_path, "non_lora_state_dict.safetensors")
    bin_path = os.path.join(model_path, "non_lora_state_dict.bin")

    if os.path.exists(st_path):
        return st_path

    if not os.path.exists(bin_path):
        return None

    print(f"  Converting .bin -> .safetensors (one-time)...")
    t0 = time.time()
    try:
        sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    except RuntimeError as e:
        print(f"  WARNING: Failed to load non_lora_state_dict.bin: {e}")
        print(f"  The file may be corrupted (e.g. truncated during save).")
        print(f"  Will skip non-LoRA weights and use base model weights instead.")
        return None
    save_file(sd, st_path)
    del sd
    print(f"  Conversion done in {time.time() - t0:.1f}s")
    return st_path


def load_non_lora_weights(model_path: str, skip_base_layer: bool = True) -> dict:
    """Load non-LoRA trainable weights from safetensors.

    Uses safetensors (memory-mapped, near-instant) instead of torch.load.
    Skips frozen LLM base_layer weights by default since they're
    identical to the base model.

    Returns empty dict if the non_lora_state_dict file is missing or corrupted,
    which is safe when non-LoRA params were frozen (e.g. TRL PEFT wrapping).
    """
    st_path = _ensure_safetensors(model_path)

    if st_path is None:
        print(f"  No valid non-LoRA weights found. Using base model weights as-is.")
        return {}

    filtered = {}
    skipped = 0
    with safe_open(st_path, framework="pt") as f:
        for key in f.keys():
            if skip_base_layer and ".base_layer." in key:
                skipped += 1
                continue
            filtered[key] = f.get_tensor(key)

    if skip_base_layer:
        print(f"  Skipped {skipped} frozen base_layer keys (already in base model)")
    return filtered


def get_base_key(lora_key: str) -> str:
    """Convert LoRA key to base model key.

    Handles multiple checkpoint formats:

    GRPO (TRL trainer, ".default" adapter):
        model.language_model.layers.0.mlp.down_proj.lora_A.default.weight
        -> model.language_model.layers.0.mlp.down_proj.weight

    GRPO (older transformers, no language_model prefix):
        model.layers.0.mlp.down_proj.lora_A.default.weight
        -> model.layers.0.mlp.down_proj.weight
        (merge_lora_inplace remaps to model.language_model.layers.* if needed)

    SFT (get_peft_model() wrapping, "base_model.model." prefix, no ".default"):
        base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight
        -> model.language_model.layers.0.mlp.down_proj.weight
    """
    # Strip lora_A/B suffix: ".lora_A.default.weight" (GRPO) or ".lora_A.weight" (SFT)
    key = re.sub(r"\.lora_[AB](\.default)?\.weight$", ".weight", lora_key)
    # Strip PeftModel wrapper prefix (SFT only): "base_model.model." -> ""
    key = re.sub(r"^base_model\.model\.", "", key)
    return key


def merge_lora_inplace(model, lora_weights: dict,
                       lora_alpha: int, lora_rank: int):
    """Merge LoRA weights directly into model parameters (no copy).

    Avoids model.state_dict() which doubles memory usage.
    Instead, walks model.named_parameters() and applies delta in-place.
    """
    scaling = lora_alpha / lora_rank
    print(f"  LoRA scaling: {lora_alpha}/{lora_rank} = {scaling}")

    # Group LoRA A and B weights by module
    lora_pairs = {}
    for key, tensor in lora_weights.items():
        base_key = get_base_key(key)
        if base_key not in lora_pairs:
            lora_pairs[base_key] = {}
        if ".lora_A." in key:
            lora_pairs[base_key]["A"] = tensor
        elif ".lora_B." in key:
            lora_pairs[base_key]["B"] = tensor

    print(f"  Found {len(lora_pairs)} LoRA module pairs")

    # Use GPU for matrix multiplication if available, then move back
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"  Using {device} for LoRA matmul")

    # Build name -> parameter lookup from model
    param_map = dict(model.named_parameters())

    merged_count = 0
    remapped_count = 0
    for base_key, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair:
            print(f"  WARNING: Incomplete LoRA pair for {base_key}")
            continue

        lookup_key = base_key
        if lookup_key not in param_map:
            # Newer transformers renamed Qwen2.5-VL internal keys:
            #   model.layers.X... -> model.language_model.layers.X...
            # Adapter may use old format; try remapping.
            remapped = re.sub(
                r"^(model)\.(layers\.)",
                r"\1.language_model.\2",
                base_key,
            )
            if remapped != base_key and remapped in param_map:
                lookup_key = remapped
                remapped_count += 1

        if lookup_key in param_map:
            W = param_map[lookup_key]
            A = pair["A"].to(device=device, dtype=W.dtype)
            B = pair["B"].to(device=device, dtype=W.dtype)
            delta = (B @ A) * scaling
            W.data.add_(delta.cpu())
            merged_count += 1
        else:
            print(f"  WARNING: Base weight not found for {base_key}")

    if remapped_count > 0:
        print(f"  Remapped {remapped_count} keys: model.layers.* -> model.language_model.layers.*")
    print(f"  Merged {merged_count}/{len(lora_pairs)} LoRA modules")


def merge_grpo_lora(args):
    model_path = args.model_path
    model_base = args.model_base
    save_path = args.save_model_path
    lora_alpha = args.lora_alpha
    lora_rank = args.lora_rank

    print("=" * 60)
    print("LoRA Weight Merger (GRPO / SFT)")
    print("=" * 60)
    print(f"  Checkpoint: {model_path}")
    print(f"  Base model: {model_base}")
    print(f"  Save path:  {save_path}")
    print(f"  LoRA:       alpha={lora_alpha}, rank={lora_rank}")
    print("=" * 60)

    total_start = time.time()

    # 1. Detect model type
    config = AutoConfig.from_pretrained(model_path)
    model_type = config.model_type
    print(f"\nModel type: {model_type}")

    # 2. Load base model
    t0 = time.time()
    print(f"\n[1/5] Loading base model...")
    model_cls = resolve_model_class(model_type)
    if model_cls is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model_cls.from_pretrained(
        model_base,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        config=config,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # 3. Load & apply non-LoRA weights (vision, merger, embed, lm_head)
    t0 = time.time()
    print(f"\n[2/5] Loading non-LoRA weights...")
    non_lora_weights = load_non_lora_weights(model_path, skip_base_layer=True)
    print(f"  Loaded {len(non_lora_weights)} tensors in {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"\n[3/5] Applying non-LoRA weights to base model...")
    model.load_state_dict(non_lora_weights, strict=False)
    del non_lora_weights
    print(f"  Done in {time.time() - t0:.1f}s")

    # 4. Load LoRA weights & merge in-place on model parameters
    t0 = time.time()
    print(f"\n[4/5] Loading LoRA weights & merging in-place...")
    lora_weights = load_lora_weights(model_path)
    print(f"  Loaded {len(lora_weights)} LoRA tensors")

    merge_lora_inplace(model, lora_weights, lora_alpha, lora_rank)
    del lora_weights
    print(f"  Done in {time.time() - t0:.1f}s")

    # 5. Save
    t0 = time.time()
    print(f"\n[5/5] Saving merged model to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=args.safe_serialization)
    processor = AutoProcessor.from_pretrained(model_path)
    processor.save_pretrained(save_path)

    # Copy base model's config.json to ensure vLLM compatibility.
    # Newer transformers adds text_config section which older vLLM may not handle.
    # model_base may be a local directory or a Hub repo id, so resolve both.
    import shutil
    base_config_path = os.path.join(model_base, "config.json")
    if not os.path.exists(base_config_path):
        try:
            base_config_path = hf_hub_download(repo_id=model_base, filename="config.json")
        except Exception as err:
            base_config_path = None
            print(f"  Could not fetch base config.json ({err}); keeping the generated one")

    if base_config_path:
        shutil.copy2(base_config_path, os.path.join(save_path, "config.json"))
        print(f"  Copied base model config.json for vLLM compatibility")

    print(f"  Done in {time.time() - t0:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Merge complete! Total: {total_time:.1f}s")
    print(f"Saved to: {save_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model (GRPO/SFT)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to LoRA checkpoint (GRPO or SFT)")
    parser.add_argument("--model-base", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--save-model-path", type=str, required=True,
                        help="Path to save merged model")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha (default: 64)")
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="LoRA rank (default: 64)")
    parser.add_argument("--safe-serialization", action="store_true", default=True,
                        help="Save in safetensors format (default: True)")
    parser.add_argument("--no-safe-serialization", action="store_false", dest="safe_serialization",
                        help="Save in PyTorch bin format instead of safetensors")

    args = parser.parse_args()
    merge_grpo_lora(args)
