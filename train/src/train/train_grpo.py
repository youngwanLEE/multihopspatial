import os
import torch
from peft import LoraConfig
import ast
import pathlib
from transformers import (
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    HfArgumentParser,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)

# ============================================================================
# FIX: Patch DeepSpeed's scaled_global_norm to fix Long dtype bug
# Root cause: complete_grad_norm_calculation_for_cpu_offload returns -1 (int)
#             when norm is invalid, and torch.tensor(-1) creates Long dtype
# ============================================================================
def patch_deepspeed_long_dtype_fix():
    """
    Fix DeepSpeed's scaled_global_norm Long dtype bug.

    The bug: When cpu_offload is enabled and norm is invalid (-1),
    torch.tensor(-1) creates a Long dtype tensor, causing
    torch.linalg.vector_norm to fail.

    Solution: Ensure norm tensor is always float dtype.
    """
    try:
        import deepspeed.runtime.zero.stage_1_and_2 as ds_zero

        original_scaled_global_norm = ds_zero.DeepSpeedZeroOptimizer.scaled_global_norm

        def fixed_scaled_global_norm(self, norm_type=2):
            assert norm_type == 2, "only L2 norm supported"
            norm_groups = []

            for i, group in enumerate(self.bit16_groups):
                if self.cpu_offload:
                    # FIX: Always create float tensor, even when -1 is returned
                    norm_value = self.complete_grad_norm_calculation_for_cpu_offload(
                        self.params_in_partition[i])
                    norm = torch.tensor(norm_value, dtype=torch.float32, device=self.device)
                    norm_groups.append(norm)
                else:
                    norm_groups.append(
                        self.get_grad_norm_direct(self.averaged_gradients[i],
                                                  self.params_in_partition[i]))

            if self.has_moe_layers:
                self._average_expert_grad_norms(norm_groups)

            return torch.linalg.vector_norm(torch.stack(norm_groups), ord=norm_type)

        ds_zero.DeepSpeedZeroOptimizer.scaled_global_norm = fixed_scaled_global_norm
        print("✅ DeepSpeed Long dtype bug fixed (torch.tensor now uses dtype=float32)")
    except Exception as e:
        print(f"⚠️ Could not patch DeepSpeed: {e}")

# Apply fix on import
patch_deepspeed_long_dtype_fix()
# ============================================================================

from src.trainer import QwenGRPOTrainer
from src.dataset import make_grpo_data_module
from src.params import DataArguments, ModelArguments, GRPOArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
from monkey_patch_forward import (
    replace_qwen2_5_with_mixed_modality_forward, 
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward
)
from monkey_patch_vision import replace_qwen2_5_vision
from src.utils import  load_reward_funcs

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(model.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = model.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    if k_llm and hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        for layer in model.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        for blk in model.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


def fix_all_non_floating_parameters(model, verbose=True):
    """
    CRITICAL: Recursively find and fix ALL non-floating point parameters/buffers
    including embedding parameters and index buffers that might be hidden
    in module hierarchies.
    """
    fixed_count = 0

    def process_module(module, prefix=""):
        nonlocal fixed_count
        for name, param in list(module._parameters.items()):
            if param is None:
                continue
            full_name = f"{prefix}.{name}" if prefix else name
            if not param.dtype.is_floating_point:
                if param.requires_grad:
                    if verbose:
                        rank0_print(f"  ⚠️  Parameter '{full_name}' dtype={param.dtype} → Setting requires_grad=False")
                    param.requires_grad = False
                    fixed_count += 1

        for name, buffer in list(module._buffers.items()):
            if buffer is None:
                continue
            # Buffers don't need gradients anyway, so just log if found
            if not buffer.dtype.is_floating_point and verbose:
                full_name = f"{prefix}.{name}" if prefix else name
                rank0_print(f"  ℹ️  Buffer '{full_name}' dtype={buffer.dtype}")

        for name, child in list(module.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            process_module(child, child_prefix)

    process_module(model)
    if verbose and fixed_count > 0:
        rank0_print(f"\n  ✅ Fixed {fixed_count} parameters with non-floating point dtype\n")
    return fixed_count


def fix_long_type_parameters(model, verbose=True):
    """
    Fix parameters with Long dtype that have requires_grad=True.
    This causes 'linalg.vector_norm: Expected floating point, Got Long' error
    during gradient norm calculation.

    More aggressive: Also fix all buffers and non-leaf parameters.
    """
    fixed_count = 0
    found_long = False

    # Check and fix all parameters
    for name, param in model.named_parameters():
        if not param.dtype.is_floating_point:
            found_long = True
            if param.requires_grad:
                if verbose:
                    rank0_print(f"  ⚠️  Parameter '{name}' dtype={param.dtype} requires_grad=True → Setting to False")
                param.requires_grad = False
                fixed_count += 1
            elif verbose:
                rank0_print(f"  ✓ Parameter '{name}' dtype={param.dtype} requires_grad=False (OK)")

    # Also check all buffers (non-parameters that should not have gradients anyway)
    for name, buffer in model.named_buffers():
        if not buffer.dtype.is_floating_point and verbose:
            rank0_print(f"  ℹ️  Buffer '{name}' dtype={buffer.dtype} (buffers don't need gradients)")

    if found_long and verbose:
        rank0_print(f"\n  Summary: Found Long dtype params/buffers. Fixed {fixed_count} parameters by disabling requires_grad.")

    return fixed_count



def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, GRPOArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set debug mode for MSR reward functions via environment variable
    if getattr(training_args, "debug_mode", False):
        os.environ["MSR_DEBUG_MODE"] = "1"
        rank0_print("🔍 DEBUG MODE ENABLED: MSR reward functions will print detailed parsing info")

    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    config = AutoConfig.from_pretrained(model_args.model_id)

    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )

    elif config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )

    elif config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )
        
    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )


    model.config.use_cache = False

    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    # Fix all non-floating point parameters that might cause gradient norm calculation errors
    # This is CRITICAL for full fine-tuning without LoRA with DeepSpeed
    rank0_print("🔧 Scanning for non-floating point parameters (potential Long dtype issues)...")
    fixed_params = fix_all_non_floating_parameters(model_to_configure, verbose=True)
    if fixed_params == 0:
        rank0_print("✅ No problematic non-floating point parameters found")

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
        model.enable_input_require_grads()

    if training_args.bits in [4,8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)

    peft_config = None

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    reward_type = getattr(training_args, 'reward_type', 'simple_sum')

    dataset_module = make_grpo_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args,
                                              reward_type=reward_type)

    if reward_type == "mcq-only":
        from src.train.reward_funcs import (
            msr_format_mcq_only,
            msr_mcq_reward,
            msr_truncation_mcq_only,
        )
        reward_funcs = [
            msr_format_mcq_only,
            msr_mcq_reward,
            msr_truncation_mcq_only,
        ]
        rank0_print(f"✅ MCQ-Only reward functions loaded: {[f.__name__ for f in reward_funcs]}")
    else:
        reward_funcs = load_reward_funcs("src.train.reward_funcs")

    trainer = QwenGRPOTrainer(
        model=model,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module["eval_dataset"],
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        peft_config=peft_config,
    )

    # ============================================================================
    # FIX: Stabilize sampling by using lower temperature and higher top_k
    # Root cause: After training step, model can produce unstable logits that
    # cause sampling to fail with "probability tensor contains inf, nan or < 0"
    # Solution: Use temperature=0.7 and top_k=50 for more stable sampling
    # Note: do_sample=True is required for GRPO diversity
    # ============================================================================
    # Check if this is a Thinking model (e.g., Qwen3-VL-4B-Thinking)
    is_thinking_model = "Thinking" in model_args.model_id

    if hasattr(trainer, 'generation_config'):
        trainer.generation_config.do_sample = True

        if is_thinking_model:
            # Thinking models need more tokens for reasoning + answer
            # Use official Thinking model settings: temperature=1.0, top_p=0.95
            trainer.generation_config.max_new_tokens = 1536  # Enough for <think>...</think> + answer
            trainer.generation_config.temperature = 0.8  # Slightly lower than official 1.0 for stability
            trainer.generation_config.top_p = 0.95
            trainer.generation_config.top_k = 50  # Keep for stability
            rank0_print(f"✅ Thinking model generation_config: max_new_tokens=1536, temperature=0.8, top_p=0.95")
        else:
            # Instruct models use standard settings
            trainer.generation_config.temperature = 0.7  # Lower temperature for stability
            trainer.generation_config.top_k = 50  # Limit vocabulary for stability
            trainer.generation_config.top_p = 0.9
            rank0_print(f"✅ Trainer generation_config updated: do_sample=True, temperature=0.7, top_k=50")

    # Fix all non-floating point parameters after trainer initialization
    # (DeepSpeed may introduce new wrapped parameters)
    if training_args.deepspeed and not training_args.lora_enable:
        rank0_print("\n🔧 FINAL CHECK: Scanning for non-floating point parameters after DeepSpeed wrapping...")
        fixed_params = fix_all_non_floating_parameters(trainer.model, verbose=True)
        if fixed_params == 0:
            rank0_print("✅ All parameters are floating point - ready for training")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        # Use require_grad_only=True to match intermediate checkpoint behavior.
        # Frozen base model params haven't changed, so no need to save them.
        # require_grad_only=False caused OOM: all 8 ranks simultaneously copy
        # the entire 8B model (~16GB each) to CPU RAM.
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
            # Save adapter_config.json for standard PEFT compatibility
            peft_config.save_pretrained(training_args.output_dir)
            # Rename model.safetensors -> adapter_model.safetensors for PEFT compatibility
            _model_st = os.path.join(training_args.output_dir, "model.safetensors")
            _adapter_st = os.path.join(training_args.output_dir, "adapter_model.safetensors")
            if os.path.exists(_model_st) and not os.path.exists(_adapter_st):
                os.rename(_model_st, _adapter_st)
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()