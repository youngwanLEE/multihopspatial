import os
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy,
    Trainer,
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS
)
from trl import GRPOTrainer
from trl.data_utils import is_conversational
from trl.trainer.utils import (
    pad,
    nanmax,
    nanmin,
    nanstd,
    selective_log_softmax,
    entropy_from_logits,
)
from trl.extras.profiling import profiling_decorator
from accelerate.utils import gather_object, is_peft_model
from src.train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from src.train.reward_funcs import preprocess_completions_for_thinking_model


def _identity_collator(features):
    """Identity collator that passes data through unchanged."""
    return features


class QwenGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super(QwenGRPOTrainer, self).__init__(*args, **kwargs)
        # Override data_collator to prevent any data processing
        self.data_collator = _identity_collator

        # Initialize GDPO reward weights from args
        if hasattr(self.args, 'gdpo_format_weight'):
            self.gdpo_reward_weights = torch.tensor([
                self.args.gdpo_format_weight,
                self.args.gdpo_mcq_weight,
                self.args.gdpo_bbox_weight,
            ])
        else:
            self.gdpo_reward_weights = torch.tensor([1.0, 1.0, 1.0])

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "assistant", "image", "images", "video", "videos", "video_kwargs", "response", "bbox_normalized"]

    def _compute_gdpo_advantages(self, rewards_per_func: torch.Tensor, num_generations: int) -> torch.Tensor:
        """
        Pure GDPO: Independent per-reward normalization → weighted sum → batch norm

        GDPO (Group Decoupled Policy Optimization) normalizes each reward independently
        at the group level, then combines them with weights, and finally applies
        batch-wise normalization.

        Args:
            rewards_per_func: Tensor of shape (batch_size, num_reward_funcs)
            num_generations: Number of generations per prompt

        Returns:
            advantages: Tensor of shape (batch_size,) with normalized advantages
        """
        device = rewards_per_func.device
        rewards_filtered = torch.nan_to_num(rewards_per_func)

        all_advantages = []
        for i in range(rewards_filtered.shape[1]):
            reward_i = rewards_filtered[:, i]

            # Group-wise statistics
            mean_grouped = reward_i.view(-1, num_generations).mean(dim=1)
            std_grouped = reward_i.view(-1, num_generations).std(dim=1)

            # Expand back to batch size
            mean_grouped = mean_grouped.repeat_interleave(num_generations, dim=0)
            std_grouped = std_grouped.repeat_interleave(num_generations, dim=0)

            # Normalize each reward independently
            advantage_i = (reward_i - mean_grouped) / (std_grouped + 1e-4)
            all_advantages.append(advantage_i)

        # Weighted combination
        combined = torch.stack(all_advantages, dim=1)
        weights = self.gdpo_reward_weights.to(device)
        pre_bn = (combined * weights.unsqueeze(0)).nansum(dim=1)

        # Batch-wise normalization
        advantages = (pre_bn - pre_bn.mean()) / (pre_bn.std() + 1e-4)
        return advantages

    def _generate_single_turn(self, prompts: list):
        """Override to include images/videos in generation for multimodal models."""
        from contextlib import nullcontext
        from trl.models.utils import unwrap_model_for_generation
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        device = self.accelerator.device

        # Get stored images/videos
        images = getattr(self, '_current_images', None)
        videos = getattr(self, '_current_videos', None)
        video_kwargs = getattr(self, '_current_video_kwargs', None)

        model_id = getattr(self.model.config, "_name_or_path", "")

        # Build processor kwargs
        # NOTE: When images/videos are present, do NOT truncate.
        # Qwen3-VL's processor validates that image token counts in input_ids match
        # the text. Truncation can cut image tokens, causing a mismatch error in
        # _check_special_mm_tokens (ValueError: Mismatch in `image` token count).
        has_multimodal = images is not None or videos is not None
        processor_kwargs = {
            "text": prompts,
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        }
        if not has_multimodal:
            processor_kwargs["max_length"] = self.max_prompt_length
            processor_kwargs["truncation"] = True

        # Add images if available
        if images is not None:
            processor_kwargs["images"] = images
            processor_kwargs["do_resize"] = True

        # Add videos if available
        if videos is not None:
            if "Qwen2.5" in model_id:
                processor_kwargs["videos"] = videos
                if video_kwargs and video_kwargs[0] is not None:
                    processor_kwargs.update(video_kwargs[0])
            elif "Qwen3" in model_id:
                batched_video_datas = []
                batched_video_metadatas = []
                for sample_videos in videos:
                    if sample_videos is None:
                        batched_video_datas.append(None)
                        batched_video_metadatas.append(None)
                    else:
                        datas, metas = zip(*sample_videos)
                        batched_video_datas.append(list(datas))
                        batched_video_metadatas.append(list(metas))
                # Only add videos to processor_kwargs if there are actual videos (not all None)
                if not all(v is None for v in batched_video_datas):
                    processor_kwargs["videos"] = batched_video_datas
                    processor_kwargs["video_metadata"] = batched_video_metadatas
                    if video_kwargs and video_kwargs[0] is not None:
                        processor_kwargs.update(video_kwargs[0])
            else:
                processor_kwargs["videos"] = videos

        # Process inputs
        generate_inputs = self.processing_class(**processor_kwargs)
        generate_inputs = Trainer._prepare_inputs(self, generate_inputs)

        # Generate completions
        with (
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=self.generation_config, disable_compile=True
            )

        # Extract prompt and completion ids
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]

        return prompt_ids, completion_ids, None, {}

    def _generate(self, prompts: list):
        """
        Wrapper method to call _generate_single_turn and return values in expected format.
        Required for TRL 0.23.x compatibility.
        """
        prompt_ids, completion_ids, sampling_logps, extra = self._generate_single_turn(prompts)
        # Return 5 values as expected by _generate_and_score_completions
        # (prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields)
        return prompt_ids, completion_ids, None, sampling_logps, extra

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Log GPU memory usage for monitoring fragmentation/leaks (rank 0 only)
        if self.accelerator.is_main_process and device.type == "cuda":
            alloc_gb = torch.cuda.memory_allocated(device) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
            frag_gb = reserved_gb - alloc_gb
            print(
                f"[GPU Memory] step={self.state.global_step} "
                f"alloc={alloc_gb:.2f}GB reserved={reserved_gb:.2f}GB frag={frag_gb:.2f}GB",
                flush=True
            )

        # Handle different input formats:
        # 1. List of dicts: [{"prompt": ..., "images": ...}, ...]
        # 2. Batched dict with "prompt" key: {"prompt": [...], "images": [...]}
        # 3. Already processed BatchFeature (no "prompt" key): error case
        if isinstance(inputs, dict):
            if "prompt" in inputs:
                # Batched dict format - convert to list of dicts
                bsz = len(inputs["prompt"])
                inputs = [
                    {k: (v[i] if v is not None else None) for k, v in inputs.items()}
                    for i in range(bsz)
                ]
            else:
                # Already processed BatchFeature - this shouldn't happen in normal flow
                raise ValueError(
                    f"Received pre-processed inputs with keys {list(inputs.keys())}. "
                    "Expected raw inputs with 'prompt' key. This may indicate the data "
                    "is being processed by the AutoProcessor before reaching the trainer. "
                    "Ensure you're passing data_collator=identity_collator to the trainer."
                )
        elif not isinstance(inputs, list):
            # Unexpected input type
            raise TypeError(
                f"Expected inputs to be list[dict] or dict, got {type(inputs).__name__}. "
                f"Sample: {str(inputs)[:200]}"
            )

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        if "videos" in inputs[0]:
            videos = [example.get("videos") for example in inputs]
            video_kwargs = [example.get("video_kwargs") for example in inputs]
        elif "video" in inputs[0]:
            videos = [[example.get("video")] if example.get("video") is not None else None for example in inputs]
            video_kwargs = [example.get("video_kwargs") for example in inputs]
        else:
            videos = None

        if videos is not None and all(v_list is None or v_list == [] for v_list in videos):
            videos = None

        # Store images/videos for use in _generate_single_turn
        self._current_images = images
        self._current_videos = videos
        self._current_video_kwargs = video_kwargs if videos is not None else None

        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
            self._generate(prompts)
        )

        # Clear stored images/videos
        self._current_images = None
        self._current_videos = None
        self._current_video_kwargs = None

        # Free GPU memory after generation to prevent fragmentation accumulation
        torch.cuda.empty_cache()

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        model_id = getattr(self.model.config, "_name_or_path", "")

        # Get forward_kwargs for models with multimodal inputs
        if images is not None or videos is not None:
            prompts_text = prompts

            processor_kwargs = dict(
                text=prompts_text,
                padding=True,
                return_tensors="pt",
                do_resize=True
            )
            if images is not None:
                processor_kwargs["images"] = images
            if videos is not None:
                if "Qwen2.5" in model_id:
                    common_vk = video_kwargs[0] if isinstance(video_kwargs, list) else video_kwargs
                    processor_kwargs["videos"] = videos
                    if common_vk is not None:
                        processor_kwargs.update(common_vk)

                elif "Qwen3" in model_id:
                    batched_video_datas = []
                    batched_video_metadatas = []
                    for sample_videos in videos:
                        if sample_videos is None:
                            batched_video_datas.append(None)
                            batched_video_metadatas.append(None)
                        else:
                            datas, metas = zip(*sample_videos)
                            batched_video_datas.append(list(datas))
                            batched_video_metadatas.append(list(metas))

                    # Only add videos to processor_kwargs if there are actual videos (not all None)
                    if not all(v is None for v in batched_video_datas):
                        processor_kwargs["videos"] = batched_video_datas
                        processor_kwargs["video_metadata"] = batched_video_metadatas

                        common_vk = video_kwargs[0] if isinstance(video_kwargs, list) else video_kwargs
                        if common_vk is not None:
                            processor_kwargs.update(common_vk)

                else:
                    processor_kwargs["videos"] = videos

            prompt_inputs = self.processing_class(**processor_kwargs)
            # Use Trainer._prepare_inputs directly to avoid recursive call through GRPOTrainer._prepare_inputs
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Free GPU memory after logps computation
        torch.cuda.empty_cache()

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Preprocess completions for Thinking models (extract answer after </think> tag)
        model_id = getattr(self.model.config, "_name_or_path", "")
        completions_for_reward = preprocess_completions_for_thinking_model(
            completions_text,
            model_id=model_id,
            debug=False  # Uses global DEBUG_MODE from reward_funcs
        )

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions_for_reward, completion_ids_list)

        # Special handling for truncation reward: use RAW completions (before preprocessing)
        # This is necessary because preprocessing removes <think> tags needed for truncation detection
        truncation_func_name = None
        if "msr_truncation_reward" in self.reward_func_names:
            truncation_func_name = "msr_truncation_reward"
        elif "msr_truncation_mcq_only" in self.reward_func_names:
            truncation_func_name = "msr_truncation_mcq_only"

        if truncation_func_name is not None:
            import os
            from accelerate.utils import gather_object
            # Dynamically import the correct truncation function
            import importlib
            reward_module = importlib.import_module("src.train.reward_funcs")
            truncation_func = getattr(reward_module, truncation_func_name)
            idx = self.reward_func_names.index(truncation_func_name)

            # Gather raw completions from all processes (same as _calculate_rewards does internally)
            if self.accelerator.num_processes > 1:
                all_completions_text = gather_object(completions_text)
            else:
                all_completions_text = completions_text

            # Recalculate truncation reward with gathered raw completions
            truncation_rewards = truncation_func(
                all_completions_text,  # gathered completions
                raw_completions=all_completions_text,  # RAW completions for detection
                debug_mode=os.environ.get("MSR_DEBUG_MODE", "0") == "1"
            )
            rewards_per_func[:, idx] = torch.tensor(truncation_rewards, device=device, dtype=rewards_per_func.dtype)

        # Determine reward type and coefficients
        reward_type = getattr(self.args, 'reward_type', 'simple_sum')
        alpha = getattr(self.args, 'msr_reward_alpha', 1.0)
        beta = getattr(self.args, 'msr_reward_beta', 1.0)
        gamma = getattr(self.args, 'truncation_penalty_weight', 1.0)

        # Check if using MSR reward functions for conditional formula
        msr_func_names = ["msr_format_reward", "msr_mcq_reward", "msr_bbox_reward"]
        is_msr_reward = all(name in self.reward_func_names for name in msr_func_names)

        # Check if using MCQ-only reward functions
        mcq_only_func_names = ["msr_format_mcq_only", "msr_mcq_reward", "msr_truncation_mcq_only"]
        is_mcq_only = reward_type == "mcq-only" and all(name in self.reward_func_names for name in mcq_only_func_names)

        # simple_sum with non-uniform weights needs explicit formula;
        # when alpha=beta=1 the result is identical to uniform weighted sum (else fallback)
        needs_weighted_simple_sum = (
            reward_type == "simple_sum"
            and is_msr_reward
            and (alpha != 1.0 or beta != 1.0)
        )

        # Handle different loss types: gdpo, grpo_bn, grpo (default)
        if self.loss_type == "gdpo":
            # GDPO: Independent per-reward normalization → weighted sum → batch norm
            # Skip MSR conditional formula entirely - each reward is normalized independently
            advantages = self._compute_gdpo_advantages(rewards_per_func, self.num_generations)

            # For logging purposes, compute combined rewards (simple sum for display)
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

        else:
            # GRPO or GRPO_BN
            if is_mcq_only:
                # MCQ-Only Formula: format + α × mcq + truncation
                format_idx = self.reward_func_names.index("msr_format_mcq_only")
                mcq_idx = self.reward_func_names.index("msr_mcq_reward")
                trunc_idx = self.reward_func_names.index("msr_truncation_mcq_only")

                format_r = rewards_per_func[:, format_idx]
                mcq_r = rewards_per_func[:, mcq_idx]
                trunc_r = rewards_per_func[:, trunc_idx]

                rewards = format_r + alpha * mcq_r + trunc_r

            elif needs_weighted_simple_sum:
                # Simple Sum with non-uniform weights: format + α × mcq + β × bbox [+ γ × truncation]
                format_idx = self.reward_func_names.index("msr_format_reward")
                mcq_idx = self.reward_func_names.index("msr_mcq_reward")
                bbox_idx = self.reward_func_names.index("msr_bbox_reward")

                format_r = rewards_per_func[:, format_idx]
                mcq_r = rewards_per_func[:, mcq_idx]
                bbox_r = rewards_per_func[:, bbox_idx]

                # Independent weighted sum: MCQ and bbox contribute independently
                rewards = format_r + alpha * mcq_r + beta * bbox_r

                # Add truncation penalty if truncation reward function is present
                if "msr_truncation_reward" in self.reward_func_names:
                    trunc_idx = self.reward_func_names.index("msr_truncation_reward")
                    trunc_r = rewards_per_func[:, trunc_idx]
                    rewards = rewards + gamma * trunc_r

                # Log formula activation (once per training)
                if not hasattr(self, '_msr_simple_sum_logged'):
                    self._msr_simple_sum_logged = True
                    has_trunc = "msr_truncation_reward" in self.reward_func_names
                    formula = f"format + α({alpha}) × mcq + β({beta}) × bbox"
                    if has_trunc:
                        formula += f" + γ({gamma}) × truncation"
                    if self.accelerator.is_main_process:
                        print(f"✅ MSR Simple Sum Reward: {formula}")

            elif is_msr_reward and reward_type == "condition_reward":
                # MSR Conditional Reward Formula: format + α × mcq × (1 + β × bbox) [+ γ × truncation]
                format_idx = self.reward_func_names.index("msr_format_reward")
                mcq_idx = self.reward_func_names.index("msr_mcq_reward")
                bbox_idx = self.reward_func_names.index("msr_bbox_reward")

                format_r = rewards_per_func[:, format_idx]
                mcq_r = rewards_per_func[:, mcq_idx]
                bbox_r = rewards_per_func[:, bbox_idx]

                # Conditional formula: when MCQ is wrong (0), bbox contribution becomes 0
                rewards = format_r + alpha * mcq_r * (1.0 + beta * bbox_r)

                # Add truncation penalty if truncation reward function is present
                if "msr_truncation_reward" in self.reward_func_names:
                    trunc_idx = self.reward_func_names.index("msr_truncation_reward")
                    trunc_r = rewards_per_func[:, trunc_idx]
                    rewards = rewards + gamma * trunc_r

                # Log conditional formula activation (once per training)
                if not hasattr(self, '_msr_formula_logged'):
                    self._msr_formula_logged = True
                    has_trunc = "msr_truncation_reward" in self.reward_func_names
                    formula = f"format + α({alpha}) × mcq × (1 + β({beta}) × bbox)"
                    if has_trunc:
                        formula += f" + γ({gamma}) × truncation"
                    if self.accelerator.is_main_process:
                        print(f"✅ MSR Conditional Reward: {formula}")
            else:
                # Uniform weighted sum (all weights = 1.0)
                # Also handles simple_sum with alpha=beta=1 (equivalent result)
                rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

                if not hasattr(self, '_uniform_sum_logged'):
                    self._uniform_sum_logged = True
                    if self.accelerator.is_main_process:
                        print(f"✅ Uniform Weighted Sum (all reward weights = 1.0)")

            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards

            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll still log group level std
                std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
                std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
            elif self.scale_rewards == "batch":
                # Compute global std
                std_rewards = rewards.std().expand_as(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)

            # GRPO_BN: Add batch normalization after group-wise normalization
            if self.loss_type == "grpo_bn":
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["image"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]

        if "pixel_values_videos" in forward_kwargs:
            output["pixel_values_videos"] = forward_kwargs["pixel_values_videos"]
        if "video_grid_thw" in forward_kwargs:
            output["video_grid_thw"] = forward_kwargs["video_grid_thw"]
        if "second_per_grid_ts" in forward_kwargs:
            output["second_per_grid_ts"] = forward_kwargs["second_per_grid_ts"]

        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output
    
    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]

            if pixel_values_videos is not None:
                model_inputs["pixel_values_videos"] = pixel_values_videos[start : start + batch_size]
            if video_grid_thw is not None:
                model_inputs["video_grid_thw"] = video_grid_thw[start : start + batch_size]
            if second_per_grid_ts is not None:
                model_inputs["second_per_grid_ts"] = second_per_grid_ts[start : start + batch_size]

            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]

            # Chunked log_softmax + gather along sequence dim — avoids
            # materializing a full (B, gen_len, vocab) log_probs intermediate
            # for large-vocab models (Qwen3-VL vocab ~152k → ~620MB per row).
            # Mathematically equivalent to selective_log_softmax/log_softmax+gather
            # since softmax is independent across positions, and gradient is identical.
            _LOGPROB_CHUNK = 256
            gen_len = logits.shape[1]
            if gen_len <= _LOGPROB_CHUNK:
                logps = selective_log_softmax(logits, completion_ids)
            else:
                chunks = []
                for i in range(0, gen_len, _LOGPROB_CHUNK):
                    c_logits = logits[:, i : i + _LOGPROB_CHUNK, :]
                    c_lps = F.log_softmax(c_logits, dim=-1)
                    c_ids = completion_ids[:, i : i + _LOGPROB_CHUNK]
                    chunks.append(c_lps.gather(2, c_ids.unsqueeze(-1)).squeeze(-1))
                    del c_lps, c_logits
                logps = torch.cat(chunks, dim=1)  # (B, gen_len)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)
            del logits

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies
    
    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values

        if video_grid_thw is not None and pixel_values_videos is not None:
            model_inputs["video_grid_thw"] = video_grid_thw
            model_inputs["pixel_values_videos"] = pixel_values_videos
        if second_per_grid_ts is not None:
            model_inputs["second_per_grid_ts"] = second_per_grid_ts

        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state
    
    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
            inputs.get("pixel_values_videos"),
            inputs.get("video_grid_thw"),
            inputs.get("second_per_grid_ts"),
        )

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss / self.current_gradient_accumulation_steps

    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type in ["grpo", "gdpo", "grpo_bn"]:
            # GRPO, GDPO, and GRPO_BN use the same loss computation
            # (The difference is in the advantage calculation done in _generate_and_score_completions)
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                                "param_group_name": "visaul_decay"
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                                "param_group_name": "visaul_non_decay"
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                                "param_group_name": "merger_decay",
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                                "param_group_name": "merger_non_decay",
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Override save_model for DeepSpeed + PEFT/LoRA to avoid OOM.

        Root cause: The parent save_model() calls accelerator.get_state_dict()
        which clones the ENTIRE model state dict (all 8B params, ~16GB) to CPU
        via clone_tensors_for_torch_save(). With 8 processes × 16GB = 128GB
        extra CPU RAM, this causes OOM during checkpoint saves when system RAM
        is already near capacity.

        Fix: Directly gather only LoRA adapter parameters (~250MB) without
        cloning the full model. This reduces checkpoint save memory overhead
        from ~128GB to ~2GB total across 8 processes.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # Only use efficient path for DeepSpeed + LoRA
        if not (self.is_deepspeed_enabled and getattr(self.args, 'lora_enable', False)):
            return super().save_model(output_dir, _internal_call=_internal_call)

        # For ZeRO-2, parameters are full copies on each rank (no cross-rank
        # gathering needed). Only rank 0 needs to save.
        if not self.args.should_save:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Get only LoRA state dict (~250MB) instead of full model (~16GB)
        lora_state_dict = get_peft_state_maybe_zero_3(
            self.model.named_parameters(),
            getattr(self.args, 'lora_bias', 'none'),
        )

        # Unwrap DeepSpeed/Accelerate wrapper to get the PeftModel
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save LoRA adapter weights + config via PEFT's save_pretrained
        unwrapped_model.save_pretrained(
            output_dir,
            state_dict=lora_state_dict,
            safe_serialization=self.args.save_safetensors,
        )

        # Save processor/tokenizer
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Push to hub if needed (mirrors parent behavior)
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        if not self.args.lora_enable:
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        non_lora = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters(),
            require_grad_only=True,
        )

        if self.args.should_save:
            torch.save(non_lora, os.path.join(output_dir, "non_lora_state_dict.bin"))
            self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))
            # Save adapter_config.json for standard PEFT compatibility
            if hasattr(self.model, 'peft_config'):
                adapter_cfg = self.model.peft_config.get('default')
                if adapter_cfg is not None:
                    adapter_cfg.save_pretrained(output_dir)
            # Rename model.safetensors -> adapter_model.safetensors for PEFT compatibility
            _model_st = os.path.join(output_dir, "model.safetensors")
            _adapter_st = os.path.join(output_dir, "adapter_model.safetensors")
            if os.path.exists(_model_st) and not os.path.exists(_adapter_st):
                os.rename(_model_st, _adapter_st)