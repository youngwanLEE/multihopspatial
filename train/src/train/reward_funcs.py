import os
import re

# Global debug mode flag - can be set via environment variable or kwargs
DEBUG_MODE = os.environ.get("MSR_DEBUG_MODE", "0") == "1"


def set_debug_mode(enabled: bool):
    """Set global debug mode for MSR reward functions."""
    global DEBUG_MODE
    DEBUG_MODE = enabled


def _debug_print(*args, **kwargs):
    """Print only when debug mode is enabled."""
    if DEBUG_MODE:
        print("[MSR_DEBUG]", *args, **kwargs)


# ============================================================================
# Thinking Model Support
# ============================================================================


def extract_answer_from_thinking(completion: str) -> str:
    """
    Extract the answer portion after </think> tag for Thinking models.

    Qwen3-VL-Thinking models output in the format:
        <think>
        ... reasoning content (step-by-step analysis) ...
        </think>
        Answer: (c) white car
        Bounding Box: [300, 145, 556, 621]

    Processing cases:
    1. Normal: <think>...</think>Answer: ... → Extract after </think>
    2. Truncated (max_tokens insufficient): <think>...Answer: ... → Use last Answer: pattern
    3. No thinking tags: Return original (Instruct model compatibility)

    Args:
        completion: Raw model completion text

    Returns:
        str: Answer portion with thinking block removed, or original if no tags
    """
    # Case 1: Normal case with closing </think> tag
    if "</think>" in completion:
        # Extract everything after </think>
        after_think = completion.split("</think>", 1)[1]
        return after_think.strip()

    # Case 2: Truncated - has <think> but no </think>
    # Try to find the last "Answer:" pattern as fallback
    if "<think>" in completion:
        # Look for Answer: pattern
        answer_matches = list(re.finditer(r"Answer:\s*\([a-d]\)", completion, re.IGNORECASE))
        if answer_matches:
            # Use the last Answer: match as the start of the answer section
            last_match = answer_matches[-1]
            return completion[last_match.start() :].strip()
        # No Answer: found in truncated thinking - return empty to signal format failure
        return ""

    # Case 3: No thinking tags - return original (Instruct model compatibility)
    return completion


def preprocess_completions_for_thinking_model(
    completions: list[str], model_id: str = "", debug: bool = False
) -> list[str]:
    """
    Preprocess completions for Thinking models by extracting answer portions.

    This function should be called before reward calculation when using
    Thinking models (e.g., Qwen3-VL-4B-Thinking).

    Args:
        completions: List of raw model completions
        model_id: Model identifier (checks for "Thinking" in name)
        debug: Whether to print debug information

    Returns:
        list[str]: Preprocessed completions with thinking blocks removed
    """
    # Check if this is a Thinking model
    is_thinking_model = "Thinking" in model_id

    if not is_thinking_model:
        return completions

    processed = []
    for i, completion in enumerate(completions):
        extracted = extract_answer_from_thinking(completion)
        processed.append(extracted)

        if debug or DEBUG_MODE:
            _debug_print(f"=== Thinking Model Preprocessing [{i}] ===")
            # _debug_print(f"  Raw completion (first 300 chars): {repr(completion[:300])}...")
            _debug_print(f"  Raw completion (first 300 chars): {repr(completion)}")
            _debug_print(f"  Has <think>: {'<think>' in completion}")
            _debug_print(f"  Has </think>: {'</think>' in completion}")
            _debug_print(
                f"  Extracted answer: {repr(extracted[:200])}..."
                if len(extracted) > 200
                else f"  Extracted answer: {repr(extracted)}"
            )

    return processed


# ============================================================================
# MSR (Multi-hop Spatial Reasoning) Reward Functions
# ============================================================================


def compute_giou(box1, box2):
    """
    Compute Generalized IoU between two boxes.

    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format

    Returns:
        GIoU value in [-1, 1]
    """
    # Ensure valid box format (x1 < x2, y1 < y2)
    box1 = [
        min(box1[0], box1[2]),
        min(box1[1], box1[3]),
        max(box1[0], box1[2]),
        max(box1[1], box1[3]),
    ]
    box2 = [
        min(box2[0], box2[2]),
        min(box2[1], box2[3]),
        max(box2[0], box2[2]),
        max(box2[1], box2[3]),
    ]

    # Intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Individual areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union
    union_area = area1 + area2 - inter_area

    # IoU
    if union_area <= 0:
        iou = 0.0
    else:
        iou = inter_area / union_area

    # Enclosing box
    x1_enc = min(box1[0], box2[0])
    y1_enc = min(box1[1], box2[1])
    x2_enc = max(box1[2], box2[2])
    y2_enc = max(box1[3], box2[3])
    enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)

    # GIoU
    if enc_area <= 0:
        giou = iou
    else:
        giou = iou - (enc_area - union_area) / enc_area

    return giou


def msr_format_reward(completions, **kwargs):
    """
    MSR Format Reward: Check if completion matches the required format.

    Expected format:
        Answer: (X) object name
        Bounding Box: [x1, y1, x2, y2]

    Returns: 1.0 if format matches, 0.0 otherwise
    """
    # Check for debug mode from kwargs or global
    debug = kwargs.get("debug_mode", DEBUG_MODE)

    # Pattern to match the expected format (flexible with whitespace)
    pattern = r"Answer:\s*\([a-d]\)\s*[\w\s\-\'\"]+.*?Bounding Box:\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"

    rewards = []
    for i, completion in enumerate(completions):
        match = re.search(pattern, completion, re.IGNORECASE | re.DOTALL)
        reward = 1.0 if match else 0.0
        rewards.append(reward)

        if debug:
            _debug_print(f"=== msr_format_reward [{i}] ===")
            _debug_print(
                f"  Raw completion: {repr(completion[:500])}..."
                if len(completion) > 500
                else f"  Raw completion: {repr(completion)}"
            )
            _debug_print(f"  Format matched: {bool(match)}")
            _debug_print(f"  Reward: {reward}")

    return rewards


def msr_mcq_reward(completions, response=None, **kwargs):
    """
    MSR MCQ Reward: Check if the MCQ answer (a/b/c/d) matches the ground truth.

    Args:
        completions: List of model outputs
        response: List of ground truth answers (e.g., "(c) white car")

    Returns: 1.0 if correct, 0.0 if wrong
    """
    debug = kwargs.get("debug_mode", DEBUG_MODE)

    if response is None:
        return [0.0] * len(completions)

    answer_pattern = r"\(([a-d])\)"
    rewards = []

    for i, (completion, gt) in enumerate(zip(completions, response)):
        # Extract answer choice from completion
        pred_match = re.search(r"Answer:\s*\(([a-d])\)", completion, re.IGNORECASE)
        gt_match = re.search(answer_pattern, gt, re.IGNORECASE)

        pred_choice = pred_match.group(1).lower() if pred_match else None
        gt_choice = gt_match.group(1).lower() if gt_match else None

        if pred_choice and gt_choice:
            reward = 1.0 if pred_choice == gt_choice else 0.0
        else:
            reward = 0.0

        rewards.append(reward)

        if debug:
            _debug_print(f"=== msr_mcq_reward [{i}] ===")
            _debug_print(f"  GT response: {gt}")
            _debug_print(f"  Predicted choice: {pred_choice}, GT choice: {gt_choice}")
            _debug_print(f"  Reward: {reward}")

    return rewards


def msr_bbox_reward(completions, bbox_normalized=None, **kwargs):
    """
    MSR Bounding Box Reward: Calculate GIoU-based reward for bounding box prediction.

    Args:
        completions: List of model outputs
        bbox_normalized: List of GT bboxes in [x1, y1, x2, y2] 0-1000 scale

    Returns: (GIoU + 1) / 2 normalized to [0, 1]
    """
    debug = kwargs.get("debug_mode", DEBUG_MODE)

    if bbox_normalized is None:
        return [0.0] * len(completions)

    bbox_pattern = r"Bounding Box:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
    rewards = []

    for i, (completion, gt_bbox) in enumerate(zip(completions, bbox_normalized)):
        match = re.search(bbox_pattern, completion, re.IGNORECASE)

        pred_bbox = None
        giou = None
        if match and gt_bbox is not None:
            try:
                pred_bbox = [int(match.group(j)) for j in range(1, 5)]
                giou = compute_giou(pred_bbox, gt_bbox)
                # Normalize from [-1, 1] to [0, 1]
                reward = (giou + 1) / 2
            except (ValueError, TypeError):
                reward = 0.0
        else:
            reward = 0.0

        rewards.append(reward)

        if debug:
            _debug_print(f"=== msr_bbox_reward [{i}] ===")
            _debug_print(f"  GT bbox: {gt_bbox}")
            _debug_print(f"  Pred bbox: {pred_bbox}")
            _debug_print(f"  GIoU: {giou}")
            _debug_print(f"  Reward: {reward}")

    return rewards


def msr_truncation_reward(completions, raw_completions=None, penalty_value=-1.0, **kwargs):
    """
    Truncation reward/penalty for Thinking model GRPO training.

    Penalizes completions that are truncated. This encourages the model to
    generate concise reasoning that fits within the max_new_tokens limit.

    NOTE: Named `_reward` (not `_penalty`) to be auto-discovered by load_reward_funcs().

    Args:
        completions: List of completion strings (may be preprocessed)
        raw_completions: List of RAW completion strings (before preprocessing)
                        If provided, uses these for truncation detection
        penalty_value: Penalty value for truncated completions (default: -1.0)

    Returns:
        List of reward values (0.0 for complete, penalty_value for truncated)

    Detection logic (checked in order):
        1. Has <think> but no </think> → Definitely truncated → penalty
        2. Has </think> but no complete BBox → Likely truncated → penalty
        3. No <think> tag (Instruct model) but no complete BBox → Truncated → penalty
        4. Otherwise → Complete → 0.0
    """
    # Use raw_completions if provided, otherwise use completions
    texts_to_check = raw_completions if raw_completions is not None else completions

    # Complete BBox pattern: Bounding Box: [num, num, num, num]
    bbox_complete_pattern = r"Bounding Box:\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"

    penalties = []
    debug = kwargs.get("debug_mode", DEBUG_MODE)

    for i, completion in enumerate(texts_to_check):
        has_think_open = "<think>" in completion
        has_think_close = "</think>" in completion
        has_complete_bbox = bool(re.search(bbox_complete_pattern, completion, re.IGNORECASE))

        if has_think_open and not has_think_close:
            # Case 1: Definitely truncated (thinking block not closed)
            penalty = penalty_value
            reason = "has <think> but no </think>"
        elif has_think_close and not has_complete_bbox:
            # Case 2: Thinking done but BBox incomplete/missing
            penalty = penalty_value
            reason = "has </think> but no complete BBox"
        elif not has_think_open and not has_complete_bbox:
            # Case 3: Instruct model output but no complete BBox
            penalty = penalty_value
            reason = "no <think> and no complete BBox"
        else:
            # Complete output
            penalty = 0.0
            reason = "complete"

        penalties.append(penalty)

        if debug:
            # Only print debug output when using raw_completions (the correct call)
            # Skip debug for preprocessed completions to avoid confusion
            using_raw = raw_completions is not None
            if using_raw:
                _debug_print(f"=== msr_truncation_reward [{i}] (RAW) ===")
                _debug_print(f"  Has <think>: {has_think_open}")
                _debug_print(f"  Has </think>: {has_think_close}")
                _debug_print(f"  Has complete BBox: {has_complete_bbox}")
                _debug_print(f"  Reward: {penalty} ({reason})")

    return penalties


# ============================================================================
# MCQ-Only Reward Functions (for --reward_type mcq-only)
# ============================================================================
# NOTE: These functions do NOT end with "_reward" so they are NOT auto-discovered
# by load_reward_funcs(). They are explicitly loaded by train_grpo.py
# when --reward_type mcq-only is specified.


def msr_format_mcq_only(completions, **kwargs):
    """
    MCQ-Only Format Reward: Check if completion has valid MCQ answer format.

    Expected format:
        Answer: (X) object name

    BBox is NOT checked.
    Returns: 1.0 if format matches, 0.0 otherwise
    """
    debug = kwargs.get("debug_mode", DEBUG_MODE)

    # Pattern: Answer: (a-d) followed by at least one word
    pattern = r"Answer:\s*\([a-d]\)\s*[\w\s\-\'\"]+?"

    rewards = []
    for i, completion in enumerate(completions):
        match = re.search(pattern, completion, re.IGNORECASE)
        reward = 1.0 if match else 0.0
        rewards.append(reward)

        if debug:
            _debug_print(f"=== msr_format_mcq_only [{i}] ===")
            _debug_print(
                f"  Raw completion: {repr(completion[:500])}..."
                if len(completion) > 500
                else f"  Raw completion: {repr(completion)}"
            )
            _debug_print(f"  Format matched: {bool(match)}")
            _debug_print(f"  Reward: {reward}")

    return rewards


def msr_truncation_mcq_only(completions, raw_completions=None, penalty_value=-1.0, **kwargs):
    """
    MCQ-Only Truncation Reward: Penalize truncated completions.

    Detection logic (uses Answer pattern instead of BBox):
        1. Has <think> but no </think> → Truncated → penalty
        2. Has </think> but no Answer: (x) pattern → Truncated → penalty
        3. No <think> (Instruct model) but no Answer: (x) → Truncated → penalty
        4. Otherwise → Complete → 0.0
    """
    texts_to_check = raw_completions if raw_completions is not None else completions

    # MCQ answer pattern: Answer: (a-d)
    answer_pattern = r"Answer:\s*\([a-d]\)"

    penalties = []
    debug = kwargs.get("debug_mode", DEBUG_MODE)

    for i, completion in enumerate(texts_to_check):
        has_think_open = "<think>" in completion
        has_think_close = "</think>" in completion
        has_answer = bool(re.search(answer_pattern, completion, re.IGNORECASE))

        if has_think_open and not has_think_close:
            penalty = penalty_value
            reason = "has <think> but no </think>"
        elif has_think_close and not has_answer:
            penalty = penalty_value
            reason = "has </think> but no Answer: (x) pattern"
        elif not has_think_open and not has_answer:
            penalty = penalty_value
            reason = "no <think> and no Answer: (x) pattern"
        else:
            penalty = 0.0
            reason = "complete"

        penalties.append(penalty)

        if debug:
            using_raw = raw_completions is not None
            if using_raw:
                _debug_print(f"=== msr_truncation_mcq_only [{i}] (RAW) ===")
                _debug_print(f"  Has <think>: {has_think_open}")
                _debug_print(f"  Has </think>: {has_think_close}")
                _debug_print(f"  Has Answer pattern: {has_answer}")
                _debug_print(f"  Reward: {penalty} ({reason})")

    return penalties
