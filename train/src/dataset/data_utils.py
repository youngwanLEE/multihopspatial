# Portions of this file (get_image_info / get_video_info) are adapted from
# 2U1/Qwen-VL-Series-Finetune (Apache-2.0):
# https://github.com/2U1/Qwen-VL-Series-Finetune
# They are thin wrappers around process_vision_info from the official
# qwen-vl-utils package (Apache-2.0).

import re

import torch
from qwen_vl_utils import process_vision_info
from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)

# Shared regex pattern for stripping bbox instruction from user queries
# Used by both SFT (sft_dataset.py) and GRPO MCQ-only (grpo_dataset.py)
BBOX_INSTRUCTION_PATTERN = re.compile(
    r"\s*And provide the bounding box coordinate of the region related to your answer\.",
    re.IGNORECASE,
)


def bbox_xywh_pixel_to_xyxy_1000(bbox_xywh, img_width, img_height):
    """Convert [x, y, w, h] pixel coords to [x1, y1, x2, y2] in 0-1000 scale."""
    x, y, w, h = bbox_xywh
    x1 = round(x / img_width * 1000)
    y1 = round(y / img_height * 1000)
    x2 = round((x + w) / img_width * 1000)
    y2 = round((y + h) / img_height * 1000)
    return [x1, y1, x2, y2]


def format_response_with_bbox(response_text, bbox_xywh, image_resolution_str):
    """Format assistant response to include bbox in benchmark-compatible format.

    Args:
        response_text: Original MCQ answer, e.g. "(b) green round container"
        bbox_xywh: [x, y, width, height] in pixel coords
        image_resolution_str: "WxH" string, e.g. "640x494"

    Returns:
        Formatted string like "Answer: (b) green round container\\nBounding Box: [876, 506, 940, 604]"
        or original response_text if conversion fails.
    """
    if bbox_xywh is None or image_resolution_str is None:
        return response_text
    try:
        w_str, h_str = image_resolution_str.split("x")
        img_w, img_h = int(w_str), int(h_str)
        if img_w <= 0 or img_h <= 0:
            return response_text
        xyxy_1000 = bbox_xywh_pixel_to_xyxy_1000(bbox_xywh, img_w, img_h)
        return f"Answer: {response_text}\nBounding Box: [{xyxy_1000[0]}, {xyxy_1000[1]}, {xyxy_1000[2]}, {xyxy_1000[3]}]"
    except (ValueError, TypeError, ZeroDivisionError):
        return response_text


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r"\n?" + re.escape(LLAVA_VIDEO_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r"\n?" + re.escape(LLAVA_IMAGE_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)


def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[: max_length - 1]
        labels = labels[: max_length - 1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels


def pad_sequence(sequences, padding_side="right", padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_image_info(image_path, min_pixel, max_pixel, width, height, image_patch_size=None):
    # Using this because of process_vision_info function
    # Note: image_patch_size is kept for API compatibility but not used in newer qwen_vl_utils
    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel,
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]

    image_input, _ = process_vision_info(messages)

    return image_input[0]


def get_video_info(
    video_path,
    min_pixels,
    max_pixels,
    width,
    height,
    fps,
    image_patch_size=None,
    return_video_metadata=False,
):
    # Using this because of process_vision_info function
    # Note: image_patch_size and return_video_metadata kept for API compatibility
    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps,
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs


def samples_per_class_from_ids(label_ids, num_classes):

    counts = torch.bincount(torch.as_tensor(label_ids, dtype=torch.long), minlength=num_classes)

    return counts.tolist()
