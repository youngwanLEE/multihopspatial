import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai, BBOX_INSTRUCTION_PATTERN


# MSR Response Format Template (default: MCQ + BBox)
MSR_RESPONSE_FORMAT = """

Please respond in the following format:
Answer: (your choice, e.g., "(a) object name")
Bounding Box: [x1, y1, x2, y2]

where [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner.
The coordinates should be in the range [0, 1000]."""

# MCQ-Only Response Format Template (no BBox)
MCQ_ONLY_RESPONSE_FORMAT = """

Please respond in the following format:
Answer: (your choice, e.g., "(a) object name")"""


def normalize_bbox_to_1000(bbox, image_resolution):
    """
    Convert [x, y, width, height] in pixels to [x1, y1, x2, y2] in 0-1000 scale.

    Args:
        bbox: [x, y, width, height] in pixel coordinates
        image_resolution: "WIDTHxHEIGHT" string (e.g., "640x480")

    Returns:
        [x1, y1, x2, y2] normalized to 0-1000 scale
    """
    img_w, img_h = map(int, image_resolution.split('x'))
    x, y, w, h = bbox

    # Convert to x1, y1, x2, y2 format
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # Normalize to 0-1000 scale
    x1_norm = int((x1 / img_w) * 1000)
    y1_norm = int((y1 / img_h) * 1000)
    x2_norm = int((x2 / img_w) * 1000)
    y2_norm = int((y2 / img_h) * 1000)

    # Clamp to [0, 1000] range
    x1_norm = max(0, min(1000, x1_norm))
    y1_norm = max(0, min(1000, y1_norm))
    x2_norm = max(0, min(1000, x2_norm))
    y2_norm = max(0, min(1000, y2_norm))

    return [x1_norm, y1_norm, x2_norm, y2_norm]

class GRPODataset(Dataset):
    """Dataset for DPO training"""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
        reward_type="simple_sum",
    ):
        super(GRPODataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.reward_type = reward_type
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
            self.return_video_metadata = True
        else:
            self.image_patch_size = 14
            self.return_video_metadata = False

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        # MSR-specific: Extract response and bbox if available
        msr_response = sources.get("response", None)
        msr_bbox = sources.get("bbox", None)
        msr_resolution = sources.get("image_resolution", None)

        # Normalize bbox to 0-1000 scale if available
        bbox_normalized = None
        if msr_bbox is not None and msr_resolution is not None:
            bbox_normalized = normalize_bbox_to_1000(msr_bbox, msr_resolution)

        if "image" in sources:
            videos = None
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
            image_folder_fallback = self.data_args.image_folder_fallback

            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                        if not os.path.exists(image_file) and image_folder_fallback:
                            image_file = os.path.join(image_folder_fallback, os.path.basename(image_file))
                image_input = get_image_info(
                        image_file, 
                        self.image_min_pixel, 
                        self.image_max_pixel, 
                        self.image_resized_w, 
                        self.image_resized_h, 
                        self.image_patch_size
                    )
                images.append(image_input)
        elif "video" in sources:
            is_video = True
            images=None

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(
                    video_file, 
                    self.video_min_pixel, 
                    self.video_max_pixel, 
                    self.video_resized_w, 
                    self.video_resized_h, 
                    self.data_args.fps,
                    self.image_patch_size,
                    return_video_metadata=self.return_video_metadata
                )
                videos.append(video_input)
        else:
            images=None
            videos=None

        conversations = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        user_input = conversations[0]
        gpt_response = conversations[1]

        # MSR-specific: Add response format instructions
        user_content = user_input['content']
        if self.reward_type == "mcq-only":
            # MCQ-only: strip bbox instruction from query, add MCQ-only format, skip bbox data
            user_content = BBOX_INSTRUCTION_PATTERN.sub('', user_content)
            user_content = user_content + MCQ_ONLY_RESPONSE_FORMAT
            bbox_normalized = None
        elif bbox_normalized is not None:
            user_content = user_content + MSR_RESPONSE_FORMAT

        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        user_message = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_content}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"

        user_prompt = system_message + user_message
        assistant_prompt = gpt_response['content']

        data_dict = dict(
            prompt=user_prompt,
            assistant=assistant_prompt,
        )

        # MSR-specific: Add response and normalized bbox for reward calculation
        if msr_response is not None:
            data_dict["response"] = msr_response
        if bbox_normalized is not None:
            data_dict["bbox_normalized"] = bbox_normalized

        # Only include images/videos keys when they have actual data
        if images is not None:
            data_dict["images"] = images
        if videos is not None:
            data_dict["videos"] = videos
            data_dict["video_kwargs"] = video_kwargs

        return data_dict
    
def make_grpo_data_module(model_id, processor, data_args, reward_type="simple_sum"):
    """Make dataset and collator for supervised fine-tuning."""
    grpo_dataset = GRPODataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id,
        reward_type=reward_type,
    )

    return dict(train_dataset=grpo_dataset,
                eval_dataset=None)