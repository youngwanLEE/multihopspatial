#!/usr/bin/env python3
"""
MultihopSpatial Benchmark - Qwen3-VL (Transformers / HF Inference)
====================================================================
Evaluates a Qwen3-VL model on the MultihopSpatial benchmark using
plain HuggingFace `transformers` generation (no vLLM required).

Both the test set (question/answer/bbox JSON + images) and the model
checkpoint are downloaded automatically on first run and cached locally
(via `huggingface_hub` / `transformers`) - no manual setup needed.

Dataset : https://huggingface.co/datasets/etri-vilab/MultihopSpatial
Model   : https://huggingface.co/etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct

Qwen3-VL Coordinate System
---------------------------
Qwen3-VL natively outputs bounding boxes in 0-1000 relative coordinates
(see the official Qwen3-VL 2D grounding cookbook). This script asks the
model for `[x1, y1, x2, y2]` and auto-converts any value > 1 from the
0-1000 scale to 0-1 normalized coordinates before computing IoU.

Usage
-----
    # Quick smoke test on 5 samples (auto-downloads data + model)
    python benchmark_qwen.py --test_samples 5

    # Full benchmark (4500 samples)
    python benchmark_qwen.py --output results_qwen3vl_4b.json

    # Point at a local checkpoint instead of the HF Hub
    python benchmark_qwen.py --model_path /path/to/local/checkpoint

    # Resume a run that was interrupted or had failed samples
    python benchmark_qwen.py --output results_qwen3vl_4b.json --resume

Requirements
------------
    pip install -r requirements.txt
    # See requirements.txt for exact pinned versions this was verified against.
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_PATH = "etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct"
DATASET_REPO_ID = "etri-vilab/MultihopSpatial"
DEFAULT_DATA_DIR = "data/multihopspatial"

# =============================================================================
# MCQ Answer Parsing
# =============================================================================
# (Lenient patterns covering the common ways models phrase "Answer: (a) ...")


def parse_mcq_answer(text: str) -> Optional[str]:
    """Extracts a normalized "(a) description" string from free-form model text."""
    if not text:
        return None

    m = re.search(r"Answer:\s*(\([a-d]\)\s*[^\n]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"Answer:\s*([a-d])\)\s*([^\n]*)", text, re.IGNORECASE)
    if m:
        letter, desc = m.group(1).lower(), m.group(2).strip()
        return f"({letter}) {desc}" if desc else f"({letter})"

    m = re.search(r"Answer:\s*([a-d])\s*$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return f"({m.group(1).lower()})"

    # Fallback: bare "(a) ..." anywhere in the text
    m = re.search(r"(\([a-d]\)\s*[^\n,\[\]]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


def extract_choice_letter(text: Optional[str]) -> Optional[str]:
    """Extracts the lowercase choice letter (a-d) from a parsed prediction string."""
    if text is None:
        return None
    m = re.search(r"\(([a-d])\)", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


# =============================================================================
# Dataset / Image Utilities
# =============================================================================


def download_dataset(data_dir: str) -> tuple[str, str]:
    """Downloads (or reuses a cached copy of) the MultihopSpatial test set.

    Returns:
        (json_path, image_root)
    """
    print(f"Fetching dataset '{DATASET_REPO_ID}' -> {data_dir} (cached after first run)")
    local_dir = snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns=["data/multihop_test_4500.json", "data/images/*"],
    )
    json_path = os.path.join(local_dir, "data", "multihop_test_4500.json")
    image_root = os.path.join(local_dir, "data", "images")
    return json_path, image_root


def remove_tags(question: str) -> str:
    """Removes <ATT>, <POS>, <REL> annotation tags from the question text."""
    cleaned = re.sub(r"</?(?:ATT|POS|REL)>", "", question)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_prompt(question: str) -> str:
    clean_q = remove_tags(question)
    return f"""{clean_q}

Please respond in the following format:
Answer: (your choice, e.g., "(a) object name")
Bounding Box: [x1, y1, x2, y2]

where [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner."""


def get_image_path(image_filename: str, image_root: str) -> str:
    while image_filename.startswith("/") or image_filename.startswith("\\"):
        image_filename = image_filename[1:]
    return os.path.join(image_root, image_filename)


def get_image_dimensions(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


# =============================================================================
# BBox / IoU Utilities
# =============================================================================


def normalized_to_pixel(bbox_norm: list, width: int, height: int) -> list:
    x1, y1, x2, y2 = bbox_norm
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def xywh_to_xyxy(bbox_xywh: list) -> list:
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def calculate_iou_xyxy(bbox1: list, bbox2: list) -> Optional[float]:
    if bbox1 is None or bbox2 is None or len(bbox1) != 4 or len(bbox2) != 4:
        return None
    try:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter_area = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area
        if union_area <= 0:
            return 0.0
        return round(inter_area / union_area, 4)
    except (TypeError, ValueError):
        return None


def calculate_iou(
    bbox_gt_xywh: list, bbox_pred_norm: list, img_width: int, img_height: int
) -> Optional[float]:
    """gt is [x, y, w, h] in pixels; pred is [x1, y1, x2, y2] normalized 0-1."""
    if (
        bbox_gt_xywh is None
        or bbox_pred_norm is None
        or len(bbox_gt_xywh) != 4
        or len(bbox_pred_norm) != 4
    ):
        return None
    try:
        gt_xyxy = xywh_to_xyxy(bbox_gt_xywh)
        pred_xyxy = normalized_to_pixel(bbox_pred_norm, img_width, img_height)
        return calculate_iou_xyxy(gt_xyxy, pred_xyxy)
    except (TypeError, ValueError):
        return None


def parse_response(response_text: str) -> tuple[Optional[str], Optional[list]]:
    """Extracts the MCQ answer and a 0-1 normalized bbox from the raw model output.

    Qwen3-VL natively outputs bbox coordinates on a 0-1000 scale regardless of
    prompt wording; any bbox with a value > 1 is auto-converted to 0-1.
    """
    prediction = parse_mcq_answer(response_text)

    bbox = None
    m = re.search(
        r"Bounding Box:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]",
        response_text,
        re.IGNORECASE,
    )
    if m:
        bbox = [float(m.group(i)) for i in range(1, 5)]
    if bbox is None:
        m = re.search(
            r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
            response_text,
        )
        if m:
            bbox = [float(m.group(i)) for i in range(1, 5)]
    if bbox is None:
        m = re.findall(
            r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", response_text
        )
        if m:
            bbox = [float(v) for v in m[0]]

    pred_bbox = None
    if bbox is not None:
        if any(v > 1 for v in bbox):
            bbox = [v / 1000.0 for v in bbox]
        pred_bbox = bbox

    return prediction, pred_bbox


def is_valid_bbox(bbox: Optional[list]) -> bool:
    if bbox is None or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    if any(v < 0 or v > 1 for v in bbox):
        return False
    return x2 > x1 and y2 > y1


def is_valid_prediction(prediction: Optional[str], pred_bbox: Optional[list]) -> bool:
    if prediction is None or extract_choice_letter(prediction) is None:
        return False
    return is_valid_bbox(pred_bbox)


def compute_score(prediction: Optional[str], answer: Optional[str]) -> bool:
    pred_letter = extract_choice_letter(prediction)
    answer_letter = extract_choice_letter(answer)
    return pred_letter is not None and pred_letter == answer_letter


def calculate_mcq_accuracy(results: list) -> dict:
    total, correct = 0, 0
    for item in results:
        answer_letter = extract_choice_letter(item.get("answer"))
        pred_letter = extract_choice_letter(item.get("prediction"))
        if answer_letter and pred_letter:
            total += 1
            if answer_letter == pred_letter:
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return {"total_evaluated": total, "correct": correct, "accuracy": round(accuracy * 100, 2)}


# =============================================================================
# Logging
# =============================================================================


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("benchmark_qwen")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# =============================================================================
# Model Loading / Inference
# =============================================================================


def load_model(model_path: str, use_flash_attention: bool = True):
    """Loads a Qwen3-VL model + processor, from a local path or an HF Hub repo id."""
    print(f"Loading model: {model_path}")
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def run_inference(
    model, processor, image_path: str, prompt: str, max_new_tokens: int, **generate_kwargs
) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **generate_kwargs)

    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] if output_text else ""


# =============================================================================
# Main Benchmark Loop
# =============================================================================


def run_benchmark(
    json_path: str,
    image_root: str,
    output_path: str,
    model_path: str,
    test_samples: Optional[int] = None,
    use_flash_attention: bool = True,
    max_new_tokens: int = 4096,
    max_retries: int = 3,
    resume: bool = False,
) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path.replace(".json", f"_{timestamp}.log")
    logger = setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("MultihopSpatial Benchmark - Qwen3-VL (Transformers)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"JSON: {json_path}")
    logger.info(f"Image Root: {image_root}")
    logger.info(f"Output: {output_path}")

    # Instruct models use greedy decoding; "Thinking" variants use the
    # sampling settings recommended in their model card.
    is_thinking = "thinking" in model_path.lower()
    if is_thinking:
        generate_kwargs = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
        logger.info("Decoding: sampling (Thinking model) temperature=0.6, top_p=0.95, top_k=20")
    else:
        generate_kwargs = {"do_sample": False}
        logger.info("Decoding: greedy (Instruct model)")

    model, processor = load_model(model_path, use_flash_attention)
    logger.info(f"Model loaded on device: {model.device}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total_count = len(data)
    if test_samples is not None:
        data = data[:test_samples]
        logger.info(f"Test mode: {test_samples} / {total_count} samples")
    else:
        logger.info(f"Full benchmark: {total_count} samples")

    existing_results = None
    failed_indices = None
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        failed_indices = {
            i for i, r in enumerate(existing_results) if "error" in r or r.get("prediction") is None
        }
        logger.info(f"Resume mode: {len(failed_indices)} failed samples to retry")

    results = existing_results if existing_results else [None] * len(data)
    success_count, error_count = 0, 0
    start_time = datetime.now()

    pbar = tqdm(data, desc="Processing", unit="sample")
    for idx, item in enumerate(pbar):
        if failed_indices is not None and idx not in failed_indices:
            continue

        image_filename = item.get("image_path", "")
        question = item.get("question", "")
        gt_bbox = item.get("bbox")
        full_image_path = get_image_path(image_filename, image_root)

        result_item = {
            **item,
            "prediction": None,
            "pred_bbox": None,
            "iou": None,
            "score": False,
            "raw_response": "",
        }

        if not os.path.exists(full_image_path):
            result_item["error"] = f"Image not found: {full_image_path}"
            logger.warning(f"[{idx}] Image not found: {full_image_path}")
            results[idx] = result_item
            error_count += 1
            continue

        try:
            img_width, img_height = get_image_dimensions(full_image_path)
            prompt = build_prompt(question)

            prediction, pred_bbox, raw_response, retry_count = None, None, "", 0
            for attempt in range(max_retries):
                raw_response = run_inference(
                    model, processor, full_image_path, prompt, max_new_tokens, **generate_kwargs
                )
                prediction, pred_bbox = parse_response(raw_response)
                if is_valid_prediction(prediction, pred_bbox):
                    break
                retry_count += 1
                if attempt < max_retries - 1:
                    logger.debug(
                        f"[{idx}] Retry {retry_count}/{max_retries - 1}: invalid prediction/bbox"
                    )

            iou = calculate_iou(gt_bbox, pred_bbox, img_width, img_height)

            result_item["prediction"] = prediction
            result_item["pred_bbox"] = pred_bbox
            result_item["iou"] = iou
            result_item["score"] = compute_score(prediction, item.get("answer"))
            result_item["raw_response"] = raw_response
            result_item["retry_count"] = retry_count

            results[idx] = result_item
            success_count += 1
            logger.debug(f"[{idx}] Prediction: {prediction} | IoU: {iou}")

        except Exception as e:
            result_item["error"] = str(e)
            logger.error(f"[{idx}] Error: {e}")
            results[idx] = result_item
            error_count += 1

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = (datetime.now() - start_time).total_seconds()
    valid_results = [r for r in results if r is not None]
    mcq_stats = calculate_mcq_accuracy(valid_results)

    valid_ious = [r["iou"] for r in valid_results if r.get("iou") is not None and r.get("score")]
    avg_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    iou50_correct = sum(
        1 for r in valid_results if r.get("score") and r.get("iou") is not None and r["iou"] >= 0.5
    )
    acc_at_iou50 = (
        round(iou50_correct / mcq_stats["total_evaluated"] * 100, 2)
        if mcq_stats["total_evaluated"]
        else 0.0
    )

    logger.info("-" * 60)
    logger.info("Benchmark Complete!")
    logger.info(f"  Success: {success_count} | Errors: {error_count} | Elapsed: {elapsed:.1f}s")
    logger.info(
        f"  MCQ Accuracy: {mcq_stats['accuracy']}% ({mcq_stats['correct']}/{mcq_stats['total_evaluated']})"
    )
    logger.info(f"  Avg IoU (MCQ-correct only): {round(avg_iou, 4)}")
    logger.info(f"  Acc@.5IoU: {acc_at_iou50}%")
    logger.info(f"  Results: {output_path}")

    return {
        "total": len(data),
        "success": success_count,
        "errors": error_count,
        "mcq_accuracy": mcq_stats["accuracy"],
        "mcq_correct": mcq_stats["correct"],
        "mcq_evaluated": mcq_stats["total_evaluated"],
        "avg_iou": round(avg_iou, 4),
        "acc_at_iou50": acc_at_iou50,
        "elapsed_time": round(elapsed, 1),
        "output_path": output_path,
        "log_path": log_file,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MultihopSpatial Benchmark - Qwen3-VL (Transformers backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_qwen.py --test_samples 5
  python benchmark_qwen.py --output results_qwen3vl_4b.json
  python benchmark_qwen.py --model_path /path/to/local/checkpoint
        """,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"HF Hub repo id or local path to a Qwen3-VL checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Local cache dir for the auto-downloaded dataset (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: results_{model}_{timestamp}.json)",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=None,
        help="Run on only N samples (omit for the full 4500-sample benchmark)",
    )
    parser.add_argument(
        "--no_flash_attention", action="store_true", help="Disable Flash Attention 2"
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--resume", action="store_true", help="Resume, retrying only failed samples"
    )
    args = parser.parse_args()

    json_path, image_root = download_dataset(args.data_dir)

    if args.output is None:
        os.makedirs("result", exist_ok=True)
        model_safe = os.path.basename(args.model_path).replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"result/{model_safe}_{ts}.json"
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    stats = run_benchmark(
        json_path=json_path,
        image_root=image_root,
        output_path=args.output,
        model_path=args.model_path,
        test_samples=args.test_samples,
        use_flash_attention=not args.no_flash_attention,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        resume=args.resume,
    )

    print("\n" + "=" * 60)
    print("Benchmark Completed!")
    print(
        f"MCQ Accuracy: {stats['mcq_accuracy']}% ({stats['mcq_correct']}/{stats['mcq_evaluated']})"
    )
    print(f"Average IoU: {stats['avg_iou']} (MCQ correct only)")
    print(f"Acc@.5IoU: {stats['acc_at_iou50']}%")
    print(f"Elapsed Time: {stats['elapsed_time']}s")
    print(f"Results: {stats['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
