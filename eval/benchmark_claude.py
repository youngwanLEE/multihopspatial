#!/usr/bin/env python3
"""
MultihopSpatial Benchmark - Claude (Batch API)
====================================================================
Evaluates an Anthropic Claude model on the MultihopSpatial benchmark
using the Batch API (50% cheaper than standard, results within 24h).

The test set (question/answer/bbox JSON + images) is downloaded
automatically on first run and cached locally - no manual setup needed.

Dataset: https://huggingface.co/datasets/etri-vilab/MultihopSpatial

Authentication
--------------
Reads the API key from the ANTHROPIC_API_KEY environment variable:
    export ANTHROPIC_API_KEY="sk-ant-..."

The key is never taken as a CLI argument, so it can't leak into your
shell history. For the same reason, don't paste one into this file -
use `export`, a .env loaded by your shell profile, or a CI secret store.

Coordinate System
------------------
Claude is prompted for normalized 0-1 [x1, y1, x2, y2] coordinates via
a `{"bbox_2d": [...]}` JSON hint, and reliably follows that convention.

Usage
-----
    # Quick smoke test on 5 samples
    python benchmark_claude.py --test_samples 5

    # Full benchmark (4500 samples, batch processing)
    python benchmark_claude.py --output results_claude.json

    # Resume a run that was interrupted or had failed samples
    python benchmark_claude.py --output results_claude.json --resume

Requirements
------------
    pip install anthropic huggingface_hub tqdm pillow
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "claude-sonnet-4-5"
DATASET_REPO_ID = "etri-vilab/MultihopSpatial"
DEFAULT_DATA_DIR = "data/multihopspatial"
BATCH_CHECK_INTERVAL = 30  # seconds between batch status polls
MAX_BATCH_SIZE = 100  # requests per batch (Anthropic allows up to 10,000)
MAX_RETRIES = 3

# =============================================================================
# MCQ Answer Parsing
# =============================================================================


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
    """Downloads (or reuses a cached copy of) the MultihopSpatial test set."""
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
    cleaned = re.sub(r"</?(?:ATT|POS|REL)>", "", question)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_prompt(question: str) -> str:
    """Prompts for normalized (0-1) [x1, y1, x2, y2] bbox coordinates."""
    clean_q = remove_tags(question)
    return f"""{clean_q}

Please respond in the following format:
Answer: (your choice, e.g., "(a) object name")
Bounding Box: {{"bbox_2d": [x1, y1, x2, y2]}}

Important: Use NORMALIZED coordinates (0.0 to 1.0).
Example: {{"bbox_2d": [0.25, 0.1, 0.75, 0.8]}}"""


def encode_image_to_base64(image_path: str) -> str:
    import base64

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_path(image_filename: str, image_root: str) -> str:
    while image_filename.startswith("/") or image_filename.startswith("\\"):
        image_filename = image_filename[1:]
    return os.path.join(image_root, image_filename)


def get_image_resolution(image_path: str) -> str:
    with Image.open(image_path) as img:
        width, height = img.size
    return f"{width}x{height}"


def get_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


# =============================================================================
# BBox / IoU Utilities (normalized 0-1 xyxy - correct for Claude)
# =============================================================================


def is_valid_normalized_bbox(bbox: Optional[list]) -> bool:
    if bbox is None or len(bbox) != 4:
        return False
    return all(0 <= v <= 1.0 for v in bbox)


def calculate_iou(
    bbox_gt_xywh: list, bbox_pred_xyxy_norm: list, img_width: int, img_height: int
) -> Optional[float]:
    """gt: [x, y, w, h] pixels. pred: [x1, y1, x2, y2] normalized 0-1."""
    if (
        bbox_gt_xywh is None
        or bbox_pred_xyxy_norm is None
        or len(bbox_gt_xywh) != 4
        or len(bbox_pred_xyxy_norm) != 4
    ):
        return None
    try:
        gt_x1, gt_y1 = bbox_gt_xywh[0], bbox_gt_xywh[1]
        gt_x2, gt_y2 = gt_x1 + bbox_gt_xywh[2], gt_y1 + bbox_gt_xywh[3]

        pred_x1 = bbox_pred_xyxy_norm[0] * img_width
        pred_y1 = bbox_pred_xyxy_norm[1] * img_height
        pred_x2 = bbox_pred_xyxy_norm[2] * img_width
        pred_y2 = bbox_pred_xyxy_norm[3] * img_height

        inter_x1, inter_y1 = max(gt_x1, pred_x1), max(gt_y1, pred_y1)
        inter_x2, inter_y2 = min(gt_x2, pred_x2), min(gt_y2, pred_y2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        gt_area = bbox_gt_xywh[2] * bbox_gt_xywh[3]
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        union_area = gt_area + pred_area - inter_area
        if union_area <= 0:
            return 0.0
        return round(inter_area / union_area, 4)
    except (TypeError, ValueError):
        return None


def parse_response(response_text: str) -> tuple[Optional[str], Optional[list]]:
    pred_bbox = None
    prediction = parse_mcq_answer(response_text)

    m = re.search(
        r'["\']bbox_2d["\']\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
        response_text,
    )
    if m:
        try:
            pred_bbox = [float(m.group(i)) for i in range(1, 5)]
        except ValueError:
            pred_bbox = None
    else:
        m = re.search(
            r"\[[\s]*([\d.]+)[\s]*,[\s]*([\d.]+)[\s]*,[\s]*([\d.]+)[\s]*,[\s]*([\d.]+)[\s]*\]",
            response_text,
        )
        if m:
            try:
                pred_bbox = [float(m.group(i)) for i in range(1, 5)]
            except ValueError:
                pred_bbox = None

    return prediction, pred_bbox


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
    accuracy = round(correct / total * 100, 2) if total > 0 else 0.0
    return {"total_evaluated": total, "correct": correct, "accuracy": accuracy}


# =============================================================================
# Logging
# =============================================================================


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("benchmark_claude")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# Batch Request Builder / Processor
# =============================================================================


def build_batch_requests(
    data: list, image_root: str, model: str, logger: logging.Logger, indices: list
) -> list:
    requests = []
    for idx in tqdm(indices, desc="Building batch requests", unit="item"):
        item = data[idx]
        image_filename = item.get("image_path", "")
        question = item.get("question", "")
        full_image_path = get_image_path(image_filename, image_root)

        if not os.path.exists(full_image_path):
            logger.warning(f"[{idx}] Image not found: {full_image_path}")
            continue

        base64_image = encode_image_to_base64(full_image_path)
        mime_type = get_mime_type(full_image_path)
        prompt = build_prompt(question)

        requests.append(
            {
                "custom_id": str(idx),
                "params": {
                    "model": model,
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime_type,
                                        "data": base64_image,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                },
            }
        )
    return requests


def process_batch(
    client: anthropic.Anthropic, batch_requests: list, batch_size: int, logger: logging.Logger
) -> tuple[dict, list]:
    results_map = {}
    all_batch_ids = []
    num_batches = (len(batch_requests) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start, end = batch_num * batch_size, min((batch_num + 1) * batch_size, len(batch_requests))
        current_batch = batch_requests[start:end]
        logger.info(
            f"Processing batch {batch_num + 1}/{num_batches} ({len(current_batch)} requests)"
        )

        try:
            message_batch = client.messages.batches.create(requests=current_batch)
            batch_id = message_batch.id
            all_batch_ids.append(batch_id)
            logger.info(f"Batch created: {batch_id}")
        except Exception as e:
            logger.error(f"Failed to create batch {batch_num + 1}: {e}")
            continue

        logger.info(f"Waiting for batch to complete (checking every {BATCH_CHECK_INTERVAL}s)...")
        while True:
            batch_status = client.messages.batches.retrieve(batch_id)
            status = batch_status.processing_status
            counts = batch_status.request_counts
            logger.info(
                f"Status: {status} | Processing: {counts.processing} | Succeeded: {counts.succeeded} | Errored: {counts.errored}"
            )
            if status == "ended":
                break
            time.sleep(BATCH_CHECK_INTERVAL)

        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            if result.result.type == "succeeded":
                results_map[custom_id] = {
                    "success": True,
                    "response": result.result.message.content[0].text,
                }
            else:
                error_msg = (
                    str(result.result.error) if hasattr(result.result, "error") else "Unknown error"
                )
                results_map[custom_id] = {"success": False, "error": error_msg}

    return results_map, all_batch_ids


# =============================================================================
# Main Benchmark Loop
# =============================================================================


def run_benchmark(
    json_path: str,
    image_root: str,
    output_path: str,
    api_key: str,
    test_samples: Optional[int] = None,
    model: str = MODEL_NAME,
    batch_size: int = MAX_BATCH_SIZE,
    resume: bool = False,
) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path.replace(".json", f"_{timestamp}.log")
    logger = setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("MultihopSpatial Benchmark - Claude (Batch)")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")

    client = anthropic.Anthropic(api_key=api_key)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total_count = len(data)
    if test_samples is not None:
        data = data[:test_samples]
        logger.info(f"Test mode: {test_samples} / {total_count} samples")
    else:
        logger.info(f"Full benchmark: {total_count} samples")

    final_results = {}
    retry_counts = {}
    all_batch_ids = []

    existing_results = None
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        failed_indices = [
            i for i, r in enumerate(existing_results) if "error" in r or r.get("prediction") is None
        ]
        if not failed_indices:
            logger.info("Resume mode: No failed samples found. Nothing to do.")
            mcq_stats = calculate_mcq_accuracy(existing_results)
            valid_ious = [
                r["iou"] for r in existing_results if r.get("iou") is not None and r.get("score")
            ]
            avg_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
            iou50 = sum(
                1
                for r in existing_results
                if r.get("score") and r.get("iou") is not None and r["iou"] >= 0.5
            )
            acc50 = (
                round(iou50 / mcq_stats["total_evaluated"] * 100, 2)
                if mcq_stats["total_evaluated"]
                else 0.0
            )
            return {
                "total": len(existing_results),
                "mcq_accuracy": mcq_stats["accuracy"],
                "mcq_correct": mcq_stats["correct"],
                "mcq_evaluated": mcq_stats["total_evaluated"],
                "avg_iou": round(avg_iou, 4),
                "acc_at_iou50": acc50,
                "output_path": output_path,
                "log_path": log_file,
            }
        logger.info(f"Resume mode: {len(failed_indices)} failed samples to retry")
        for i, r in enumerate(existing_results):
            if i not in failed_indices:
                final_results[i] = r
        pending_indices = failed_indices
    else:
        pending_indices = list(range(len(data)))

    for attempt in range(MAX_RETRIES + 1):
        if not pending_indices:
            break
        logger.info("-" * 60)
        logger.info(
            f"{'Initial run' if attempt == 0 else f'Retry {attempt}/{MAX_RETRIES}'}: {len(pending_indices)} samples"
        )

        batch_requests = build_batch_requests(data, image_root, model, logger, pending_indices)
        if not batch_requests:
            logger.warning("No valid requests to process")
            break

        results_map, batch_ids = process_batch(client, batch_requests, batch_size, logger)
        all_batch_ids.extend(batch_ids)

        new_pending = []
        for idx in pending_indices:
            custom_id = str(idx)
            item = data[idx]
            image_filename = item.get("image_path", "")
            full_image_path = get_image_path(image_filename, image_root)
            if os.path.exists(full_image_path):
                width, height = map(int, get_image_resolution(full_image_path).split("x"))
            else:
                width, height = 640, 480

            if custom_id not in results_map:
                final_results[idx] = {
                    **item,
                    "prediction": None,
                    "pred_bbox": None,
                    "iou": None,
                    "score": False,
                    "retries": attempt,
                    "error": "Not processed (image not found)",
                }
                continue

            batch_result = results_map[custom_id]
            if not batch_result["success"]:
                if attempt < MAX_RETRIES:
                    new_pending.append(idx)
                    retry_counts[idx] = attempt + 1
                else:
                    final_results[idx] = {
                        **item,
                        "prediction": None,
                        "pred_bbox": None,
                        "iou": None,
                        "score": False,
                        "retries": attempt,
                        "error": batch_result["error"],
                    }
                continue

            response_text = batch_result["response"]
            prediction, pred_bbox = parse_response(response_text)

            needs_retry = (
                prediction is None or pred_bbox is None or not is_valid_normalized_bbox(pred_bbox)
            )

            if needs_retry and attempt < MAX_RETRIES:
                new_pending.append(idx)
                retry_counts[idx] = attempt + 1
            else:
                gt_bbox = item.get("bbox")
                iou = (
                    calculate_iou(gt_bbox, pred_bbox, width, height)
                    if pred_bbox and is_valid_normalized_bbox(pred_bbox)
                    else None
                )
                final_results[idx] = {
                    **item,
                    "prediction": prediction,
                    "pred_bbox": pred_bbox,
                    "iou": iou,
                    "score": compute_score(prediction, item.get("answer")),
                    "raw_response": response_text,
                    "retries": retry_counts.get(idx, 0),
                }

        pending_indices = new_pending
        logger.info(f"Completed: {len(final_results)}, Pending for retry: {len(pending_indices)}")

        interim = [
            final_results.get(i, {**data[i], "error": "Not processed"}) for i in range(len(data))
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(interim, f, ensure_ascii=False, indent=2)

    results = [
        final_results.get(i, {**data[i], "error": "Not processed"}) for i in range(len(data))
    ]
    mcq_stats = calculate_mcq_accuracy(results)
    valid_ious = [r["iou"] for r in results if r.get("iou") is not None and r.get("score")]
    avg_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    iou50 = sum(
        1 for r in results if r.get("score") and r.get("iou") is not None and r["iou"] >= 0.5
    )
    acc50 = (
        round(iou50 / mcq_stats["total_evaluated"] * 100, 2)
        if mcq_stats["total_evaluated"]
        else 0.0
    )

    logger.info("-" * 60)
    logger.info("Benchmark Complete!")
    logger.info(
        f"  MCQ Accuracy: {mcq_stats['accuracy']}% ({mcq_stats['correct']}/{mcq_stats['total_evaluated']})"
    )
    logger.info(f"  Avg IoU (MCQ-correct only): {round(avg_iou, 4)}")
    logger.info(f"  Acc@.5IoU: {acc50}%")
    logger.info(f"  Results: {output_path}")

    return {
        "total": len(data),
        "mcq_accuracy": mcq_stats["accuracy"],
        "mcq_correct": mcq_stats["correct"],
        "mcq_evaluated": mcq_stats["total_evaluated"],
        "avg_iou": round(avg_iou, 4),
        "acc_at_iou50": acc50,
        "output_path": output_path,
        "log_path": log_file,
        "batch_ids": all_batch_ids,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MultihopSpatial Benchmark - Claude (Batch API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_claude.py --test_samples 5
  python benchmark_claude.py --output results_claude.json
  python benchmark_claude.py --model claude-opus-4-5 --output results_claude_opus.json
        """,
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME, help=f"Claude model name (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Local cache dir for the auto-downloaded dataset (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--test_samples", type=int, default=None, help="Run on only N samples")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=MAX_BATCH_SIZE,
        help=f"Requests per batch (default: {MAX_BATCH_SIZE})",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume, retrying only failed samples"
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Get a key at https://console.anthropic.com/ and `export ANTHROPIC_API_KEY=...`"
        )

    json_path, image_root = download_dataset(args.data_dir)

    if args.output is None:
        os.makedirs("result", exist_ok=True)
        model_safe = args.model.replace("/", "_").replace(" ", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"result/{model_safe}_{ts}.json"
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    stats = run_benchmark(
        json_path,
        image_root,
        args.output,
        api_key,
        args.test_samples,
        args.model,
        args.batch_size,
        args.resume,
    )

    print("\n" + "=" * 60)
    print("Benchmark Completed!")
    print(
        f"MCQ Accuracy: {stats['mcq_accuracy']}% ({stats['mcq_correct']}/{stats['mcq_evaluated']})"
    )
    print(f"Average IoU: {stats['avg_iou']} (MCQ correct only)")
    print(f"Acc@.5IoU: {stats['acc_at_iou50']}%")
    print(f"Results: {stats['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
