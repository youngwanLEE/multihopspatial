#!/usr/bin/env python3
"""
MultihopSpatial Benchmark - GPT (Async Parallel)
====================================================================
Evaluates an OpenAI GPT model on the MultihopSpatial benchmark using
async parallel requests for fast throughput.

The test set (question/answer/bbox JSON + images) is downloaded
automatically on first run and cached locally - no manual setup needed.

Dataset: https://huggingface.co/datasets/etri-vilab/MultihopSpatial

Authentication
--------------
Reads the API key from the OPENAI_API_KEY environment variable:
    export OPENAI_API_KEY="sk-..."

The key is never taken as a CLI argument, so it can't leak into your
shell history. For the same reason, don't paste one into this file -
use `export`, a .env loaded by your shell profile, or a CI secret store.

Coordinate System
------------------
GPT is prompted for normalized 0-1 [x1, y1, x2, y2] coordinates via a
`{"bbox_2d": [...]}` JSON hint, and reliably follows that convention.

Usage
-----
    # Quick smoke test on 5 samples
    python benchmark_gpt.py --test_samples 5

    # Full benchmark (4500 samples, 10 concurrent requests)
    python benchmark_gpt.py --output results_gpt.json --concurrency 10

    # A reasoning-effort model (e.g. gpt-5.2)
    python benchmark_gpt.py --model gpt-5.2 --reasoning_effort high

    # Resume a run that was interrupted or had failed samples
    python benchmark_gpt.py --output results_gpt.json --resume

Requirements
------------
    pip install openai huggingface_hub tqdm pillow
"""

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from openai import AsyncOpenAI
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "gpt-5.2"
DATASET_REPO_ID = "etri-vilab/MultihopSpatial"
DEFAULT_DATA_DIR = "data/multihopspatial"
DEFAULT_CONCURRENCY = 5
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


# =============================================================================
# BBox / IoU Utilities (normalized 0-1 xyxy - correct for GPT)
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
    logger = logging.getLogger("benchmark_gpt")
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
# Async API Call
# =============================================================================


async def call_gpt_api_async(
    client: AsyncOpenAI,
    prompt: str,
    image_path: str,
    model: str,
    semaphore: asyncio.Semaphore,
    reasoning_effort: Optional[str] = None,
) -> tuple[str, Optional[int]]:
    async with semaphore:
        base64_image = encode_image_to_base64(image_path)
        ext = Path(image_path).suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")

        kwargs = dict(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=32000 if reasoning_effort else 4000,
        )
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content

        reasoning_tokens = None
        if response.usage and hasattr(response.usage, "completion_tokens_details"):
            details = response.usage.completion_tokens_details
            if details and hasattr(details, "reasoning_tokens"):
                reasoning_tokens = details.reasoning_tokens

        if content is None:
            if hasattr(message, "refusal") and message.refusal:
                return f"[REFUSAL]: {message.refusal}", reasoning_tokens
            return "", reasoning_tokens
        return content, reasoning_tokens


async def process_single_item(
    client: AsyncOpenAI,
    item: dict,
    image_root: str,
    model: str,
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
    logger: logging.Logger,
    reasoning_effort: Optional[str] = None,
    max_retries: int = MAX_RETRIES,
) -> dict:
    image_filename = item.get("image_path", "")
    question = item.get("question", "")
    full_image_path = get_image_path(image_filename, image_root)

    if not os.path.exists(full_image_path):
        logger.warning(f"[{idx + 1}/{total}] Image not found: {full_image_path}")
        return {
            **item,
            "prediction": None,
            "pred_bbox": None,
            "iou": None,
            "score": False,
            "error": f"Image not found: {full_image_path}",
        }

    width, height = map(int, get_image_resolution(full_image_path).split("x"))
    prompt = build_prompt(question)

    last_response, retry_reasons = None, []
    for attempt in range(max_retries):
        try:
            response_text, reasoning_tokens = await call_gpt_api_async(
                client, prompt, full_image_path, model, semaphore, reasoning_effort
            )
            last_response = response_text
            prediction, pred_bbox = parse_response(response_text)

            needs_retry = (
                prediction is None or pred_bbox is None or not is_valid_normalized_bbox(pred_bbox)
            )
            if needs_retry:
                retry_reasons.append(f"Attempt {attempt + 1}: invalid prediction/bbox")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                    continue

            gt_bbox = item.get("bbox")
            iou = calculate_iou(gt_bbox, pred_bbox, width, height)
            result_item = {
                **item,
                "prediction": prediction,
                "pred_bbox": pred_bbox,
                "iou": iou,
                "score": compute_score(prediction, item.get("answer")),
                "raw_response": response_text,
                "retries": attempt,
            }
            if reasoning_tokens:
                result_item["reasoning_tokens"] = reasoning_tokens
            if retry_reasons:
                result_item["retry_reasons"] = retry_reasons
            return result_item

        except Exception as e:
            retry_reasons.append(f"Attempt {attempt + 1}: Exception - {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue

    logger.warning(f"[{idx + 1}/{total}] All {max_retries} retries failed")
    prediction, pred_bbox = parse_response(last_response) if last_response else (None, None)
    gt_bbox = item.get("bbox")
    iou = (
        calculate_iou(gt_bbox, pred_bbox, width, height)
        if pred_bbox and is_valid_normalized_bbox(pred_bbox)
        else None
    )
    return {
        **item,
        "prediction": prediction,
        "pred_bbox": pred_bbox,
        "iou": iou,
        "score": compute_score(prediction, item.get("answer")),
        "raw_response": last_response or "",
        "retries": max_retries,
        "retry_reasons": retry_reasons,
        "error": f"All {max_retries} retries failed",
    }


# =============================================================================
# Main Benchmark Loop
# =============================================================================


async def run_benchmark_async(
    json_path: str,
    image_root: str,
    output_path: str,
    api_key: str,
    test_samples: Optional[int] = None,
    model: str = MODEL_NAME,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_retries: int = MAX_RETRIES,
    resume: bool = False,
    reasoning_effort: Optional[str] = None,
) -> dict:
    log_file = output_path.replace(".json", ".log")
    logger = setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("MultihopSpatial Benchmark - GPT (Async)")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    if reasoning_effort:
        logger.info(f"Reasoning Effort: {reasoning_effort}")
    logger.info(f"Concurrency: {concurrency}")

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

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
            existing_data = json.load(f)
        existing_results = (
            existing_data["results"]
            if isinstance(existing_data, dict) and "results" in existing_data
            else existing_data
        )
        failed_indices = [
            i
            for i, r in enumerate(existing_results)
            if r is None or "error" in r or r.get("prediction") is None
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

    results = existing_results if existing_results is not None else [None] * len(data)
    metadata = {
        "model_name": model,
        "input_json": json_path,
        "test_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_lock = asyncio.Lock()

    async def process_and_save(idx, item):
        result = await process_single_item(
            client,
            item,
            image_root,
            model,
            semaphore,
            idx,
            len(data),
            logger,
            reasoning_effort,
            max_retries,
        )
        results[idx] = result
        async with save_lock:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"metadata": metadata, "results": results}, f, ensure_ascii=False, indent=2
                )
        return result

    if failed_indices is not None:
        tasks = [process_and_save(idx, data[idx]) for idx in failed_indices]
    else:
        tasks = [process_and_save(idx, item) for idx, item in enumerate(data)]

    logger.info(f"Processing {len(tasks)} samples with {concurrency} concurrent requests...")
    start_time = datetime.now()
    await tqdm_asyncio.gather(*tasks, desc="Processing", unit="sample")
    elapsed = (datetime.now() - start_time).total_seconds()

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
    logger.info(f"  Elapsed: {elapsed:.1f}s")
    logger.info(f"  Results: {output_path}")

    return {
        "total": len(data),
        "mcq_accuracy": mcq_stats["accuracy"],
        "mcq_correct": mcq_stats["correct"],
        "mcq_evaluated": mcq_stats["total_evaluated"],
        "avg_iou": round(avg_iou, 4),
        "acc_at_iou50": acc50,
        "elapsed_time": round(elapsed, 1),
        "output_path": output_path,
        "log_path": log_file,
    }


def run_benchmark(*args, **kwargs) -> dict:
    return asyncio.run(run_benchmark_async(*args, **kwargs))


def main():
    parser = argparse.ArgumentParser(
        description="MultihopSpatial Benchmark - GPT (Async Parallel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_gpt.py --test_samples 5
  python benchmark_gpt.py --output results_gpt.json --concurrency 10
  python benchmark_gpt.py --model gpt-5.2 --reasoning_effort high
        """,
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME, help=f"GPT model name (default: {MODEL_NAME})"
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
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES)
    parser.add_argument(
        "--resume", action="store_true", help="Resume, retrying only failed samples"
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort for thinking models (e.g. gpt-5.2)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY environment variable is not set. "
            "Get a key at https://platform.openai.com/ and `export OPENAI_API_KEY=...`"
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
        args.concurrency,
        args.max_retries,
        args.resume,
        args.reasoning_effort,
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
