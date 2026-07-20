#!/usr/bin/env python3
"""
MultihopSpatial Benchmark - Gemini (Async Parallel)
====================================================================
Evaluates a Google Gemini model on the MultihopSpatial benchmark using
async parallel requests for fast throughput.

The test set (question/answer/bbox JSON + images) is downloaded
automatically on first run and cached locally - no manual setup needed.

Dataset: https://huggingface.co/datasets/etri-vilab/MultihopSpatial

Authentication
--------------
Reads the API key from the GEMINI_API_KEY environment variable:
    export GEMINI_API_KEY="AIza..."

The key is never taken as a CLI argument, so it can't leak into your
shell history. For the same reason, don't paste one into this file -
use `export`, a .env loaded by your shell profile, or a CI secret store.

Coordinate System (important - differs from Claude/GPT!)
----------------------------------------------------------
Gemini's spatial-understanding convention returns bounding boxes as
normalized **[y1, x1, y2, x2]** ("yxyx"), NOT [x1, y1, x2, y2] like
Claude/GPT/most other VLMs. This is documented in Google's own
cookbook: https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb

This script's prompt explicitly asks for the yxyx convention (matching
what Gemini actually returns), and `calculate_iou` interprets the
returned bbox as [y1, x1, y2, x2] accordingly. Using the xyxy
interpretation here (as some earlier internal scripts did) silently
produces near-zero IoU for every sample - it's an easy trap.

Despite the prompt explicitly requesting normalized (0-1) coordinates,
Gemini also sometimes ignores that and returns its other native scale
(0-1000) instead - the same quirk Qwen3-VL has. `parse_response()`
auto-detects and rescales any bbox with a value > 1.

Usage
-----
    # Quick smoke test on 5 samples
    python benchmark_gemini.py --test_samples 5

    # Full benchmark (4500 samples, 10 concurrent requests)
    python benchmark_gemini.py --output results_gemini.json --concurrency 10

    # Resume a run that was interrupted or had failed samples
    python benchmark_gemini.py --output results_gemini.json --resume

Note on rate limits
--------------------
`--concurrency` bounds how many requests are in flight at once, not the
request *rate* - free-tier Gemini API keys are capped at a low
requests-per-minute quota (e.g. 5/min for gemini-3-flash), which even
`--concurrency 2` can exceed in bursts. On a free-tier key, pass a low
concurrency (e.g. `--concurrency 1`) and expect retries/waits; on a
paid-tier key (the expected setup for a full 4500-sample run) this
isn't a concern.

Requirements
------------
    pip install google-genai huggingface_hub tqdm pillow
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

from google import genai
from google.genai import types
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "gemini-3-flash-preview"
DATASET_REPO_ID = "etri-vilab/MultihopSpatial"
DEFAULT_DATA_DIR = "data/multihopspatial"
DEFAULT_CONCURRENCY = 10
MAX_RETRIES = 5  # 503 errors get exponential backoff

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
    """Prompts for normalized (0-1) [y1, x1, y2, x2] bbox coordinates - Gemini's
    native convention (see module docstring)."""
    clean_q = remove_tags(question)
    return f"""{clean_q}

Please respond in the following format:
Answer: (your choice, e.g., "(a) object name")
Bounding Box: {{"bbox_2d": [y1, x1, y2, x2]}}

Important: Use NORMALIZED coordinates (0.0 to 1.0), in [y1, x1, y2, x2] order
(top-left y, top-left x, bottom-right y, bottom-right x).
Example: {{"bbox_2d": [0.1, 0.25, 0.8, 0.75]}}"""


def get_image_path(image_filename: str, image_root: str) -> str:
    while image_filename.startswith("/") or image_filename.startswith("\\"):
        image_filename = image_filename[1:]
    return os.path.join(image_root, image_filename)


def get_image_resolution(image_path: str) -> str:
    with Image.open(image_path) as img:
        width, height = img.size
    return f"{width}x{height}"


# =============================================================================
# BBox / IoU Utilities
# =============================================================================
# NOTE: pred_bbox here is [y1, x1, y2, x2] normalized (0-1) - Gemini's native
# convention. Do NOT reuse the xyxy calculate_iou from the Claude/GPT scripts.


def is_valid_normalized_bbox(bbox: Optional[list]) -> bool:
    if bbox is None or len(bbox) != 4:
        return False
    return all(0 <= v <= 1.0 for v in bbox)


def calculate_iou(
    bbox_gt_xywh: list, bbox_pred_yxyx_norm: list, img_width: int, img_height: int
) -> Optional[float]:
    """gt: [x, y, w, h] pixels. pred: [y1, x1, y2, x2] normalized 0-1 (Gemini convention).

    Reference: https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb
    """
    if (
        bbox_gt_xywh is None
        or bbox_pred_yxyx_norm is None
        or len(bbox_gt_xywh) != 4
        or len(bbox_pred_yxyx_norm) != 4
    ):
        return None
    try:
        gx, gy, gw, gh = [float(v) for v in bbox_gt_xywh]
        if gw <= 0 or gh <= 0:
            return None
        gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh

        y1, x1, y2, x2 = bbox_pred_yxyx_norm
        px1, py1 = x1 * img_width, y1 * img_height
        px2, py2 = x2 * img_width, y2 * img_height

        if px2 <= px1 or py2 <= py1:
            return 0.0

        ix1, iy1 = max(gx1, px1), max(gy1, py1)
        ix2, iy2 = min(gx2, px2), min(gy2, py2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

        gt_area = gw * gh
        pred_area = (px2 - px1) * (py2 - py1)
        union = gt_area + pred_area - inter
        if union <= 0:
            return 0.0
        return round(inter / union, 4)
    except (TypeError, ValueError):
        return None


def parse_response(response_text: str) -> tuple[Optional[str], Optional[list]]:
    """Returns (prediction, pred_bbox) where pred_bbox is [y1, x1, y2, x2] normalized.

    Despite the prompt explicitly requesting normalized (0-1) coordinates,
    Gemini sometimes ignores that and returns its other native scale (0-1000)
    instead - the same quirk Qwen3-VL has. Any bbox with a value > 1 is
    auto-converted from 0-1000 to 0-1 before being returned.
    """
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

    if pred_bbox is not None and any(v > 1 for v in pred_bbox):
        pred_bbox = [v / 1000.0 for v in pred_bbox]

    return prediction, pred_bbox


def compute_score(prediction: Optional[str], answer: Optional[str]) -> bool:
    pred_letter = extract_choice_letter(prediction)
    answer_letter = extract_choice_letter(answer)
    return pred_letter is not None and pred_letter == answer_letter


def extract_retry_delay(error_text: str) -> Optional[float]:
    """Extracts the server-suggested retry delay (seconds) from a 429 error message.

    Free-tier keys hit low per-minute quotas quickly; the API tells us exactly
    how long to wait ('retryDelay': '55s' or 'retry in 55.6s') - respecting
    that is far more reliable than a fixed exponential backoff, which can
    expire before the quota window actually resets.
    """
    m = re.search(r"'retryDelay':\s*'([\d.]+)s'", error_text)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in ([\d.]+)s", error_text)
    if m:
        return float(m.group(1))
    return None


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
    logger = logging.getLogger("benchmark_gemini")
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
# Gemini API Call (Async)
# =============================================================================


async def call_gemini_api_async(
    client: genai.Client,
    model_name: str,
    prompt: str,
    image_path: str,
    semaphore: asyncio.Semaphore,
    thinking: bool = False,
) -> tuple[str, Optional[str]]:
    async with semaphore:
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            ext = Path(image_path).suffix.lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(ext, "image/jpeg")

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ]

            config = None
            if thinking:
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=8000, include_thoughts=True
                    )
                )

            loop = asyncio.get_event_loop()
            if config:
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_name, contents=contents, config=config
                    ),
                )
            else:
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(model=model_name, contents=contents),
                )

            text_parts, thinking_parts = [], []
            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "thought") and part.thought:
                        thinking_parts.append(part.text)
                    elif part.text:
                        text_parts.append(part.text)

            return "\n".join(text_parts), ("\n".join(thinking_parts) if thinking_parts else None)

        except Exception as e:
            return f"[ERROR]: {e}", None


async def process_single_item(
    idx: int,
    item: dict,
    client: genai.Client,
    model_name: str,
    image_root: str,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    max_retries: int = MAX_RETRIES,
    thinking: bool = False,
) -> dict:
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
        "retries": 0,
    }

    if not os.path.exists(full_image_path):
        result_item["error"] = f"Image not found: {full_image_path}"
        logger.warning(f"[{idx}] Image not found: {full_image_path}")
        return result_item

    try:
        width, height = map(int, get_image_resolution(full_image_path).split("x"))
        prompt = build_prompt(question)

        for attempt in range(max_retries + 1):
            raw_response, thinking_summary = await call_gemini_api_async(
                client, model_name, prompt, full_image_path, semaphore, thinking
            )

            if raw_response.startswith("[ERROR]:"):
                if attempt < max_retries:
                    result_item["retries"] = attempt + 1
                    server_delay = extract_retry_delay(raw_response)
                    wait_time = server_delay + 1 if server_delay is not None else 2**attempt
                    logger.debug(
                        f"[{idx}] API error, retry {attempt + 1}/{max_retries} after {wait_time}s"
                        f"{' (server-suggested)' if server_delay is not None else ''}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                result_item["error"] = raw_response
                return result_item

            prediction, pred_bbox = parse_response(raw_response)
            needs_retry = (
                prediction is None or pred_bbox is None or not is_valid_normalized_bbox(pred_bbox)
            )

            if needs_retry and attempt < max_retries:
                result_item["retries"] = attempt + 1
                await asyncio.sleep(1)
                continue

            result_item["prediction"] = prediction
            result_item["pred_bbox"] = pred_bbox
            result_item["raw_response"] = raw_response
            result_item["score"] = compute_score(prediction, result_item["answer"])
            if pred_bbox and is_valid_normalized_bbox(pred_bbox):
                result_item["iou"] = calculate_iou(gt_bbox, pred_bbox, width, height)
            if thinking_summary:
                result_item["thinking_summary"] = thinking_summary
            if needs_retry:
                result_item["retry_failed"] = "invalid prediction/bbox after all retries"
            break

    except Exception as e:
        result_item["error"] = str(e)
        logger.error(f"[{idx}] Error: {e}")

    return result_item


# =============================================================================
# Main Benchmark Loop
# =============================================================================


async def run_benchmark_async(
    json_path: str,
    image_root: str,
    output_path: str,
    api_key: str,
    test_samples: Optional[int] = None,
    model_name: str = MODEL_NAME,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_retries: int = MAX_RETRIES,
    resume: bool = False,
    thinking: bool = False,
) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path.replace(".json", f"_{timestamp}.log")
    logger = setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("MultihopSpatial Benchmark - Gemini (Async)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    if thinking:
        logger.info("Thinking: enabled")
    logger.info(f"Concurrency: {concurrency}")

    client = genai.Client(api_key=api_key)

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

    semaphore = asyncio.Semaphore(concurrency)
    results = existing_results if existing_results is not None else [None] * len(data)
    save_lock = asyncio.Lock()

    async def process_and_save(idx, item):
        result = await process_single_item(
            idx, item, client, model_name, image_root, semaphore, logger, max_retries, thinking
        )
        results[idx] = result
        async with save_lock:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        return result

    if failed_indices is not None:
        tasks = [process_and_save(idx, data[idx]) for idx in failed_indices]
    else:
        tasks = [process_and_save(idx, item) for idx, item in enumerate(data)]

    logger.info(f"Processing {len(tasks)} samples with concurrency {concurrency}...")
    start_time = datetime.now()
    await tqdm_asyncio.gather(*tasks, desc="Processing")
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
        description="MultihopSpatial Benchmark - Gemini (Async)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_gemini.py --test_samples 5
  python benchmark_gemini.py --output results_gemini.json --concurrency 20
  python benchmark_gemini.py --model gemini-3-pro-preview --thinking
        """,
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME, help=f"Gemini model name (default: {MODEL_NAME})"
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
        "--thinking", action="store_true", help="Enable thinking mode (thinking_budget=8000)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a key at https://aistudio.google.com/ and `export GEMINI_API_KEY=...`"
        )

    json_path, image_root = download_dataset(args.data_dir)

    if args.output is None:
        os.makedirs("result", exist_ok=True)
        model_safe = args.model.split("/")[-1].replace(" ", "_")
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
        args.thinking,
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
