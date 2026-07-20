#!/usr/bin/env python3
"""
MultihopSpatial Benchmark - Qwen3-VL (vLLM Batched Inference)
====================================================================
Evaluates a Qwen3-VL model on the MultihopSpatial benchmark using vLLM
for fast batched inference (PagedAttention + continuous batching).
Much faster than the plain-transformers `benchmark_qwen.py` script.

Both the test set (question/answer/bbox JSON + images) and the model
checkpoint are downloaded automatically on first run and cached locally
(via `huggingface_hub` / vLLM's own HF integration) - no manual setup.

Dataset : https://huggingface.co/datasets/etri-vilab/MultihopSpatial
Model   : https://huggingface.co/etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct

Coordinate System
------------------
Qwen3-VL natively outputs bounding boxes on a 0-1000 relative scale
(per the official Qwen3-VL 2D grounding cookbook). Any bbox value > 1
returned by the model is auto-converted to 0-1 normalized coordinates
before computing IoU.

Sampling
--------
By default, sampling parameters are loaded from the checkpoint's
`generation_config.json` (temperature/top_p/top_k). Pass `--greedy` to
force deterministic greedy decoding (temperature=0) instead.

Usage
-----
    # Quick smoke test on 5 samples (auto-downloads data + model)
    python benchmark_qwen_vllm.py --test_samples 5

    # Full benchmark (4500 samples), single GPU
    python benchmark_qwen_vllm.py --output results_qwen3vl_4b

    # Multi-GPU tensor parallelism (e.g. for the 32B checkpoint)
    python benchmark_qwen_vllm.py \\
        --model_path /path/to/32B/checkpoint \\
        --gpus 0,1,2,3,4,5,6,7 --max_model_len 32768

    # Force greedy decoding instead of the checkpoint's sampling config
    python benchmark_qwen_vllm.py --greedy

    # Resume a run that was interrupted or had failed samples
    python benchmark_qwen_vllm.py --output results_qwen3vl_4b --resume

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
import subprocess
from datetime import datetime
from typing import Optional

from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_PATH = "etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct"
DATASET_REPO_ID = "etri-vilab/MultihopSpatial"
DEFAULT_DATA_DIR = "data/multihopspatial"


def _has_nvlink() -> bool:
    """Detects NVLink among visible GPUs via `nvidia-smi topo -m`.

    On PCIe-only systems (no NVLink), vLLM's tensor-parallel P2P transport
    can hang during NCCL communicator setup. If no NVLink is detected we
    disable P2P and fall back to socket-based communication.
    """
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False
        lines = result.stdout.strip().splitlines()
        header_idx = next((i for i, line in enumerate(lines) if "GPU0" in line), None)
        if header_idx is None:
            return False
        visible_set = {int(x.strip()) for x in visible.split(",")} if visible else None
        for line in lines[header_idx + 1 :]:
            if not line.startswith("GPU"):
                break
            parts = line.split()
            try:
                row_gpu = int(parts[0].replace("GPU", ""))
            except ValueError:
                continue
            if visible_set is not None and row_gpu not in visible_set:
                continue
            if any(cell.startswith("NV") for cell in parts[1:]):
                return True
    except Exception:
        pass
    return False


if "NCCL_P2P_DISABLE" not in os.environ and not _has_nvlink():
    os.environ["NCCL_P2P_DISABLE"] = "1"
    logging.getLogger(__name__).info(
        "No NVLink detected — setting NCCL_P2P_DISABLE=1 to avoid PCIe P2P hangs."
    )

from vllm import LLM, SamplingParams  # noqa: E402  (import after NCCL env var is set)

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
    clean_q = remove_tags(question)
    return f"""{clean_q}

Please respond in the following format:
Answer: (your choice, e.g., "(a) object name")
Bounding Box: [x1, y1, x2, y2]

where [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner."""


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
    """Extracts the MCQ answer and a 0-1 normalized bbox from the raw model output."""
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


def calculate_full_metrics(results: list) -> dict:
    total, correct = 0, 0
    for r in results:
        answer_letter = extract_choice_letter(r.get("answer"))
        pred_letter = extract_choice_letter(r.get("prediction"))
        if answer_letter and pred_letter:
            total += 1
            if answer_letter == pred_letter:
                correct += 1
    accuracy = round(correct / total * 100, 2) if total else 0.0

    valid_ious = [r["iou"] for r in results if r.get("iou") is not None and r.get("score")]
    avg_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    iou50_correct = sum(
        1 for r in results if r.get("score") and r.get("iou") is not None and r["iou"] >= 0.5
    )
    acc_at_iou50 = round(iou50_correct / total * 100, 2) if total else 0.0

    return {
        "total_evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
        "acc_at_iou50": acc_at_iou50,
        "avg_iou": round(avg_iou, 4),
    }


def log_metrics_table(logger, valid_results: list) -> dict:
    overall = calculate_full_metrics(valid_results)
    logger.info("Overall Performance:")
    logger.info(f"  Evaluated: {overall['total_evaluated']}")
    logger.info(
        f"  MCQ Acc: {overall['accuracy']}% ({overall['correct']}/{overall['total_evaluated']})"
    )
    logger.info(f"  Acc@.5IoU: {overall['acc_at_iou50']}%")
    logger.info(f"  Avg IoU: {overall['avg_iou']} (MCQ correct only)")

    hops = sorted(set(r.get("hop", "") for r in valid_results if r.get("hop")))
    views = sorted(set(r.get("view", "") for r in valid_results if r.get("view")))
    if not hops:
        return overall

    logger.info("-" * 60)
    logger.info(f"  {'Split':<20} {'N':>6} {'MCQ Acc':>9} {'Acc@.5IoU':>10} {'Avg IoU':>9}")
    for hop in hops:
        hop_results = [r for r in valid_results if r.get("hop") == hop]
        m = calculate_full_metrics(hop_results)
        logger.info(
            f"  {hop:<20} {m['total_evaluated']:>6} {m['accuracy']:>8.2f}% {m['acc_at_iou50']:>9.2f}% {m['avg_iou']:>8.4f}"
        )
        for view in views:
            hv_results = [r for r in hop_results if r.get("view") == view]
            if not hv_results:
                continue
            hv = calculate_full_metrics(hv_results)
            logger.info(
                f"    {hop}/{view:<16} {hv['total_evaluated']:>6} {hv['accuracy']:>8.2f}% {hv['acc_at_iou50']:>9.2f}% {hv['avg_iou']:>8.4f}"
            )

    return overall


# =============================================================================
# Sampling Parameters
# =============================================================================


def build_sampling_params(
    model_path: str,
    max_tokens: int,
    greedy: bool,
    temperature_override: Optional[float],
    seed: Optional[int],
    logger: logging.Logger,
) -> SamplingParams:
    """Priority: --greedy > --temperature > generation_config.json > fallback (t=0)."""
    params = {}
    if greedy:
        params["temperature"] = 0.0
        logger.info("Sampling: --greedy -> temperature=0.0")
    elif temperature_override is not None:
        params["temperature"] = temperature_override
        logger.info(f"Sampling: --temperature {temperature_override} (CLI override)")
    else:
        config_path = (
            os.path.join(model_path, "generation_config.json")
            if os.path.isdir(model_path)
            else None
        )
        config = None
        if config_path and os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)
        if config:
            if config.get("do_sample") is False:
                params["temperature"] = 0.0
            for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
                if key in config and key not in params:
                    params[key] = config[key]
            logger.info("Sampling: loaded from generation_config.json")
        else:
            params = {"temperature": 0.0}
            logger.info("Sampling: no generation_config.json found, defaulting to greedy")

    params["max_tokens"] = max_tokens
    if seed is not None:
        params["seed"] = seed
    logger.info(f"  -> {params}")
    return SamplingParams(**params)


# =============================================================================
# Logging
# =============================================================================


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("benchmark_qwen_vllm")
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
# Main Benchmark Loop
# =============================================================================


def run_benchmark(
    json_path: str,
    image_root: str,
    output_path: str,
    model_path: str,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_new_tokens: int = 8192,
    greedy: bool = False,
    temperature_override: Optional[float] = None,
    seed: Optional[int] = None,
    test_samples: Optional[int] = None,
    max_retries: int = 3,
    resume: bool = False,
    batch_size: int = 100,
    enforce_eager: bool = False,
    max_num_seqs: Optional[int] = None,
) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(output_path)
    log_file = os.path.join(output_dir, f"log_{timestamp}.log")
    logger = setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("MultihopSpatial Benchmark - Qwen3-VL (vLLM)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"JSON: {json_path}")
    logger.info(f"Image Root: {image_root}")
    logger.info(f"Tensor Parallel Size: {tensor_parallel_size}")
    logger.info(f"Max Model Len: {max_model_len}")
    logger.info(f"Max New Tokens: {max_new_tokens}")

    logger.info("Loading vLLM model...")
    start_load = datetime.now()
    llm_kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        allowed_local_media_path="/",
        enforce_eager=enforce_eager,
    )
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    llm = LLM(**llm_kwargs)
    sampling_params = build_sampling_params(
        model_path, max_new_tokens, greedy, temperature_override, seed, logger
    )
    logger.info(f"vLLM model loaded in {(datetime.now() - start_load).total_seconds():.1f}s")

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
            i
            for i, r in enumerate(existing_results)
            if r is None
            or "error" in r
            or not is_valid_prediction(r.get("prediction"), r.get("pred_bbox"))
        }
        logger.info(f"Resume mode: {len(failed_indices)} failed samples to retry")

    results = existing_results if existing_results else [None] * len(data)

    logger.info("Preparing conversations...")
    process_indices, conversations, image_dims = [], [], {}
    for idx, item in enumerate(tqdm(data, desc="Preparing", unit="sample")):
        if failed_indices is not None and idx not in failed_indices:
            continue

        image_path = os.path.join(image_root, item.get("image_path", "").lstrip("/\\"))
        if not os.path.exists(image_path):
            results[idx] = {
                **item,
                "prediction": None,
                "pred_bbox": None,
                "iou": None,
                "score": False,
                "raw_response": "",
                "error": f"Image not found: {image_path}",
            }
            logger.warning(f"[{idx}] Image not found: {image_path}")
            continue

        try:
            img_width, img_height = get_image_dimensions(image_path)
            image_dims[idx] = (img_width, img_height)
        except Exception as e:
            results[idx] = {
                **item,
                "prediction": None,
                "pred_bbox": None,
                "iou": None,
                "score": False,
                "raw_response": "",
                "error": f"Failed to read image: {e}",
            }
            continue

        prompt = build_prompt(item.get("question", ""))
        real_image_path = os.path.realpath(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{real_image_path}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        process_indices.append(idx)
        conversations.append(messages)

    logger.info(f"Prepared {len(conversations)} conversations for inference")

    start_time = datetime.now()
    pending = list(zip(process_indices, conversations))
    retry_counts = {idx: 0 for idx in process_indices}

    def _save_and_log(label: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        completed = [r for r in results if r is not None]
        m = calculate_full_metrics(completed)
        logger.info(
            f"  {label} | Saved | Acc: {m['accuracy']}% | IoU: {m['avg_iou']:.3f} | "
            f"Acc@.5IoU: {m['acc_at_iou50']}% | Done: {len(completed)}/{len(data)}"
        )

    for attempt in range(max_retries):
        if not pending:
            break
        num_chunks = (len(pending) + batch_size - 1) // batch_size
        logger.info(
            f"Round {attempt + 1}/{max_retries}: {len(pending)} samples in {num_chunks} chunks"
        )
        next_pending = []

        for chunk_start in range(0, len(pending), batch_size):
            chunk = pending[chunk_start : chunk_start + batch_size]
            chunk_num = chunk_start // batch_size + 1
            chunk_indices = [p[0] for p in chunk]
            chunk_convos = [p[1] for p in chunk]

            logger.info(f"  Chunk {chunk_num}/{num_chunks}: Inferring {len(chunk)} samples...")
            outputs = llm.chat(messages=chunk_convos, sampling_params=sampling_params)

            for i, output in enumerate(outputs):
                idx = chunk_indices[i]
                item = data[idx]
                raw_response = output.outputs[0].text if output.outputs else ""
                answer = item.get("answer", "")
                prediction, pred_bbox = parse_response(raw_response)
                img_width, img_height = image_dims[idx]

                if is_valid_prediction(prediction, pred_bbox) or attempt == max_retries - 1:
                    gt_bbox = item.get("bbox")
                    iou = calculate_iou(gt_bbox, pred_bbox, img_width, img_height)
                    results[idx] = {
                        **item,
                        "answer": answer,
                        "prediction": prediction,
                        "pred_bbox": pred_bbox,
                        "iou": iou,
                        "score": compute_score(prediction, answer),
                        "raw_response": raw_response,
                        "retry_count": retry_counts[idx],
                    }
                else:
                    retry_counts[idx] += 1
                    next_pending.append((idx, chunk_convos[i]))

            _save_and_log(f"Chunk {chunk_num}/{num_chunks}")

        pending = next_pending

    elapsed = (datetime.now() - start_time).total_seconds()
    valid_results = [r for r in results if r is not None]
    success_count = sum(
        1 for r in valid_results if r.get("prediction") is not None and "error" not in r
    )
    error_count = len(valid_results) - success_count

    logger.info("-" * 60)
    logger.info("Benchmark Complete!")
    logger.info(
        f"  Success: {success_count} | Errors: {error_count} | "
        f"Elapsed: {elapsed:.1f}s ({elapsed / len(data):.2f}s/sample)"
    )
    logger.info("-" * 60)
    overall = log_metrics_table(logger, valid_results)
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Log saved to: {log_file}")

    return {
        **overall,
        "mcq_accuracy": overall["accuracy"],
        "mcq_correct": overall["correct"],
        "mcq_evaluated": overall["total_evaluated"],
        "total": len(data),
        "success": success_count,
        "errors": error_count,
        "elapsed_time": round(elapsed, 1),
        "output_path": output_path,
        "log_path": log_file,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MultihopSpatial Benchmark - Qwen3-VL (vLLM backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_qwen_vllm.py --test_samples 5
  python benchmark_qwen_vllm.py --output results_qwen3vl_4b
  python benchmark_qwen_vllm.py --model_path /path/to/32B --gpus 0,1,2,3,4,5,6,7 --max_model_len 32768
  python benchmark_qwen_vllm.py --greedy
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
        help="Output directory for results.json + log (default: auto-generated)",
    )
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help='Comma-separated GPU ids, e.g. "0,1,2,3". Sets CUDA_VISIBLE_DEVICES and tensor_parallel_size.',
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument(
        "--temperature", type=float, default=None, help="Explicit sampling temperature override"
    )
    parser.add_argument(
        "--greedy", action="store_true", help="Force greedy decoding (temperature=0)"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test_samples", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if args.tensor_parallel_size is None:
            args.tensor_parallel_size = len(args.gpus.split(","))
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = 1

    json_path, image_root = download_dataset(args.data_dir)

    if args.output is None:
        model_safe = os.path.basename(args.model_path).replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"result/{model_safe}_{ts}"
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    args.output = os.path.join(output_dir, "results.json")

    stats = run_benchmark(
        json_path=json_path,
        image_root=image_root,
        output_path=args.output,
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        temperature_override=args.temperature,
        seed=args.seed,
        test_samples=args.test_samples,
        max_retries=args.max_retries,
        resume=args.resume,
        batch_size=args.batch_size,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
    )

    print("\n" + "=" * 60)
    print("Benchmark Completed!")
    print(
        f"Overall MCQ Acc: {stats['mcq_accuracy']}% ({stats['mcq_correct']}/{stats['mcq_evaluated']})"
    )
    print(f"Overall Avg IoU: {stats['avg_iou']} (MCQ correct only)")
    print(f"Overall Acc@.5IoU: {stats['acc_at_iou50']}%")
    print(f"Elapsed Time: {stats['elapsed_time']}s")
    print(f"Results: {stats['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
