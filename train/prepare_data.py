#!/usr/bin/env python3
"""Prepare the MultihopSpatial training set for GRPO training.

Downloads the train split (JSON + images) from the HuggingFace Hub and converts
it into the format the GRPO dataset loader expects. Run this once before
training; the download is cached, so re-running is cheap.

Usage:
    # Default: downloads to data/multihopspatial, writes the converted JSON there
    python prepare_data.py

    # Custom locations
    python prepare_data.py --data_dir /path/to/cache --output /path/to/train_grpo.json

The converted JSON is a list of records, each carrying the fields
`src/dataset/grpo_dataset.py` reads:

    image             image filename, resolved against --image_folder at train time
    conversations     LLaVA-style [human, gpt] turn pair holding the prompt/answer
    response          ground-truth MCQ answer, used by the MCQ reward
    bbox              ground-truth box as [x, y, w, h] in pixels, used by the bbox
                      reward (normalized to a 0-1000 scale by the dataset loader)
    image_resolution  "WIDTHxHEIGHT", required to normalize the bbox

A note on the question text: the source data carries both `question` (plain) and
`question_tag`, which marks the spatial relation with <ATT>/<POS>/<REL> tags.
Training uses `question_tag`. The tags are part of the prompt the released
checkpoints were trained on, so keeping them is what reproduces those results.

Note that evaluation strips the tags (see eval/benchmark_qwen.py:remove_tags),
so the models are trained with tagged prompts and evaluated on untagged ones.
That asymmetry is inherited from the original training setup and is preserved
here deliberately: changing it would no longer reproduce the released models.
"""

import argparse
import json
import os
import time

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

DATASET_REPO_ID = "etri-vilab/MultihopSpatial"
TRAIN_JSON_NAME = "multihop_train_6791.json"
DEFAULT_DATA_DIR = "data/multihopspatial"
DEFAULT_OUTPUT = "data/multihopspatial/multihop_train_6791_grpo.json"


def download_dataset(data_dir: str, max_retries: int = 10) -> tuple[str, str]:
    """Downloads (or reuses a cached copy of) the MultihopSpatial train set.

    The split is ~6,500 files, which is large enough to trip the Hub's
    per-5-minute request quota on a free account. A 429 is not fatal: already
    downloaded files are cached, so we wait out the window and resume where we
    left off. Each retry makes progress, so this converges even on a slow account.

    Returns:
        (json_path, image_root)
    """
    print(f"Fetching dataset '{DATASET_REPO_ID}' -> {data_dir} (cached after first run)")

    for attempt in range(1, max_retries + 1):
        try:
            local_dir = snapshot_download(
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                local_dir=data_dir,
                allow_patterns=[f"data/{TRAIN_JSON_NAME}", "data/images/*"],
            )
            break
        except HfHubHTTPError as err:
            is_rate_limited = err.response is not None and err.response.status_code == 429
            if not is_rate_limited or attempt == max_retries:
                raise
            wait_s = 5 * 60 + 15  # the quota window, plus a little slack
            print(
                f"\nHit the Hub rate limit (attempt {attempt}/{max_retries}). "
                f"Already-downloaded files are kept; waiting {wait_s}s to resume..."
            )
            time.sleep(wait_s)
    else:
        raise SystemExit(f"Download did not finish after {max_retries} attempts.")
    json_path = os.path.join(local_dir, "data", TRAIN_JSON_NAME)
    image_root = os.path.join(local_dir, "data", "images")
    return json_path, image_root


def to_grpo_format(records: list[dict]) -> list[dict]:
    """Converts raw MultihopSpatial records into the GRPO training format."""
    converted = []
    for record in records:
        question = record["question_tag"]
        answer = record["answer"]

        converted.append({
            "id": record.get("id"),
            "image": record["image_path"],
            "image_resolution": record["image_resolution"],
            "bbox": record["bbox"],
            "response": answer,
            "view": record.get("view"),
            "hop": record.get("hop"),
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        })
    return converted


def verify(converted: list[dict], image_root: str) -> None:
    """Fails loudly if the converted data would break training."""
    required = ("image", "conversations", "response", "bbox", "image_resolution")
    missing_fields = [
        i for i, r in enumerate(converted)
        if any(r.get(k) is None for k in required)
    ]
    if missing_fields:
        raise SystemExit(
            f"{len(missing_fields)} records are missing required fields "
            f"(first few: {missing_fields[:5]})"
        )

    missing_images = [
        r["image"] for r in converted
        if not os.path.exists(os.path.join(image_root, r["image"]))
    ]
    if missing_images:
        raise SystemExit(
            f"{len(missing_images)} referenced images are absent from {image_root} "
            f"(first few: {missing_images[:5]})"
        )

    bad_bbox = [
        r["id"] for r in converted
        if not (isinstance(r["bbox"], list) and len(r["bbox"]) == 4)
    ]
    if bad_bbox:
        raise SystemExit(f"{len(bad_bbox)} records have a malformed bbox (first few: {bad_bbox[:5]})")

    print(f"Verified {len(converted)} records: fields present, images resolve, bboxes well-formed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help=f"Where to cache the downloaded dataset (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Path for the converted training JSON (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    json_path, image_root = download_dataset(args.data_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} training records from {json_path}")

    converted = to_grpo_format(records)
    verify(converted, image_root)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(converted)} records -> {args.output}")
    print("\nTrain with:")
    print(f"    bash train_grpo_qwen3vl_4b.sh --data_path {args.output} --image_folder {image_root}")


if __name__ == "__main__":
    main()
