#!/usr/bin/env python3
"""Prepare HER-Dataset for OpenRLHF SFT training.

This script converts the Hugging Face dataset
`ChengyuDu0123/HER-Dataset` into a local JSONL file that can be used by
OpenRLHF's `json@/path/to/data` loader.

For `sft_multi_turn` and `sft_single_turn`, each output sample keeps the
full `messages` list so it can be used with:

    --dataset json@/path/to/data_dir
    --input_key messages
    --apply_chat_template
    --multiturn

The script also preserves a few useful metadata fields when present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_DATASET = "ChengyuDu0123/HER-Dataset"
SUPPORTED_CONFIGS = {"sft_multi_turn", "sft_single_turn"}
ALLOWED_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HER-Dataset to OpenRLHF JSONL format.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sft_multi_turn",
        choices=sorted(SUPPORTED_CONFIGS),
        help="HER-Dataset config to export.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Source split to read from Hugging Face datasets. Supports slicing like train[:5000].",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples written.",
    )
    return parser.parse_args()


def normalize_message(message: dict[str, Any]) -> dict[str, str]:
    role = str(message.get("role", "")).strip().lower()
    if role not in ALLOWED_ROLES:
        raise ValueError(f"Unsupported role: {role!r}")

    content = message.get("content", "")
    if content is None:
        content = ""
    content = str(content).strip()

    return {"role": role, "content": content}


def convert_example(example: dict[str, Any]) -> dict[str, Any]:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Example is missing a non-empty 'messages' list.")

    converted = {
        "messages": [normalize_message(message) for message in messages],
    }

    for key in ("trace_id", "character", "source_id", "turn_index"):
        if key in example and example[key] is not None:
            converted[key] = example[key]

    return converted


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    rows = [convert_example(example) for example in dataset]
    print(111, len(rows))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
        print(222, len(rows))

    write_jsonl(args.output_file, rows)

    print(f"Wrote {len(rows)} samples to {args.output_file}")
    print("Use with OpenRLHF:")
    print(f"  --dataset json@{args.output_file.parent}")
    print("  --input_key messages --apply_chat_template --multiturn")


if __name__ == "__main__":
    main()
