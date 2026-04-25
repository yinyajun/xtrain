#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    load_dataset_split,
    maybe_filter_dataset,
    maybe_sample_dataset,
    save_json,
    write_jsonl,
)


def parse_component(spec: str) -> dict[str, Any]:
    parts = spec.split("|")
    if len(parts) != 5:
        raise ValueError(
            "Each --component must have exactly five pipe-separated fields: "
            "dataset|config|split|max_samples|filter_field=value1,value2"
        )

    dataset_name, config_name, split, max_samples_raw, filter_spec = parts
    max_samples = None if max_samples_raw in {"", "-", "all"} else int(max_samples_raw)

    filter_field = None
    filter_values = None
    if filter_spec not in {"", "-"}:
        filter_field, raw_values = filter_spec.split("=", maxsplit=1)
        filter_values = [value for value in raw_values.split(",") if value]

    return {
        "dataset_name": dataset_name,
        "config_name": None if config_name in {"", "-"} else config_name,
        "split": split,
        "max_samples": max_samples,
        "filter_field": filter_field,
        "filter_values": filter_values,
    }


def ensure_messages_column(dataset: Any, dataset_label: str, messages_field: str) -> None:
    if messages_field in dataset.column_names:
        return
    raise ValueError(
        f"Expected column {messages_field!r} in {dataset_label}. "
        f"Available: {', '.join(dataset.column_names)}"
    )


def render_token_ids(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)


def run_mix(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    components_stats: list[dict[str, Any]] = []

    for raw_spec in args.component:
        spec = parse_component(raw_spec)
        dataset = load_dataset_split(spec["dataset_name"], spec["config_name"], spec["split"])
        dataset = maybe_filter_dataset(dataset, spec["filter_field"], spec["filter_values"])
        dataset = maybe_sample_dataset(dataset, spec["max_samples"], args.seed)

        ensure_messages_column(dataset, spec["dataset_name"], args.messages_field)
        rows.extend(dataset.to_list())
        components_stats.append(
            {
                "dataset_name": spec["dataset_name"],
                "config_name": spec["config_name"],
                "split": spec["split"],
                "rows": len(dataset),
                "filter_field": spec["filter_field"],
                "filter_values": spec["filter_values"],
            }
        )

    random.Random(args.seed).shuffle(rows)
    output_jsonl = Path(args.output_jsonl)
    write_jsonl(output_jsonl, rows)
    save_json(output_jsonl.with_suffix(".stats.json"), {"rows": len(rows), "components": components_stats})
    print(f"Wrote {len(rows)} rows to {output_jsonl}")


def run_token_match(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    reference_dataset = load_dataset_split(args.reference_dataset, args.reference_dataset_config, args.reference_split)
    reference_dataset = maybe_sample_dataset(reference_dataset, args.reference_max_samples, args.seed)

    candidate_dataset = load_dataset_split(args.candidate_dataset, args.candidate_dataset_config, args.candidate_split)
    candidate_dataset = maybe_sample_dataset(candidate_dataset, args.candidate_max_samples, args.seed)
    candidate_dataset = candidate_dataset.shuffle(seed=args.seed)

    ensure_messages_column(reference_dataset, args.reference_dataset, args.messages_field)
    ensure_messages_column(candidate_dataset, args.candidate_dataset, args.messages_field)

    reference_token_lengths = [
        min(len(render_token_ids(tokenizer, row[args.messages_field])), args.max_length) for row in reference_dataset
    ]
    target_budget = sum(reference_token_lengths)

    selected_rows = []
    selected_budget = 0
    selected_examples = 0
    dropped_overlong = 0

    for row in candidate_dataset:
        messages = row[args.messages_field]
        raw_len = len(render_token_ids(tokenizer, messages))
        if args.drop_overlong and raw_len > args.max_length:
            dropped_overlong += 1
            continue

        selected_rows.append(row)
        selected_budget += min(raw_len, args.max_length)
        selected_examples += 1
        if selected_budget >= target_budget:
            break

    output_jsonl = Path(args.output_jsonl)
    write_jsonl(output_jsonl, selected_rows)
    save_json(
        output_jsonl.with_suffix(".stats.json"),
        {
            "model_name_or_path": args.model_name_or_path,
            "reference_dataset": args.reference_dataset,
            "reference_split": args.reference_split,
            "reference_examples": len(reference_dataset),
            "reference_token_budget": target_budget,
            "candidate_dataset": args.candidate_dataset,
            "candidate_split": args.candidate_split,
            "selected_examples": selected_examples,
            "selected_token_budget": selected_budget,
            "drop_overlong": args.drop_overlong,
            "dropped_overlong": dropped_overlong,
        },
    )
    print(
        json.dumps(
            {
                "output_jsonl": str(output_jsonl),
                "selected_examples": selected_examples,
                "selected_token_budget": selected_budget,
                "reference_token_budget": target_budget,
            },
            indent=2,
        )
    )


def run_holdout_split(args: argparse.Namespace) -> None:
    dataset = load_dataset_split(args.dataset, args.dataset_config, args.split)
    dataset = maybe_filter_dataset(dataset, args.filter_field, args.filter_values)
    dataset = maybe_sample_dataset(dataset, args.max_samples, args.seed)

    ensure_messages_column(dataset, args.dataset, args.messages_field)

    if args.eval_samples <= 0:
        raise ValueError("--eval_samples must be a positive integer.")
    if len(dataset) <= args.eval_samples:
        raise ValueError(
            f"Need more rows than eval_samples to build a holdout split. "
            f"Got rows={len(dataset)} eval_samples={args.eval_samples}."
        )

    dataset = dataset.shuffle(seed=args.seed)
    eval_dataset = dataset.select(range(args.eval_samples))
    train_dataset = dataset.select(range(args.eval_samples, len(dataset)))

    train_output_jsonl = Path(args.train_output_jsonl)
    eval_output_jsonl = Path(args.eval_output_jsonl)
    write_jsonl(train_output_jsonl, train_dataset.to_list())
    write_jsonl(eval_output_jsonl, eval_dataset.to_list())
    save_json(
        train_output_jsonl.with_suffix(".split_stats.json"),
        {
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "seed": args.seed,
            "messages_field": args.messages_field,
            "source_rows": len(dataset),
            "train_rows": len(train_dataset),
            "eval_rows": len(eval_dataset),
            "filter_field": args.filter_field,
            "filter_values": args.filter_values,
            "train_output_jsonl": str(train_output_jsonl),
            "eval_output_jsonl": str(eval_output_jsonl),
        },
    )
    print(
        json.dumps(
            {
                "train_output_jsonl": str(train_output_jsonl),
                "eval_output_jsonl": str(eval_output_jsonl),
                "train_rows": len(train_dataset),
                "eval_rows": len(eval_dataset),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一的数据准备工具，包含数据混合、token budget 对齐和 holdout 切分。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mix_parser = subparsers.add_parser("mix", help="把多个 SFT 数据切片混合成一份 JSONL。")
    mix_parser.add_argument(
        "--component",
        action="append",
        required=True,
        help="单个数据组件，格式为 dataset|config|split|max_samples|filter_field=v1,v2。",
    )
    mix_parser.add_argument("--output_jsonl", required=True, help="输出的混合数据 jsonl 路径。")
    mix_parser.add_argument("--seed", type=int, default=42, help="打乱数据时使用的随机种子。")
    mix_parser.add_argument("--messages_field", default="messages", help="对话消息字段名，默认要求是 messages。")
    mix_parser.set_defaults(handler=run_mix)

    token_match_parser = subparsers.add_parser(
        "token-match",
        help="生成一份与参考数据集 token budget 对齐的对话子集。",
    )
    token_match_parser.add_argument("--model_name_or_path", required=True, help="用于计算 token 长度的模型或 tokenizer。")
    token_match_parser.add_argument("--reference_dataset", required=True, help="参考数据集名称或本地路径。")
    token_match_parser.add_argument("--reference_dataset_config", default=None, help="参考数据集的 config 名称。")
    token_match_parser.add_argument("--reference_split", default="train", help="参考数据集 split。")
    token_match_parser.add_argument("--reference_max_samples", type=int, default=None, help="参考数据集最多抽多少条。")
    token_match_parser.add_argument("--candidate_dataset", required=True, help="候选数据集名称或本地路径。")
    token_match_parser.add_argument("--candidate_dataset_config", default=None, help="候选数据集的 config 名称。")
    token_match_parser.add_argument("--candidate_split", default="train", help="候选数据集 split。")
    token_match_parser.add_argument("--candidate_max_samples", type=int, default=None, help="候选数据集最多读取多少条。")
    token_match_parser.add_argument("--messages_field", default="messages", help="对话字段名。")
    token_match_parser.add_argument("--max_length", type=int, default=2048, help="统计 token 时的截断长度上限。")
    token_match_parser.add_argument("--drop_overlong", action="store_true", help="是否直接丢弃超过 max_length 的样本。")
    token_match_parser.add_argument("--seed", type=int, default=42, help="候选数据集 shuffle 的随机种子。")
    token_match_parser.add_argument("--output_jsonl", required=True, help="输出的 token 对齐子集路径。")
    token_match_parser.set_defaults(handler=run_token_match)

    holdout_parser = subparsers.add_parser(
        "holdout-split",
        help="从单个 split 切出一份本地 held-out train/eval JSONL。",
    )
    holdout_parser.add_argument("--dataset", required=True, help="源数据集名称或本地路径。")
    holdout_parser.add_argument("--dataset_config", default=None, help="源数据集的 config 名称。")
    holdout_parser.add_argument("--split", default="train", help="要切分的源 split。")
    holdout_parser.add_argument("--max_samples", type=int, default=None, help="切分前最多读取多少条。")
    holdout_parser.add_argument("--eval_samples", type=int, required=True, help="held-out eval 集样本数。")
    holdout_parser.add_argument("--messages_field", default="messages", help="对话字段名。")
    holdout_parser.add_argument("--filter_field", default=None, help="可选过滤字段。")
    holdout_parser.add_argument("--filter_values", nargs="*", default=None, help="过滤字段允许的值列表。")
    holdout_parser.add_argument("--seed", type=int, default=42, help="切分时使用的随机种子。")
    holdout_parser.add_argument("--train_output_jsonl", required=True, help="输出的 train jsonl 路径。")
    holdout_parser.add_argument("--eval_output_jsonl", required=True, help="输出的 eval jsonl 路径。")
    holdout_parser.set_defaults(handler=run_holdout_split)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
