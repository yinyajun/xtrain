#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_conversational_dataset(dataset: Any, messages_field: str) -> Any:
    # 统一把上游数据规整成只包含 `messages` 一列，减少后续 trainer/tokenizer 的分支处理。
    if messages_field not in dataset.column_names:
        raise ValueError(
            f"Expected conversational column {messages_field!r}. "
            f"Available columns: {', '.join(dataset.column_names)}"
        )
    if messages_field != "messages":
        dataset = dataset.rename_column(messages_field, "messages")

    extra_columns = [column for column in dataset.column_names if column != "messages"]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)
    return dataset


def load_dataset_split(
        dataset_name_or_path: str,
        dataset_config: str | None,
        split: str,
) -> Any:
    from datasets import load_dataset, load_from_disk

    if dataset_config in {"", "-"}:
        dataset_config = None
    path = Path(dataset_name_or_path)

    if path.exists():
        if path.is_file() and path.suffix in {".jsonl", ".json"}:
            return load_dataset("json", data_files={split: str(path)}, split=split)
        if path.is_dir():
            dataset_or_dict = load_from_disk(str(path))
            try:
                return dataset_or_dict[split]
            except Exception as exc:
                raise ValueError(f"Local dataset at {path} does not contain split {split!r}.") from exc
        raise ValueError(f"Unsupported local dataset path: {dataset_name_or_path}")

    return load_dataset(dataset_name_or_path, name=dataset_config, split=split)


def maybe_sample_dataset(dataset: Any, max_samples: int | None, seed: int) -> Any:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def maybe_filter_dataset(dataset: Any, field: str | None, values: list[str] | None) -> Any:
    if not field or not values:
        return dataset
    allowed = set(values)
    return dataset.filter(lambda example: example.get(field) in allowed)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
