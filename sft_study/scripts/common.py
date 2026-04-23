#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant finetuned by Yajun"
KNOWN_SYSTEM_PROMPT_LITERALS = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "You are a helpful assistant.",
)


SPLIT_ALIASES = {
    "train_sft": ("train_sft", "train"),
    "test_sft": ("test_sft", "test", "validation"),
    "validation": ("validation", "test", "eval"),
    "train": ("train", "train_sft"),
    "test": ("test", "test_sft", "validation"),
}


def ensure_packages() -> None:
    missing = []
    for module_name in ("datasets", "transformers", "trl", "peft", "torch"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install them with `pip install -r sft_study/requirements.txt`."
        )


def _import_datasets():
    from datasets import load_dataset, load_dataset_builder, load_from_disk

    return load_dataset, load_dataset_builder, load_from_disk


def resolve_split_name(dataset_name: str, dataset_config: str | None, requested_split: str) -> str:
    if dataset_config in {"", "-"}:
        dataset_config = None
    load_dataset, load_dataset_builder, _ = _import_datasets()
    del load_dataset
    try:
        builder = load_dataset_builder(dataset_name, name=dataset_config)
        available = tuple(builder.info.splits.keys())
    except Exception:
        return requested_split

    if requested_split in available:
        return requested_split

    for candidate in SPLIT_ALIASES.get(requested_split, (requested_split,)):
        if candidate in available:
            return candidate

    raise ValueError(
        f"Split {requested_split!r} not found for dataset {dataset_name!r}. "
        f"Available splits: {', '.join(available) or '(unknown)'}"
    )


def load_dataset_split(
    dataset_name_or_path: str,
    dataset_config: str | None,
    split: str,
) -> Any:
    if dataset_config in {"", "-"}:
        dataset_config = None
    load_dataset, _, load_from_disk = _import_datasets()
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

    actual_split = resolve_split_name(dataset_name_or_path, dataset_config, split)
    return load_dataset(dataset_name_or_path, name=dataset_config, split=actual_split)


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


def replace_default_system_prompt(chat_template: str | None, system_prompt: str | None) -> str | None:
    if not chat_template or not system_prompt:
        return chat_template

    patched = chat_template
    for literal in KNOWN_SYSTEM_PROMPT_LITERALS:
        patched = patched.replace(literal, system_prompt)
    return patched


def apply_default_system_prompt_to_tokenizer(tokenizer: Any, system_prompt: str | None) -> bool:
    original = getattr(tokenizer, "chat_template", None)
    patched = replace_default_system_prompt(original, system_prompt)
    if patched is None or patched == original:
        return False
    tokenizer.chat_template = patched
    return True


def export_tokenizer_with_system_prompt(
    tokenizer_name_or_path: str,
    output_dir: str | Path,
    system_prompt: str | None,
) -> tuple[str, bool]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    changed = apply_default_system_prompt_to_tokenizer(tokenizer, system_prompt)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(target)
    return str(target), changed
