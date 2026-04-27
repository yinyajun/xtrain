#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.resources
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
sys.path.append(str(SCRIPT_DIR.parent))

from common import load_dataset_split


GENERATION_BLOCK_MARKERS = ("{% generation %}", "{% endgeneration %}")
KNOWN_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def ensure_packages(*module_names: str) -> None:
    missing = []
    for module_name in module_names:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise SystemExit("Missing required packages: " + ", ".join(missing))


def _has_training_generation_blocks(chat_template: str | None) -> bool:
    return bool(chat_template and all(marker in chat_template for marker in GENERATION_BLOCK_MARKERS))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text_if_exists(path: str | None) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return target.read_text(encoding="utf-8")


def _load_trl_template(template_name: str) -> str:
    template_path = importlib.resources.files("trl").joinpath("chat_templates", template_name)
    return template_path.read_text(encoding="utf-8")


def _load_tokenizer(tokenizer_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _normalize_dataset(dataset: Any, messages_field: str) -> Any:
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


def _resolve_template_context(args: argparse.Namespace) -> dict[str, Any]:
    run_config: dict[str, Any] = {}
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    if checkpoint_dir and (checkpoint_dir / "run_config.json").exists():
        run_config = _load_json(checkpoint_dir / "run_config.json")

    model_name_or_path = args.model_name_or_path or run_config.get("model_name_or_path")
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if tokenizer_name_or_path is None:
        if checkpoint_dir and (checkpoint_dir / "tokenizer.json").exists():
            tokenizer_name_or_path = str(checkpoint_dir)
        else:
            tokenizer_name_or_path = model_name_or_path

    chat_template_file = None
    if checkpoint_dir:
        candidate = checkpoint_dir / "chat_template.jinja"
        if candidate.exists():
            chat_template_file = str(candidate)

    if not tokenizer_name_or_path:
        raise SystemExit("Unable to resolve tokenizer. Pass --tokenizer_name_or_path, --model_name_or_path, or --checkpoint_dir.")

    return {
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "model_name_or_path": model_name_or_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "chat_template_file": chat_template_file,
    }


def _resolve_examples_context(args: argparse.Namespace) -> dict[str, Any]:
    run_config: dict[str, Any] = {}
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    if checkpoint_dir:
        run_config = _load_json(checkpoint_dir / "run_config.json")

    dataset = args.dataset or run_config.get("train_dataset")
    dataset_config = args.dataset_config if args.dataset_config is not None else run_config.get("train_dataset_config")
    split = args.split or run_config.get("train_split") or "train"
    messages_field = args.messages_field or run_config.get("messages_field") or "messages"

    tokenizer_name_or_path = args.tokenizer_name_or_path
    if tokenizer_name_or_path is None:
        if checkpoint_dir and (checkpoint_dir / "tokenizer.json").exists():
            tokenizer_name_or_path = str(checkpoint_dir)
        else:
            tokenizer_name_or_path = run_config.get("tokenizer_name_or_path") or run_config.get("model_name_or_path")

    chat_template_file = args.chat_template_file
    if chat_template_file is None and checkpoint_dir:
        candidate = checkpoint_dir / "chat_template.jinja"
        if candidate.exists():
            chat_template_file = str(candidate)

    output_file = args.output_file
    if output_file is None and checkpoint_dir:
        output_file = str(checkpoint_dir / "train_examples_preview.md")

    if not dataset:
        raise SystemExit("Unable to resolve dataset. Pass --dataset or --checkpoint_dir with a valid run_config.json.")
    if not tokenizer_name_or_path:
        raise SystemExit("Unable to resolve tokenizer. Pass --tokenizer_name_or_path or use --checkpoint_dir.")

    return {
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "messages_field": messages_field,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "chat_template_file": chat_template_file,
        "output_file": output_file,
    }


def _special_tokens_payload(tokenizer: Any) -> dict[str, Any]:
    payload = {
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
    }
    for token in KNOWN_SPECIAL_TOKENS:
        payload[f"{token}_id"] = tokenizer.convert_tokens_to_ids(token)
    return payload


def run_template(args: argparse.Namespace) -> None:
    ensure_packages("transformers", "trl")
    context = _resolve_template_context(args)
    tokenizer = _load_tokenizer(context["tokenizer_name_or_path"])
    tokenizer_template = tokenizer.chat_template or ""
    explicit_template = _read_text_if_exists(context["chat_template_file"])
    trl_template = _load_trl_template(args.template_name)

    payload = {
        "resolved_context": context,
        "special_tokens": _special_tokens_payload(tokenizer),
        "template": {
            "template_name": args.template_name,
            "tokenizer_has_chat_template": bool(tokenizer_template),
            "explicit_template_file": context["chat_template_file"],
            "tokenizer_equals_explicit": tokenizer_template == explicit_template if explicit_template is not None else None,
            "tokenizer_equals_trl": tokenizer_template == trl_template,
        },
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print()
    print("=== Tokenizer chat_template ===")
    print(tokenizer_template)
    if explicit_template is not None:
        print()
        print("=== Explicit chat_template.jinja ===")
        print(explicit_template)
    print()
    print(f"=== TRL Template: {args.template_name} ===")
    print(trl_template)


def _render_with_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    assistant_only_loss: bool,
    chat_template: str | None = None,
) -> dict[str, Any]:
    effective_template = chat_template if chat_template is not None else getattr(tokenizer, "chat_template", None)
    rendered_text = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=False,
    )

    encoding_kwargs: dict[str, Any] = {
        "conversation": messages,
        "chat_template": chat_template,
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    assistant_mask_error = None
    if assistant_only_loss and _has_training_generation_blocks(effective_template):
        encoding_kwargs["return_assistant_tokens_mask"] = True
    try:
        encoding = tokenizer.apply_chat_template(**encoding_kwargs)
    except TypeError:
        assistant_mask_error = "tokenizer.apply_chat_template does not support return_assistant_tokens_mask in this environment."
        encoding_kwargs.pop("return_assistant_tokens_mask", None)
        encoding = tokenizer.apply_chat_template(**encoding_kwargs)
    except Exception as exc:
        assistant_mask_error = str(exc)
        encoding_kwargs.pop("return_assistant_tokens_mask", None)
        encoding = tokenizer.apply_chat_template(**encoding_kwargs)

    input_ids = list(encoding["input_ids"])
    assistant_masks = list(encoding.get("assistant_masks", [])) if encoding.get("assistant_masks") is not None else None
    loss_masks = [1] * len(input_ids)
    if assistant_only_loss and assistant_masks is not None:
        loss_masks = [1 if value else 0 for value in assistant_masks]

    return {
        "rendered_text": rendered_text,
        "input_ids": input_ids,
        "loss_masks": loss_masks,
        "assistant_mask_error": assistant_mask_error,
    }


def _find_mask_spans(mask: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def _role_after_im_start(tokenizer: Any, input_ids: list[int], idx: int) -> str:
    preview = tokenizer.decode(input_ids[idx + 1: idx + 8], skip_special_tokens=False)
    return preview.split("\n", 1)[0].strip() or "(unknown)"


def _render_boundary_summary(tokenizer: Any, input_ids: list[int], loss_masks: list[int]) -> str:
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    lines: list[str] = []

    for idx, token_id in enumerate(input_ids):
        if token_id == im_start_id:
            lines.append(f"[IM_START @{idx}] {_role_after_im_start(tokenizer, input_ids, idx)}")
        elif token_id == im_end_id:
            lines.append(f"[IM_END   @{idx}]")

    for start, end in _find_mask_spans(loss_masks):
        snippet = tokenizer.decode(input_ids[start:end + 1], skip_special_tokens=False).replace("\n", "\\n")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        lines.append(f"[LOSS     @{start}-{end}] {snippet}")

    return "\n".join(lines)


def _render_highlighted_text(rendered_text: str) -> str:
    highlighted = rendered_text.replace("<|im_start|>", "\n[IM_START] ")
    highlighted = highlighted.replace("<|im_end|>", " [IM_END]\n")

    def promote_role(match: re.Match[str]) -> str:
        return f"\n[IM_START {match.group(1)}]"

    highlighted = re.sub(r"\n\[IM_START\]\s*([^\n]+)", promote_role, highlighted)
    highlighted = re.sub(r"\n{3,}", "\n\n", highlighted).strip()
    return highlighted


def build_examples_report(args: argparse.Namespace) -> str:
    ensure_packages("datasets", "transformers")
    context = _resolve_examples_context(args)
    dataset = load_dataset_split(context["dataset"], context["dataset_config"], context["split"])
    dataset = _normalize_dataset(dataset, context["messages_field"])
    dataset = dataset.select(range(min(args.num_examples, len(dataset))))

    tokenizer = _load_tokenizer(context["tokenizer_name_or_path"])
    explicit_template = _read_text_if_exists(context["chat_template_file"])

    report_lines = [
        "# Training Example Preview",
        "",
        "这份报告展示的是 `messages -> chat template -> token ids` 这条训练前处理链。",
        "",
        "## Resolved Context",
        "",
        f"- checkpoint_dir: `{context['checkpoint_dir'] or '(none)'}`",
        f"- dataset: `{context['dataset']}`",
        f"- dataset_config: `{context['dataset_config']}`",
        f"- split: `{context['split']}`",
        f"- messages_field: `{context['messages_field']}`",
        f"- num_examples: `{len(dataset)}`",
        f"- assistant_only_loss: `{args.assistant_only_loss}`",
        "",
        "## Template Context",
        "",
        f"- tokenizer: `{context['tokenizer_name_or_path']}`",
        f"- explicit jinja file: `{context['chat_template_file'] or '(none)'}`",
        f"- tokenizer.chat_template has generation blocks: `{_has_training_generation_blocks(getattr(tokenizer, 'chat_template', None))}`",
        f"- tokenizer.chat_template == explicit jinja file: `{getattr(tokenizer, 'chat_template', None) == explicit_template if explicit_template is not None else 'n/a'}`",
        "",
        "## Special Tokens",
        "",
    ]
    for key, value in _special_tokens_payload(tokenizer).items():
        report_lines.append(f"- {key}: `{value}`")

    for example_idx, row in enumerate(dataset):
        messages = row["messages"]
        current_render = _render_with_template(
            tokenizer=tokenizer,
            messages=messages,
            assistant_only_loss=args.assistant_only_loss,
            chat_template=None,
        )
        explicit_render = None
        if explicit_template is not None:
            explicit_render = _render_with_template(
                tokenizer=tokenizer,
                messages=messages,
                assistant_only_loss=args.assistant_only_loss,
                chat_template=explicit_template,
            )

        report_lines.extend(
            [
                "",
                f"## Example {example_idx}",
                "",
                f"- total_tokens: `{len(current_render['input_ids'])}`",
                f"- loss_tokens: `{sum(current_render['loss_masks'])}`",
                f"- message_count: `{len(messages)}`",
            ]
        )
        if current_render["assistant_mask_error"]:
            report_lines.append(f"- assistant mask note: `{current_render['assistant_mask_error']}`")

        report_lines.extend(
            [
                "",
                "### Messages",
                "",
                "```json",
                json.dumps(messages, ensure_ascii=False, indent=2),
                "```",
                "",
                "### Rendered With Current tokenizer.chat_template",
                "",
                "```text",
                current_render["rendered_text"],
                "```",
                "",
                "### Boundary View",
                "",
                "```text",
                _render_boundary_summary(tokenizer, current_render["input_ids"], current_render["loss_masks"]),
                "```",
                "",
                "### Boundary-Highlighted Render",
                "",
                "```text",
                _render_highlighted_text(current_render["rendered_text"]),
                "```",
            ]
        )

        if explicit_render is not None and current_render["rendered_text"] != explicit_render["rendered_text"]:
            report_lines.extend(
                [
                    "",
                    "### Rendered With Explicit chat_template.jinja",
                    "",
                    "```text",
                    explicit_render["rendered_text"],
                    "```",
                ]
            )

    return "\n".join(report_lines) + "\n"


def run_examples(args: argparse.Namespace) -> None:
    report = build_examples_report(args)
    context = _resolve_examples_context(args)
    if context["output_file"]:
        target = Path(context["output_file"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"Wrote training example preview to {target}")
        return
    print(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一的模板/训练样本诊断工具。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser(
        "template",
        help="查看 tokenizer special tokens 和 chat template，并和 TRL 模板做对比。",
    )
    template_source = template_parser.add_mutually_exclusive_group(required=True)
    template_source.add_argument("--checkpoint_dir", default=None, help="训练输出目录。优先读取目录里的 tokenizer。")
    template_source.add_argument("--model_name_or_path", default=None, help="底模名称或本地模型路径。")
    template_parser.add_argument("--tokenizer_name_or_path", default=None, help="可选 tokenizer 路径；不填时自动推断。")
    template_parser.add_argument(
        "--template_name",
        default="qwen2_5_training.jinja",
        help="TRL chat_templates 目录里的模板文件名，默认 qwen2_5_training.jinja。",
    )
    template_parser.set_defaults(handler=run_template)

    examples_parser = subparsers.add_parser(
        "examples",
        help="查看训练样本在 chat template 下的渲染结果和 assistant-only loss 边界。",
    )
    example_source = examples_parser.add_mutually_exclusive_group(required=True)
    example_source.add_argument(
        "--checkpoint_dir",
        default=None,
        help="训练输出目录。会自动读取 run_config.json，并优先使用目录里的 tokenizer/chat_template.jinja。",
    )
    example_source.add_argument("--dataset", default=None, help="数据集名称或本地路径。")
    examples_parser.add_argument("--dataset_config", default=None, help="数据集 config 名称。")
    examples_parser.add_argument("--split", default=None, help="数据集 split；checkpoint 模式下默认读 run_config.json 里的 train_split。")
    examples_parser.add_argument("--messages_field", default=None, help="消息字段名；checkpoint 模式下默认读 run_config.json。")
    examples_parser.add_argument("--tokenizer_name_or_path", default=None, help="tokenizer 路径；不填时自动推断。")
    examples_parser.add_argument(
        "--chat_template_file",
        default=None,
        help="可选 jinja 模板文件路径；不填时会优先尝试使用 checkpoint_dir/chat_template.jinja。",
    )
    examples_parser.add_argument("--num_examples", type=int, default=10, help="查看前多少条样本，默认 10。")
    examples_parser.add_argument(
        "--assistant_only_loss",
        dest="assistant_only_loss",
        action="store_true",
        help="打印 assistant-only loss mask（默认开启）。",
    )
    examples_parser.add_argument(
        "--no_assistant_only_loss",
        dest="assistant_only_loss",
        action="store_false",
        help="不打印 assistant-only loss mask，所有 token 都视为参与 loss。",
    )
    examples_parser.add_argument(
        "--output_file",
        default=None,
        help="可选输出文件路径。checkpoint 模式下默认写到 <checkpoint_dir>/train_examples_preview.md。",
    )
    examples_parser.set_defaults(handler=run_examples, assistant_only_loss=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
