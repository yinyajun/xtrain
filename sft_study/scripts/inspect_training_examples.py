#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from common import apply_default_system_prompt_to_tokenizer, load_dataset_split


KNOWN_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
GENERATION_BLOCK_MARKERS = ("{% generation %}", "{% endgeneration %}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="查看训练数据前 N 条样本在 chat template 下渲染后的文本、token 和 assistant-only loss mask。"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--checkpoint_dir",
        default=None,
        help="训练输出目录。会自动读取 run_config.json，并优先使用目录里的 tokenizer/chat_template.jinja。",
    )
    source_group.add_argument("--dataset", default=None, help="数据集名称或本地路径。")

    parser.add_argument("--dataset_config", default=None, help="数据集 config 名称。")
    parser.add_argument("--split", default=None, help="数据集 split；checkpoint 模式下默认读 run_config.json 里的 train_split。")
    parser.add_argument("--messages_field", default=None, help="消息字段名；checkpoint 模式下默认读 run_config.json。")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="tokenizer 路径；不填时自动推断。")
    parser.add_argument(
        "--compare_tokenizer_name_or_path",
        default=None,
        help="可选的对比 tokenizer 路径。常用于对比 checkpoint 里保存的训练模板 vs 原始 tokenizer 模板。",
    )
    parser.add_argument(
        "--chat_template_file",
        default=None,
        help="可选 jinja 模板文件路径；不填时会优先尝试使用 checkpoint_dir/chat_template.jinja。",
    )
    parser.add_argument(
        "--default_system_prompt",
        default=None,
        help="当加载原始 tokenizer 做对比时，可选地覆盖默认 system prompt。",
    )
    parser.add_argument("--num_examples", type=int, default=10, help="查看前多少条样本，默认 10。")
    parser.add_argument(
        "--assistant_only_loss",
        dest="assistant_only_loss",
        action="store_true",
        help="打印 assistant-only loss mask（默认开启）。",
    )
    parser.add_argument(
        "--no_assistant_only_loss",
        dest="assistant_only_loss",
        action="store_false",
        help="不打印 assistant-only loss mask，所有 token 都视为参与 loss。",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="可选输出文件路径。checkpoint 模式下默认写到 <checkpoint_dir>/train_examples_preview.md。",
    )
    parser.set_defaults(assistant_only_loss=True)
    return parser.parse_args()


def ensure_local_packages() -> None:
    missing = []
    for module_name in ("datasets", "transformers"):
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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _resolve_context(args: argparse.Namespace) -> dict[str, Any]:
    run_config: dict[str, Any] = {}
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    if checkpoint_dir:
        run_config = _load_json(checkpoint_dir / "run_config.json")

    dataset = args.dataset or run_config.get("train_dataset")
    dataset_config = args.dataset_config
    if dataset_config is None:
        dataset_config = run_config.get("train_dataset_config")
    split = args.split or run_config.get("train_split") or "train"
    messages_field = args.messages_field or run_config.get("messages_field") or "messages"
    default_system_prompt = args.default_system_prompt
    if default_system_prompt is None:
        default_system_prompt = run_config.get("default_system_prompt")

    tokenizer_name_or_path = args.tokenizer_name_or_path
    if tokenizer_name_or_path is None:
        if checkpoint_dir and (checkpoint_dir / "tokenizer.json").exists():
            tokenizer_name_or_path = str(checkpoint_dir)
        else:
            tokenizer_name_or_path = run_config.get("tokenizer_name_or_path") or run_config.get("model_name_or_path")

    compare_tokenizer_name_or_path = args.compare_tokenizer_name_or_path
    if compare_tokenizer_name_or_path is None and checkpoint_dir:
        compare_tokenizer_name_or_path = run_config.get("tokenizer_name_or_path") or run_config.get("model_name_or_path")
        if compare_tokenizer_name_or_path == tokenizer_name_or_path:
            compare_tokenizer_name_or_path = None

    chat_template_file = args.chat_template_file
    if chat_template_file is None and checkpoint_dir:
        candidate = checkpoint_dir / "chat_template.jinja"
        if candidate.exists():
            chat_template_file = str(candidate)

    if not dataset:
        raise ValueError("Unable to resolve dataset. Pass --dataset or --checkpoint_dir with a valid run_config.json.")
    if not tokenizer_name_or_path:
        raise ValueError("Unable to resolve tokenizer. Pass --tokenizer_name_or_path or use --checkpoint_dir.")

    output_file = args.output_file
    if output_file is None and checkpoint_dir:
        output_file = str(checkpoint_dir / "train_examples_preview.md")

    return {
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "run_config": run_config,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "messages_field": messages_field,
        "default_system_prompt": default_system_prompt,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "compare_tokenizer_name_or_path": compare_tokenizer_name_or_path,
        "chat_template_file": chat_template_file,
        "output_file": output_file,
    }


def _load_tokenizer(
    tokenizer_name_or_path: str,
    default_system_prompt: str | None,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_default_system_prompt_to_tokenizer(tokenizer, default_system_prompt)
    return tokenizer


def _read_text_if_exists(path: str | None) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return target.read_text(encoding="utf-8")


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
    has_generation_blocks = bool(
        effective_template and all(marker in effective_template for marker in GENERATION_BLOCK_MARKERS)
    )
    if assistant_only_loss and has_generation_blocks:
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
    assistant_masks = encoding.get("assistant_masks")
    if assistant_masks is not None:
        assistant_masks = list(assistant_masks)

    loss_masks = [1] * len(input_ids)
    if assistant_only_loss and assistant_masks is not None:
        loss_masks = [1 if value else 0 for value in assistant_masks]

    labels = [token_id if loss_mask else -100 for token_id, loss_mask in zip(input_ids, loss_masks)]
    return {
        "rendered_text": rendered_text,
        "input_ids": input_ids,
        "assistant_masks": assistant_masks,
        "loss_masks": loss_masks,
        "labels": labels,
        "assistant_mask_error": assistant_mask_error,
    }


def _compare_summary(
    current_render: dict[str, Any],
    jinja_render: dict[str, Any] | None,
    compare_render: dict[str, Any] | None,
) -> list[str]:
    lines = []
    if jinja_render is not None:
        same_text = current_render["rendered_text"] == jinja_render["rendered_text"]
        same_ids = current_render["input_ids"] == jinja_render["input_ids"]
        lines.append(
            f"- 当前 tokenizer.chat_template vs 显式 `chat_template.jinja`：same_text={same_text}, same_token_ids={same_ids}"
        )
    if compare_render is not None:
        same_text = current_render["rendered_text"] == compare_render["rendered_text"]
        same_ids = current_render["input_ids"] == compare_render["input_ids"]
        lines.append(f"- 当前训练 tokenizer vs 对比 tokenizer：same_text={same_text}, same_token_ids={same_ids}")
    return lines


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


def _render_boundary_summary(
    tokenizer: Any,
    input_ids: list[int],
    loss_masks: list[int],
) -> str:
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    lines: list[str] = []

    for idx, token_id in enumerate(input_ids):
        if token_id == im_start_id:
            role = _role_after_im_start(tokenizer, input_ids, idx)
            lines.append(f"[IM_START @{idx}] {role}")
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
        role = match.group(1)
        return f"\n[IM_START {role}]"

    highlighted = re.sub(r"\n\[IM_START\]\s*([^\n]+)", promote_role, highlighted)
    highlighted = re.sub(r"\n{3,}", "\n\n", highlighted).strip()
    return highlighted


def _template_meta_section(
    tokenizer: Any,
    tokenizer_name_or_path: str,
    chat_template_file: str | None,
    explicit_template: str | None,
    compare_tokenizer: Any | None,
    compare_tokenizer_name_or_path: str | None,
) -> str:
    sections = [
        "## Template Context",
        "",
        f"- current tokenizer: `{tokenizer_name_or_path}`",
        f"- current tokenizer has chat_template: `{bool(getattr(tokenizer, 'chat_template', None))}`",
        f"- explicit jinja file: `{chat_template_file or '(none)'}`",
        f"- tokenizer.chat_template == explicit jinja file: `{getattr(tokenizer, 'chat_template', None) == explicit_template if explicit_template is not None else 'n/a'}`",
        f"- compare tokenizer: `{compare_tokenizer_name_or_path or '(none)'}`",
    ]
    if compare_tokenizer is not None:
        sections.append(
            f"- current tokenizer.chat_template == compare tokenizer.chat_template: "
            f"`{getattr(tokenizer, 'chat_template', None) == getattr(compare_tokenizer, 'chat_template', None)}`"
        )
    sections.extend(
        [
            "",
            "## Special Tokens",
            "",
            f"- bos_token: `{getattr(tokenizer, 'bos_token', None)}` / `{getattr(tokenizer, 'bos_token_id', None)}`",
            f"- eos_token: `{getattr(tokenizer, 'eos_token', None)}` / `{getattr(tokenizer, 'eos_token_id', None)}`",
            f"- pad_token: `{getattr(tokenizer, 'pad_token', None)}` / `{getattr(tokenizer, 'pad_token_id', None)}`",
        ]
    )
    for token in KNOWN_SPECIAL_TOKENS:
        sections.append(f"- {token}: `{tokenizer.convert_tokens_to_ids(token)}`")
    return "\n".join(sections)


def build_report(args: argparse.Namespace) -> str:
    ensure_local_packages()
    context = _resolve_context(args)
    dataset = load_dataset_split(context["dataset"], context["dataset_config"], context["split"])
    dataset = _normalize_dataset(dataset, context["messages_field"])
    dataset = dataset.select(range(min(args.num_examples, len(dataset))))

    tokenizer = _load_tokenizer(context["tokenizer_name_or_path"], context["default_system_prompt"])
    explicit_template = _read_text_if_exists(context["chat_template_file"])

    compare_tokenizer = None
    compare_tokenizer_error = None
    if context["compare_tokenizer_name_or_path"]:
        try:
            compare_tokenizer = _load_tokenizer(
                context["compare_tokenizer_name_or_path"],
                context["default_system_prompt"],
            )
        except Exception as exc:
            compare_tokenizer_error = str(exc)

    report_lines = [
        "# Training Example Preview",
        "",
        "这份报告展示的是 `messages -> chat template -> token ids` 这条训练前处理链。",
        "",
        "说明：",
        "- `Boundary View` 会把 `<|im_start|>` / `<|im_end|>` 和 assistant loss span 单独拎出来。",
        "- `loss_mask=1` 表示这个位置会参与 loss；在 `assistant_only_loss=True` 时，通常只有 assistant 回复正文会是 1。",
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
        _template_meta_section(
            tokenizer=tokenizer,
            tokenizer_name_or_path=context["tokenizer_name_or_path"],
            chat_template_file=context["chat_template_file"],
            explicit_template=explicit_template,
            compare_tokenizer=compare_tokenizer,
            compare_tokenizer_name_or_path=context["compare_tokenizer_name_or_path"],
        ),
    ]

    if compare_tokenizer_error:
        report_lines.extend(
            [
                "",
                "## Compare Tokenizer Load Error",
                "",
                f"- `{compare_tokenizer_error}`",
            ]
        )

    for example_idx, row in enumerate(dataset):
        messages = row["messages"]
        current_render = _render_with_template(
            tokenizer=tokenizer,
            messages=messages,
            assistant_only_loss=args.assistant_only_loss,
            chat_template=None,
        )
        jinja_render = None
        if explicit_template is not None:
            jinja_render = _render_with_template(
                tokenizer=tokenizer,
                messages=messages,
                assistant_only_loss=args.assistant_only_loss,
                chat_template=explicit_template,
            )
        compare_render = None
        if compare_tokenizer is not None:
            compare_render = _render_with_template(
                tokenizer=compare_tokenizer,
                messages=messages,
                assistant_only_loss=args.assistant_only_loss,
                chat_template=None,
            )

        loss_token_count = sum(current_render["loss_masks"])
        report_lines.extend(
            [
                "",
                f"## Example {example_idx}",
                "",
                f"- total_tokens: `{len(current_render['input_ids'])}`",
                f"- loss_tokens: `{loss_token_count}`",
                f"- message_count: `{len(messages)}`",
            ]
        )
        report_lines.extend(_compare_summary(current_render, jinja_render, compare_render))

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
                _render_boundary_summary(
                    tokenizer=tokenizer,
                    input_ids=current_render["input_ids"],
                    loss_masks=current_render["loss_masks"],
                ),
                "```",
                "",
                "### Boundary-Highlighted Render",
                "",
                "```text",
                _render_highlighted_text(current_render["rendered_text"]),
                "```",
            ]
        )

        if jinja_render is not None and current_render["rendered_text"] != jinja_render["rendered_text"]:
            report_lines.extend(
                [
                    "",
                    "### Rendered With Explicit chat_template.jinja",
                    "",
                    "```text",
                    jinja_render["rendered_text"],
                    "```",
                ]
            )

        if compare_render is not None and current_render["rendered_text"] != compare_render["rendered_text"]:
            report_lines.extend(
                [
                    "",
                    "### Rendered With Compare Tokenizer",
                    "",
                    "```text",
                    compare_render["rendered_text"],
                    "```",
                ]
            )

    return "\n".join(report_lines) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args)

    context = _resolve_context(args)
    output_file = context["output_file"]
    if output_file:
        target = Path(output_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"Wrote training example preview to {target}")
        return

    print(report)


if __name__ == "__main__":
    main()
