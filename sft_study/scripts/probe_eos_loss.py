#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from common import apply_default_system_prompt_to_tokenizer


GENERATION_BLOCK_MARKERS = ("{% generation %}", "{% endgeneration %}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造一条 assistant 为空的合成样本，检查 <|im_end|> 是否进入 assistant-only loss。"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="可选训练输出目录；会自动读取 run_config.json，并优先使用目录里的 tokenizer。",
    )
    parser.add_argument("--tokenizer_name_or_path", default=None, help="可选 tokenizer 路径；不填时自动推断。")
    parser.add_argument(
        "--default_system_prompt",
        default=None,
        help="可选默认 system prompt；当 tokenizer 模板内置默认 system 提示时可覆盖。",
    )
    parser.add_argument(
        "--user_text",
        default="Reply with nothing.",
        help="合成 user 消息内容。",
    )
    parser.add_argument(
        "--assistant_text",
        default="",
        help="合成 assistant 消息内容；默认空字符串，对应“正文为空，只观察模板里的 eos”。",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="可选输出文件路径；checkpoint 模式下默认写到 <checkpoint_dir>/eos_loss_probe.md。",
    )
    return parser.parse_args()


def ensure_local_packages() -> None:
    missing = []
    for module_name in ("transformers",):
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


def _resolve_context(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    run_config: dict[str, Any] = {}
    if checkpoint_dir:
        run_config = _load_json(checkpoint_dir / "run_config.json")

    tokenizer_name_or_path = args.tokenizer_name_or_path
    if tokenizer_name_or_path is None:
        if checkpoint_dir and (checkpoint_dir / "tokenizer.json").exists():
            tokenizer_name_or_path = str(checkpoint_dir)
        else:
            tokenizer_name_or_path = run_config.get("tokenizer_name_or_path") or run_config.get("model_name_or_path")

    default_system_prompt = args.default_system_prompt
    if default_system_prompt is None:
        default_system_prompt = run_config.get("default_system_prompt")

    output_file = args.output_file
    if output_file is None and checkpoint_dir:
        output_file = str(checkpoint_dir / "eos_loss_probe.md")

    if not tokenizer_name_or_path:
        raise ValueError("Unable to resolve tokenizer. Pass --tokenizer_name_or_path or --checkpoint_dir.")

    return {
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "run_config": run_config,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "default_system_prompt": default_system_prompt,
        "output_file": output_file,
    }


def _load_tokenizer(tokenizer_name_or_path: str, default_system_prompt: str | None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_default_system_prompt_to_tokenizer(tokenizer, default_system_prompt)
    return tokenizer


def _has_generation_blocks(chat_template: str | None) -> bool:
    if not chat_template:
        return False
    return all(marker in chat_template for marker in GENERATION_BLOCK_MARKERS)


def _tokenize_conversation(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    encoding_kwargs = {
        "conversation": messages,
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    if _has_generation_blocks(getattr(tokenizer, "chat_template", None)):
        encoding_kwargs["return_assistant_tokens_mask"] = True
    encoding = tokenizer.apply_chat_template(**encoding_kwargs)
    return {
        "rendered_text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
        "input_ids": list(encoding["input_ids"]),
        "assistant_masks": list(encoding.get("assistant_masks", [])),
    }


def _collect_supervised_targets(
    tokenizer: Any,
    input_ids: list[int],
    assistant_masks: list[int],
) -> list[dict[str, Any]]:
    eos_token_id = tokenizer.eos_token_id
    targets: list[dict[str, Any]] = []
    for source_idx in range(len(input_ids) - 1):
        target_idx = source_idx + 1
        if assistant_masks and not assistant_masks[target_idx]:
            continue
        target_id = input_ids[target_idx]
        piece = tokenizer.decode([target_id], skip_special_tokens=False).replace("\n", "\\n")
        targets.append(
            {
                "source_idx": source_idx,
                "target_idx": target_idx,
                "target_id": target_id,
                "piece": piece,
                "is_eos": target_id == eos_token_id,
            }
        )
    return targets


def _render_tail_table(targets: list[dict[str, Any]]) -> str:
    lines = [
        "| source_idx | target_idx | target_id | piece | is_eos |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for item in targets:
        piece = item["piece"].replace("|", "\\|")
        lines.append(
            f"| {item['source_idx']} | {item['target_idx']} | {item['target_id']} | `{piece}` | `{item['is_eos']}` |"
        )
    return "\n".join(lines)


def _toy_loss(targets: list[dict[str, Any]], eos_bad_prob: float = 0.01, other_good_prob: float = 0.999999) -> dict[str, float]:
    perfect = []
    eos_bad = []
    for item in targets:
        perfect.append(-math.log(other_good_prob))
        if item["is_eos"]:
            eos_bad.append(-math.log(eos_bad_prob))
        else:
            eos_bad.append(-math.log(other_good_prob))
    perfect_loss = sum(perfect) / len(perfect) if perfect else 0.0
    eos_bad_loss = sum(eos_bad) / len(eos_bad) if eos_bad else 0.0
    return {
        "perfect_loss": perfect_loss,
        "eos_bad_loss": eos_bad_loss,
    }


def build_report(args: argparse.Namespace) -> str:
    ensure_local_packages()
    context = _resolve_context(args)
    tokenizer = _load_tokenizer(context["tokenizer_name_or_path"], context["default_system_prompt"])

    messages = [{"role": "user", "content": args.user_text}, {"role": "assistant", "content": args.assistant_text}]
    tokenized = _tokenize_conversation(tokenizer, messages)
    input_ids = tokenized["input_ids"]
    assistant_masks = tokenized["assistant_masks"]
    supervised_targets = _collect_supervised_targets(tokenizer, input_ids, assistant_masks)
    eos_targets = [item for item in supervised_targets if item["is_eos"]]
    toy = _toy_loss(supervised_targets)

    assistant_positions = [idx for idx, value in enumerate(assistant_masks) if value]
    tail_targets = supervised_targets[-8:] if len(supervised_targets) > 8 else supervised_targets

    lines = [
        "# EOS Loss Probe",
        "",
        "这个实验构造一条合成样本：assistant 正文为空字符串。",
        "在当前训练模板下，这并不意味着 assistant span 完全为空；通常会变成 `\\n<|im_end|>\\n` 这段。",
        "",
        "## Context",
        "",
        f"- checkpoint_dir: `{context['checkpoint_dir'] or '(none)'}`",
        f"- tokenizer: `{context['tokenizer_name_or_path']}`",
        f"- eos_token: `{tokenizer.eos_token}` / `{tokenizer.eos_token_id}`",
        f"- assistant_only_template: `{_has_generation_blocks(getattr(tokenizer, 'chat_template', None))}`",
        "",
        "## Synthetic Messages",
        "",
        "```json",
        json.dumps(messages, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Rendered Text",
        "",
        "```text",
        tokenized["rendered_text"],
        "```",
        "",
        "## Mask Summary",
        "",
        f"- assistant_mask_positions: `{assistant_positions}`",
        f"- supervised_target_count: `{len(supervised_targets)}`",
        f"- eos_target_count: `{len(eos_targets)}`",
        f"- eos_target_indices: `{[item['target_idx'] for item in eos_targets]}`",
        "",
        "## Supervised Target Tail",
        "",
        _render_tail_table(tail_targets),
        "",
        "## Toy Loss Probe",
        "",
        "- 设定：非 eos target 一律“几乎预测正确”，只让 eos target 的真类概率掉到 `0.01`。",
        f"- perfect_loss: `{toy['perfect_loss']:.6f}`",
        f"- eos_bad_loss: `{toy['eos_bad_loss']:.6f}`",
        "",
        "如果 `eos_bad_loss` 明显大于 `perfect_loss`，说明 eos target 确实在 supervised targets 里；",
        "否则，eos 没有进入这条样本的 loss 统计。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args)
    context = _resolve_context(args)
    output_file = context["output_file"]
    if output_file:
        target = Path(output_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"Wrote eos loss probe to {target}")
        return
    print(report)


if __name__ == "__main__":
    main()
