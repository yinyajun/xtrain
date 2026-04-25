#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.resources
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common import save_json


def load_trl_template(template_name: str) -> str:
    template_path = importlib.resources.files("trl").joinpath("chat_templates", template_name)
    return template_path.read_text(encoding="utf-8")


def run_special_tokens(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="cpu")

    original = {
        "tokenizer_source": tokenizer_source,
        "model_name_or_path": args.model_name_or_path,
        "tokenizer": {
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
        },
        "model_config": {
            "eos_token_id": model.config.eos_token_id,
            "pad_token_id": model.config.pad_token_id,
            "bos_token_id": model.config.bos_token_id,
        },
        "generation_config": {
            "eos_token_id": model.generation_config.eos_token_id,
            "pad_token_id": model.generation_config.pad_token_id,
            "bos_token_id": model.generation_config.bos_token_id,
        },
    }

    simulated = None
    if args.eos_token:
        eos_token_id = tokenizer.convert_tokens_to_ids(args.eos_token)
        simulated = {
            "requested_eos_token": args.eos_token,
            "requested_eos_token_id": eos_token_id,
            "would_align_to": {
                "eos_token_id": eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
            },
        }

    payload = {
        "original": original,
        "simulated": simulated,
        "known_token_ids": {
            "<|im_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "<|endoftext|>": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            "<|im_start|>": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        },
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.save_path:
        save_json(args.save_path, payload)


def run_chat_template(args: argparse.Namespace) -> None:
    trl_template = load_trl_template(args.template_name)
    print(f"===== TRL TEMPLATE: {args.template_name} =====")
    print(trl_template)

    tokenizer_template = None
    if args.tokenizer_name_or_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
        tokenizer_template = tokenizer.chat_template or ""
        print()
        print(f"===== TOKENIZER TEMPLATE: {args.tokenizer_name_or_path} =====")
        print(tokenizer_template)

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        trl_path = save_dir / args.template_name
        trl_path.write_text(trl_template, encoding="utf-8")

        if args.tokenizer_name_or_path:
            safe_name = args.tokenizer_name_or_path.replace("/", "__")
            tokenizer_path = save_dir / f"{safe_name}.chat_template.jinja"
            tokenizer_path.write_text(tokenizer_template or "", encoding="utf-8")

        print()
        print(f"Saved templates to {save_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一的模型诊断工具，包含 special tokens 和 chat template 两类检查。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tokens_parser = subparsers.add_parser(
        "special-tokens",
        help="检查 tokenizer、model config 和 generation config 的 special tokens 是否一致。",
    )
    tokens_parser.add_argument("--model_name_or_path", required=True, help="底模名称或本地模型路径。")
    tokens_parser.add_argument("--tokenizer_name_or_path", default=None, help="可选 tokenizer 路径；不填时默认与模型同源。")
    tokens_parser.add_argument("--eos_token", default=None, help="可选：模拟训练脚本里显式设置的 eos_token。")
    tokens_parser.add_argument("--save_path", default=None, help="可选：把检查结果保存成 JSON。")
    tokens_parser.set_defaults(handler=run_special_tokens)

    template_parser = subparsers.add_parser(
        "chat-template",
        help="查看 TRL 内置 chat template，或与某个 tokenizer 当前模板做对比。",
    )
    template_parser.add_argument(
        "--template_name",
        default="qwen2_5_training.jinja",
        help="TRL chat_templates 目录里的模板文件名，默认 qwen2_5_training.jinja。",
    )
    template_parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        help="可选 tokenizer 路径；如果提供，会额外打印该 tokenizer 当前的 chat_template。",
    )
    template_parser.add_argument(
        "--save_dir",
        default=None,
        help="可选输出目录；如果提供，会把模板内容保存成文件，方便你本地 diff。",
    )
    template_parser.set_defaults(handler=run_chat_template)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
