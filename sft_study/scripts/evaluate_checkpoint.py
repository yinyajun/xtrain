#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from common import DEFAULT_SYSTEM_PROMPT, save_json


DEFAULT_BENCHMARKS = ["ifeval", "gsm8k", "mmlu", "cmmlu"]


def parse_args() -> argparse.Namespace:
    # 这个脚本是“评估入口总控”。
    # 它会根据 checkpoint 或显式传入的模型参数，串联固定 prompt 生成和 lm-eval。
    parser = argparse.ArgumentParser(
        description="对 base 模型或 SFT checkpoint 统一执行固定 prompt 生成和 benchmark 评估。"
    )
    parser.add_argument("--checkpoint_dir", default=None, help="SFT 输出目录；如果存在 run_config.json，会自动补齐模型上下文。")
    parser.add_argument("--model_name_or_path", default=None, help="底模名称或本地模型路径。")
    parser.add_argument("--adapter_path", default=None, help="可选 LoRA adapter 路径。")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="可选 tokenizer 路径。")
    parser.add_argument("--output_dir", default=None, help="评估结果输出目录。")
    parser.add_argument("--prompts_file", default=None, help="固定 prompt 文件；不填则使用仓库默认样例。")
    parser.add_argument("--benchmarks", nargs="*", default=DEFAULT_BENCHMARKS, help="要运行的 benchmark 列表。")
    parser.add_argument("--device", default="auto", help="运行设备，例如 auto / cuda:0 / cpu。")
    parser.add_argument("--batch_size", default="auto", help="评估 batch size。")
    parser.add_argument("--dtype", default=None, help="可选推理 dtype。")
    parser.add_argument("--attn_implementation", default=None, help="可选注意力实现，例如 flash_attention_2。")
    parser.add_argument("--default_system_prompt", default=DEFAULT_SYSTEM_PROMPT, help="统一覆盖 chat template 里的默认 system prompt。")
    parser.add_argument("--revision", default=None, help="模型 revision。")
    parser.add_argument("--trust_remote_code", action="store_true", help="是否允许 remote code。")
    parser.add_argument("--load_in_4bit", action="store_true", help="是否以 4-bit 方式加载模型。")
    parser.add_argument("--limit", default=None, help="可选 benchmark limit。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="固定 prompt 生成时的最大新 token 数。")
    parser.add_argument("--temperature", type=float, default=0.0, help="固定 prompt 生成温度。")
    parser.add_argument("--top_p", type=float, default=1.0, help="固定 prompt 生成的 top_p。")
    parser.add_argument("--think_end_token", default=None, help="模型思维链终止 token。")
    parser.add_argument("--skip_fixed_prompts", action="store_true", help="跳过固定 prompt 生成。")
    parser.add_argument("--skip_benchmarks", action="store_true", help="跳过 benchmark 评估。")
    parser.add_argument("--log_samples", action="store_true", help="让 lm-eval 输出样本级日志。")
    parser.add_argument("--extra_model_arg", action="append", default=[], help="额外透传给 model_args 的参数。")
    parser.add_argument("--extra_eval_arg", action="append", default=[], help="额外透传给 lm-eval 的参数。")
    parser.add_argument("--dry_run", action="store_true", help="只打印子命令，不真正执行。")
    return parser.parse_args()


def load_checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    run_config = checkpoint_dir / "run_config.json"
    if not run_config.exists():
        raise SystemExit(
            f"Checkpoint directory {checkpoint_dir} does not contain run_config.json. "
            "Pass --model_name_or_path and --adapter_path explicitly instead."
        )
    return json.loads(run_config.read_text(encoding="utf-8"))


def resolve_context(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    checkpoint_metadata: dict[str, Any] | None = None
    if checkpoint_dir is not None:
        checkpoint_metadata = load_checkpoint_metadata(checkpoint_dir)

    model_name_or_path = args.model_name_or_path or (
        checkpoint_metadata.get("model_name_or_path") if checkpoint_metadata else None
    )
    adapter_path = args.adapter_path or (str(checkpoint_dir) if checkpoint_dir else None)
    tokenizer_name_or_path = args.tokenizer_name_or_path or adapter_path or model_name_or_path

    if not model_name_or_path:
        raise SystemExit("Could not resolve model_name_or_path. Pass --model_name_or_path or --checkpoint_dir.")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif checkpoint_dir is not None:
        output_dir = checkpoint_dir / "eval"
    else:
        raise SystemExit("Pass --output_dir when evaluating without --checkpoint_dir.")

    prompts_file = (
        Path(args.prompts_file).resolve()
        if args.prompts_file
        else Path(__file__).resolve().parent.parent / "data" / "fixed_prompts.jsonl"
    )

    return {
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "output_dir": output_dir,
        "prompts_file": prompts_file,
    }


def run_command(command: list[str], dry_run: bool) -> None:
    print(f"$ {shlex.join(command)}")
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    context = resolve_context(args)

    output_dir: Path = context["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_config = {
        "checkpoint_dir": context["checkpoint_dir"],
        "model_name_or_path": context["model_name_or_path"],
        "adapter_path": context["adapter_path"],
        "tokenizer_name_or_path": context["tokenizer_name_or_path"],
        "prompts_file": str(context["prompts_file"]),
        "benchmarks": args.benchmarks,
        "device": args.device,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "default_system_prompt": args.default_system_prompt,
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "load_in_4bit": args.load_in_4bit,
        "limit": args.limit,
        "think_end_token": args.think_end_token,
    }
    save_json(output_dir / "evaluation_config.json", evaluation_config)

    scripts_dir = Path(__file__).resolve().parent
    if not args.skip_fixed_prompts:
        fixed_prompt_command = [
            sys.executable,
            str(scripts_dir / "generate_fixed_prompts.py"),
            "--model_name_or_path",
            context["model_name_or_path"],
            "--tokenizer_name_or_path",
            context["tokenizer_name_or_path"],
            "--prompts_file",
            str(context["prompts_file"]),
            "--output_file",
            str(output_dir / "fixed_prompts.jsonl"),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--top_p",
            str(args.top_p),
            "--default_system_prompt",
            args.default_system_prompt,
        ]
        if args.attn_implementation:
            fixed_prompt_command.extend(["--attn_implementation", args.attn_implementation])
        if context["adapter_path"]:
            fixed_prompt_command.extend(["--adapter_path", context["adapter_path"]])
        run_command(fixed_prompt_command, args.dry_run)

    if not args.skip_benchmarks:
        benchmark_command = [
            sys.executable,
            str(scripts_dir / "run_lm_eval.py"),
            "--model_name_or_path",
            context["model_name_or_path"],
            "--output_dir",
            str(output_dir / "benchmarks"),
            "--device",
            args.device,
            "--batch_size",
            args.batch_size,
            "--benchmarks",
            *args.benchmarks,
        ]
        if context["adapter_path"]:
            benchmark_command.extend(["--adapter_path", context["adapter_path"]])
        if context["tokenizer_name_or_path"]:
            benchmark_command.extend(["--tokenizer_name_or_path", context["tokenizer_name_or_path"]])
        if args.dtype:
            benchmark_command.extend(["--dtype", args.dtype])
        if args.attn_implementation:
            benchmark_command.extend(["--attn_implementation", args.attn_implementation])
        if args.default_system_prompt:
            benchmark_command.extend(["--default_system_prompt", args.default_system_prompt])
        if args.revision:
            benchmark_command.extend(["--revision", args.revision])
        if args.trust_remote_code:
            benchmark_command.append("--trust_remote_code")
        if args.load_in_4bit:
            benchmark_command.append("--load_in_4bit")
        if args.limit is not None:
            benchmark_command.extend(["--limit", str(args.limit)])
        if args.think_end_token:
            benchmark_command.extend(["--think_end_token", args.think_end_token])
        if args.log_samples:
            benchmark_command.append("--log_samples")
        if args.dry_run:
            benchmark_command.append("--dry_run")
        for value in args.extra_model_arg:
            benchmark_command.extend(["--extra_model_arg", value])
        for value in args.extra_eval_arg:
            benchmark_command.extend(["--extra_eval_arg", value])
        run_command(benchmark_command, args.dry_run)

    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
