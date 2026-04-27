#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from common import save_json


DEFAULT_BENCHMARKS = ["ifeval", "gsm8k", "mmlu", "cmmlu"]
LM_EVAL_COMMAND = [sys.executable, "-m", "lm_eval"]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    tasks: tuple[str, ...]
    num_fewshot: int | None = None
    apply_chat_template: bool = False
    gen_kwargs: dict[str, Any] | None = None


BENCHMARK_SPECS = {
    "ifeval": BenchmarkSpec(
        name="ifeval",
        tasks=("ifeval",),
        num_fewshot=0,
        apply_chat_template=True,
        gen_kwargs={"temperature": 0.0, "do_sample": False, "max_gen_toks": 256},
    ),
    "gsm8k": BenchmarkSpec(
        name="gsm8k",
        tasks=("gsm8k_cot",),
        num_fewshot=8,
        apply_chat_template=True,
        gen_kwargs={"temperature": 0.0, "do_sample": False},
    ),
    "mmlu": BenchmarkSpec(
        name="mmlu",
        tasks=("mmlu",),
        num_fewshot=5,
    ),
    "cmmlu": BenchmarkSpec(
        name="cmmlu",
        tasks=("cmmlu",),
        num_fewshot=5,
    ),
}


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--limit", default=None, help="可选 benchmark limit。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="固定 prompt 生成时的最大新 token 数。")
    parser.add_argument("--temperature", type=float, default=0.0, help="固定 prompt 生成温度。")
    parser.add_argument("--top_p", type=float, default=1.0, help="固定 prompt 生成的 top_p。")
    parser.add_argument("--skip_fixed_prompts", action="store_true", help="跳过固定 prompt 生成。")
    parser.add_argument("--skip_benchmarks", action="store_true", help="跳过 benchmark 评估。")
    parser.add_argument("--log_samples", action="store_true", help="让 lm-eval 输出样本级日志。")
    parser.add_argument("--dry_run", action="store_true", help="只打印子命令，不真正执行。")
    return parser.parse_args()


def resolve_context(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    checkpoint_metadata: dict[str, Any] | None = None
    if checkpoint_dir is not None:
        run_config = checkpoint_dir / "run_config.json"
        if not run_config.exists():
            raise SystemExit(
                f"Checkpoint directory {checkpoint_dir} does not contain run_config.json. "
                "Pass --model_name_or_path and --adapter_path explicitly instead."
            )
        checkpoint_metadata = json.loads(run_config.read_text(encoding="utf-8"))

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


def build_model_args(
    model_name_or_path: str,
    adapter_path: str | None,
    tokenizer_name_or_path: str | None,
    dtype: str | None,
    attn_implementation: str | None,
) -> str:
    model_args = [f"pretrained={model_name_or_path}"]
    if adapter_path:
        model_args.append(f"peft={adapter_path}")
    model_args.append(f"tokenizer={tokenizer_name_or_path or model_name_or_path}")
    if dtype:
        model_args.append(f"dtype={dtype}")
    if attn_implementation:
        model_args.append(f"attn_implementation={attn_implementation}")
    return ",".join(model_args)


def build_benchmark_command(
    *,
    model_name_or_path: str,
    adapter_path: str | None,
    tokenizer_name_or_path: str | None,
    device: str,
    batch_size: str,
    dtype: str | None,
    attn_implementation: str | None,
    limit: str | None,
    log_samples: bool,
    spec: BenchmarkSpec,
    output_dir: Path,
) -> list[str]:
    command = [
        *LM_EVAL_COMMAND,
        "--model",
        "hf",
        "--model_args",
        build_model_args(
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        ),
        "--tasks",
        ",".join(spec.tasks),
        "--device",
        device,
        "--batch_size",
        batch_size,
        "--output_path",
        str(output_dir),
    ]
    if spec.num_fewshot is not None:
        command.extend(["--num_fewshot", str(spec.num_fewshot)])
    if spec.apply_chat_template:
        command.append("--apply_chat_template")
    if spec.gen_kwargs:
        gen_kwargs = ",".join(
            f"{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in spec.gen_kwargs.items()
        )
        command.extend(["--gen_kwargs", gen_kwargs])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if log_samples:
        command.append("--log_samples")
    return command


def run_benchmark_suite(
    *,
    model_name_or_path: str,
    adapter_path: str | None,
    tokenizer_name_or_path: str | None,
    output_dir: Path,
    benchmarks: list[str],
    device: str = "auto",
    batch_size: str = "auto",
    dtype: str | None = None,
    attn_implementation: str | None = None,
    limit: str | None = None,
    log_samples: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "tokenizer_name_or_path": tokenizer_name_or_path or model_name_or_path,
        "benchmarks": [],
    }

    for benchmark_name in benchmarks:
        spec = BENCHMARK_SPECS[benchmark_name]
        benchmark_output_dir = output_dir / spec.name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        command = build_benchmark_command(
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            device=device,
            batch_size=batch_size,
            dtype=dtype,
            attn_implementation=attn_implementation,
            limit=limit,
            log_samples=log_samples,
            spec=spec,
            output_dir=benchmark_output_dir,
        )

        result: dict[str, Any] = {
            "benchmark": spec.name,
            "tasks": list(spec.tasks),
            "command": command,
            "output_dir": str(benchmark_output_dir),
        }
        summary["benchmarks"].append(result)

        print(f"$ {shlex.join(command)}")
        if dry_run:
            result["skipped"] = True
            continue

        completed = subprocess.run(command, env=os.environ.copy(), check=False)
        result["returncode"] = completed.returncode
        if completed.returncode != 0:
            raise SystemExit(f"lm-eval failed for benchmark {spec.name!r} with exit code {completed.returncode}.")

    save_json(output_dir / "summary.json", summary)
    return summary


def run_command(command: list[str], dry_run: bool) -> None:
    print(f"$ {shlex.join(command)}")
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    context = resolve_context(args)
    benchmarks = args.benchmarks
    for name in benchmarks:
        if name not in BENCHMARK_SPECS:
            allowed = ", ".join(sorted(BENCHMARK_SPECS))
            raise SystemExit(f"Unknown benchmark {name!r}. Allowed values: {allowed}")

    output_dir: Path = context["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_config = {
        "checkpoint_dir": context["checkpoint_dir"],
        "model_name_or_path": context["model_name_or_path"],
        "adapter_path": context["adapter_path"],
        "tokenizer_name_or_path": context["tokenizer_name_or_path"],
        "prompts_file": str(context["prompts_file"]),
        "benchmarks": benchmarks,
        "device": args.device,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "limit": args.limit,
    }
    save_json(output_dir / "evaluation_config.json", evaluation_config)

    scripts_dir = Path(__file__).resolve().parent
    if not args.skip_fixed_prompts:
        fixed_prompt_command = [
            sys.executable,
            str(scripts_dir / "generate.py"),
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
        ]
        if args.attn_implementation:
            fixed_prompt_command.extend(["--attn_implementation", args.attn_implementation])
        if context["adapter_path"]:
            fixed_prompt_command.extend(["--adapter_path", context["adapter_path"]])
        run_command(fixed_prompt_command, args.dry_run)

    if not args.skip_benchmarks:
        run_benchmark_suite(
            model_name_or_path=context["model_name_or_path"],
            adapter_path=context["adapter_path"],
            tokenizer_name_or_path=context["tokenizer_name_or_path"],
            output_dir=output_dir / "benchmarks",
            benchmarks=benchmarks,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            limit=args.limit,
            log_samples=args.log_samples,
            dry_run=args.dry_run,
        )

    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
