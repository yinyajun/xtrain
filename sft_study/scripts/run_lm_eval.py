#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from common import DEFAULT_SYSTEM_PROMPT, export_tokenizer_with_system_prompt, save_json


DEFAULT_BENCHMARKS = ["ifeval", "gsm8k", "mmlu", "cmmlu"]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    tasks: tuple[str, ...]
    num_fewshot: int | None = None
    apply_chat_template: bool = False
    batch_size: str | None = None
    gen_kwargs: dict[str, Any] | None = None
    unsafe_code: bool = False


BENCHMARK_SPECS = {
    "ifeval": BenchmarkSpec(
        name="ifeval",
        tasks=("ifeval",),
        num_fewshot=0,
        apply_chat_template=True,
        # IFEval 更偏指令遵循，不需要超长生成。
        # 把最大生成长度收紧到 256，能显著减少单样本耗时。
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
    # 这个脚本封装了 lm-eval，目的是把常用 benchmark 跑法标准化。
    # 默认不再包含 HumanEval，避免执行代码测试，只保留 ifeval/gsm8k/mmlu/cmmlu。
    parser = argparse.ArgumentParser(description="运行 lm-eval 基准测试，评估底模或 LoRA adapter。")
    parser.add_argument("--model_name_or_path", required=True, help="底模名称或本地模型路径。")
    parser.add_argument("--adapter_path", default=None, help="可选 LoRA adapter 路径。")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="可选 tokenizer 路径。")
    parser.add_argument("--output_dir", required=True, help="lm-eval 输出目录。")
    parser.add_argument("--benchmarks", nargs="*", default=DEFAULT_BENCHMARKS, help="要运行的 benchmark 名称列表。")
    parser.add_argument("--model_backend", default="hf", help="lm-eval 使用的模型后端，默认 hf。")
    parser.add_argument("--device", default="auto", help="运行设备，例如 auto / cuda:0 / cpu。")
    parser.add_argument("--batch_size", default="auto", help="评估 batch size，可传 auto。")
    parser.add_argument("--dtype", default=None, help="可选推理 dtype，例如 bfloat16。")
    parser.add_argument("--attn_implementation", default=None, help="可选注意力实现，例如 flash_attention_2。")
    parser.add_argument("--default_system_prompt", default=DEFAULT_SYSTEM_PROMPT, help="统一覆盖 chat template 里的默认 system prompt。")
    parser.add_argument("--revision", default=None, help="模型仓库 revision。")
    parser.add_argument("--trust_remote_code", action="store_true", help="是否允许加载 remote code。")
    parser.add_argument("--load_in_4bit", action="store_true", help="是否以 4-bit 方式加载模型。")
    parser.add_argument("--limit", default=None, help="可选的 lm-eval limit，接受整数或浮点语义。")
    parser.add_argument("--think_end_token", default=None, help="部分模型的思维链终止 token。")
    parser.add_argument("--log_samples", action="store_true", help="是否让 lm-eval 保存样本级日志。")
    parser.add_argument("--extra_model_arg", action="append", default=[], help="额外透传给 model_args 的键值对。")
    parser.add_argument("--extra_eval_arg", action="append", default=[], help="额外透传给 lm-eval CLI 的参数。")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令，不真正执行。")
    return parser.parse_args()


def ensure_lm_eval_installed() -> None:
    if importlib.util.find_spec("lm_eval") is None:
        raise SystemExit(
            "Missing required package: lm_eval. Install benchmark dependencies with "
            "`pip install -r sft_study/requirements.txt`."
        )


def normalize_benchmarks(raw_values: list[str]) -> list[str]:
    values: list[str] = []
    for raw in raw_values:
        for piece in raw.split(","):
            value = piece.strip()
            if not value:
                continue
            if value == "all":
                values.extend(DEFAULT_BENCHMARKS)
            else:
                values.append(value)

    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in BENCHMARK_SPECS:
            allowed = ", ".join(sorted(BENCHMARK_SPECS))
            raise SystemExit(f"Unknown benchmark {value!r}. Allowed values: {allowed}")
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def detect_lm_eval_base_command() -> list[str]:
    probe = subprocess.run(
        [sys.executable, "-m", "lm_eval", "run", "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", "lm_eval", "run"]
    return [sys.executable, "-m", "lm_eval"]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def build_model_args(args: argparse.Namespace) -> str:
    model_args = [f"pretrained={args.model_name_or_path}"]
    if args.adapter_path:
        model_args.append(f"peft={args.adapter_path}")
    tokenizer_name_or_path = getattr(args, "resolved_tokenizer_name_or_path", None) or args.tokenizer_name_or_path
    if tokenizer_name_or_path:
        model_args.append(f"tokenizer={tokenizer_name_or_path}")
    if args.dtype:
        model_args.append(f"dtype={args.dtype}")
    if args.attn_implementation:
        model_args.append(f"attn_implementation={args.attn_implementation}")
    if args.revision:
        model_args.append(f"revision={args.revision}")
    if args.trust_remote_code:
        model_args.append("trust_remote_code=true")
    if args.load_in_4bit:
        model_args.append("load_in_4bit=true")
    if args.think_end_token:
        model_args.append(f"think_end_token={args.think_end_token}")
    model_args.extend(args.extra_model_arg)
    return ",".join(model_args)


def build_command(
    base_cmd: list[str],
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    benchmark_output_dir: Path,
) -> list[str]:
    command = [
        *base_cmd,
        "--model",
        args.model_backend,
        "--model_args",
        build_model_args(args),
        "--tasks",
        ",".join(spec.tasks),
        "--device",
        resolve_device(args.device),
        "--batch_size",
        spec.batch_size or args.batch_size,
        "--output_path",
        str(benchmark_output_dir),
    ]

    if spec.num_fewshot is not None:
        command.extend(["--num_fewshot", str(spec.num_fewshot)])
    if spec.apply_chat_template:
        command.append("--apply_chat_template")
    if spec.gen_kwargs:
        gen_kwargs = ",".join(f"{key}={format_cli_value(value)}" for key, value in spec.gen_kwargs.items())
        command.extend(["--gen_kwargs", gen_kwargs])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.log_samples:
        command.append("--log_samples")
    if spec.unsafe_code:
        command.append("--confirm_run_unsafe_code")
    command.extend(args.extra_eval_arg)
    return command


def find_results_file(output_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for path in sorted(output_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and ("results" in payload or "groups" in payload):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: (0 if "result" in path.name else 1, len(path.parts), path.name))
    return candidates[0]


def extract_numeric_metrics(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for section_name in ("groups", "results"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        for task_name, metrics in section.items():
            if not isinstance(metrics, dict):
                continue
            numeric_metrics: dict[str, float] = {}
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                lowered = metric_name.lower()
                if lowered.endswith("_stderr") or lowered == "stderr":
                    continue
                numeric_metrics[metric_name] = float(value)
            if numeric_metrics:
                summary[task_name] = numeric_metrics
    return summary


def run_benchmark(
    base_cmd: list[str],
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark_output_dir = output_dir / spec.name
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(base_cmd, args, spec, benchmark_output_dir)
    env = os.environ.copy()
    if spec.unsafe_code:
        env["HF_ALLOW_CODE_EVAL"] = "1"

    result: dict[str, Any] = {
        "benchmark": spec.name,
        "tasks": list(spec.tasks),
        "command": command,
        "output_dir": str(benchmark_output_dir),
    }

    if args.dry_run:
        result["skipped"] = True
        return result

    completed = subprocess.run(command, env=env, check=False)
    result["returncode"] = completed.returncode
    if completed.returncode != 0:
        raise SystemExit(f"lm-eval failed for benchmark {spec.name!r} with exit code {completed.returncode}.")

    results_file = find_results_file(benchmark_output_dir)
    if results_file is not None:
        payload = json.loads(results_file.read_text(encoding="utf-8"))
        result["results_file"] = str(results_file)
        result["metrics"] = extract_numeric_metrics(payload)
    return result


def main() -> None:
    args = parse_args()
    ensure_lm_eval_installed()

    benchmarks = normalize_benchmarks(args.benchmarks)
    base_cmd = detect_lm_eval_base_command()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_source = args.tokenizer_name_or_path or args.model_name_or_path
    resolved_tokenizer_name_or_path, system_prompt_overridden = export_tokenizer_with_system_prompt(
        tokenizer_name_or_path=tokenizer_source,
        output_dir=output_dir / "_tokenizer",
        system_prompt=args.default_system_prompt,
    )
    args.resolved_tokenizer_name_or_path = resolved_tokenizer_name_or_path

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "resolved_tokenizer_name_or_path": resolved_tokenizer_name_or_path,
        "default_system_prompt": args.default_system_prompt,
        "system_prompt_overridden": system_prompt_overridden,
        "benchmarks": [],
    }

    for benchmark_name in benchmarks:
        spec = BENCHMARK_SPECS[benchmark_name]
        result = run_benchmark(base_cmd, args, spec, output_dir)
        summary["benchmarks"].append(result)

    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
