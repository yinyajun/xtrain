#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="运行 lm-eval 基准测试，评估底模或 LoRA adapter。")
    parser.add_argument("--model_name_or_path", required=True, help="底模名称或本地模型路径。")
    parser.add_argument("--adapter_path", default=None, help="可选 LoRA adapter 路径。")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="可选 tokenizer 路径。")
    parser.add_argument("--output_dir", required=True, help="lm-eval 输出目录。")
    parser.add_argument("--benchmarks", nargs="*", default=DEFAULT_BENCHMARKS, help="要运行的 benchmark 名称列表。")
    parser.add_argument("--device", default="auto", help="运行设备，例如 auto / cuda:0 / cpu。")
    parser.add_argument("--batch_size", default="auto", help="评估 batch size，可传 auto。")
    parser.add_argument("--dtype", default=None, help="可选推理 dtype，例如 bfloat16。")
    parser.add_argument("--attn_implementation", default=None, help="可选注意力实现，例如 flash_attention_2。")
    parser.add_argument("--limit", default=None, help="可选的 lm-eval limit，接受整数或浮点语义。")
    parser.add_argument("--log_samples", action="store_true", help="是否让 lm-eval 保存样本级日志。")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令，不真正执行。")
    return parser.parse_args()


def ensure_lm_eval_installed() -> None:
    try:
        import lm_eval  # noqa: F401
    except ImportError:
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

    tokenizer_name_or_path = args.tokenizer_name_or_path or args.model_name_or_path
    model_args.append(f"tokenizer={tokenizer_name_or_path}")

    if args.dtype:
        model_args.append(f"dtype={args.dtype}")
    if args.attn_implementation:
        model_args.append(f"attn_implementation={args.attn_implementation}")
    return ",".join(model_args)


def build_command(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    benchmark_output_dir: Path,
) -> list[str]:
    command = [
        *LM_EVAL_COMMAND,
        "--model",
        "hf",
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
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark_output_dir = output_dir / spec.name
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(args, spec, benchmark_output_dir)
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path or args.model_name_or_path,
        "benchmarks": [],
    }

    for benchmark_name in benchmarks:
        spec = BENCHMARK_SPECS[benchmark_name]
        result = run_benchmark(args, spec, output_dir)
        summary["benchmarks"].append(result)

    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
