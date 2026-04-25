#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
sys.path.append(str(SCRIPT_DIR.parent))

from common import (
    _has_training_generation_blocks,
    load_dataset_split,
    maybe_sample_dataset,
    read_jsonl,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="监控 checkpoint 的 <|im_end|> NLL 与自然停止行为。")
    parser.add_argument("--checkpoint_dir", required=True, help="训练输出目录或某个 checkpoint-xxx 目录。")
    parser.add_argument("--dataset_name", default=None, help="可选覆盖数据集名称或本地路径。")
    parser.add_argument("--dataset_config", default=None, help="可选覆盖数据集 config。")
    parser.add_argument("--split", default=None, help="可选覆盖 split；默认优先用 run_config 里的 eval split。")
    parser.add_argument("--messages_field", default=None, help="可选覆盖消息字段名。")
    parser.add_argument("--max_dataset_samples", type=int, default=32, help="普通 assistant NLL 监控最多抽多少条样本。")
    parser.add_argument("--prompts_file", default=None, help="fixed prompts jsonl；不填默认仓库自带样例。")
    parser.add_argument("--max_prompt_samples", type=int, default=12, help="自然停止监控最多跑多少条 fixed prompts。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="固定 prompts 生成的最大 token 数。")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度；0 表示贪心。")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--user_text", default="Reply with nothing.", help="空 assistant probe 的 user 文本。")
    parser.add_argument("--assistant_text", default="", help="空 assistant probe 的 assistant 文本。")
    parser.add_argument("--output_json", default=None, help="可选输出 json 路径。")
    return parser.parse_args()
def find_run_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "run_config.json").exists():
            return candidate
    raise SystemExit(f"Could not find run_config.json from {path}.")


def resolve_context(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint_dir).resolve()
    run_root = find_run_root(checkpoint_path)
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))

    adapter_path = checkpoint_path if (checkpoint_path / "adapter_config.json").exists() else run_root
    tokenizer_path = checkpoint_path if (checkpoint_path / "tokenizer_config.json").exists() else run_root
    prompts_file = (
        Path(args.prompts_file).resolve()
        if args.prompts_file
        else ROOT_DIR / "data" / "fixed_prompts.jsonl"
    )

    dataset_name = args.dataset_name or run_config.get("eval_dataset") or run_config["train_dataset"]
    dataset_config = args.dataset_config if args.dataset_config is not None else run_config.get("eval_dataset_config")
    split = args.split or run_config.get("eval_split") or "test"
    messages_field = args.messages_field or run_config.get("messages_field") or "messages"

    return {
        "checkpoint_dir": str(checkpoint_path),
        "run_root": str(run_root),
        "model_name_or_path": run_config["model_name_or_path"],
        "adapter_path": str(adapter_path),
        "tokenizer_name_or_path": str(tokenizer_path),
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "messages_field": messages_field,
        "prompts_file": str(prompts_file),
    }


def choose_device_and_dtype(torch: Any) -> tuple[str, Any]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_tokenizer_and_model(context: dict[str, Any]) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(context["tokenizer_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device, dtype = choose_device_and_dtype(torch)
    model = AutoModelForCausalLM.from_pretrained(context["model_name_or_path"], torch_dtype=dtype)

    adapter_loaded = False
    if Path(context["adapter_path"]).joinpath("adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, context["adapter_path"])
        adapter_loaded = True

    model = model.to(device)
    model.eval()
    return tokenizer, model, device if not adapter_loaded else f"{device}+adapter"


def encode_conversation(tokenizer: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    encode_kwargs = {
        "conversation": messages,
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    if _has_training_generation_blocks(getattr(tokenizer, "chat_template", None)):
        encode_kwargs["return_assistant_tokens_mask"] = True

    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoding = tokenizer.apply_chat_template(**encode_kwargs)
    return {
        "rendered": rendered,
        "input_ids": list(encoding["input_ids"]),
        "assistant_masks": list(encoding.get("assistant_masks", [])),
    }


def compute_per_token_nll(model: Any, input_ids: list[int], labels: list[int]) -> list[float | None]:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    input_ids_tensor = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(input_ids_tensor)
    labels_tensor = torch.tensor([labels], device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask, labels=labels_tensor)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels_tensor[:, 1:].contiguous()
        per_token_nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

    values: list[float | None] = []
    for idx in range(shift_labels.shape[-1]):
        label_value = int(shift_labels[0, idx].item())
        if label_value == -100:
            values.append(None)
        else:
            values.append(float(per_token_nll[0, idx].item()))
    return values


def collect_normal_im_end_nlls(
    dataset: Any,
    messages_field: str,
    tokenizer: Any,
    model: Any,
    im_end_token_id: int | None,
) -> dict[str, Any]:
    if im_end_token_id is None or im_end_token_id < 0:
        return {
            "samples_evaluated": len(dataset),
            "samples_with_im_end_target": 0,
            "im_end_target_count": 0,
            "avg_nll": None,
            "sample_rows": [],
        }

    sample_rows: list[dict[str, Any]] = []
    all_nlls: list[float] = []

    for sample_index, example in enumerate(dataset):
        messages = example[messages_field]
        encoded = encode_conversation(tokenizer, messages)
        input_ids = encoded["input_ids"]
        assistant_masks = encoded["assistant_masks"]
        if not assistant_masks:
            continue

        labels = [token_id if assistant_masks[idx] else -100 for idx, token_id in enumerate(input_ids)]
        per_token_nll = compute_per_token_nll(model, input_ids, labels)

        sample_nlls: list[float] = []
        for target_idx in range(1, len(input_ids)):
            if not assistant_masks[target_idx]:
                continue
            if input_ids[target_idx] != im_end_token_id:
                continue
            nll_value = per_token_nll[target_idx - 1]
            if nll_value is None:
                continue
            sample_nlls.append(nll_value)
            all_nlls.append(nll_value)

        if sample_nlls:
            sample_rows.append(
                {
                    "sample_index": sample_index,
                    "message_count": len(messages),
                    "im_end_target_count": len(sample_nlls),
                    "avg_nll": round(mean(sample_nlls), 6),
                }
            )

    return {
        "samples_evaluated": len(dataset),
        "samples_with_im_end_target": len(sample_rows),
        "im_end_target_count": len(all_nlls),
        "avg_nll": round(mean(all_nlls), 6) if all_nlls else None,
        "min_nll": round(min(all_nlls), 6) if all_nlls else None,
        "max_nll": round(max(all_nlls), 6) if all_nlls else None,
        "sample_rows": sample_rows[:10],
    }


def probe_empty_assistant_nll(
    tokenizer: Any,
    model: Any,
    user_text: str,
    assistant_text: str,
    im_end_token_id: int | None,
) -> dict[str, Any]:
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    encoded = encode_conversation(tokenizer, messages)
    input_ids = encoded["input_ids"]
    assistant_masks = encoded["assistant_masks"]
    labels = [token_id if not assistant_masks or assistant_masks[idx] else -100 for idx, token_id in enumerate(input_ids)]
    per_token_nll = compute_per_token_nll(model, input_ids, labels)

    rows = []
    eos_nlls: list[float] = []
    for target_idx in range(1, len(input_ids)):
        if assistant_masks and not assistant_masks[target_idx]:
            continue
        token_id = input_ids[target_idx]
        piece = tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")
        nll_value = per_token_nll[target_idx - 1]
        if nll_value is None:
            continue
        is_im_end = token_id == im_end_token_id
        rows.append(
            {
                "source_idx": target_idx - 1,
                "target_idx": target_idx,
                "token_id": token_id,
                "piece": piece,
                "is_im_end": is_im_end,
                "nll": round(nll_value, 6),
            }
        )
        if is_im_end:
            eos_nlls.append(nll_value)

    return {
        "user_text": user_text,
        "assistant_text": assistant_text,
        "assistant_mask_positions": [idx for idx, value in enumerate(assistant_masks) if value],
        "im_end_target_count": len(eos_nlls),
        "avg_nll": round(mean(eos_nlls), 6) if eos_nlls else None,
        "rendered": encoded["rendered"],
        "targets": rows,
    }


def token_positions(token_ids: list[int], target_id: int | None) -> list[int]:
    if target_id is None or target_id < 0:
        return []
    return [index for index, token_id in enumerate(token_ids) if token_id == target_id]


def run_natural_stop_probe(
    prompts: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    native_eos_id: int | None,
    im_end_token_id: int | None,
    endoftext_token_id: int | None,
) -> dict[str, Any]:
    import torch

    prompt_rows = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(prompt["messages"], tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(rendered, return_tensors="pt").to(next(model.parameters()).device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = model.generate(**model_inputs, **generation_kwargs)

        prompt_length = model_inputs["input_ids"].shape[-1]
        completion_ids = output_ids[0][prompt_length:].detach().cpu().tolist()
        native_eos_positions = token_positions(completion_ids, native_eos_id)
        im_end_positions = token_positions(completion_ids, im_end_token_id)
        endoftext_positions = token_positions(completion_ids, endoftext_token_id)
        prompt_rows.append(
            {
                "id": prompt.get("id"),
                "completion_token_count": len(completion_ids),
                "stopped_early": len(completion_ids) < max_new_tokens,
                "native_eos_positions": native_eos_positions,
                "im_end_token_positions": im_end_positions,
                "endoftext_token_positions": endoftext_positions,
            }
        )

    return {
        "prompt_count": len(prompt_rows),
        "stopped_early_rate": round(mean(1.0 if row["stopped_early"] else 0.0 for row in prompt_rows), 6) if prompt_rows else None,
        "native_eos_stop_rate": round(mean(1.0 if row["native_eos_positions"] else 0.0 for row in prompt_rows), 6) if prompt_rows else None,
        "im_end_generated_rate": round(mean(1.0 if row["im_end_token_positions"] else 0.0 for row in prompt_rows), 6) if prompt_rows else None,
        "endoftext_generated_rate": round(mean(1.0 if row["endoftext_token_positions"] else 0.0 for row in prompt_rows), 6) if prompt_rows else None,
        "mean_completion_tokens": round(mean(row["completion_token_count"] for row in prompt_rows), 6) if prompt_rows else None,
        "prompt_rows": prompt_rows,
    }


def main() -> None:
    args = parse_args()
    context = resolve_context(args)

    dataset = load_dataset_split(context["dataset_name"], context["dataset_config"], context["split"])
    dataset = maybe_sample_dataset(dataset, args.max_dataset_samples, args.seed)

    tokenizer, model, device_label = load_tokenizer_and_model(context)
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    native_eos_id = tokenizer.eos_token_id

    prompts = read_jsonl(context["prompts_file"])[: args.max_prompt_samples]

    report = {
        "context": context,
        "tokenizer": {
            "eos_token": tokenizer.eos_token,
            "eos_token_id": native_eos_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "im_end_token_id": im_end_token_id,
            "endoftext_token_id": endoftext_token_id,
        },
        "runtime": {
            "device": device_label,
            "dataset_sample_count": len(dataset),
            "prompt_sample_count": len(prompts),
        },
        "normal_assistant_im_end": collect_normal_im_end_nlls(
            dataset=dataset,
            messages_field=context["messages_field"],
            tokenizer=tokenizer,
            model=model,
            im_end_token_id=im_end_token_id,
        ),
        "empty_assistant_im_end": probe_empty_assistant_nll(
            tokenizer=tokenizer,
            model=model,
            user_text=args.user_text,
            assistant_text=args.assistant_text,
            im_end_token_id=im_end_token_id,
        ),
        "natural_stop": run_natural_stop_probe(
            prompts=prompts,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            native_eos_id=native_eos_id,
            im_end_token_id=im_end_token_id,
            endoftext_token_id=endoftext_token_id,
        ),
    }

    if args.output_json:
        save_json(args.output_json, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
