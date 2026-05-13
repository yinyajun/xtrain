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

from dataset_utils import (
    _has_training_generation_blocks,
    load_dataset_split,
    maybe_sample_dataset,
    read_jsonl,
    save_json,
)


"""
Unified checkpoint monitor.

适用方法：
- single：调一条 fixed prompt，看渲染、生成文本、stop token 位置和前几步 top-k。
- compare：同一条 fixed prompt 下，对比 base model 和 SFT checkpoint 的 stop token 倾向。
- stop：批量监控 checkpoint 的 <|im_end|> NLL、空 assistant probe 和自然停止率。

这个文件刻意不再承担 chat template / 训练样本预览职责；那类检查适合放回训练入口或更轻量的
数据预览工具里。这里专注回答一个问题：模型训练后，生成和停止行为是否正常。
"""


def load_checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    run_config = checkpoint_dir / "run_config.json"
    if not run_config.exists():
        raise SystemExit(
            f"Checkpoint directory {checkpoint_dir} does not contain run_config.json. "
            "Pass --model_name_or_path and --adapter_path explicitly where supported."
        )
    return json.loads(run_config.read_text(encoding="utf-8"))


def find_run_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "run_config.json").exists():
            return candidate
    raise SystemExit(f"Could not find run_config.json from {path}.")


def default_prompts_file(args: argparse.Namespace) -> str:
    return str(Path(args.prompts_file).resolve() if args.prompts_file else ROOT_DIR / "data" / "fixed_prompts.jsonl")


def select_prompt(prompts: list[dict[str, Any]], prompt_id: str | None, prompt_index: int) -> dict[str, Any]:
    if prompt_id:
        for prompt in prompts:
            if prompt.get("id") == prompt_id:
                return prompt
        raise SystemExit(f"Prompt id {prompt_id!r} not found in prompts file.")

    if prompt_index < 0 or prompt_index >= len(prompts):
        raise SystemExit(f"Prompt index {prompt_index} is out of range for {len(prompts)} prompts.")
    return prompts[prompt_index]


def token_positions(token_ids: list[int], target_id: int | None) -> list[int]:
    if target_id is None or target_id < 0:
        return []
    return [index for index, token_id in enumerate(token_ids) if token_id == target_id]


def decode_token_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")


def choose_dtype(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_tokenizer(tokenizer_name_or_path: str, revision: str | None, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name_or_path: str,
    adapter_path: str | None,
    attn_implementation: str | None,
    revision: str | None,
    trust_remote_code: bool,
    device_map: str | None = "auto",
) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    dtype = choose_dtype(torch)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
    }
    if torch.cuda.is_available() and device_map:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if adapter_path and Path(adapter_path).joinpath("adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    if not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
    model.eval()
    return model


def tracked_token_ids(tokenizer: Any) -> dict[str, int | None]:
    return {
        "native_eos": tokenizer.eos_token_id,
        "im_start": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "endoftext": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    }


def tokenizer_summary(tokenizer: Any) -> dict[str, Any]:
    tracked = tracked_token_ids(tokenizer)
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "im_start_token_id": tracked["im_start"],
        "im_end_token_id": tracked["im_end"],
        "endoftext_token_id": tracked["endoftext"],
    }


def generation_kwargs(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        kwargs["do_sample"] = False
    return kwargs


def compute_token_rank_and_logit(step_logits: Any, token_id: int | None) -> tuple[int | None, float | None]:
    if token_id is None or token_id < 0:
        return None, None
    token_logit = float(step_logits[token_id].item())
    higher_than_token = int((step_logits > step_logits[token_id]).sum().item())
    return higher_than_token + 1, token_logit


def inspect_top_k_next_tokens(
    model: Any,
    tokenizer: Any,
    output_ids: Any,
    prompt_length: int,
    inspect_steps: int,
    top_k: int,
) -> list[dict[str, Any]]:
    import torch

    if inspect_steps <= 0 or top_k <= 0:
        return []

    attention_mask = torch.ones_like(output_ids)
    with torch.inference_mode():
        logits = model(input_ids=output_ids, attention_mask=attention_mask).logits[0]

    full_ids = output_ids[0].detach().cpu().tolist()
    completion_length = len(full_ids) - prompt_length
    limit = min(inspect_steps, completion_length)
    tracked = tracked_token_ids(tokenizer)
    summaries: list[dict[str, Any]] = []

    for step in range(limit):
        source_index = prompt_length - 1 + step
        chosen_index = prompt_length + step
        chosen_token_id = full_ids[chosen_index]
        step_logits = logits[source_index]
        top_values, top_indices = torch.topk(step_logits, k=min(top_k, step_logits.shape[-1]))

        tracked_tokens: dict[str, Any] = {}
        for name, token_id in tracked.items():
            rank, logit = compute_token_rank_and_logit(step_logits, token_id)
            tracked_tokens[name] = {
                "token_id": token_id,
                "piece": decode_token_piece(tokenizer, token_id) if token_id is not None and token_id >= 0 else None,
                "rank": rank,
                "logit": round(logit, 6) if logit is not None else None,
                "is_chosen": token_id == chosen_token_id if token_id is not None else False,
            }

        summaries.append(
            {
                "step": step,
                "source_index": source_index,
                "source_piece": decode_token_piece(tokenizer, full_ids[source_index]),
                "chosen_index": chosen_index,
                "chosen_token_id": chosen_token_id,
                "chosen_piece": decode_token_piece(tokenizer, chosen_token_id),
                "chosen_is_native_eos": chosen_token_id == tokenizer.eos_token_id,
                "tracked_tokens": tracked_tokens,
                "top_k": [
                    {
                        "rank": rank,
                        "token_id": candidate_id,
                        "piece": decode_token_piece(tokenizer, candidate_id),
                        "logit": round(float(logit_value), 6),
                        "is_chosen": candidate_id == chosen_token_id,
                        "is_native_eos": candidate_id == tokenizer.eos_token_id,
                    }
                    for rank, (candidate_id, logit_value) in enumerate(
                        zip(top_indices.detach().cpu().tolist(), top_values.detach().cpu().tolist()),
                        start=1,
                    )
                ],
            }
        )

    return summaries


def generate_for_prompt(model: Any, tokenizer: Any, prompt: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    rendered_prompt = tokenizer.apply_chat_template(prompt["messages"], tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
    kwargs = generation_kwargs(args, tokenizer)
    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, **kwargs)

    prompt_length = model_inputs["input_ids"].shape[-1]
    completion_ids = output_ids[0][prompt_length:]
    completion_ids_list = completion_ids.detach().cpu().tolist()
    tracked = tracked_token_ids(tokenizer)
    return {
        "generation_args": kwargs,
        "rendered_prompt": rendered_prompt,
        "output_ids": output_ids,
        "prompt_length": prompt_length,
        "completion_ids": completion_ids_list,
        "raw_completion": tokenizer.decode(completion_ids, skip_special_tokens=False),
        "clean_completion": tokenizer.decode(completion_ids, skip_special_tokens=True).strip(),
        "completion_token_count": len(completion_ids_list),
        "native_eos_token_positions": token_positions(completion_ids_list, tracked["native_eos"]),
        "im_start_token_positions": token_positions(completion_ids_list, tracked["im_start"]),
        "im_end_token_positions": token_positions(completion_ids_list, tracked["im_end"]),
        "endoftext_token_positions": token_positions(completion_ids_list, tracked["endoftext"]),
    }


def resolve_single_context(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    metadata = load_checkpoint_metadata(checkpoint_dir) if checkpoint_dir else None
    model_name_or_path = args.model_name_or_path or (metadata.get("model_name_or_path") if metadata else None)
    adapter_path = args.adapter_path or (str(checkpoint_dir) if checkpoint_dir else None)
    tokenizer_name_or_path = args.tokenizer_name_or_path or adapter_path or model_name_or_path
    if not model_name_or_path:
        raise SystemExit("Could not resolve model_name_or_path. Pass --model_name_or_path or --checkpoint_dir.")
    return {
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "prompts_file": default_prompts_file(args),
    }


def run_single(args: argparse.Namespace) -> None:
    # 适用于“某条固定 prompt 为什么这样生成/为什么停不住”的深挖。
    from transformers import set_seed

    set_seed(args.seed)
    context = resolve_single_context(args)
    prompt = select_prompt(read_jsonl(context["prompts_file"]), args.prompt_id, args.prompt_index)
    tokenizer = load_tokenizer(context["tokenizer_name_or_path"], args.revision, args.trust_remote_code)
    model = load_model(
        context["model_name_or_path"],
        context["adapter_path"],
        args.attn_implementation,
        args.revision,
        args.trust_remote_code,
    )
    generated = generate_for_prompt(model, tokenizer, prompt, args)
    report = {
        "context": context,
        "prompt": {"id": prompt.get("id"), "tags": prompt.get("tags", []), "messages": prompt.get("messages", [])},
        "tokenizer": tokenizer_summary(tokenizer),
        "generation": {key: value for key, value in generated.items() if key not in {"output_ids"}},
        "top_k_next_tokens": inspect_top_k_next_tokens(
            model=model,
            tokenizer=tokenizer,
            output_ids=generated["output_ids"],
            prompt_length=generated["prompt_length"],
            inspect_steps=args.inspect_steps,
            top_k=args.top_k,
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def resolve_compare_context(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    metadata = load_checkpoint_metadata(checkpoint_dir)
    base_model_name_or_path = args.base_model_name_or_path or metadata["model_name_or_path"]
    base_tokenizer_name_or_path = args.base_tokenizer_name_or_path or metadata.get("tokenizer_name_or_path") or base_model_name_or_path
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "prompts_file": default_prompts_file(args),
        "base_model_name_or_path": base_model_name_or_path,
        "base_tokenizer_name_or_path": base_tokenizer_name_or_path,
        "checkpoint_model_name_or_path": metadata["model_name_or_path"],
        "checkpoint_tokenizer_name_or_path": str(checkpoint_dir),
        "checkpoint_adapter_path": str(checkpoint_dir),
    }


def run_generation_case(
    label: str,
    model_name_or_path: str,
    adapter_path: str | None,
    tokenizer_name_or_path: str,
    prompt: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from transformers import set_seed

    set_seed(args.seed)
    tokenizer = load_tokenizer(tokenizer_name_or_path, args.revision, args.trust_remote_code)
    model = load_model(model_name_or_path, adapter_path, args.attn_implementation, args.revision, args.trust_remote_code)
    generated = generate_for_prompt(model, tokenizer, prompt, args)
    top_k = inspect_top_k_next_tokens(model, tokenizer, generated["output_ids"], generated["prompt_length"], args.inspect_steps, args.top_k)
    return {
        "label": label,
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "tokenizer": tokenizer_summary(tokenizer),
        "clean_completion": generated["clean_completion"],
        "completion_token_count": generated["completion_token_count"],
        "native_eos_token_positions": generated["native_eos_token_positions"],
        "im_end_token_positions": generated["im_end_token_positions"],
        "endoftext_token_positions": generated["endoftext_token_positions"],
        "top_k_next_tokens": top_k,
    }


def tracked_delta(base_step: dict[str, Any], checkpoint_step: dict[str, Any], token_name: str) -> dict[str, Any]:
    base_token = base_step.get("tracked_tokens", {}).get(token_name, {})
    checkpoint_token = checkpoint_step.get("tracked_tokens", {}).get(token_name, {})
    base_rank = base_token.get("rank")
    checkpoint_rank = checkpoint_token.get("rank")
    base_logit = base_token.get("logit")
    checkpoint_logit = checkpoint_token.get("logit")
    return {
        "rank": None if base_rank is None or checkpoint_rank is None else checkpoint_rank - base_rank,
        "logit": None if base_logit is None or checkpoint_logit is None else round(checkpoint_logit - base_logit, 6),
    }


def build_step_comparison(base_case: dict[str, Any], checkpoint_case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    limit = min(len(base_case["top_k_next_tokens"]), len(checkpoint_case["top_k_next_tokens"]))
    for step in range(limit):
        base_step = base_case["top_k_next_tokens"][step]
        checkpoint_step = checkpoint_case["top_k_next_tokens"][step]
        rows.append(
            {
                "step": step,
                "base_chosen_piece": base_step["chosen_piece"],
                "checkpoint_chosen_piece": checkpoint_step["chosen_piece"],
                "delta": {
                    "native_eos": tracked_delta(base_step, checkpoint_step, "native_eos"),
                    "im_end": tracked_delta(base_step, checkpoint_step, "im_end"),
                    "endoftext": tracked_delta(base_step, checkpoint_step, "endoftext"),
                },
                "base_tracked_tokens": base_step["tracked_tokens"],
                "checkpoint_tracked_tokens": checkpoint_step["tracked_tokens"],
            }
        )
    return rows


def run_compare(args: argparse.Namespace) -> None:
    # 适用于判断“SFT 后模型是否更会在对话边界输出 <|im_end|> / eos”。
    context = resolve_compare_context(args)
    prompt = select_prompt(read_jsonl(context["prompts_file"]), args.prompt_id, args.prompt_index)
    base_case = run_generation_case(
        "base",
        context["base_model_name_or_path"],
        None,
        context["base_tokenizer_name_or_path"],
        prompt,
        args,
    )
    checkpoint_case = run_generation_case(
        "checkpoint",
        context["checkpoint_model_name_or_path"],
        context["checkpoint_adapter_path"],
        context["checkpoint_tokenizer_name_or_path"],
        prompt,
        args,
    )
    report = {
        "context": context,
        "prompt": {"id": prompt.get("id"), "tags": prompt.get("tags", []), "messages": prompt.get("messages", [])},
        "base": base_case,
        "checkpoint": checkpoint_case,
        "step_comparison": build_step_comparison(base_case, checkpoint_case),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def resolve_stop_context(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint_dir).resolve()
    run_root = find_run_root(checkpoint_path)
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    adapter_path = checkpoint_path if (checkpoint_path / "adapter_config.json").exists() else run_root
    tokenizer_path = checkpoint_path if (checkpoint_path / "tokenizer_config.json").exists() else run_root
    dataset_name = args.dataset_name or run_config.get("eval_dataset") or run_config["train_dataset"]
    dataset_config = args.dataset_config if args.dataset_config is not None else run_config.get("eval_dataset_config")
    return {
        "checkpoint_dir": str(checkpoint_path),
        "run_root": str(run_root),
        "model_name_or_path": run_config["model_name_or_path"],
        "adapter_path": str(adapter_path),
        "tokenizer_name_or_path": str(tokenizer_path),
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": args.split or run_config.get("eval_split") or "test",
        "messages_field": args.messages_field or run_config.get("messages_field") or "messages",
        "prompts_file": default_prompts_file(args),
    }


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
        values.append(None if label_value == -100 else float(per_token_nll[0, idx].item()))
    return values


def collect_normal_im_end_nlls(dataset: Any, messages_field: str, tokenizer: Any, model: Any, im_end_token_id: int | None) -> dict[str, Any]:
    if im_end_token_id is None or im_end_token_id < 0:
        return {"samples_evaluated": len(dataset), "samples_with_im_end_target": 0, "im_end_target_count": 0, "avg_nll": None}

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
        sample_nlls = []
        for target_idx in range(1, len(input_ids)):
            if assistant_masks[target_idx] and input_ids[target_idx] == im_end_token_id:
                nll_value = per_token_nll[target_idx - 1]
                if nll_value is not None:
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


def probe_empty_assistant_nll(tokenizer: Any, model: Any, user_text: str, assistant_text: str, im_end_token_id: int | None) -> dict[str, Any]:
    messages = [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}]
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
        nll_value = per_token_nll[target_idx - 1]
        if nll_value is None:
            continue
        is_im_end = token_id == im_end_token_id
        rows.append(
            {
                "source_idx": target_idx - 1,
                "target_idx": target_idx,
                "token_id": token_id,
                "piece": decode_token_piece(tokenizer, token_id),
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


def run_natural_stop_probe(prompts: list[dict[str, Any]], tokenizer: Any, model: Any, args: argparse.Namespace) -> dict[str, Any]:
    tracked = tracked_token_ids(tokenizer)
    prompt_rows = []
    for prompt in prompts:
        generated = generate_for_prompt(model, tokenizer, prompt, args)
        prompt_rows.append(
            {
                "id": prompt.get("id"),
                "completion_token_count": generated["completion_token_count"],
                "stopped_early": generated["completion_token_count"] < args.max_new_tokens,
                "native_eos_positions": token_positions(generated["completion_ids"], tracked["native_eos"]),
                "im_end_token_positions": token_positions(generated["completion_ids"], tracked["im_end"]),
                "endoftext_token_positions": token_positions(generated["completion_ids"], tracked["endoftext"]),
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


def run_stop(args: argparse.Namespace) -> None:
    # 适用于训练后做健康检查：停止 token 是否学到、固定 prompt 是否自然停止。
    from transformers import set_seed

    set_seed(args.seed)
    context = resolve_stop_context(args)
    dataset = load_dataset_split(context["dataset_name"], context["dataset_config"], context["split"])
    dataset = maybe_sample_dataset(dataset, args.max_dataset_samples, args.seed)
    tokenizer = load_tokenizer(context["tokenizer_name_or_path"], None, False)
    model = load_model(context["model_name_or_path"], context["adapter_path"], None, None, False, device_map=None)
    tracked = tracked_token_ids(tokenizer)
    prompts = read_jsonl(context["prompts_file"])[: args.max_prompt_samples]

    report = {
        "context": context,
        "tokenizer": tokenizer_summary(tokenizer),
        "runtime": {"dataset_sample_count": len(dataset), "prompt_sample_count": len(prompts)},
        "normal_assistant_im_end": collect_normal_im_end_nlls(
            dataset=dataset,
            messages_field=context["messages_field"],
            tokenizer=tokenizer,
            model=model,
            im_end_token_id=tracked["im_end"],
        ),
        "empty_assistant_im_end": probe_empty_assistant_nll(tokenizer, model, args.user_text, args.assistant_text, tracked["im_end"]),
        "natural_stop": run_natural_stop_probe(prompts, tokenizer, model, args),
    }
    if args.output_json:
        save_json(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompts_file", default=None, help="fixed prompts jsonl；不填默认使用 sft_study/data/fixed_prompts.jsonl。")
    parser.add_argument("--prompt_id", default=None, help="按 prompt id 选择样本。")
    parser.add_argument("--prompt_index", type=int, default=0, help="如果不传 --prompt_id，就按索引选样本。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最多生成多少 token。")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度；0 表示贪心。")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p。")
    parser.add_argument("--attn_implementation", default=None, help="可选注意力实现，例如 flash_attention_2。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--revision", default=None, help="模型 revision。")
    parser.add_argument("--trust_remote_code", action="store_true", help="是否允许 remote code。")
    parser.add_argument("--inspect_steps", type=int, default=8, help="打印前多少个生成位置的 top-k next-token 候选。")
    parser.add_argument("--top_k", type=int, default=10, help="每个生成位置打印多少个 top-k token。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一的 SFT checkpoint 生成/停止行为监控工具。",
        epilog=(
            "Examples:\n"
            "  python sft_study/scripts/debug/monitor.py single --checkpoint_dir sft_study/outputs/e1_no_robots_smoke --prompt_id one_word_capital\n"
            "  python sft_study/scripts/debug/monitor.py compare --checkpoint_dir sft_study/outputs/e1_no_robots_smoke --prompt_index 0\n"
            "  python sft_study/scripts/debug/monitor.py stop --checkpoint_dir sft_study/outputs/e1_no_robots_smoke"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="调试单条 fixed prompt 的生成细节。")
    single.add_argument("--checkpoint_dir", default=None, help="SFT 输出目录；若存在 run_config.json，会自动补齐模型上下文。")
    single.add_argument("--model_name_or_path", default=None, help="底模名称或本地模型路径。")
    single.add_argument("--adapter_path", default=None, help="可选 LoRA adapter 路径。")
    single.add_argument("--tokenizer_name_or_path", default=None, help="可选 tokenizer 路径。")
    add_generation_args(single)
    single.set_defaults(handler=run_single)

    compare = subparsers.add_parser("compare", help="对比 base model 和 checkpoint 的停止 token 倾向。")
    compare.add_argument("--checkpoint_dir", required=True, help="SFT 输出目录；会自动读取 run_config.json。")
    compare.add_argument("--base_model_name_or_path", default=None, help="可选 base model；不填默认用 run_config.json 里的底模。")
    compare.add_argument("--base_tokenizer_name_or_path", default=None, help="可选 base tokenizer；不填默认用底模路径。")
    add_generation_args(compare)
    compare.set_defaults(handler=run_compare)

    stop = subparsers.add_parser("stop", help="批量监控 checkpoint 的 <|im_end|> NLL 与自然停止行为。")
    stop.add_argument("--checkpoint_dir", required=True, help="训练输出目录或某个 checkpoint-xxx 目录。")
    stop.add_argument("--dataset_name", default=None, help="可选覆盖数据集名称或本地路径。")
    stop.add_argument("--dataset_config", default=None, help="可选覆盖数据集 config。")
    stop.add_argument("--split", default=None, help="可选覆盖 split；默认优先用 run_config 里的 eval split。")
    stop.add_argument("--messages_field", default=None, help="可选覆盖消息字段名。")
    stop.add_argument("--max_dataset_samples", type=int, default=32, help="普通 assistant NLL 监控最多抽多少条样本。")
    stop.add_argument("--prompts_file", default=None, help="fixed prompts jsonl；不填默认使用 sft_study/data/fixed_prompts.jsonl。")
    stop.add_argument("--max_prompt_samples", type=int, default=12, help="自然停止监控最多跑多少条 fixed prompts。")
    stop.add_argument("--max_new_tokens", type=int, default=256, help="固定 prompts 生成的最大新 token 数。")
    stop.add_argument("--temperature", type=float, default=0.0, help="生成温度；0 表示贪心。")
    stop.add_argument("--top_p", type=float, default=1.0, help="top_p。")
    stop.add_argument("--seed", type=int, default=42, help="随机种子。")
    stop.add_argument("--user_text", default="Reply with nothing.", help="空 assistant probe 的 user 文本。")
    stop.add_argument("--assistant_text", default="", help="空 assistant probe 的 assistant 文本。")
    stop.add_argument("--output_json", default=None, help="可选输出 json 路径。")
    stop.set_defaults(handler=run_stop)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
