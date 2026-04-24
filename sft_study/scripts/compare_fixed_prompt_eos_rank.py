#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from common import DEFAULT_SYSTEM_PROMPT, apply_default_system_prompt_to_tokenizer, ensure_packages, read_jsonl
from debug_single_fixed_prompt import inspect_top_k_next_tokens, load_checkpoint_metadata, select_prompt, token_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比 base model 和 checkpoint 在同一条 fixed prompt 上的 native eos 与固定 stop token 倾向。"
    )
    parser.add_argument("--checkpoint_dir", required=True, help="SFT 输出目录；会自动读取 run_config.json。")
    parser.add_argument("--base_model_name_or_path", default=None, help="可选 base model；不填默认用 run_config.json 里的底模。")
    parser.add_argument("--base_tokenizer_name_or_path", default=None, help="可选 base tokenizer；不填默认用底模路径。")
    parser.add_argument("--prompts_file", default=None, help="fixed prompts jsonl；不填默认仓库自带样例。")
    parser.add_argument("--prompt_id", default=None, help="按 prompt id 选择样本。")
    parser.add_argument("--prompt_index", type=int, default=0, help="如果不传 --prompt_id，就按索引选样本。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最多生成多少 token。")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度；0 表示贪心。")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p。")
    parser.add_argument("--attn_implementation", default=None, help="可选注意力实现，例如 flash_attention_2。")
    parser.add_argument("--default_system_prompt", default=None, help="统一覆盖默认 system prompt。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--revision", default=None, help="模型 revision。")
    parser.add_argument("--trust_remote_code", action="store_true", help="是否允许 remote code。")
    parser.add_argument("--inspect_steps", type=int, default=8, help="比较前多少个生成 step。")
    parser.add_argument("--top_k", type=int, default=10, help="每个 step 保存多少个 top-k 候选。")
    return parser.parse_args()


def resolve_context(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    metadata = load_checkpoint_metadata(checkpoint_dir)
    prompts_file = (
        Path(args.prompts_file).resolve()
        if args.prompts_file
        else Path(__file__).resolve().parent.parent / "data" / "fixed_prompts.jsonl"
    )
    default_system_prompt = args.default_system_prompt or metadata.get("default_system_prompt") or DEFAULT_SYSTEM_PROMPT
    base_model_name_or_path = args.base_model_name_or_path or metadata["model_name_or_path"]
    base_tokenizer_name_or_path = args.base_tokenizer_name_or_path or metadata.get("tokenizer_name_or_path") or base_model_name_or_path
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "prompts_file": str(prompts_file),
        "default_system_prompt": default_system_prompt,
        "base_model_name_or_path": base_model_name_or_path,
        "base_tokenizer_name_or_path": base_tokenizer_name_or_path,
        "checkpoint_model_name_or_path": metadata["model_name_or_path"],
        "checkpoint_tokenizer_name_or_path": str(checkpoint_dir),
        "checkpoint_adapter_path": str(checkpoint_dir),
    }


def load_tokenizer(tokenizer_name_or_path: str, default_system_prompt: str, revision: str | None, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_default_system_prompt_to_tokenizer(tokenizer, default_system_prompt)
    return tokenizer


def build_tracked_token_ids(tokenizer: Any) -> dict[str, int | None]:
    return {
        "native_eos": tokenizer.eos_token_id,
        "im_end": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "endoftext": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    }


def summarize_tokenizer(tokenizer: Any, tracked_token_ids: dict[str, int | None]) -> dict[str, Any]:
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "im_end_token_id": tracked_token_ids["im_end"],
        "endoftext_token_id": tracked_token_ids["endoftext"],
    }


def load_model(
    model_name_or_path: str,
    adapter_path: str | None,
    attn_implementation: str | None,
    revision: str | None,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModelForCausalLM

    model_dtype = None
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def run_case(
    label: str,
    model_name_or_path: str,
    adapter_path: str | None,
    tokenizer_name_or_path: str,
    prompt: dict[str, Any],
    args: argparse.Namespace,
    default_system_prompt: str,
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    set_seed(args.seed)
    tokenizer = load_tokenizer(
        tokenizer_name_or_path=tokenizer_name_or_path,
        default_system_prompt=default_system_prompt,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    tracked_token_ids = build_tracked_token_ids(tokenizer)
    model = load_model(
        model_name_or_path=model_name_or_path,
        adapter_path=adapter_path,
        attn_implementation=args.attn_implementation,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )

    rendered_prompt = tokenizer.apply_chat_template(prompt["messages"], tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = model_inputs["input_ids"].shape[-1]
    completion_ids = output_ids[0][prompt_length:]
    completion_ids_list = completion_ids.detach().cpu().tolist()
    top_k_debug = inspect_top_k_next_tokens(
        model=model,
        tokenizer=tokenizer,
        output_ids=output_ids,
        prompt_length=prompt_length,
        inspect_steps=args.inspect_steps,
        top_k=args.top_k,
        tracked_token_ids=tracked_token_ids,
    )

    return {
        "label": label,
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "tokenizer": summarize_tokenizer(tokenizer, tracked_token_ids),
        "clean_completion": tokenizer.decode(completion_ids, skip_special_tokens=True).strip(),
        "completion_token_count": len(completion_ids_list),
        "native_eos_token_positions": token_positions(completion_ids_list, tracked_token_ids["native_eos"]),
        "im_end_token_positions": token_positions(completion_ids_list, tracked_token_ids["im_end"]),
        "endoftext_token_positions": token_positions(completion_ids_list, tracked_token_ids["endoftext"]),
        "top_k_next_tokens": top_k_debug,
    }


def summarize_tracked_tokens(step: dict[str, Any]) -> dict[str, Any]:
    tracked = step.get("tracked_tokens", {})
    return {
        "native_eos": tracked.get("native_eos"),
        "im_end": tracked.get("im_end"),
        "endoftext": tracked.get("endoftext"),
    }


def build_tracked_delta(
    base_step: dict[str, Any],
    checkpoint_step: dict[str, Any],
    token_name: str,
) -> dict[str, Any]:
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
    base_steps = base_case["top_k_next_tokens"]
    checkpoint_steps = checkpoint_case["top_k_next_tokens"]
    limit = min(len(base_steps), len(checkpoint_steps))
    rows = []
    for step in range(limit):
        base_step = base_steps[step]
        checkpoint_step = checkpoint_steps[step]
        rows.append(
            {
                "step": step,
                "base": {
                    "chosen_piece": base_step["chosen_piece"],
                    "chosen_is_native_eos": base_step["chosen_is_eos"],
                    "native_eos_rank": base_step["eos_rank"],
                    "native_eos_logit": base_step["eos_logit"],
                    "tracked_tokens": summarize_tracked_tokens(base_step),
                },
                "checkpoint": {
                    "chosen_piece": checkpoint_step["chosen_piece"],
                    "chosen_is_native_eos": checkpoint_step["chosen_is_eos"],
                    "native_eos_rank": checkpoint_step["eos_rank"],
                    "native_eos_logit": checkpoint_step["eos_logit"],
                    "tracked_tokens": summarize_tracked_tokens(checkpoint_step),
                },
                "delta": {
                    "native_eos_rank": None
                    if base_step["eos_rank"] is None or checkpoint_step["eos_rank"] is None
                    else checkpoint_step["eos_rank"] - base_step["eos_rank"],
                    "native_eos_logit": None
                    if base_step["eos_logit"] is None or checkpoint_step["eos_logit"] is None
                    else round(checkpoint_step["eos_logit"] - base_step["eos_logit"], 6),
                    "im_end": build_tracked_delta(base_step, checkpoint_step, "im_end"),
                    "endoftext": build_tracked_delta(base_step, checkpoint_step, "endoftext"),
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    ensure_packages()
    context = resolve_context(args)

    prompts = read_jsonl(context["prompts_file"])
    prompt = select_prompt(prompts, args.prompt_id, args.prompt_index)

    base_case = run_case(
        label="base",
        model_name_or_path=context["base_model_name_or_path"],
        adapter_path=None,
        tokenizer_name_or_path=context["base_tokenizer_name_or_path"],
        prompt=prompt,
        args=args,
        default_system_prompt=context["default_system_prompt"],
    )
    checkpoint_case = run_case(
        label="checkpoint",
        model_name_or_path=context["checkpoint_model_name_or_path"],
        adapter_path=context["checkpoint_adapter_path"],
        tokenizer_name_or_path=context["checkpoint_tokenizer_name_or_path"],
        prompt=prompt,
        args=args,
        default_system_prompt=context["default_system_prompt"],
    )

    report = {
        "context": context,
        "prompt": {
            "id": prompt.get("id"),
            "tags": prompt.get("tags", []),
            "messages": prompt.get("messages", []),
        },
        "base": base_case,
        "checkpoint": checkpoint_case,
        "step_comparison": build_step_comparison(base_case, checkpoint_case),
    }

    print(json.dumps(report["context"], ensure_ascii=False, indent=2))
    print()
    print("=== Prompt ===")
    print(json.dumps(report["prompt"], ensure_ascii=False, indent=2))
    print()
    print("=== Base Summary ===")
    print(json.dumps(
        {
            "model_name_or_path": base_case["model_name_or_path"],
            "tokenizer_name_or_path": base_case["tokenizer_name_or_path"],
            "tokenizer": base_case["tokenizer"],
            "completion_token_count": base_case["completion_token_count"],
            "native_eos_token_positions": base_case["native_eos_token_positions"],
            "im_end_token_positions": base_case["im_end_token_positions"],
            "endoftext_token_positions": base_case["endoftext_token_positions"],
            "clean_completion": base_case["clean_completion"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print("=== Checkpoint Summary ===")
    print(json.dumps(
        {
            "model_name_or_path": checkpoint_case["model_name_or_path"],
            "adapter_path": checkpoint_case["adapter_path"],
            "tokenizer_name_or_path": checkpoint_case["tokenizer_name_or_path"],
            "tokenizer": checkpoint_case["tokenizer"],
            "completion_token_count": checkpoint_case["completion_token_count"],
            "native_eos_token_positions": checkpoint_case["native_eos_token_positions"],
            "im_end_token_positions": checkpoint_case["im_end_token_positions"],
            "endoftext_token_positions": checkpoint_case["endoftext_token_positions"],
            "clean_completion": checkpoint_case["clean_completion"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print("=== Step Comparison ===")
    print(json.dumps(report["step_comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
