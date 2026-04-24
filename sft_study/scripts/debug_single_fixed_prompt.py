#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from common import DEFAULT_SYSTEM_PROMPT, apply_default_system_prompt_to_tokenizer, ensure_packages, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调试单条 fixed prompt 的生成过程，打印 prompt 渲染结果、special tokens 和 raw decode。"
    )
    parser.add_argument("--checkpoint_dir", default=None, help="SFT 输出目录；若存在 run_config.json，会自动补齐模型上下文。")
    parser.add_argument("--model_name_or_path", default=None, help="底模名称或本地模型路径。")
    parser.add_argument("--adapter_path", default=None, help="可选 LoRA adapter 路径。")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="可选 tokenizer 路径。")
    parser.add_argument("--prompts_file", default=None, help="fixed prompts jsonl；不填默认仓库自带样例。")
    parser.add_argument("--prompt_id", default=None, help="按 prompt id 选择样本。")
    parser.add_argument("--prompt_index", type=int, default=0, help="如果不传 --prompt_id，就按索引选样本。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最多生成多少 token。")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度；0 表示贪心。")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p。")
    parser.add_argument("--attn_implementation", default=None, help="可选注意力实现，例如 flash_attention_2。")
    parser.add_argument("--default_system_prompt", default=DEFAULT_SYSTEM_PROMPT, help="统一覆盖默认 system prompt。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--revision", default=None, help="模型 revision。")
    parser.add_argument("--trust_remote_code", action="store_true", help="是否允许 remote code。")
    parser.add_argument("--inspect_steps", type=int, default=8, help="额外打印前多少个生成位置的 top-k next-token 候选。")
    parser.add_argument("--top_k", type=int, default=10, help="每个生成位置打印多少个 top-k token。")
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
    prompts_file = (
        Path(args.prompts_file).resolve()
        if args.prompts_file
        else Path(__file__).resolve().parent.parent / "data" / "fixed_prompts.jsonl"
    )

    if not model_name_or_path:
        raise SystemExit("Could not resolve model_name_or_path. Pass --model_name_or_path or --checkpoint_dir.")

    return {
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "prompts_file": str(prompts_file),
    }


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
    if target_id is None:
        return []
    return [index for index, token_id in enumerate(token_ids) if token_id == target_id]


def decode_token_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")


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
    summaries: list[dict[str, Any]] = []

    for step in range(limit):
        source_index = prompt_length - 1 + step
        chosen_index = prompt_length + step
        chosen_token_id = full_ids[chosen_index]
        step_logits = logits[source_index]
        top_values, top_indices = torch.topk(step_logits, k=min(top_k, step_logits.shape[-1]))
        candidates = []
        for rank, (candidate_id, logit_value) in enumerate(
            zip(top_indices.detach().cpu().tolist(), top_values.detach().cpu().tolist()),
            start=1,
        ):
            candidates.append(
                {
                    "rank": rank,
                    "token_id": candidate_id,
                    "piece": decode_token_piece(tokenizer, candidate_id),
                    "logit": round(float(logit_value), 6),
                    "is_chosen": candidate_id == chosen_token_id,
                    "is_eos": candidate_id == tokenizer.eos_token_id,
                }
            )

        summaries.append(
            {
                "step": step,
                "source_index": source_index,
                "source_piece": decode_token_piece(tokenizer, full_ids[source_index]),
                "chosen_index": chosen_index,
                "chosen_token_id": chosen_token_id,
                "chosen_piece": decode_token_piece(tokenizer, chosen_token_id),
                "chosen_is_eos": chosen_token_id == tokenizer.eos_token_id,
                "top_k": candidates,
            }
        )

    return summaries


def main() -> None:
    args = parse_args()
    context = resolve_context(args)
    ensure_packages()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)

    prompts = read_jsonl(context["prompts_file"])
    prompt = select_prompt(prompts, args.prompt_id, args.prompt_index)

    tokenizer = AutoTokenizer.from_pretrained(
        context["tokenizer_name_or_path"],
        use_fast=True,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_default_system_prompt_to_tokenizer(tokenizer, args.default_system_prompt)

    model_dtype = None
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        context["model_name_or_path"],
        **model_kwargs,
    )
    if context["adapter_path"]:
        model = PeftModel.from_pretrained(model, context["adapter_path"])

    rendered_prompt = tokenizer.apply_chat_template(
        prompt["messages"],
        tokenize=False,
        add_generation_prompt=True,
    )
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
    raw_completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
    clean_completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    top_k_debug = inspect_top_k_next_tokens(
        model=model,
        tokenizer=tokenizer,
        output_ids=output_ids,
        prompt_length=prompt_length,
        inspect_steps=args.inspect_steps,
        top_k=args.top_k,
    )

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    report = {
        "resolved_context": context,
        "prompt": {
            "id": prompt.get("id"),
            "tags": prompt.get("tags", []),
            "messages": prompt.get("messages", []),
        },
        "tokenizer": {
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "im_start_token_id": im_start_id,
            "im_end_token_id": im_end_id,
            "endoftext_token_id": endoftext_id,
        },
        "generation": {
            "args": generation_kwargs,
            "completion_token_count": len(completion_ids_list),
            "eos_token_positions": token_positions(completion_ids_list, tokenizer.eos_token_id),
            "im_end_token_positions": token_positions(completion_ids_list, im_end_id),
            "im_start_token_positions": token_positions(completion_ids_list, im_start_id),
            "endoftext_token_positions": token_positions(completion_ids_list, endoftext_id),
        },
        "rendered_prompt": rendered_prompt,
        "raw_completion": raw_completion,
        "clean_completion": clean_completion,
        "top_k_next_tokens": top_k_debug,
    }

    print(json.dumps(report["resolved_context"], ensure_ascii=False, indent=2))
    print()
    print("=== Prompt ===")
    print(json.dumps(report["prompt"], ensure_ascii=False, indent=2))
    print()
    print("=== Tokenizer ===")
    print(json.dumps(report["tokenizer"], ensure_ascii=False, indent=2))
    print()
    print("=== Generation ===")
    print(json.dumps(report["generation"], ensure_ascii=False, indent=2))
    print()
    print("=== Rendered Prompt ===")
    print(report["rendered_prompt"])
    print()
    print("=== Raw Completion (skip_special_tokens=False) ===")
    print(report["raw_completion"])
    print()
    print("=== Clean Completion (skip_special_tokens=True) ===")
    print(report["clean_completion"])
    if report["top_k_next_tokens"]:
        print()
        print("=== Top-K Next Tokens ===")
        print(json.dumps(report["top_k_next_tokens"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
