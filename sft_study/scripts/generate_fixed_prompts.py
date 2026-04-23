#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common import DEFAULT_SYSTEM_PROMPT, apply_default_system_prompt_to_tokenizer, ensure_packages, read_jsonl


def parse_args() -> argparse.Namespace:
    # 这个脚本只做一件事：用固定 prompt 跑生成，方便比较训练前后输出变化。
    # 它既能直接评底模，也能加载某个 LoRA adapter 后再生成。
    parser = argparse.ArgumentParser(description="对固定 prompt 集合做生成，用于训练前后行为对比。")
    parser.add_argument("--model_name_or_path", required=True, help="底模名称或本地模型路径。")
    parser.add_argument("--adapter_path", default=None, help="可选 LoRA adapter 路径；不填则直接跑底模。")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="可选 tokenizer 路径；不填时自动回退到 adapter 或底模。")
    parser.add_argument("--prompts_file", required=True, help="固定 prompt 的 jsonl 文件。")
    parser.add_argument("--output_file", required=True, help="生成结果输出文件。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="每条样本最多生成多少 token。")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度；0 表示贪心解码。")
    parser.add_argument("--top_p", type=float, default=1.0, help="nucleus sampling 的 top_p。")
    parser.add_argument("--attn_implementation", default=None, help="可选注意力实现，例如 flash_attention_2。")
    parser.add_argument("--default_system_prompt", default=DEFAULT_SYSTEM_PROMPT, help="统一覆盖 chat template 里的默认 system prompt。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_packages()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    prompts = read_jsonl(args.prompts_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_source = args.tokenizer_name_or_path or args.adapter_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_default_system_prompt_to_tokenizer(tokenizer, args.default_system_prompt)

    model_dtype = None
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs = {
        "dtype": model_dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    device = model.device

    with output_path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            rendered = tokenizer.apply_chat_template(
                prompt["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer(rendered, return_tensors="pt").to(device)
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
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            row = {
                "id": prompt["id"],
                "tags": prompt.get("tags", []),
                "messages": prompt["messages"],
                "response": completion_text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote generations to {output_path}")


if __name__ == "__main__":
    main()
