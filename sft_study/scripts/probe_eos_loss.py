#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common import apply_default_system_prompt_to_tokenizer


GENERATION_BLOCK_MARKERS = ("{% generation %}", "{% endgeneration %}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用一条空 assistant 样本检查 <|im_end|> 是否进入真实训练 loss。")
    parser.add_argument("--checkpoint_dir", required=True, help="训练输出目录，默认直接使用目录里的 tokenizer。")
    parser.add_argument("--user_text", default="Reply with nothing.", help="合成 user 内容。")
    parser.add_argument("--assistant_text", default="", help="合成 assistant 内容，默认空字符串。")
    return parser.parse_args()


def has_generation_blocks(chat_template: str | None) -> bool:
    return bool(chat_template and all(marker in chat_template for marker in GENERATION_BLOCK_MARKERS))


def choose_device_and_dtype(torch: object) -> tuple[str, object]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def main() -> None:
    args = parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    run_config = json.loads((checkpoint_dir / "run_config.json").read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_default_system_prompt_to_tokenizer(tokenizer, run_config.get("default_system_prompt"))

    messages = [
        {"role": "user", "content": args.user_text},
        {"role": "assistant", "content": args.assistant_text},
    ]

    encode_kwargs = {
        "conversation": messages,
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    if has_generation_blocks(getattr(tokenizer, "chat_template", None)):
        encode_kwargs["return_assistant_tokens_mask"] = True

    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoding = tokenizer.apply_chat_template(**encode_kwargs)

    input_ids = list(encoding["input_ids"])
    assistant_masks = list(encoding.get("assistant_masks", []))
    eos_token_id = tokenizer.eos_token_id
    labels = [token_id if not assistant_masks or assistant_masks[idx] else -100 for idx, token_id in enumerate(input_ids)]

    device, dtype = choose_device_and_dtype(torch)
    model = AutoModelForCausalLM.from_pretrained(run_config["model_name_or_path"], torch_dtype=dtype)

    adapter_loaded = False
    if (checkpoint_dir / "adapter_config.json").exists():
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(checkpoint_dir))
            adapter_loaded = True
        except ImportError:
            raise SystemExit("Adapter exists but `peft` is not installed. Install `peft` to measure the real checkpoint loss.")

    model = model.to(device)
    model.eval()

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

    print("=== rendered ===")
    print(rendered)
    print()
    print("=== model ===")
    print(f"base_model={run_config['model_name_or_path']}")
    print(f"adapter_loaded={adapter_loaded}")
    print(f"device={device}")
    print(f"loss={outputs.loss.item():.6f}")
    print()
    print("=== assistant mask positions ===")
    print([idx for idx, value in enumerate(assistant_masks) if value])
    print()
    print("=== supervised targets ===")
    for source_idx in range(len(input_ids) - 1):
        target_idx = source_idx + 1
        if assistant_masks and not assistant_masks[target_idx]:
            continue
        target_id = input_ids[target_idx]
        piece = tokenizer.decode([target_id], skip_special_tokens=False).replace("\n", "\\n")
        is_eos = target_id == eos_token_id
        nll = float(per_token_nll[0, source_idx].item())
        print(
            f"source_idx={source_idx} target_idx={target_idx} "
            f"target_id={target_id} piece={piece!r} is_eos={is_eos} nll={nll:.6f}"
        )


if __name__ == "__main__":
    main()
