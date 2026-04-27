#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    _normalize_conversational_dataset,
    load_dataset_split,
    maybe_filter_dataset,
    maybe_sample_dataset,
    save_json,
)


def parse_args() -> argparse.Namespace:
    # 这个脚本是整套实验的统一训练入口。
    # 设计目标是：用一套参数接口，既能直接读 Hugging Face 数据集，也能读本地 jsonl，
    # 并且支持 LoRA / QLoRA、固定 prompt 对比和后续可重复实验。
    parser = argparse.ArgumentParser(description="使用 TRL SFTTrainer 训练对话式 SFT 模型。")

    # 模型与输出位置：定义底模、训练产物输出目录，以及本次实验名称。
    parser.add_argument("--model_name_or_path", required=True, help="底模名称或本地模型路径。")
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        help="可选 tokenizer 路径。不填时默认复用 model_name_or_path。",
    )
    parser.add_argument(
        "--chat_template_path",
        required=True,
        help="显式指定训练用 chat template。可传本地 jinja 文件路径，或 TRL 支持的模板来源。",
    )
    parser.add_argument("--output_dir", required=True, help="训练输出目录，会写入 adapter、tokenizer 和配置。")
    parser.add_argument("--run_name", default=None, help="可选的实验名称，便于日志平台或输出目录识别。")
    parser.add_argument("--deepspeed_config", default=None, help="可选 DeepSpeed 配置文件路径。")

    # 数据集入口：支持远程数据集和本地数据，train/eval 可以分别指定。
    parser.add_argument("--train_dataset", required=True, help="训练集名称或本地路径。")
    parser.add_argument("--train_dataset_config", default=None, help="Hugging Face 数据集的 config 名称；没有可留空。")
    parser.add_argument("--train_split", default="train", help="训练 split 名称，例如 train / train_sft。")
    parser.add_argument("--eval_dataset", default=None, help="验证集名称或本地路径；不填时默认复用训练集来源。")
    parser.add_argument("--eval_dataset_config", default=None, help="验证集 config 名称；不填时默认复用训练集 config。")
    parser.add_argument("--eval_split", default="test", help="验证 split 名称，例如 test / validation / test_sft。")

    # 数据读取与裁剪：控制消息字段名、抽样大小、字段过滤和随机种子。
    parser.add_argument("--messages_field", default="messages", help="对话消息所在字段名，默认使用 messages。")
    parser.add_argument("--max_train_samples", type=int, default=None, help="训练集最多保留多少条样本；不填表示全量。")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="验证集最多保留多少条样本；不填表示全量。")
    parser.add_argument("--filter_field", default=None, help="可选过滤字段，例如 source 或 language。")
    parser.add_argument("--filter_values", nargs="*", default=None, help="过滤字段允许的值列表。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，影响 shuffle 和抽样结果。")

    # 训练超参：这里是最常改的一组参数，决定上下文长度、学习率和 batch 规模。
    parser.add_argument("--max_length", type=int, default=2048, help="单条样本的最大 token 长度。")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="优化器学习率。")
    parser.add_argument("--lr_scheduler_type", default="cosine", help="学习率调度器类型，例如 cosine / linear。")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="warmup 步数占总步数的比例。")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减系数。")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="训练轮数。")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每张卡的训练 batch size。")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="每张卡的验证 batch size。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数，用来放大有效 batch。")
    parser.add_argument("--logging_steps", type=int, default=10, help="每隔多少步打印一次训练日志。")
    parser.add_argument("--eval_steps", type=int, default=100, help="每隔多少步做一次验证。")
    parser.add_argument("--save_steps", type=int, default=100, help="每隔多少步保存一次 checkpoint。")
    parser.add_argument("--save_total_limit", type=int, default=1, help="最多保留多少个 checkpoint。")
    assistant_loss_group = parser.add_mutually_exclusive_group()
    assistant_loss_group.add_argument(
        "--assistant_only_loss",
        dest="assistant_only_loss",
        action="store_true",
        help="只在 assistant 回复部分计算 loss。默认开启，更符合聊天 SFT 的常见设置。",
    )
    assistant_loss_group.add_argument(
        "--no_assistant_only_loss",
        dest="assistant_only_loss",
        action="store_false",
        help="关闭 assistant-only loss，改为整条序列都参与 loss。",
    )
    parser.add_argument("--packing", action="store_true", help="是否开启样本 packing；学习阶段默认建议关闭。")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="是否开启梯度检查点以节省显存。")
    parser.add_argument("--report_to", default="none", help="日志上报目标，例如 none / wandb。")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="DataLoader worker 数量。")

    # LoRA 配置：控制适配器大小和注入模块。
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank。")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha。")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout。")
    parser.add_argument(
        "--target_modules",
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        help="要注入 LoRA 的模块列表。",
    )

    # 模型加载细节：控制量化、注意力实现和 Qwen 系列常见的 EOS 设置。
    parser.add_argument("--quantization", choices=("auto", "none", "4bit"), default="auto",
                        help="量化策略；auto 会在 CUDA 上优先尝试 4-bit。")
    parser.add_argument("--attn_implementation", default=None, help="可选注意力实现，例如 flash_attention_2。")
    parser.add_argument("--eos_token", default="<|im_end|>", help="显式指定 EOS token，Qwen chat template 场景建议设置。")

    parser.set_defaults(assistant_only_loss=True)
    return parser.parse_args()


def _prepare_model_init_kwargs(args: argparse.Namespace, torch: Any, BitsAndBytesConfig: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    quantization = args.quantization
    if quantization == "auto" and not torch.cuda.is_available():
        quantization = "none"

    if quantization == "4bit" or quantization == "auto":
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        # FlashAttention 2 要求模型实际加载 dtype 为 fp16/bf16。
        # 仅设置 bnb_4bit_compute_dtype 还不够，未量化模块仍可能保持 float32。
        kwargs["dtype"] = compute_dtype
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    return kwargs


def main() -> None:
    args = parse_args()

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train_dataset = load_dataset_split(args.train_dataset, args.train_dataset_config, args.train_split)
    train_dataset = maybe_filter_dataset(train_dataset, args.filter_field, args.filter_values)
    train_dataset = maybe_sample_dataset(train_dataset, args.max_train_samples, args.seed)

    eval_dataset_name = args.eval_dataset or args.train_dataset
    eval_dataset_config = args.eval_dataset_config if args.eval_dataset else args.train_dataset_config
    eval_dataset = load_dataset_split(eval_dataset_name, eval_dataset_config, args.eval_split)
    eval_dataset = maybe_filter_dataset(eval_dataset, args.filter_field, args.filter_values)
    eval_dataset = maybe_sample_dataset(eval_dataset, args.max_eval_samples, args.seed)

    train_dataset = _normalize_conversational_dataset(train_dataset, args.messages_field)
    eval_dataset = _normalize_conversational_dataset(eval_dataset, args.messages_field)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_source = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    template_path = Path(args.chat_template_path)
    if template_path.is_file():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    model_init_kwargs = _prepare_model_init_kwargs(args, torch, BitsAndBytesConfig)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        deepspeed=args.deepspeed_config,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        max_length=args.max_length,
        packing=args.packing,
        assistant_only_loss=args.assistant_only_loss,
        chat_template_path=args.chat_template_path,
        eos_token=args.eos_token,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        seed=args.seed,
        model_init_kwargs=model_init_kwargs,
    )

    metadata = dict(vars(args))
    metadata.update(
        {
            "tokenizer_name_or_path": tokenizer_source,
            "eval_dataset": eval_dataset_name,
            "eval_dataset_config": eval_dataset_config,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "messages_field": "messages",
            "dataset_columns_after_normalization": train_dataset.column_names,
            "model_init_kwargs": {key: str(value) for key, value in model_init_kwargs.items()},
        }
    )
    save_json(output_dir / "run_config.json", metadata)

    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    trainer.save_model()
    trained_processing_class = getattr(trainer, "processing_class", None) or tokenizer
    trained_processing_class.save_pretrained(str(output_dir))
    save_json(output_dir / "train_result.json", train_result.metrics)
    print(json.dumps(train_result.metrics, indent=2))


if __name__ == "__main__":
    main()
