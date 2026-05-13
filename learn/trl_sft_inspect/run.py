from typing import Any
from pathlib import Path

from datasets import disable_caching, load_dataset
from transformers import AutoTokenizer, set_seed
from trl import SFTConfig

from debug import TrainerDebugPrinter
from model import qwen25_smoke
from trainer import InspectSFTTrainer, my_loss


def main() -> None:
    debugger = TrainerDebugPrinter("inspect.run")
    # 调试 Trainer 数据流时必须关闭 datasets.map 缓存，否则 _prepare_dataset 可能直接复用旧的 input_ids。
    disable_caching()

    # 使用真实数据集小切片 + 真实 Qwen tokenizer + mock Qwen 模型跑一个 epoch。
    # 这里直接改参数即可。若使用本地 jsonl，可以把 dataset_name 改为 "json"，
    # 并传 dataset_kwargs={"data_files": "你的文件.jsonl"}。
    dataset_name = "trl-lib/Capybara"
    dataset_split = "train"
    dataset_kwargs: dict[str, Any] | None = None
    tokenizer_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir = None
    num_examples: int | None = 1
    max_length = 1024
    learning_rate = 5e-5
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.03
    weight_decay = 0.0
    seed = 42
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 1
    gradient_checkpointing = False
    dataloader_num_workers = 0
    assistant_only_loss = True
    chat_template_path: str | None = None
    packing = False

    # mock Qwen 是随机初始化的；必须在创建模型前设 seed，否则每次运行 loss 都会不一样。
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen/LLaMA 类 CausalLM 常见做法：没有 pad token 时复用 eos。
    tokenizer.padding_side = "right"

    if chat_template_path is not None:
        template_path = Path(chat_template_path)
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    train_dataset = load_dataset(dataset_name, split=dataset_split, **(dataset_kwargs or {}))
    if num_examples is not None:
        start = 1
        end = min(start + num_examples, len(train_dataset))
        train_dataset = train_dataset.select(range(start, end))

    debugger.dump("raw train_dataset", train_dataset)
    debugger.dump("raw train_dataset[0]", train_dataset[0])

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        max_length=max_length,
        packing=packing,
        assistant_only_loss=assistant_only_loss,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        seed=seed,
    )

    model = qwen25_smoke(tokenizer)
    trainer = InspectSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        compute_loss_func=my_loss,
    )
    trainer.train()


if __name__ == "__main__":
    main()
