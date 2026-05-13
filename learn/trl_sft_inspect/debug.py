from typing import Any
from pprint import pformat

import torch


class TrainerDebugPrinter:
    """Trainer 调试打印工具，和具体 Trainer 逻辑解耦。"""

    YELLOW = "\033[33m"
    RESET = "\033[0m"

    def __init__(
            self,
            owner_name: str = "Trainer",
            width: int = 120,
            compact: bool = False,
            max_list_items: int = 8,
            max_string_chars: int = 240,
            numbers_per_line: int = 40,
            max_tensor_items: int = 4096,
    ) -> None:
        self.owner_name = owner_name
        self.width = width
        self.compact = compact
        self.max_list_items = max_list_items
        self.max_string_chars = max_string_chars
        self.numbers_per_line = numbers_per_line
        self.max_tensor_items = max_tensor_items

    def call(self, name: str, **values: Any) -> None:
        # 统一的调试打印入口，方便观察 Trainer 生命周期中各个 hook 的入参。
        self.section(name)
        for key, value in values.items():
            self.item(key, value)

    def section(self, title: str) -> None:
        # 用明显边界隔开不同调试区块，避免 pprint 输出混在训练日志里。
        line = "=" * 96
        block = f"\n{line}\n[{self.owner_name}] {title}\n{line}"
        print(f"{self.YELLOW}{block}{self.RESET}", flush=True)

    def item(self, key: str, value: Any) -> None:
        # 每个字段单独成块，字段名固定在左侧，值统一缩进。
        rendered = self.repr(value)
        print(f"\n--- {key} ---", flush=True)
        print(self._indent(rendered), flush=True)

    def dump(self, title: str, value: Any, *, full: bool = False) -> None:
        # 默认打印摘要，避免长 input_ids / masks / 文本把训练日志冲掉；必要时可 full=True 展开。
        self.section(title)
        value = value if full else self.summarize(value)
        print(self._indent(self.format_dump_value(value)), flush=True)

    def format_dump_value(self, value: Any) -> str:
        # pformat 会把多行字符串转义成 "\n"；这里对 dict 做定制渲染，让 token 列表真正换行。
        if isinstance(value, dict):
            lines = ["{"]
            for key, item in value.items():
                rendered = self.format_dump_value(item)
                if "\n" in rendered:
                    lines.append(f"  {key!r}:")
                    lines.append(self._indent(rendered, "    "))
                else:
                    lines.append(f"  {key!r}: {rendered},")
            lines.append("}")
            return "\n".join(lines)
        if isinstance(value, str) and "\n" in value:
            return value
        return pformat(value, width=self.width, compact=self.compact, sort_dicts=False)

    def summarize(self, value: Any) -> Any:
        # 将复杂对象转成可读摘要：保留结构，压缩长文本和长序列。
        if isinstance(value, torch.Tensor):
            return self.repr(value)
        if isinstance(value, dict):
            return {key: self.summarize(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self.summarize_list(value))
        if isinstance(value, list):
            return self.summarize_list(value)
        if isinstance(value, str) and len(value) > self.max_string_chars:
            return f"{value[:self.max_string_chars]}... <str len={len(value)}>"
        return value

    def summarize_list(self, value: list[Any] | tuple[Any, ...]) -> Any:
        # 数值长列表通常是 token ids / masks，按固定列数排版，方便肉眼看位置。
        if len(value) > self.max_list_items and all(isinstance(item, int) for item in value):
            return self.format_number_list(value)
        if len(value) <= self.max_list_items:
            return [self.summarize(item) for item in value]
        head_count = self.max_list_items // 2
        tail_count = self.max_list_items - head_count
        return {
            "type": type(value).__name__,
            "len": len(value),
            "head": [self.summarize(item) for item in value[:head_count]],
            "tail": [self.summarize(item) for item in value[-tail_count:]],
        }

    def format_number_list(self, value: list[Any] | tuple[Any, ...]) -> str:
        # 每行固定数量数字；数字宽度按当前列表最大宽度对齐。
        width = max(len(str(item)) for item in value) if value else 1
        lines = [f"{type(value).__name__}(len={len(value)}) ["]
        for start in range(0, len(value), self.numbers_per_line):
            chunk = value[start: start + self.numbers_per_line]
            rendered = ", ".join(f"{item:>{width}}" for item in chunk)
            comma = "," if start + self.numbers_per_line < len(value) else ""
            lines.append(f"  {rendered}{comma}")
        lines.append("]")
        return "\n".join(lines)

    def format_tensor(self, value: torch.Tensor) -> str:
        # 小/中型 tensor 直接打印具体数值；超大 tensor 才退回 shape 摘要。
        meta = (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device}, requires_grad={value.requires_grad})"
        )
        if value.numel() > self.max_tensor_items:
            return meta

        data = value.detach().cpu()
        if data.ndim == 0:
            return f"{meta}\nvalue={data.item()!r}"
        if data.ndim == 1:
            return f"{meta}\n{self.format_number_list(data.tolist())}"
        if data.ndim == 2:
            lines = [meta]
            rows = data.tolist()
            for row_index, row in enumerate(rows):
                lines.append(f"row {row_index}:")
                lines.append(self._indent(self.format_number_list(row), "  "))
            return "\n".join(lines)
        return f"{meta}\n{pformat(data.tolist(), width=self.width, compact=self.compact, sort_dicts=False)}"

    def repr(self, value: Any) -> str:
        # Tensor 默认打印具体数值，超过 max_tensor_items 才只打印形状摘要。
        if isinstance(value, torch.Tensor):
            return self.format_tensor(value)
        if value.__class__.__name__.endswith("Config") and hasattr(value, "to_dict"):
            keys = [
                "output_dir",
                "num_train_epochs",
                "max_steps",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "lr_scheduler_type",
                "warmup_steps",
                "warmup_ratio",
                "weight_decay",
                "gradient_checkpointing",
                "dataloader_num_workers",
                "max_length",
                "packing",
                "padding_free",
                "assistant_only_loss",
                "completion_only_loss",
                "loss_type",
            ]
            values = value.to_dict()
            summary = {key: values.get(key) for key in keys if key in values}
            return f"{type(value).__name__}({summary})"
        # dict/list/tuple 递归打印摘要，复杂 batch 也能快速看清结构。
        if isinstance(value, dict):
            items = ", ".join(f"{k}={self.repr(v)}" for k, v in value.items())
            return "{" + items + "}"
        if isinstance(value, (list, tuple)):
            preview = ", ".join(self.repr(v) for v in value[:3])
            suffix = ", ..." if len(value) > 3 else ""
            return f"{type(value).__name__}(len={len(value)}, [{preview}{suffix}])"
        return repr(value)

    @staticmethod
    def _indent(text: str, prefix: str = "  ") -> str:
        return "\n".join(prefix + line for line in text.splitlines())
