import dataclasses
from typing import Any, Callable

import torch
import torch.nn as nn
from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, ProcessorMixin, TrainingArguments, DataCollator, PreTrainedModel
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer import EvalPrediction, TrainerCallback
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import pad, selective_log_softmax
from peft import PeftConfig, PeftModel

from debug import TrainerDebugPrinter


def my_loss(outputs, labels, num_items_in_batch=None):
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100
    shift_labels[~loss_mask] = 0
    logprobs = selective_log_softmax(outputs.logits, shift_labels)
    # probs = logprobs.exp().detach()
    # weight = probs
    # per_token_loss = -weight * logprobs
    # if num_items_in_batch is None:
    #     num_items_in_batch = loss_mask.sum()
    # loss0 = (per_token_loss * loss_mask).sum() / num_items_in_batch
    #
    # weight = probs * (1 - probs)
    # per_token_loss = - weight * logprobs
    # if num_items_in_batch is None:
    #     num_items_in_batch = loss_mask.sum()
    # loss1 = (per_token_loss * loss_mask).sum() / num_items_in_batch
    per_token_loss = - logprobs
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch

    return loss


@dataclasses.dataclass
class InspectDataCollator(DataCollatorMixin):
    """可改造版 SFT data collator。

    这里保留真实 SFT collator 的主要逻辑，方便继续插入 debug 或修改 labels/mask/padding 行为。
    """

    pad_token_id: int
    max_length: int | None = None
    truncation_mode: str = "keep_start"
    completion_only_loss: bool = True
    padding_free: bool = False
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"
    debug_max_calls: int = 3

    def __post_init__(self) -> None:
        self.debugger = TrainerDebugPrinter(type(self).__name__)
        self.call_count = 0

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        self.call_count += 1
        should_debug = self.call_count <= self.debug_max_calls
        if should_debug:
            self.debugger.dump(f"call #{self.call_count} raw examples", examples)

        input_ids = [example["input_ids"] for example in examples]
        batch_seq_lengths = [example["seq_lengths"] for example in examples] if "seq_lengths" in examples[0] else None
        labels = [example.get("labels", example["input_ids"]) for example in examples]
        completion_mask = (
            [example["completion_mask"] for example in examples]
            if self.completion_only_loss and "completion_mask" in examples[0]
            else None
        )
        assistant_masks = (
            [example["assistant_masks"] for example in examples] if "assistant_masks" in examples[0] else None
        )
        if should_debug:
            self.debugger.dump(
                f"call #{self.call_count} extracted fields",
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "completion_mask": completion_mask,
                    "assistant_masks": assistant_masks,
                    "batch_seq_lengths": batch_seq_lengths,
                },
            )

        if self.max_length is not None and not self.padding_free:
            if self.truncation_mode == "keep_start":
                sl = slice(None, self.max_length)
            elif self.truncation_mode == "keep_end":
                sl = slice(-self.max_length, None)
            else:
                raise ValueError(
                    f"Unsupported truncation mode: {self.truncation_mode}, expected 'keep_start' or 'keep_end'"
                )

            input_ids = [ids[sl] for ids in input_ids]
            labels = [lbl[sl] for lbl in labels]
            if completion_mask is not None:
                completion_mask = [m[sl] for m in completion_mask]
            if assistant_masks is not None:
                assistant_masks = [m[sl] for m in assistant_masks]
            if should_debug:
                self.debugger.dump(
                    f"call #{self.call_count} after truncation",
                    {
                        "max_length": self.max_length,
                        "truncation_mode": self.truncation_mode,
                        "input_ids": input_ids,
                        "labels": labels,
                        "completion_mask": completion_mask,
                        "assistant_masks": assistant_masks,
                    },
                )

        input_ids = [torch.tensor(ids) for ids in input_ids]
        labels = [torch.tensor(lbl) for lbl in labels]
        if completion_mask is not None:
            completion_mask = [torch.tensor(m) for m in completion_mask]
        if assistant_masks is not None:
            assistant_masks = [torch.tensor(m) for m in assistant_masks]
        if should_debug:
            self.debugger.call(
                f"call #{self.call_count} tensors",
                input_ids=input_ids,
                labels=labels,
                completion_mask=completion_mask,
                assistant_masks=assistant_masks,
            )

        if self.padding_free:
            if batch_seq_lengths is not None:
                position_ids = self.get_position_ids_from_packed_seq_lengths(batch_seq_lengths)
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        else:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]
        if should_debug:
            self.debugger.call(
                f"call #{self.call_count} masks before padding",
                attention_mask=None if self.padding_free else attention_mask,
                position_ids=position_ids if self.padding_free else None,
            )

        output = {}
        if self.padding_free:
            input_ids = [torch.cat(input_ids, dim=0)]
            labels = [torch.cat(labels, dim=0)]
            position_ids = [torch.cat(position_ids, dim=0)]
            if completion_mask is not None:
                completion_mask = [torch.cat(completion_mask, dim=0)]
            if assistant_masks is not None:
                assistant_masks = [torch.cat(assistant_masks, dim=0)]
            if should_debug:
                self.debugger.call(
                    f"call #{self.call_count} after padding-free flatten",
                    input_ids=input_ids,
                    labels=labels,
                    position_ids=position_ids,
                    completion_mask=completion_mask,
                    assistant_masks=assistant_masks,
                )

        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["labels"] = pad(
            labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )
        if self.padding_free:
            output["position_ids"] = pad(
                position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][output["position_ids"] == 0] = -100
        else:
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
        if should_debug:
            self.debugger.call(f"call #{self.call_count} after padding", output=output)
        if completion_mask is not None:
            completion_mask = pad(
                completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][completion_mask == 0] = -100
            if should_debug:
                self.debugger.call(
                    f"call #{self.call_count} after completion_mask",
                    completion_mask=completion_mask,
                    labels=output["labels"],
                )
        if assistant_masks is not None:
            assistant_masks = pad(
                assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][assistant_masks == 0] = -100
            if should_debug:
                self.debugger.call(
                    f"call #{self.call_count} after assistant_masks",
                    assistant_masks=assistant_masks,
                    labels=output["labels"],
                )
        if should_debug:
            self.debugger.call(f"call #{self.call_count} final output", output=output)
        return output

    @staticmethod
    def get_position_ids_from_packed_seq_lengths(batch_seq_lengths: list[list[int]]) -> list[torch.Tensor]:
        """根据 packed seq_lengths 生成每个 token 的 position_ids。"""
        example_lengths = [sum(seq_lengths) for seq_lengths in batch_seq_lengths]
        batch_seq_lengths = torch.tensor(
            [seq_length for seq_lengths in batch_seq_lengths for seq_length in seq_lengths]
        )
        position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_lengths.dtype)
        position_ids[0] = 0
        position_ids[batch_seq_lengths[:-1].cumsum(0)] = -(batch_seq_lengths[:-1] - 1)
        position_ids = position_ids.cumsum(0)
        return list(position_ids.split(example_lengths))


class InspectSFTTrainer(SFTTrainer):
    def __init__(
            self,
            model: "str | PreTrainedModel | PeftModel",
            args: SFTConfig | TrainingArguments | None = None,
            data_collator: DataCollator | None = None,
            train_dataset: Dataset | IterableDataset | None = None,
            eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
            processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
            compute_loss_func: Callable | None = None,
            compute_metrics: Callable[[EvalPrediction], dict] | None = None,
            callbacks: list[TrainerCallback] | None = None,
            optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
            optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
            peft_config: "PeftConfig | None" = None,
            formatting_func: Callable[[dict], str] | None = None,
    ) -> None:
        """初始化 Trainer，并在父类完成数据预处理后打印处理后的训练集。

        SFTTrainer.__init__ 会在内部调用 _prepare_dataset，把原始 dataset 转成带 input_ids/mask 的 dataset。
        这里重载 __init__ 的目的不是改训练逻辑，而是在 super().__init__ 前准备调试器和计数器，
        并在 super().__init__ 后观察 self.train_dataset 的最终形态。
        """
        self.debugger = TrainerDebugPrinter(type(self).__name__)
        self._tokenize_call_count = 0

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )
        if data_collator is None:
            self.data_collator = InspectDataCollator(
                pad_token_id=self._tokenizer.pad_token_id,
                max_length=None if self.padding_free else self.args.max_length,
                completion_only_loss=self.completion_only_loss,
                padding_free=self.padding_free,
                pad_to_multiple_of=self.args.pad_to_multiple_of,
            )

    def _tokenize(
            self,
            processing_class: PreTrainedTokenizerBase | ProcessorMixin,
            input: str | list,
            **kwargs,
    ) -> dict[str, list]:
        """观察单条样本的 tokenization 过程。

        父类 _prepare_dataset 里的 tokenize_fn 会调用 self._tokenize：
        - input 是 list[dict] 时，表示 conversational messages，会走 apply_chat_template。
        - input 是 str 时，表示普通文本，会走 tokenizer(text=...)。

        这里重载它是为了打印首次调用时的原始 input、chat template 渲染文本和 kwargs，
        然后继续调用父类实现，保持真实 tokenization 行为不变。
        """
        self._tokenize_call_count += 1
        # tokenization 会被 dataset.map 调很多次，只展开前几次，避免日志爆炸；总次数在 __init__ 末尾汇总。

        result = super()._tokenize(processing_class, input, **kwargs)
        if self._tokenize_call_count <= 1:
            self.debugger.call(
                f"_tokenize #{self._tokenize_call_count}",
                input=input,
                output=processing_class.apply_chat_template(
                    input,
                    tokenize=False,
                    return_dict=True,
                    chat_template=self.chat_template,
                ),
                result=result,
                kwargs=kwargs,
            )
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """观察 loss 计算入口。

        training_step 会把 batch inputs 传进 compute_loss。
        在 SFTTrainer 中，父类 compute_loss 会执行 model forward，并根据 labels / compute_loss_func 算 loss，
        同时 SFTTrainer 还会记录 token accuracy、entropy 等指标。

        这里重载它只是为了查看进入 loss 前的 inputs，比如 input_ids、labels、attention_mask 的形状和 device。
        """
        # loss 计算入口：这里只观察入参，实际 loss 仍由父类实现。
        self.debugger.call(
            "compute_loss",
            model=type(model).__name__,
            inputs=inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _prepare_dataset(
            self,
            dataset: Dataset | IterableDataset,
            processing_class: PreTrainedTokenizerBase | ProcessorMixin,
            args: SFTConfig,
            packing: bool,
            formatting_func: Callable[[dict], str] | None,
            dataset_name: str,
    ) -> Dataset | IterableDataset:
        """观察 SFTTrainer 的 dataset 预处理入口和结果。

        SFTTrainer.__init__ 会调用这个方法，把原始 dataset 处理成训练用 dataset。
        父类实现会根据数据格式执行：
        - formatting_func
        - ChatML 转换
        - EOS 补齐
        - tokenize
        - packing

        这里重载它是为了打印进入预处理前的 dataset/参数，以及父类处理后的 dataset/input_ids/mask。
        """
        # dataset 预处理入口：可以观察原始 dataset、packing、tokenizer/processor 等配置。
        self.debugger.call(
            "_prepare_dataset",
            dataset=dataset,
            processing_class=type(processing_class).__name__,
            args=args,
            packing=packing,
            formatting_func=formatting_func,
            dataset_name=dataset_name,
        )
        dataset = super()._prepare_dataset(
            dataset,
            processing_class,
            args,
            packing,
            formatting_func,
            dataset_name,
        )

        self.debugger.dump("_prepare_dataset result", dataset)
        self.debugger.dump("_prepare_dataset result[0]", dataset[0])
        return dataset
