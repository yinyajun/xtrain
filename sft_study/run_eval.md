# Run Eval

评估和生成入口都在 `sft_study/run_eval/`。这里不做训练。

## Base Fixed Prompts

```bash
bash sft_study/run_eval/e0_fixed_prompts_base.sh
```

输出：

```text
sft_study/outputs/e0_base_fixed_prompts.jsonl
```

## Checkpoint Fixed Prompts

```bash
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke \
  bash sft_study/run_eval/generate_fixed_prompts.sh
```

默认输出：

```text
<checkpoint_dir>/eval/fixed_prompts.jsonl
```

常用变量：

- `CHECKPOINT_DIR`：checkpoint 目录，会读取 `run_config.json`
- `MODEL`：不传 checkpoint 时手动指定 base model
- `ADAPTER_PATH` / `TOKENIZER_PATH`：可覆盖 adapter/tokenizer
- `PROMPTS_FILE`：默认 `sft_study/data/fixed_prompts.jsonl`
- `OUTPUT_FILE`：可覆盖输出文件
- `MAX_NEW_TOKENS` / `TEMPERATURE` / `TOP_P`

## Checkpoint Benchmark

```bash
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke \
  bash sft_study/run_eval/checkpoint_benchmark.sh
```

只跑部分 benchmark：

```bash
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke \
BENCHMARKS="ifeval gsm8k" \
LIMIT=100 \
  bash sft_study/run_eval/checkpoint_benchmark.sh
```

评估 base 模型：

```bash
MODEL=Qwen/Qwen2.5-7B \
OUTPUT_DIR=sft_study/outputs/e0_base_eval \
BENCHMARKS="ifeval" \
LIMIT=100 \
  bash sft_study/run_eval/checkpoint_benchmark.sh
```

常用变量：

- `BENCHMARKS`：默认 `ifeval gsm8k mmlu cmmlu`
- `LIMIT`：每个 lm-eval task 的样本上限
- `DEVICE`：默认 `auto`
- `BATCH_SIZE`：默认 `auto`
- `ATTN_IMPLEMENTATION`：可选 `flash_attention_2`
