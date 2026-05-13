# Runs

这份文档只记录 `sft_study/runs/` 下 shell 脚本的常用跑法。环境安装看 `install.md`，实验动机和观察重点看 `experiments.md`。

通用环境变量：

- `PYTHON_BIN`：Python 解释器，默认 `python3`
- `MODEL`：底模，默认多为 `Qwen/Qwen2.5-7B`
- `TOKENIZER_PATH`：可选 tokenizer 路径
- `CHAT_TEMPLATE_PATH`：训练用 chat template，默认 `chat_templates/qwen2_5_training.jinja`
- `REPORT_TO`：日志上报，默认 `none`；可设为 `wandb`
- `ATTN_IMPLEMENTATION`：可选注意力实现，例如 `flash_attention_2`

## E0 Base Fixed Prompts

脚本：

```bash
bash sft_study/runs/e0_fixed_prompts_base.sh
```

用途：不训练，直接生成 base 模型在固定 prompts 上的输出。

常用覆盖：

```bash
MODEL=Qwen/Qwen2.5-7B \
  bash sft_study/runs/e0_fixed_prompts_base.sh
```

输出：

```text
sft_study/outputs/e0_base_fixed_prompts.jsonl
```

## Generate Fixed Prompts

脚本：

```bash
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke \
  bash sft_study/runs/generate_fixed_prompts.sh
```

用途：给某个 checkpoint 单独生成 fixed prompts。

常用覆盖：

```bash
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke \
MAX_NEW_TOKENS=512 \
  bash sft_study/runs/generate_fixed_prompts.sh
```

评估 base 模型：

```bash
MODEL=Qwen/Qwen2.5-7B \
OUTPUT_FILE=sft_study/outputs/e0_base_fixed_prompts.jsonl \
  bash sft_study/runs/generate_fixed_prompts.sh
```

主要变量：

- `CHECKPOINT_DIR`：训练输出目录；脚本会读 `run_config.json`
- `MODEL`：不传 `CHECKPOINT_DIR` 时必须传
- `ADAPTER_PATH`：可选 LoRA adapter 路径，默认等于 `CHECKPOINT_DIR`
- `TOKENIZER_PATH`：默认等于 `CHECKPOINT_DIR`
- `PROMPTS_FILE`：默认 `sft_study/data/fixed_prompts.jsonl`
- `OUTPUT_FILE`：默认 `<checkpoint_dir>/eval/fixed_prompts.jsonl`
- `MAX_NEW_TOKENS` / `TEMPERATURE` / `TOP_P`：生成参数

## Eval Checkpoint

脚本：

```bash
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke \
  bash sft_study/runs/checkpoint_benchmark.sh
```

用途：跑标准 benchmark。这个脚本不生成 fixed prompts。

只跑部分 benchmark：

```bash
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke \
BENCHMARKS="ifeval gsm8k" \
  bash sft_study/runs/checkpoint_benchmark.sh
```

快速 smoke test：

```bash
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke \
LIMIT=20 \
  bash sft_study/runs/checkpoint_benchmark.sh
```

评估 base 模型：

```bash
MODEL=Qwen/Qwen2.5-7B \
OUTPUT_DIR=sft_study/outputs/e0_base_eval \
  bash sft_study/runs/checkpoint_benchmark.sh
```

主要变量：

- `CHECKPOINT_DIR`：checkpoint 输出目录；会读取 `run_config.json`
- `MODEL` / `ADAPTER_PATH` / `TOKENIZER_PATH`：手动指定模型上下文
- `OUTPUT_DIR`：默认 `<checkpoint_dir>/eval`
- `BENCHMARKS`：默认 `ifeval gsm8k mmlu cmmlu`
- `DEVICE` / `BATCH_SIZE` / `LIMIT`：传给 `lm-eval`

## E1 No Robots Smoke

脚本：

```bash
bash sft_study/runs/e1_no_robots_smoke.sh
```

用途：最小 SFT 冒烟实验，训练 `HuggingFaceH4/no_robots` 的小样本版本。

常用覆盖：

```bash
REPORT_TO=wandb \
WANDB_PROJECT=sft-study \
  bash sft_study/runs/e1_no_robots_smoke.sh
```

输出：

```text
sft_study/outputs/e1_no_robots_smoke
```

训练结束后会自动调用 `generate_fixed_prompts.sh`。

## E1 No Robots Full

脚本：

```bash
bash sft_study/runs/e1_no_robots_full.sh
```

用途：`no_robots` 的较完整训练版本，用来和 smoke 结果对照。

输出：

```text
sft_study/outputs/e1_no_robots_full
```

## E2 Prepare UltraChat Token Match

脚本：

```bash
bash sft_study/runs/e2_prepare_ultrachat_token_match.sh
```

用途：按 `no_robots` 的 token budget，从 `UltraChat` 中抽取对齐子集。

常用覆盖：

```bash
SEED=42 \
  bash sft_study/runs/e2_prepare_ultrachat_token_match.sh
```

输出：

```text
sft_study/artifacts/datasets/e2_ultrachat_token_matched_train.jsonl
```

## E2 No Robots Matched

脚本：

```bash
bash sft_study/runs/e2_no_robots_matched.sh
```

用途：E2 对照实验 A 组，训练 token 规模与 UltraChat matched 对齐的 `no_robots`。

输出：

```text
sft_study/outputs/e2_no_robots_matched
```

## E2 UltraChat Matched

脚本：

```bash
bash sft_study/runs/e2_ultrachat_matched.sh
```

用途：E2 对照实验 B 组，使用 token matched 后的 UltraChat 本地 JSONL。

常用覆盖：

```bash
TRAIN_JSONL=sft_study/artifacts/datasets/e2_ultrachat_token_matched_train.jsonl \
  bash sft_study/runs/e2_ultrachat_matched.sh
```

输出：

```text
sft_study/outputs/e2_ultrachat_matched
```

## E3 Smol Magpie 20K

脚本：

```bash
bash sft_study/runs/e3_smol_magpie_20k.sh
```

用途：训练 `HuggingFaceTB/smoltalk` 的 `smol-magpie-ultra` 配置，观察 synthetic 主线数据。

输出：

```text
sft_study/outputs/e3_smol_magpie_20k
```

## E4A Smol Constraints

脚本：

```bash
bash sft_study/runs/e4a_smol_constraints.sh
```

用途：把 `smol-constraints` 切片与 `smol-magpie-ultra` 混合，观察格式/约束类数据影响。

输出：

```text
sft_study/outputs/e4a_smol_constraints
```

常用变量：

- `COMMON_EVAL_CONFIG`：默认 `smol-magpie-ultra`

## E4B SystemChats 30K

脚本：

```bash
bash sft_study/runs/e4b_systemchats_30k.sh
```

用途：把 `systemchats-30k` 切片加入主线 synthetic 数据，观察 system/message 形态影响。

输出：

```text
sft_study/outputs/e4b_systemchats_30k
```

## E4C Numina CoT 100K

脚本：

```bash
bash sft_study/runs/e4c_numina_cot_100k.sh
```

用途：把 `numina-cot-100k` 切片加入主线 synthetic 数据，观察数学/推理链数据影响。

输出：

```text
sft_study/outputs/e4c_numina_cot_100k
```

## E5 Tulu3 100K

脚本：

```bash
bash sft_study/runs/e5_tulu3_100k.sh
```

用途：从 `allenai/tulu-3-sft-mixture` 切出本地 train/eval，训练 recipe 级混合数据。

输出：

```text
sft_study/outputs/e5_tulu3_100k
```

中间数据：

```text
sft_study/artifacts/datasets/e5_tulu3_train.jsonl
sft_study/artifacts/datasets/e5_tulu3_eval.jsonl
```

## 常见组合

首次学习闭环：

```bash
bash sft_study/runs/e0_fixed_prompts_base.sh
bash sft_study/runs/e1_no_robots_smoke.sh
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke LIMIT=20 \
  bash sft_study/runs/checkpoint_benchmark.sh
```

E2 对照：

```bash
bash sft_study/runs/e2_prepare_ultrachat_token_match.sh
bash sft_study/runs/e2_no_robots_matched.sh
bash sft_study/runs/e2_ultrachat_matched.sh
```

启用 Flash Attention 2：

```bash
ATTN_IMPLEMENTATION=flash_attention_2 \
  bash sft_study/runs/e1_no_robots_smoke.sh
```
