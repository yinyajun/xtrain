# SFT Study Workspace

这个目录是给“系统学习 SFT”单独开的。

主线目标只有三个：

1. 看懂 `base -> instruct` 的训练现象
2. 建立对主流公开 SFT 数据集的手感
3. 学会把“loss 下降”和“能力真的变好”分开看

## 为什么主线不用 instruct 模型

主线统一从 `Qwen/Qwen2.5-7B` 这种 base 模型开始，而不是 instruct 模型。

原因很简单：

- 你要学的是 SFT 在做什么
- 如果起点已经是 instruct，很多现象会被“模型本来就会”掩盖掉
- continued SFT 值得单独做，但不适合作为第一套学习实验

## 目录结构

```text
sft_study/
  experiments.md
  requirements.txt
  requirements-extra.txt
  data/
    fixed_prompts.jsonl
  scripts/
    common.py
    train_sft.py
    dataset_utils.py
    generate.py
    benchmark.py
    debug/
      compare_fixed_prompt_eos_rank.py
      debug_single_fixed_prompt.py
      inspect_model.py
      monitor_stop_behavior.py
      probe_eos_loss.py
  runs/
    e0_fixed_prompts_base.sh
    eval_checkpoint.sh
    e1_no_robots_smoke.sh
    e1_no_robots_full.sh
    e2_prepare_ultrachat_token_match.sh
    e2_no_robots_matched.sh
    e2_ultrachat_matched.sh
    e3_smol_magpie_20k.sh
    e4a_smol_constraints.sh
    e4b_systemchats_30k.sh
    e4c_numina_cot_100k.sh
    e5_tulu3_100k.sh
```

辅助脚本现在统一成两类入口：

- `scripts/dataset_utils.py`
  负责数据准备，子命令是 `mix`、`token-match` 和 `holdout-split`
- `scripts/debug/inspect_model.py`
  负责模板/样本诊断，子命令是 `template` 和 `examples`

## 统一约定

- 主线模型：`Qwen/Qwen2.5-7B`
- 训练方式：LoRA 为主；脚本支持自动切到 4-bit QLoRA
- 框架：`TRL SFTTrainer`
- 默认 `max_length=2048`
- 默认 `packing=False`
- 默认 `assistant_only_loss=True`
- 默认 `eos_token="<|im_end|>"`
- 每个实验都保留固定 prompt 输出

训练后的统一评测入口：

- `runs/eval_checkpoint.sh`
- 会生成固定 prompt 输出，并跑 `IFEval / GSM8K / MMLU / CMMLU`

训练类实验统一支持 `wandb`：

- 默认 `REPORT_TO=none`
- 想开启时，在命令前加：
  `REPORT_TO=wandb WANDB_PROJECT=sft-study`
- `run_name` 已经在各个 `runs/*.sh` 里固定好了，会直接作为 W&B run name 使用

如果你已经安装了 `flash-attn`，所有训练/评估脚本都支持用环境变量显式打开 `Flash Attention 2`：

```bash
ATTN_IMPLEMENTATION=flash_attention_2 bash sft_study/runs/e1_no_robots_smoke.sh
```

评估同理：

```bash
ATTN_IMPLEMENTATION=flash_attention_2 bash sft_study/runs/eval_checkpoint.sh
```

说明：

- 安装了 `flash-attn` 也不会自动启用，必须显式传 `ATTN_IMPLEMENTATION=flash_attention_2`
- 训练和评估都会把这个值透传给底层 `transformers`
- 如果显卡、dtype 或模型组合不兼容，先去掉这个环境变量再排查

为什么显式写 `eos_token`：

- `Qwen` 自带 chat template
- 但官方 TRL 文档明确建议给 `Qwen2.5` 系列显式设 `eos_token="<|im_end|>"`
- 否则容易出现停不住或者对话边界异常

## 评估最小闭环

每个实验至少做三件事：

1. 保存 train/eval loss
2. 跑固定 prompt 生成
3. 跑 `IFEval`

现在把 benchmark 也统一收进闭环：

- `IFEval`：必跑，最贴近 SFT 对 instruction following 的影响
- `GSM8K`：只在数学切片实验里重点看
- `MMLU/CMMLU`：看通用能力有没有明显回归
默认 benchmark 配置：

- `IFEval`：`ifeval`，0-shot，chat template
- `GSM8K`：`gsm8k_cot`，8-shot，chat template
- `MMLU`：`mmlu`，5-shot
- `CMMLU`：`cmmlu`，5-shot

注意：

- benchmark 由 `lm-eval` 统一执行

## 实验顺序

建议严格按这个顺序跑：

1. `E0` 看 base 模型原始输出
2. `E1` 跑最标准的干净单轮 SFT
3. `E2` 做单轮 instruction vs 多轮 chat 对照
4. `E3` 跑现代 synthetic 主线数据
5. `E4` 做能力切片 ablation
6. `E5` 最后再看 recipe 级混合数据

## E0：不训练，先留基线

问题：

- base 模型原始状态下到底有多不像一个助手？

数据：

- 不训练
- 直接用 `data/fixed_prompts.jsonl`

跑法：

- `runs/e0_fixed_prompts_base.sh`

运行命令：

```bash
bash sft_study/runs/e0_fixed_prompts_base.sh
```

如果你已经装了 `flash-attn`：

```bash
ATTN_IMPLEMENTATION=flash_attention_2 bash sft_study/runs/e0_fixed_prompts_base.sh
```

主要观察：

- 回答是否像补全器
- 是否不稳定
- 是否不擅长格式约束

如果你想给 base 模型也留一份 benchmark 基线：

```bash
MODEL=Qwen/Qwen2.5-7B \
OUTPUT_DIR=sft_study/outputs/e0_base_eval \
  bash sft_study/runs/eval_checkpoint.sh
```

如果你已经装了 `flash-attn`：

```bash
ATTN_IMPLEMENTATION=flash_attention_2 \
MODEL=Qwen/Qwen2.5-7B \
OUTPUT_DIR=sft_study/outputs/e0_base_eval \
  bash sft_study/runs/eval_checkpoint.sh
```

## E1：No Robots 干净单轮基线

问题：

- 最教科书式的 `base -> instruct` 长什么样？

数据：

- `HuggingFaceH4/no_robots`
- 训练：`train_sft`
- 验证：`test_sft`

脚本：

- 冒烟版：`runs/e1_no_robots_smoke.sh`
- 完整版：`runs/e1_no_robots_full.sh`

运行命令：

```bash
bash sft_study/runs/e1_no_robots_smoke.sh
```

如果你已经装了 `flash-attn`：

```bash
ATTN_IMPLEMENTATION=flash_attention_2 bash sft_study/runs/e1_no_robots_smoke.sh
```

```bash
bash sft_study/runs/e1_no_robots_full.sh
```

如果要记到 W&B：

```bash
REPORT_TO=wandb WANDB_PROJECT=sft-study \
  bash sft_study/runs/e1_no_robots_smoke.sh
```

建议先跑：

- `2k train + 200 eval`
- 再跑完整 `9500 + 500`

主要观察：

- train loss 是否稳定下降
- eval loss 是否同步下降
- 固定 prompt 是否明显更像助手
- `IFEval` 是否明显上涨

成功标准：

- 输出从“像续写器”变成“像助手”
- 没有明显模板崩坏

## E2：No Robots vs UltraChat 200k

问题：

- 单轮 instruction 数据和多轮 chat 数据到底教会模型什么？

数据：

- A 组：`HuggingFaceH4/no_robots`
- B 组：`HuggingFaceH4/ultrachat_200k`

关键约束：

- 不按样本数对齐
- 按 token budget 对齐

为什么：

- `UltraChat` 天然更长、轮次更多
- 如果只按条数对齐，结论会被长度差带偏

脚本：

- 先准备对齐子集：`runs/e2_prepare_ultrachat_token_match.sh`
- A 组训练：`runs/e2_no_robots_matched.sh`
- B 组训练：`runs/e2_ultrachat_matched.sh`

额外控制：

- 两组都固定成 `1 epoch`
- `UltraChat` 子集按 `No Robots 9500` 的 token budget 对齐
- 对照时不要拿 `E1 full` 直接和 `E2 UltraChat` 比

运行命令：

```bash
bash sft_study/runs/e2_prepare_ultrachat_token_match.sh
bash sft_study/runs/e2_no_robots_matched.sh
bash sft_study/runs/e2_ultrachat_matched.sh
```

如果要记到 W&B：

```bash
bash sft_study/runs/e2_prepare_ultrachat_token_match.sh
REPORT_TO=wandb WANDB_PROJECT=sft-study \
  bash sft_study/runs/e2_no_robots_matched.sh
REPORT_TO=wandb WANDB_PROJECT=sft-study \
  bash sft_study/runs/e2_ultrachat_matched.sh
```

主要观察：

- `No Robots` 是否更干净地执行单轮任务
- `UltraChat` 是否更擅长 follow-up 和延续对话
- 哪边 loss 更好看，哪边行为更像聊天助手

最值得学的点：

- 更低的 loss 不一定代表更好的多轮体验

## E3：SmolTalk 的现代 synthetic 主线

问题：

- 现在更主流的高质量 synthetic SFT 数据是什么感觉？

数据：

- `HuggingFaceTB/smoltalk`
- config：`smol-magpie-ultra`

脚本：

- `runs/e3_smol_magpie_20k.sh`

运行命令：

```bash
bash sft_study/runs/e3_smol_magpie_20k.sh
```

如果要记到 W&B：

```bash
REPORT_TO=wandb WANDB_PROJECT=sft-study \
  bash sft_study/runs/e3_smol_magpie_20k.sh
```

建议规模：

- 先跑 `20k`
- 如果感觉稳定，再补一档 `100k`

主要观察：

- 和 `No Robots` 相比，风格是否更“现代助手”
- `IFEval` 和固定 prompt 是否更平衡
- 数据规模变大后曲线是否更平滑

## E4：能力切片 ablation

问题：

- 加什么数据，会拉什么能力？

核心思路：

- 固定一个主干数据 `smol-magpie-ultra`
- 每次只加一种切片
- 新增切片先按 token budget 对齐到 `5k smol-magpie-ultra`
- train-time eval 统一看 `smol-magpie-ultra/test`，避免每个实验换 eval 集

### E4a：加约束遵循

数据混合：

- `smol-magpie-ultra` 15k
- `smol-constraints` token-matched 到 `5k smol-magpie-ultra`

脚本：

- `runs/e4a_smol_constraints.sh`

运行命令：

```bash
bash sft_study/runs/e4a_smol_constraints.sh
```

如果要记到 W&B：

```bash
REPORT_TO=wandb WANDB_PROJECT=sft-study \
  bash sft_study/runs/e4a_smol_constraints.sh
```

关注：

- 公共 eval loss 是否稳定
- `IFEval`
- 固定 prompt 里的格式、长度、关键词约束

### E4b：加 system prompt 跟随

数据混合：

- `smol-magpie-ultra` 15k
- `systemchats-30k` token-matched 到 `5k smol-magpie-ultra`

脚本：

- `runs/e4b_systemchats_30k.sh`

运行命令：

```bash
bash sft_study/runs/e4b_systemchats_30k.sh
```

如果要记到 W&B：

```bash
REPORT_TO=wandb WANDB_PROJECT=sft-study \
  bash sft_study/runs/e4b_systemchats_30k.sh
```

关注：

- 公共 eval loss 是否稳定
- system 约束是否更稳
- 多轮中是否更能记住前置设定

### E4c：加数学 CoT

数据混合：

- `smol-magpie-ultra` 15k
- `numina-cot-100k` token-matched 到 `5k smol-magpie-ultra`

脚本：

- `runs/e4c_numina_cot_100k.sh`

运行命令：

```bash
bash sft_study/runs/e4c_numina_cot_100k.sh
```

如果要记到 W&B：

```bash
REPORT_TO=wandb WANDB_PROJECT=sft-study \
  bash sft_study/runs/e4c_numina_cot_100k.sh
```

关注：

- 公共 eval loss 是否稳定
- `GSM8K`
- 固定 prompt 里的推理完整度

## E5：Tulu 3 SFT mixture

问题：

- recipe 级混合数据和单一数据集的感觉有什么不同？

数据：

- `allenai/tulu-3-sft-mixture`

脚本：

- `runs/e5_tulu3_100k.sh`

运行命令：

```bash
bash sft_study/runs/e5_tulu3_100k.sh
```

如果要记到 W&B：

```bash
REPORT_TO=wandb WANDB_PROJECT=sft-study \
  bash sft_study/runs/e5_tulu3_100k.sh
```

建议做法：

- 先只抽 `100k`
- 先从同一个 `train` split 里切一份本地 held-out eval，再训练
- 不要第一轮就尝试整套 90 多万

主要观察：

- 指标是否更均衡
- 风格是否更杂、更泛化
- 和 `E3/E4` 相比，单项能力是否反而没那么突出

## 你每次都要记下来的东西

- 用的模型和数据
- `seed`
- token budget
- 训练步数
- train/eval loss 的形状
- 固定 prompt 的训练前后输出
- 一句话结论：
  这次学到的是“模型更会做任务了”，还是“模型只是更像助手了”

关键对照实验建议至少补到 `3` 个 seed：

- `E2` 的 `No Robots matched vs UltraChat matched`
- `E4` 任意一个切片实验
- `E5` 的 recipe mixture

## 启动建议

第一次建议只跑这三个：

1. `runs/e0_fixed_prompts_base.sh`
2. `runs/e1_no_robots_smoke.sh`
3. `runs/e2_prepare_ultrachat_token_match.sh && runs/e2_no_robots_matched.sh && runs/e2_ultrachat_matched.sh`

这样你会最快建立两个核心直觉：

- SFT 怎么把 base 模型拉成 instruct
- 不同数据形态到底在教模型什么

训练完任意一个 checkpoint 后，统一补评测：

```bash
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke \
  bash sft_study/runs/eval_checkpoint.sh
```

如果你这次只想快速看 fixed case，不想跑 benchmark：

```bash
python sft_study/scripts/benchmark.py \
  --checkpoint_dir sft_study/outputs/e1_no_robots_smoke \
  --skip_benchmarks
```

说明：

- 这条命令会自动从 `checkpoint_dir/run_config.json` 里恢复底模信息
- 会把 `checkpoint_dir` 本身当成 `adapter_path` 来加载
- 默认输出到 `sft_study/outputs/e1_no_robots_smoke/eval/fixed_prompts.jsonl`

如果你只想跑部分 benchmark：

```bash
CHECKPOINT_DIR=sft_study/outputs/e1_no_robots_smoke \
BENCHMARKS="ifeval gsm8k" \
  bash sft_study/runs/eval_checkpoint.sh
```

输出约定：

- 固定 prompts：`<eval_output_dir>/fixed_prompts.jsonl`
- 单条样本调试：`python sft_study/scripts/debug/debug_single_fixed_prompt.py --checkpoint_dir <checkpoint_dir> --prompt_id one_word_capital`
  用来排查“为什么停不住”这类问题；会打印渲染后的 prompt、tokenizer special tokens、raw completion（`skip_special_tokens=False`）以及 `<|im_end|>` 在生成序列中的位置
- eos rank 对比：`python sft_study/scripts/debug/compare_fixed_prompt_eos_rank.py --checkpoint_dir <checkpoint_dir> --prompt_id one_word_capital`
  用来对比 base model 和 checkpoint 在同一条 fixed prompt 上的 `chosen token / native eos / <|im_end|> / <|endoftext|>` 倾向，避免把不同 tokenizer 的 eos 混在一起看
- 停止行为监控：`python sft_study/scripts/debug/monitor_stop_behavior.py --checkpoint_dir <checkpoint_dir>`
  会同时给出普通 assistant 回复末尾 `<|im_end|>` 的平均 NLL、空 assistant probe 的 `<|im_end|>` NLL，以及 fixed prompts 的自然停止率，适合按 checkpoint 或 epoch 纵向看趋势
- 模板和训练样本检查：`python sft_study/scripts/debug/inspect_model.py examples --checkpoint_dir <checkpoint_dir> --num_examples 10`
  会直接读取训练集前 N 条，把 `messages -> chat template -> token ids` 这条链渲染出来，并额外对比 checkpoint 里保存的 `chat_template.jinja`
- 固定 prompts viewer：直接打开 `sft_study/fixed_prompts_viewer.html`
  在浏览器里选择一个或多个 `fixed_prompts.jsonl` 文件后，就能以更易读的卡片形式查看内容；不需要 build，也不需要起服务
- benchmark 原始结果：`<eval_output_dir>/benchmarks/`
- 汇总：`<eval_output_dir>/benchmarks/summary.json`

## 依赖

建议先安装基础依赖：

```bash
pip install -r sft_study/requirements.txt
```

如果你是 Linux + NVIDIA，并且想额外启用 `flash-attn` 这类可选加速，先装前置依赖：

```bash
pip install -r sft_study/requirements-extra.txt
```

再单独安装 `flash-attn`：

```bash
pip install flash-attn --no-build-isolation
```

说明：

- `requirements.txt` 放基础训练与评估依赖，也包含 `wandb`
- `requirements-extra.txt` 只放可选 GPU 加速前置依赖
- `flash-attn` 需要单独安装，是因为它常常要配合 `--no-build-isolation`
- `flash-attn` 通常只建议在 Linux + CUDA 环境安装
- 如果你是本机 macOS / CPU / MPS 环境，默认会退回普通 LoRA 配置，但速度会慢很多

## 参考

- TRL SFTTrainer: https://huggingface.co/docs/trl/en/sft_trainer
- Qwen2.5-7B: https://huggingface.co/Qwen/Qwen2.5-7B
- No Robots: https://huggingface.co/datasets/HuggingFaceH4/no_robots
- UltraChat 200k: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- SmolTalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- Tulu 3 SFT mixture: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- IFEval task in lm-eval: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ifeval/README.md
