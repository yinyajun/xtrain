# SFT Study Distributed

这套目录是 `sft_study` 的分布式独立版本。

目标很简单：

1. 用两卡跑和 `sft_study` 类似的主线实验
2. 保持目录、脚本、配置完全独立
3. 优先验证两卡训练是否比单卡更划算

## 目录结构

```text
sft_study_distributed/
  experiments.md
  deepspeed_zero2_2gpu.json
  chat_templates/
    qwen2_5_training.jinja
  data/
    fixed_prompts.jsonl
  scripts/
    benchmark.py
    common.py
    dataset_tools.py
    generate.py
    train_sft.py
  runs/
    e3_smol_magpie_20k_2gpu.sh
    e4a_smol_constraints_2gpu.sh
    e4b_systemchats_30k_2gpu.sh
    e4c_numina_cot_100k_2gpu.sh
    e5_tulu3_100k_2gpu.sh
```

## 当前约定

- 默认两卡：`NPROC_PER_NODE=2`
- 默认用：`torch.distributed.run + DeepSpeed ZeRO-2`
- 训练入口：`scripts/train_sft.py`
- 评估入口：`scripts/benchmark.py`
- 默认只跑 fixed prompts，不自动跑 benchmark

## 运行方式

E3:

```bash
bash sft_study_distributed/runs/e3_smol_magpie_20k_2gpu.sh
```

E4a:

```bash
bash sft_study_distributed/runs/e4a_smol_constraints_2gpu.sh
```

E4b:

```bash
bash sft_study_distributed/runs/e4b_systemchats_30k_2gpu.sh
```

E4c:

```bash
bash sft_study_distributed/runs/e4c_numina_cot_100k_2gpu.sh
```

E5:

```bash
bash sft_study_distributed/runs/e5_tulu3_100k_2gpu.sh
```

如果要记到 W&B：

```bash
REPORT_TO=wandb WANDB_PROJECT=sft-study-distributed \
  bash sft_study_distributed/runs/e3_smol_magpie_20k_2gpu.sh
```

## 说明

- 这套目录不再调用 `sft_study` 下的脚本
- `artifacts/` 和 `outputs/` 都会写在 `sft_study_distributed` 自己目录下
- 如果后面要扩成更多多卡实验，就继续在这里加，不回写 `sft_study`
