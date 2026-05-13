# Run Distributed

分布式训练入口都在 `sft_study/run_distributed/`。这里的脚本都保持 `e*_2gpu.sh` 形式，每个脚本都是独立完整的训练命令，不复用 `run_single/`。

```text
torchrun
DeepSpeed ZeRO-2
```

通用变量：

- `NPROC_PER_NODE`：默认 `2`
- `DEEPSPEED_CONFIG`：默认 `sft_study/deepspeed_zero2.json`
- `MODEL` / `TOKENIZER_PATH` / `ATTN_IMPLEMENTATION`：同单卡脚本

脚本文件名仍然保留 `e*_2gpu.sh`，方便区分它们是分布式版本；实际的输出目录和 W&B `run_name` 会使用 `NPROC_PER_NODE` 自动加后缀，比如 `2gpu`、`4gpu`。

## 常用方式

```bash
bash sft_study/run_distributed/e3_smol_magpie_20k_2gpu.sh
```

换卡数：

```bash
NPROC_PER_NODE=4 bash sft_study/run_distributed/e3_smol_magpie_20k_2gpu.sh
```

上面这条命令会写到 `outputs/e3_smol_magpie_20k_4gpu`，`run_name` 也是 `e3_smol_magpie_20k_4gpu`。

## 脚本列表

```bash
bash sft_study/run_distributed/e1_no_robots_smoke_2gpu.sh
bash sft_study/run_distributed/e1_no_robots_full_2gpu.sh
bash sft_study/run_distributed/e2_no_robots_matched_2gpu.sh
bash sft_study/run_distributed/e2_ultrachat_matched_2gpu.sh
bash sft_study/run_distributed/e3_smol_magpie_20k_2gpu.sh
bash sft_study/run_distributed/e4a_smol_constraints_2gpu.sh
bash sft_study/run_distributed/e4b_systemchats_30k_2gpu.sh
bash sft_study/run_distributed/e4c_numina_cot_100k_2gpu.sh
bash sft_study/run_distributed/e5_tulu3_100k_2gpu.sh
```

E2 的 UltraChat matched 会在训练脚本里自动准备数据：

```bash
bash sft_study/run_distributed/e2_ultrachat_matched_2gpu.sh
```

训练完成后，脚本会自动生成 fixed prompts。标准 benchmark 单独用 `run_eval/checkpoint_benchmark.sh`。
