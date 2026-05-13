# Run Single

单卡训练入口都在 `sft_study/run_single/`。这些脚本直接调用 `scripts/train_sft.py`，训练结束后会自动调用 `run_eval/generate_fixed_prompts.sh` 生成 fixed prompts。

通用变量：

- `MODEL`：默认 `Qwen/Qwen2.5-7B`
- `TOKENIZER_PATH`：可选 tokenizer 路径
- `ATTN_IMPLEMENTATION`：可选 `flash_attention_2`

## 常用顺序

```bash
bash sft_study/run_single/e1_no_robots_smoke.sh
bash sft_study/run_single/e2_prepare_ultrachat_token_match.sh
bash sft_study/run_single/e2_no_robots_matched.sh
bash sft_study/run_single/e2_ultrachat_matched.sh
```

## 脚本列表

`e1_no_robots_smoke.sh`

最小 SFT 冒烟实验：

```bash
bash sft_study/run_single/e1_no_robots_smoke.sh
```

`e1_no_robots_full.sh`

较完整的 No Robots 实验：

```bash
bash sft_study/run_single/e1_no_robots_full.sh
```

`e2_prepare_ultrachat_token_match.sh`

准备 UltraChat token-matched 本地 JSONL：

```bash
bash sft_study/run_single/e2_prepare_ultrachat_token_match.sh
```

`e2_no_robots_matched.sh`

E2 对照 A 组：

```bash
bash sft_study/run_single/e2_no_robots_matched.sh
```

`e2_ultrachat_matched.sh`

E2 对照 B 组：

```bash
bash sft_study/run_single/e2_ultrachat_matched.sh
```

`e3_smol_magpie_20k.sh`

SmolTalk synthetic 主线：

```bash
bash sft_study/run_single/e3_smol_magpie_20k.sh
```

`e4a_smol_constraints.sh`

约束/格式切片实验：

```bash
bash sft_study/run_single/e4a_smol_constraints.sh
```

`e4b_systemchats_30k.sh`

SystemChats 切片实验：

```bash
bash sft_study/run_single/e4b_systemchats_30k.sh
```

`e4c_numina_cot_100k.sh`

Numina CoT 数学推理切片实验：

```bash
bash sft_study/run_single/e4c_numina_cot_100k.sh
```

`e5_tulu3_100k.sh`

Tulu3 recipe mixture 实验：

```bash
bash sft_study/run_single/e5_tulu3_100k.sh
```

## W&B

```bash
bash sft_study/run_single/e1_no_robots_smoke.sh
```
