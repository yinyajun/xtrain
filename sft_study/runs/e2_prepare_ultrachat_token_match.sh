#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
SEED="${SEED:-42}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/dataset_utils.py" token-match \
  --model_name_or_path "$MODEL" \
  --reference_dataset "HuggingFaceH4/no_robots" \
  --reference_split "train_sft" \
  --reference_max_samples 9500 \
  --candidate_dataset "HuggingFaceH4/ultrachat_200k" \
  --candidate_split "train_sft" \
  --drop_overlong \
  --seed "$SEED" \
  --output_jsonl "$ROOT_DIR/artifacts/datasets/e2_ultrachat_token_matched_train.jsonl"
