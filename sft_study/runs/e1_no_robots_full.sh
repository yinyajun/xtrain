#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
REPORT_TO="${REPORT_TO:-none}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_sft.py"
  --model_name_or_path "$MODEL" \
  --train_dataset "HuggingFaceH4/no_robots" \
  --train_dataset_config "" \
  --train_split "train_sft" \
  --eval_dataset "HuggingFaceH4/no_robots" \
  --eval_dataset_config "" \
  --eval_split "test_sft" \
  --output_dir "$ROOT_DIR/outputs/e1_no_robots_full" \
  --run_name "e1_no_robots_full" \
  --max_train_samples 9500 \
  --max_eval_samples 500 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --report_to "$REPORT_TO" \
  --gradient_checkpointing
)

if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  CMD+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi

if [[ -n "$TOKENIZER_PATH" ]]; then
  CMD+=(--tokenizer_name_or_path "$TOKENIZER_PATH")
fi

"${CMD[@]}"
