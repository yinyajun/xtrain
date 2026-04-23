#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
MIX_JSONL="$ROOT_DIR/artifacts/datasets/e4c_numina_cot_mix.jsonl"
REPORT_TO="${REPORT_TO:-none}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/dataset_tools.py" mix \
  --component "HuggingFaceTB/smoltalk|smol-magpie-ultra|train|15000|-" \
  --component "HuggingFaceTB/smoltalk|numina-cot-100k|train|5000|-" \
  --output_jsonl "$MIX_JSONL"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_sft.py"
  --model_name_or_path "$MODEL" \
  --train_dataset "$MIX_JSONL" \
  --train_split "train" \
  --eval_dataset "HuggingFaceTB/smoltalk" \
  --eval_dataset_config "numina-cot-100k" \
  --eval_split "test" \
  --output_dir "$ROOT_DIR/outputs/e4c_numina_cot_100k" \
  --run_name "e4c_numina_cot_100k" \
  --max_eval_samples 1000 \
  --num_train_epochs 1 \
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
