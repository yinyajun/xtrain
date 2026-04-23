#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
MODEL="${MODEL:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
BENCHMARKS="${BENCHMARKS:-ifeval gsm8k mmlu cmmlu}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
LIMIT="${LIMIT:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

read -r -a BENCHMARK_ARRAY <<< "$BENCHMARKS"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluate_checkpoint.py"
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
  --benchmarks "${BENCHMARK_ARRAY[@]}"
)

if [[ -n "$CHECKPOINT_DIR" ]]; then
  CMD+=(--checkpoint_dir "$CHECKPOINT_DIR")
fi

if [[ -n "$MODEL" ]]; then
  CMD+=(--model_name_or_path "$MODEL")
fi

if [[ -n "$ADAPTER_PATH" ]]; then
  CMD+=(--adapter_path "$ADAPTER_PATH")
fi

if [[ -n "$TOKENIZER_PATH" ]]; then
  CMD+=(--tokenizer_name_or_path "$TOKENIZER_PATH")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output_dir "$OUTPUT_DIR")
fi

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  CMD+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi

"${CMD[@]}"
