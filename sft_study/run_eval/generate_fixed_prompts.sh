#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
MODEL="${MODEL:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
PROMPTS_FILE="${PROMPTS_FILE:-$ROOT_DIR/data/fixed_prompts.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

if [[ -n "$CHECKPOINT_DIR" ]]; then
  if [[ ! -f "$CHECKPOINT_DIR/run_config.json" ]]; then
    echo "CHECKPOINT_DIR must contain run_config.json: $CHECKPOINT_DIR" >&2
    exit 1
  fi
  MODEL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["model_name_or_path"])' "$CHECKPOINT_DIR/run_config.json")"
  ADAPTER_PATH="${ADAPTER_PATH:-$CHECKPOINT_DIR}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-$CHECKPOINT_DIR}"
  OUTPUT_FILE="${OUTPUT_FILE:-$CHECKPOINT_DIR/eval/fixed_prompts.jsonl}"
fi

if [[ -z "$MODEL" ]]; then
  echo "Pass MODEL or CHECKPOINT_DIR." >&2
  exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
  echo "Pass OUTPUT_FILE when generating without CHECKPOINT_DIR." >&2
  exit 1
fi

CMD=(
  python3 "$ROOT_DIR/scripts/generate.py"
  --model_name_or_path "$MODEL"
  --prompts_file "$PROMPTS_FILE"
  --output_file "$OUTPUT_FILE"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
)

if [[ -n "$ADAPTER_PATH" ]]; then
  CMD+=(--adapter_path "$ADAPTER_PATH")
fi

if [[ -n "$TOKENIZER_PATH" ]]; then
  CMD+=(--tokenizer_name_or_path "$TOKENIZER_PATH")
fi

if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  CMD+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi

"${CMD[@]}"
