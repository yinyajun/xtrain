#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/generate_fixed_prompts.py"
  --model_name_or_path "$MODEL" \
  --prompts_file "$ROOT_DIR/data/fixed_prompts.jsonl" \
  --output_file "$ROOT_DIR/outputs/e0_base_fixed_prompts.jsonl" \
  --max_new_tokens 256
)

if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  CMD+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi

"${CMD[@]}"
