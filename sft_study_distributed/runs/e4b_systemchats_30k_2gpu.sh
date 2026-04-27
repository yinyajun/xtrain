#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-$ROOT_DIR/chat_templates/qwen2_5_training.jinja}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$ROOT_DIR/deepspeed_zero2_2gpu.json}"
MIX_JSONL="${MIX_JSONL:-$ROOT_DIR/artifacts/datasets/e4b_systemchats_mix.jsonl}"
SLICE_JSONL="${SLICE_JSONL:-$ROOT_DIR/artifacts/datasets/e4b_systemchats_token_matched.jsonl}"
COMMON_EVAL_CONFIG="${COMMON_EVAL_CONFIG:-smol-magpie-ultra}"

REPORT_TO="${REPORT_TO:-none}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/e4b_systemchats_30k_2gpu}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/dataset_tools.py" token-match \
  --model_name_or_path "$MODEL" \
  --reference_dataset "HuggingFaceTB/smoltalk" \
  --reference_dataset_config "smol-magpie-ultra" \
  --reference_split "train" \
  --reference_max_samples 5000 \
  --candidate_dataset "HuggingFaceTB/smoltalk" \
  --candidate_dataset_config "systemchats-30k" \
  --candidate_split "train" \
  --drop_overlong \
  --seed 42 \
  --output_jsonl "$SLICE_JSONL"

"$PYTHON_BIN" "$ROOT_DIR/scripts/dataset_tools.py" mix \
  --component "HuggingFaceTB/smoltalk|smol-magpie-ultra|train|15000|-" \
  --component "$SLICE_JSONL|-|train|all|-" \
  --seed 42 \
  --output_jsonl "$MIX_JSONL"

CMD=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nproc_per_node "$NPROC_PER_NODE"
  --master_port "$MASTER_PORT"
  "$ROOT_DIR/scripts/train_sft.py"
  --model_name_or_path "$MODEL"
  --chat_template_path "$CHAT_TEMPLATE_PATH"
  --deepspeed_config "$DEEPSPEED_CONFIG"
  --train_dataset "$MIX_JSONL"
  --train_split "train"
  --eval_dataset "HuggingFaceTB/smoltalk"
  --eval_dataset_config "$COMMON_EVAL_CONFIG"
  --eval_split "test"
  --output_dir "$OUTPUT_DIR"
  --run_name "e4b_systemchats_30k_2gpu"
  --max_eval_samples 1000
  --num_train_epochs 1
  --per_device_train_batch_size 2
  --per_device_eval_batch_size 2
  --gradient_accumulation_steps 16
  --logging_steps 10
  --eval_steps 100
  --save_steps 500
  --save_total_limit 1
  --max_length 2048
  --learning_rate 1e-4
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  --quantization auto
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.05
  --target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj lm_head
  --report_to "$REPORT_TO"
  --gradient_checkpointing
  --dataloader_num_workers 4
  --seed 42
)

if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  CMD+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi

if [[ -n "$TOKENIZER_PATH" ]]; then
  CMD+=(--tokenizer_name_or_path "$TOKENIZER_PATH")
fi

"${CMD[@]}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/benchmark.py" \
  --checkpoint_dir "$OUTPUT_DIR" \
  --skip_benchmarks
