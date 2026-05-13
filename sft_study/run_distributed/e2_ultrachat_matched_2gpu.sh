#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
RUN_SUFFIX="${NPROC_PER_NODE}gpu"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$ROOT_DIR/deepspeed_zero2.json}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
SEED="${SEED:-42}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/artifacts/datasets/e2_ultrachat_token_matched_train.jsonl}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
OUTPUT_DIR="$ROOT_DIR/outputs/e2_ultrachat_matched_$RUN_SUFFIX"

python3 "$ROOT_DIR/scripts/dataset_utils.py" token-match \
  --model_name_or_path "$MODEL" \
  --reference_dataset "HuggingFaceH4/no_robots" \
  --reference_split "train" \
  --reference_max_samples 9500 \
  --candidate_dataset "HuggingFaceH4/ultrachat_200k" \
  --candidate_split "train_sft" \
  --drop_overlong \
  --seed "$SEED" \
  --output_jsonl "$TRAIN_JSONL"

CMD=(
  torchrun
  --nproc_per_node "$NPROC_PER_NODE"
  "$ROOT_DIR/scripts/train_sft.py"
  --deepspeed_config "$DEEPSPEED_CONFIG"
  --model_name_or_path "$MODEL" \
  --chat_template_path "$ROOT_DIR/chat_templates/qwen2_5_training.jinja" \
  --train_dataset "$TRAIN_JSONL" \
  --train_split "train" \
  --eval_dataset "HuggingFaceH4/ultrachat_200k" \
  --eval_split "test_sft" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "e2_ultrachat_matched_$RUN_SUFFIX" \
  --num_train_epochs 1 \
  --max_eval_samples 500 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 1 \
  --max_length 2048 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --quantization auto \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj lm_head \
  --report_to wandb \
  --gradient_checkpointing \
  --seed "$SEED"
)

if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  CMD+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi

if [[ -n "$TOKENIZER_PATH" ]]; then
  CMD+=(--tokenizer_name_or_path "$TOKENIZER_PATH")
fi

"${CMD[@]}"

CHECKPOINT_DIR="$OUTPUT_DIR" ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" bash "$ROOT_DIR/run_eval/generate_fixed_prompts.sh"
