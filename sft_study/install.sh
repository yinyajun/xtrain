#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_FLASH_ATTN=0
INSTALL_DEEPSPEED=0

usage() {
  cat <<'EOF'
Usage:
  bash sft_study/install.sh
  bash sft_study/install.sh --flash-attn
  bash sft_study/install.sh --deepspeed
  bash sft_study/install.sh --deepspeed --flash-attn

Options:
  --deepspeed   Install optional DeepSpeed for distributed training.
  --flash-attn  Install optional FlashAttention 2 dependencies and package.
  -h, --help    Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deepspeed)
      INSTALL_DEEPSPEED=1
      shift
      ;;
    --flash-attn)
      INSTALL_FLASH_ATTN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "[install] Installing base requirements..."
python3 -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ "$INSTALL_DEEPSPEED" -eq 1 ]]; then
  echo "[install] Installing DeepSpeed..."
  python3 -m pip install deepspeed
fi

if [[ "$INSTALL_FLASH_ATTN" -eq 1 ]]; then
  echo "[install] Installing FlashAttention 2 prerequisites..."
  python3 -m pip install ninja packaging psutil
  echo "[install] Installing flash-attn..."
  python3 -m pip install flash-attn --no-build-isolation
fi

echo "[install] Done."
