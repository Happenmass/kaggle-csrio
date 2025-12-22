#!/usr/bin/env bash
set -euo pipefail

# Torch DDP launch helper for dino_finetune_scripts.py
# Usage: ./train_ddp.sh [extra-args passed to dino_finetune_scripts.py]

NUM_PROCS=${NUM_PROCS:-2}                # GPUs per node
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

SCRIPT=${SCRIPT:-dinov3_train_rebuild.py} # pretraining with LoRA adapters
IMAGE_ROOT=${IMAGE_ROOT:-csiro-biomass/train}
OUTPUT_DIR=${OUTPUT_DIR:-dino_rebuild_runs}

echo "Launching torchrun with ${NUM_PROCS} processes on ${SCRIPT}"
torchrun --nproc_per_node="${NUM_PROCS}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT}" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" \
  "$@"
