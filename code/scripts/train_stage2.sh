#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
DATA_ROOT=${1:?usage: train_stage2.sh DATASET_B_CLEAN_ROOT STAGE1_MODEL OUTPUT_ROOT [GPU]}
STAGE1_MODEL=${2:?usage: train_stage2.sh DATASET_B_CLEAN_ROOT STAGE1_MODEL OUTPUT_ROOT [GPU]}
OUTPUT_ROOT=${3:?usage: train_stage2.sh DATASET_B_CLEAN_ROOT STAGE1_MODEL OUTPUT_ROOT [GPU]}
GPU=${4:-0}
SMOKE_ARGS=()
[[ "${SMOKE:-0}" == 1 ]] && SMOKE_ARGS+=(--smoke)
CUDA_VISIBLE_DEVICES="$GPU" python "$ROOT/code/cli.py" train-stage \
    --stage stage2_uniform_b0080_b0090 --init_checkpoint "$STAGE1_MODEL" \
    --data_root "$DATA_ROOT" --output_root "$OUTPUT_ROOT" "${SMOKE_ARGS[@]}"
