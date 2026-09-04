#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
STAGE1_ROOT=${1:?usage: train_two_stage.sh JOINT_AB_CLEAN_ROOT DATASET_B_CLEAN_ROOT OUTPUT_ROOT INITIAL_CHECKPOINT [GPU]}
STAGE2_ROOT=${2:?usage: train_two_stage.sh JOINT_AB_CLEAN_ROOT DATASET_B_CLEAN_ROOT OUTPUT_ROOT INITIAL_CHECKPOINT [GPU]}
OUTPUT_ROOT=${3:?usage: train_two_stage.sh JOINT_AB_CLEAN_ROOT DATASET_B_CLEAN_ROOT OUTPUT_ROOT INITIAL_CHECKPOINT [GPU]}
INITIAL_CHECKPOINT=${4:?usage: train_two_stage.sh JOINT_AB_CLEAN_ROOT DATASET_B_CLEAN_ROOT OUTPUT_ROOT INITIAL_CHECKPOINT [GPU]}
GPU=${5:-0}
SMOKE_ARGS=()
[[ "${SMOKE:-0}" == 1 ]] && SMOKE_ARGS+=(--smoke)
CUDA_VISIBLE_DEVICES="$GPU" python "$ROOT/code/cli.py" train-two-stage \
    --initial_checkpoint "$INITIAL_CHECKPOINT" --stage1_data_root "$STAGE1_ROOT" \
    --stage2_data_root "$STAGE2_ROOT" --output_root "$OUTPUT_ROOT" "${SMOKE_ARGS[@]}"
