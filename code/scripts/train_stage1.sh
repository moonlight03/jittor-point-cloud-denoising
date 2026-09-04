#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
DATA_ROOT=${1:?usage: train_stage1.sh JOINT_AB_CLEAN_ROOT OUTPUT_ROOT INITIAL_CHECKPOINT [GPU]}
OUTPUT_ROOT=${2:?usage: train_stage1.sh JOINT_AB_CLEAN_ROOT OUTPUT_ROOT INITIAL_CHECKPOINT [GPU]}
CHECKPOINT=${3:?usage: train_stage1.sh JOINT_AB_CLEAN_ROOT OUTPUT_ROOT INITIAL_CHECKPOINT [GPU]}
GPU=${4:-0}
SMOKE_ARGS=()
[[ "${SMOKE:-0}" == 1 ]] && SMOKE_ARGS+=(--smoke)
CUDA_VISIBLE_DEVICES="$GPU" python "$ROOT/code/cli.py" train-stage \
    --stage stage1_ab_fixed_broad --init_checkpoint "$CHECKPOINT" \
    --data_root "$DATA_ROOT" --output_root "$OUTPUT_ROOT" "${SMOKE_ARGS[@]}"
