#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

DATA_ROOT=${1:?usage: infer_single_gpu.sh DATA_ROOT RUN_ROOT [GPU] [CHECKPOINT]}
RUN_ROOT=${2:?usage: infer_single_gpu.sh DATA_ROOT RUN_ROOT [GPU] [CHECKPOINT]}
GPU=${3:-0}
CHECKPOINT=${4:-$ROOT/checkpoints/model_epoch_0009.pkl}
EXPECTED_COUNT=${EXPECTED_COUNT:-200}

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to reuse existing run root: $RUN_ROOT" >&2
    exit 4
fi
mkdir -p "$RUN_ROOT/reports" "$RUN_ROOT/logs"
CUDA_VISIBLE_DEVICES="$GPU" python "$ROOT/code/cli.py" infer \
    --ckpt "$CHECKPOINT" \
    --data_root "$DATA_ROOT" \
    --out_root "$RUN_ROOT/predictions" \
    --patch_size 2048 --patch_batch_size 12 --niters 2 --seed_k 8 \
    --stitch poly --stitch_power 1.0 --rotation identity --seed 2023 \
    --patch_radius_mode none --out_json "$RUN_ROOT/reports/inference.json" \
    >"$RUN_ROOT/logs/inference.log" 2>&1
python "$ROOT/code/cli.py" validate \
    --pred_root "$RUN_ROOT/predictions" --data_root "$DATA_ROOT" \
    --expected_count "$EXPECTED_COUNT" --out_json "$RUN_ROOT/reports/validation.json"
