#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

DATA_ROOT=${1:?usage: infer_four_gpu.sh DATA_ROOT RUN_ROOT [CHECKPOINT]}
RUN_ROOT=${2:?usage: infer_four_gpu.sh DATA_ROOT RUN_ROOT [CHECKPOINT]}
CHECKPOINT=${3:-$ROOT/checkpoints/model_epoch_0009.pkl}
EXPECTED_COUNT=${EXPECTED_COUNT:-200}
GPU_ID_TEXT=${GPU_IDS:-0,1,2,3}
IFS=',' read -r -a GPUS <<<"$GPU_ID_TEXT"
SHARDS=${#GPUS[@]}

if [[ "$SHARDS" -ne 4 ]]; then
    echo "GPU_IDS must contain exactly four comma-separated GPU IDs" >&2
    exit 2
fi

# 默认把完整推理流程交给 nohup 后台进程，当前终端立即返回。
if [[ "${JITTOR_INFER_DETACHED:-0}" != "1" ]]; then
    if [[ -e "$RUN_ROOT" ]]; then
        echo "Refusing to reuse existing run root: $RUN_ROOT" >&2
        exit 4
    fi
    mkdir -p "$(dirname "$RUN_ROOT")"
    mkdir "$RUN_ROOT"
    mkdir -p "$RUN_ROOT/parts" "$RUN_ROOT/reports" "$RUN_ROOT/logs"

    JITTOR_INFER_DETACHED=1 GPU_IDS="$GPU_ID_TEXT" EXPECTED_COUNT="$EXPECTED_COUNT" \
        nohup bash "$0" "$DATA_ROOT" "$RUN_ROOT" "$CHECKPOINT" \
        >"$RUN_ROOT/logs/inference.log" 2>&1 </dev/null &
    background_pid=$!
    printf '%s\n' "$background_pid" >"$RUN_ROOT/inference.pid"

    echo "四卡推理已成功启动到后台"
    echo "pid=$background_pid"
    echo "run_root=$RUN_ROOT"
    echo "log=$RUN_ROOT/logs/inference.log"
    exit 0
fi

worker_pids=()

# 只处理异常退出时的子进程清理，不创建额外监控任务。
cleanup_workers() {
    local pid
    for pid in "${worker_pids[@]:-}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
}
trap 'cleanup_workers; exit 130' INT
trap 'cleanup_workers; exit 143' TERM

for shard in 0 1 2 3; do
    gpu=${GPUS[$shard]}
    setsid env CUDA_VISIBLE_DEVICES="$gpu" \
        python "$ROOT/code/cli.py" infer \
        --ckpt "$CHECKPOINT" \
        --data_root "$DATA_ROOT" \
        --out_root "$RUN_ROOT/parts/gpu${gpu}" \
        --patch_size 2048 --patch_batch_size 12 --niters 2 --seed_k 8 \
        --stitch poly --stitch_power 1.0 --rotation identity --seed 2023 \
        --patch_radius_mode none --num_shards 4 --shard_index "$shard" \
        --out_json "$RUN_ROOT/reports/infer_gpu${gpu}.json" \
        >"$RUN_ROOT/logs/infer_gpu${gpu}.log" 2>&1 &
    worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    cleanup_workers
    echo "At least one Jittor inference shard failed; inspect $RUN_ROOT/logs" >&2
    exit "$status"
fi

python "$ROOT/code/cli.py" merge \
    --part_roots "$RUN_ROOT/parts/gpu${GPUS[0]}" "$RUN_ROOT/parts/gpu${GPUS[1]}" \
                 "$RUN_ROOT/parts/gpu${GPUS[2]}" "$RUN_ROOT/parts/gpu${GPUS[3]}" \
    --out_root "$RUN_ROOT/predictions" --expected_count "$EXPECTED_COUNT" \
    --out_json "$RUN_ROOT/reports/merge.json"
python "$ROOT/code/cli.py" validate \
    --pred_root "$RUN_ROOT/predictions" --data_root "$DATA_ROOT" \
    --expected_count "$EXPECTED_COUNT" --out_json "$RUN_ROOT/reports/validation.json"

echo "四卡推理运行成功：${EXPECTED_COUNT}/${EXPECTED_COUNT}"
echo "预测目录: $RUN_ROOT/predictions"
