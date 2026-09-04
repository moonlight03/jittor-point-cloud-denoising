#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

if [[ $# -lt 4 ]]; then
    echo "usage: prepare_meshes.sh DATASET_TAG MESH_ROOT OUTPUT_ROOT SPLIT_FILE [SPLIT_FILE ...]" >&2
    exit 2
fi

DATASET_TAG=$1
MESH_ROOT=$2
OUTPUT_ROOT=$3
shift 3
SPLIT_ARGS=()
for split_file in "$@"; do
    SPLIT_ARGS+=(--split-file "$split_file")
done

python "$ROOT/code/prepare_data.py" meshes \
    --dataset-tag "$DATASET_TAG" --mesh-root "$MESH_ROOT" \
    --output-root "$OUTPUT_ROOT" --points "${POINTS:-50000}" \
    --seed "${SEED:-2023}" --workers "${WORKERS:-8}" "${SPLIT_ARGS[@]}"
