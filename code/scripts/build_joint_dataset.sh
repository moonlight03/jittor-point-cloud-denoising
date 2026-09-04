#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

A_ROOT=${1:?usage: build_joint_dataset.sh A_NPY_ROOT B_NPY_ROOT OUTPUT_ROOT}
B_ROOT=${2:?usage: build_joint_dataset.sh A_NPY_ROOT B_NPY_ROOT OUTPUT_ROOT}
OUTPUT_ROOT=${3:?usage: build_joint_dataset.sh A_NPY_ROOT B_NPY_ROOT OUTPUT_ROOT}

python "$ROOT/code/prepare_data.py" joint \
    --source "a=$A_ROOT" --source "b=$B_ROOT" --output-root "$OUTPUT_ROOT" \
    --expected-count "${EXPECTED_COUNT:-35632}" --points "${POINTS:-50000}" \
    --mode "${MATERIALIZE_MODE:-symlink}"
