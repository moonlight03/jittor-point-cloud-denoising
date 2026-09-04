#!/usr/bin/env bash

PDLTS_JITTOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PDLTS_JITTOR_ROOT

# 始终使用调用者当前环境中的 python、编译器和 CUDA，不覆盖环境配置。
if ! command -v python >/dev/null 2>&1; then
    echo "python is not available; activate a Jittor environment first" >&2
    return 2 2>/dev/null || exit 2
fi

# 不在交付代码中生成字节码缓存。
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PDLTS_JITTOR_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
