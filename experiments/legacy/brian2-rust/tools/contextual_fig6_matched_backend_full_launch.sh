#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "hk-prod-model-ae09-94" ]]; then
    echo "Refusing to simulate off the approved remote host" >&2
    exit 2
fi

workspace=/atlas-home/0003/workspace/contextual-dendritic-gating-20260921
job="$workspace/fig6-matched-cython-full-v1"
cd "$workspace"

exec env PATH="$workspace/tools:$PATH" MPLBACKEND=Agg \
    taskset -c 169 nice -n 15 \
    "$workspace/.paper-venv-fig6-py310/bin/python" \
    "$job/tools/contextual_fig6_matched_backend_full_launcher.py" \
    --driver "$job/tools/contextual_dendritic_fig6_corrected_full_job.py" \
    --extension "$workspace/fig6-final-checkpoint-state-audit-v1/cython-queue-build-v2/lib/brian2/synapses/cythonspikequeue.cpython-310-x86_64-linux-gnu.so" \
    --frozen-prefix "$workspace/fig6-corrected-full-v1/prefix-comparison-v1.json" \
    --matched-prefix "$job/matched-prefix-comparison-v1.json" \
    --launcher-preflight "$job/launcher-preflight-run-v2.json" \
    -- \
    --paper-repo "$job/paper-repository" \
    --candidate-arrays "$workspace/fig6-numpy-preprocessing-probe-v1/numpy126-arrays.npz" \
    --sort-report "$workspace/fig6-sort-prefix-v1/numpy244-indices-v2.json" \
    --prefix-comparison "$workspace/fig6-corrected-full-v1/prefix-comparison-v1.json" \
    --official-report "$job/official-report-v1.json" \
    --audit-report "$job/corrected-audit-v1.json" \
    --reproduction-id fig6-matched-cython-full-v1
