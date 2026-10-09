#!/usr/bin/env bash
set -euo pipefail

# Host-specific launch record for the paper's largest explicit/general
# configuration. Run from the dedicated scale-20480 directory on Linux 23.
root=$(pwd)
python_bin=/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python
upstream=/atlas-home/0003/workspace/nmda-skaar-2025-20260915/nmda_skaar_2025_upstream
runner_script="$root/scripts/run_cpp_timed_fixture.py"
output="$root/original_brian2_scale20480_cpu8.npz"
project="$root/cpp/brian_benchmark_explicit_standalone_2048017_8"

test -x "$python_bin"
test -f "$upstream/brian_benchmark_explicit.py"
test -f "$runner_script"
test ! -e "$output"
test ! -e "${output%.npz}.json"
test ! -e "$project"

mkdir -p "$root/cpp"
git -C "$upstream" rev-parse HEAD > "$root/upstream_commit.txt"
sha256sum "$upstream/brian_benchmark_explicit.py" > "$root/upstream_source.sha256"
{
  date -u +started_utc=%Y-%m-%dT%H:%M:%SZ
  uname -a
  "$python_bin" --version
  gcc --version | head -1
  free -h
  df -h "$root"
} > "$root/launch_environment.txt"

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$root/cpp"
taskset -c 0-7 /usr/bin/time -v "$python_bin" "$runner_script" \
  --upstream "$upstream" \
  --scale 8.0 \
  --runner-id 2048017 \
  --threads 8 \
  --output "$output" \
  > "$root/original_brian2_scale20480_cpu8.log" \
  2> "$root/original_brian2_scale20480_cpu8.time.log"

date -u +completed_utc=%Y-%m-%dT%H:%M:%SZ > "$root/completed.txt"
