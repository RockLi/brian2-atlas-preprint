#!/usr/bin/env bash
set -euo pipefail

base=/data/brick2/brian2-mpi-region-20260907
thread_stage="$base/nmda-skaar-2025-cpu-thread-scaling-2560-20260917"
large_stage="$base/nmda-skaar-2025-scale20480-stage"
python_bin=/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python
project="$thread_stage/cpp/brian_benchmark_explicit_standalone_25600917_8"
project16="$thread_stage/cpp16/brian_benchmark_explicit_standalone_25600918_16"
thread_artifact="$thread_stage/rust/artifact"
thread_binary="$thread_artifact/native/b2-native-targetcpu-native"
large_artifact="$large_stage/rust/rust_explicit_20480_aot_2048018"
large_binary="$large_artifact/native/b2-native-targetcpu-native"
large_output="$large_stage/balanced_replay_targetcpu_native_cpu8_20260918"

test -x "$thread_binary"
test -x "$large_binary"
test ! -e "$large_output"
grep -q 'target-cpu=native' "$thread_stage/rust198_targetcpu_native_compile.json"
grep -q 'target-cpu=native' "$large_stage/rust198_targetcpu_native_compile.json"
test "$(awk '{print $1}' "$thread_stage/rust198_targetcpu_native_equivalence_hashes.txt" | sort -u | wc -l)" -eq 1
test -x "$project16/main"
grep -q '_nb_threads = 16;' "$project16/synapses_classes.h"
grep -q 'omp_set_num_threads(16);' "$project16/main.cpp"

for threads in 1 2 4 8 16; do
    if [ "$threads" -eq 1 ]; then
        cpu_list=0
    else
        cpu_list="0-$((threads - 1))"
    fi
    if [ "$threads" -eq 16 ]; then
        replay_project="$project16"
        replay_binary="$project16/main"
        output_name=replays_native_cpu16_regenerated
    else
        replay_project="$project"
        replay_binary="$project/thread_binaries/main_t${threads}"
        output_name="replays_native_cpu${threads}"
    fi
    "$python_bin" "$thread_stage/tools/replay_cpu_pair.py" \
        --cpp-project "$replay_project" \
        --cpp-binary "$replay_binary" \
        --rust-artifact "$thread_artifact" \
        --rust-binary "$thread_binary" \
        --output-dir "$thread_stage/$output_name" \
        --repeats 5 --threads "$threads" --cpu-list "$cpu_list" \
        > "$thread_stage/${output_name}.driver.log" 2>&1
done

date -u +completed_utc=%Y-%m-%dT%H:%M:%SZ \
    > "$thread_stage/thread_scaling_native_completed.txt"

"$python_bin" "$large_stage/tools/replay_cpu_pair.py" \
    --cpp-project "$large_stage/cpp/brian_benchmark_explicit_standalone_2048017_8" \
    --rust-artifact "$large_artifact" \
    --rust-binary "$large_binary" \
    --output-dir "$large_output" \
    --repeats 5 --threads 8 --cpu-list 0-7 \
    > "$large_stage/balanced_replay_targetcpu_native_cpu8_20260918.driver.log" 2>&1

sha256sum \
    "$large_artifact/rust/results.bin" \
    "$large_artifact/rust_balanced_replay_targetcpu_native_cpu8_20260918_warmup_0_rust/results.bin" \
    > "$large_stage/rust198_targetcpu_native_equivalence_hashes.txt"
date -u +completed_utc=%Y-%m-%dT%H:%M:%SZ \
    > "$large_stage/targetcpu_native_campaign_completed.txt"
