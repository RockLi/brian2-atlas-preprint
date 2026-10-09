#!/usr/bin/env bash
set -euo pipefail

# Linux 23 follow-on for the NMDA 2025 validation.  It waits for the formal
# scale-20,480 CPU8 replay so the two controlled measurements cannot contend.
base=/data/brick2/brian2-mpi-region-20260907
formal_stage="$base/nmda-skaar-2025-scale20480-stage"
formal_pid=$(cat "$formal_stage/balanced_replay_cpu8_20260917.pid")
formal_out="$formal_stage/balanced_replay_cpu8_20260917"
stage="$base/nmda-skaar-2025-cpu-thread-scaling-2560-20260917"
snapshot="$base/nmda-skaar-2025-scale10240/validation_10240_20260916"
python_bin=/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python
upstream=/atlas-home/0003/workspace/nmda-skaar-2025-20260915/nmda_skaar_2025_upstream
cpp_script="$formal_stage/scripts/run_cpp_timed_fixture.py"
rust_script="$snapshot/benchmarks/published/nmda_skaar_2025/scripts/run_rust_fixture.py"
runner="$snapshot/cargo_target_rust198/release/b2-runner"
rust_bin_dir="$base/rustup-1.98.1/toolchains/1.98.1-x86_64-unknown-linux-gnu/bin"

test ! -e "$stage"
mkdir -p "$stage/tools" "$stage/cpp" "$stage/rust_work" "$stage/rust"
cp "$formal_stage/tools/measure_binary.py" "$stage/tools/"
cp "$formal_stage/tools/measure_rust_binary.py" "$stage/tools/"
cp "$formal_stage/tools/replay_cpu_pair.py" "$stage/tools/"

while kill -0 "$formal_pid" 2>/dev/null; do
    sleep 30
done

"$python_bin" -c 'import json, pathlib, sys; rows=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert len(rows)==12 and all(row["exit_code"]==0 for row in rows)' "$formal_out/schedule.json"

export PATH="$rust_bin_dir:$PATH"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
    date -u +started_utc=%Y-%m-%dT%H:%M:%SZ
    uname -a
    lscpu
    "$python_bin" --version
    "$python_bin" -c 'import brian2; print("Brian2", brian2.__version__)'
    rustc --version
    gcc --version | head -1
    git -C "$upstream" rev-parse HEAD
} > "$stage/environment.txt"

cd "$stage/cpp"
taskset -c 0-7 /usr/bin/time -v "$python_bin" "$cpp_script" \
    --upstream "$upstream" --scale 1.0 --runner-id 25600917 \
    --threads 8 --output "$stage/original_brian2_scale2560_cpu8.npz" \
    > "$stage/original_brian2_scale2560_cpu8.log" \
    2> "$stage/original_brian2_scale2560_cpu8.time.log"

cd "$stage/rust_work"
taskset -c 0-7 /usr/bin/time -v "$python_bin" "$rust_script" \
    --upstream "$upstream" --script brian_benchmark_explicit.py --scale 1.0 \
    --threads 8 --thread-affinity required --runner "$runner" --engine aot \
    --artifact "$stage/rust/artifact" --output "$stage/rust_scale2560_cpu8.npz" \
    > "$stage/rust_scale2560_cpu8.log" \
    2> "$stage/rust_scale2560_cpu8.time.log"

project="$stage/cpp/brian_benchmark_explicit_standalone_25600917_8"
mkdir "$project/thread_binaries"
cp -p "$project/main.cpp" "$project/main.cpp.cpu8.original"
cp -p "$project/main.o" "$project/main.o.cpu8.original"
cp -p "$project/main" "$project/main.cpu8.original"
cp -p "$project/main" "$project/thread_binaries/main_t8"

for threads in 1 2 4; do
    sed "s/omp_set_num_threads(8);/omp_set_num_threads(${threads});/" \
        "$project/main.cpp.cpu8.original" > "$project/main.cpp"
    grep -q "omp_set_num_threads(${threads});" "$project/main.cpp"
    make -C "$project" main.o main
    cp -p "$project/main" "$project/thread_binaries/main_t${threads}"
done

cp -p "$project/main.cpp.cpu8.original" "$project/main.cpp"
cp -p "$project/main.o.cpu8.original" "$project/main.o"
cp -p "$project/main.cpu8.original" "$project/main"
sha256sum "$project"/thread_binaries/main_t* > "$stage/thread_binaries.sha256"

# Brian2 bakes the OpenMP thread count into more than main.cpp. In particular,
# SynapticPathway allocates one queue per generation-time thread. A 16-thread
# executable must therefore be regenerated, not obtained by patching the
# eight-thread main.cpp (which indexes eight queues with omp_get_thread_num()).
mkdir "$stage/cpp16"
cd "$stage/cpp16"
taskset -c 0-15 /usr/bin/time -v "$python_bin" "$cpp_script" \
    --upstream "$upstream" --scale 1.0 --runner-id 25600918 \
    --threads 16 --output "$stage/original_brian2_scale2560_cpu16.npz" \
    > "$stage/original_brian2_scale2560_cpu16.log" \
    2> "$stage/original_brian2_scale2560_cpu16.time.log"
project16="$stage/cpp16/brian_benchmark_explicit_standalone_25600918_16"
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
    else
        replay_project="$project"
        replay_binary="$project/thread_binaries/main_t${threads}"
    fi
    "$python_bin" "$stage/tools/replay_cpu_pair.py" \
        --cpp-project "$replay_project" \
        --cpp-binary "$replay_binary" \
        --rust-artifact "$stage/rust/artifact" \
        --output-dir "$stage/replays_cpu${threads}" \
        --repeats 5 --threads "$threads" --cpu-list "$cpu_list" \
        > "$stage/replays_cpu${threads}.driver.log" 2>&1
done

date -u +completed_utc=%Y-%m-%dT%H:%M:%SZ > "$stage/completed.txt"
