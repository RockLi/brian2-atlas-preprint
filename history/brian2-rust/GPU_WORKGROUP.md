# Explicit bounded single-workgroup execution

Use `executor.run(dag_execution="workgroup")` on `MetalExecutor` or
`CudaExecutor`, or select `metal_dag_execution="workgroup"` /
`cuda_dag_execution="workgroup"` with `set_device("rust_standalone", ...)`.
GPU numeric mode remains explicitly `float32`. `auto` never selects this mode.
A model must have an explicit DAG. Both portable Function bodies and native
scalar Metal/CUDA Functions are supported. Native declarations are preserved
verbatim inside each private stage namespace, before their callers; workgroup
source transforms apply only to generated code. Independent-population fused
plans remain rejected.

This experimental physical program runs the original ordered stage schedule
inside exactly one 128-thread workgroup/block. Each thread visits strided lanes;
all threads meet at a device-memory barrier after every stage. The host still
uses canonical f64 clock coalescence to construct the stage/tick table. This is
not grid-wide synchronization: launching more blocks would violate the design.
The original arithmetic, edge order, stage order and explicit f32 precision are
preserved. It does not repair f32/f64 threshold disagreements.

The runtime packs typed slots at 16-byte-aligned offsets, uploads fresh state,
executes one dispatch, reads back the arena and fault flag, and unpacks writable
slots. Numeric faults still prevent result publication. It retains the compiled
program within one executor, but retains no GPU data. Maximums are 64 stages,
8,192 logical stage launches, 16,000,000 lane visits and 16 MiB total buffers,
also subject to the caller's smaller memory budget. These work limits are not a
hard runtime deadline. Metal checks the compiled pipeline's thread capacity.

`workgroup-plan.json` binds the ordinary plan SHA, emitted program SHA,
schedule SHA, slot offsets/sizes, resource bounds and single-group geometry.
Runtime metadata records packing/unpacking, program compilation/replay,
allocated bytes, logical stages and the one physical dispatch. GPU timing alone
excludes host packing/unpacking; use complete replay wall time for comparisons.

The ordinary stage pipelines still compile during executor construction. The
workgroup compiles lazily on its first run. `gpu_compile_reuse=True` only reuses
the ordinary stage programs across Device activations; each new executor still
compiles its workgroup. `gpu_buffer_reuse=True` does not retain workgroup data.
Store/restore and continuation reconstruct fresh inputs as usual. Entering
workgroup mode releases ordinary resident allocations. Compiled workgroups
remain reusable until explicit close or an exception inside packed execution.
A nonfinite result rejected during later numeric validation may retain immutable
compiled code until close; it does not retain writable GPU data.

The purpose is to measure the launch-overhead/parallelism tradeoff. A single
block cannot exploit all GPU multiprocessors. In two isolated M3 rounds the
64-neuron workload improved, while 512/1,024-neuron cases regressed substantially.
Do not enable it by default or infer a general GPU speedup. See the full paired
measurements, correctness checks and NVIDIA results in
[workgroup evidence](execution-plan-evidence/gpu-workgroup/README.md).

Synchronization references: [CUDA block synchronization](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)
and [Metal device-memory barriers](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon).

Native Function workgroup coverage, exact typed replay, nonfinite-result rejection
and L4/A100 retained-call recovery are documented in
[native workgroup evidence](execution-plan-evidence/gpu-native-workgroup/README.md).
