# Metal DAG buffer reuse

`MetalExecutor` can retain its DAG data buffers between replays. Each replay
still starts from the validated initial snapshot. Mutable state, event queues,
monitors and scratch buffers are reset; immutable topology and input tables stay
on the GPU. Readback returns writable result and error-check buffers. Known
internal scratch (pathway delay rings/cursors, active-edge lists/counts, linked
gathers and floating last-spike scratch) stays on the GPU. All writable buffers
still reset, including omitted scratch. Unknown buffer kinds are conservatively
read back. The writable set is the union across the complete task graph.

```python
from brian2_rust.metal import MetalExecutor

with MetalExecutor(model, "metal-output", numeric_mode="float32",
                   event_delivery="sparse", dag_execution="auto") as executor:
    first = executor.run()
    replay = executor.run()
    print(replay["metal_runtime"])
    direct_control = executor.run(dag_execution="direct")
```

The default remains direct execution: paired timings show that reduced transfer
work does not guarantee lower end-to-end time for every workload. The executor
modes are:

- `auto`: retain the DAG buffers when their total size fits
  `max_buffer_bytes`; otherwise use direct execution.
- `resident`: require the total size to fit, or raise `MemoryError`.
- `direct` (default): allocate and upload all buffers for every replay, then read
  back result/error buffers. Switching to this mode releases retained storage.

`max_buffer_bytes` retains its existing per-buffer execution check. In addition,
it bounds the sum of retained GPU data buffers. It is not a process RSS limit:
host snapshots, result arrays, command buffers and driver overhead are separate,
and direct execution retains its previous per-buffer budget semantics.

`metal_runtime` records selected/requested mode, whether buffers were reused,
allocated/uploaded/read-back bytes, retained bytes, total buffer bytes and the
writable subset and readback buffer count. Allocation counters cover the planned `MTLBuffer` data objects;
command encoders and command buffers are still created for each bounded chunk.
The same report is included in `summary.json` when publishing Metal results.

Lowering the memory budget releases a cache that no longer fits. Native dispatch
errors and incompatible storage shapes also release it; the next valid replay
reinitializes GPU storage. `close()` releases retained buffers and pipelines and
is idempotent. Context-manager use provides cleanup on failures.

The dispatch sequence, scalar kernels and precision profile are unchanged.
Independent population kernels keep their existing execution path.
This change supports replays through one retained executor; it does not keep
state resident across separate Brian `Device.run` activations, implement a
persistent GPU kernel or remove host command encoding. CUDA's existing resident
and Graph policies are separate.

See [verification and paired timings](execution-plan-evidence/metal-buffers/README.md).

## DAG synchronization experiment

`MetalExecutor(..., dag_synchronization="explicit")` preserves the default
per-dispatch buffer barrier. `executor.run(dag_synchronization="tracked")`
selects automatic synchronization, using a serial compute encoder and directly
bound hazard-tracked buffers. The bridge checks both prerequisites before
dispatch. The constructor also accepts `"tracked"`; the run argument overrides
it for one replay. These choices are independent of buffer retention.

This relies on the existing `MTLCommandQueue` / `MTLCommandEncoder` API contract,
not Metal 4 synchronization, untracked heaps, indirect buffers or concurrent
encoders. See [Apple resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization)
and [serial dispatch](https://developer.apple.com/documentation/metal/mtldispatchtype/serial).

The `metal_runtime` report additionally records `synchronization`, `dispatch_type`,
`hazard_tracking`, `dispatches` and `explicit_barriers`. Both policies execute
the same kernel and clock schedule. Automatic tracking does not mean that the
driver performs no synchronization.

Two M3 paired runs did not establish a repeatable general speedup. Explicit
barriers therefore remain the default. The selectable path and counters support
workload-specific diagnosis; this is not a persistent-kernel optimization.
See [synchronization measurements](execution-plan-evidence/metal-synchronization/README.md).

For optional allocation reuse between separate Brian `Network.run()` calls,
see [Device GPU buffer reuse](GPU_BUFFER_REUSE.md). It transfers compatible
allocations to a freshly validated executor; it does not retain continuation
state exclusively on the GPU or reuse old kernels.

## Explicit indirect command replay

`MetalExecutor(..., dag_execution="indirect")`,
`executor.run(dag_execution="indirect")`, or Device option
`metal_dag_execution="indirect"` selects a bounded experimental indirect command
buffer. `auto` does not select it. An explicit DAG and explicit synchronization
are required; independent temporally fused plans and tracked-only synchronization
are rejected. CUDA's separate execution modes are unchanged.

The first run creates pipelines supporting indirect compute, a constant int64
tick buffer and one cached command per logical stage activation. It uses the
same host-f64 clock coalescence, stage order, source kernels, lane geometry and
64-coalesced-tick command chunks as the ordinary path. Every indirect command
has a barrier before it, including the first command of a chunk. Buffer resources
are explicitly declared to the encoder. This preserves stage completion before
the next command; it does not remove the GPU's synchronization work. See Apple's
[indirect compute command API](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand)
and [barrier contract](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand/setbarrier()).

Subsequent runs reset writable data and read back result/error buffers while reusing indirect commands
and compatible data buffers. Native kernels retain their float32 arithmetic;
original Function source is unchanged. The first indirect run includes lazy
pipeline/command preparation in command wall time, not in the executor's ordinary
`compile_seconds`. Treat it as cold setup when evaluating repeated execution.

Limits are 64 stages and 8,192 dispatches. The sum of planned retained data bytes,
the indirect buffer's reported `allocatedSize`, and the tick buffer's reported
`allocatedSize` must fit `max_buffer_bytes`. This is not a process or total driver
memory limit: pipeline objects and transient host/driver storage are outside it.
`indirect_bytes`, `indirect_commands_encoded` and `indirect_commands_reused`
expose command-cache behavior; `resident_bytes` includes indirect/tick storage,
while `total_buffer_bytes` retains its existing data-buffer-only meaning.

Switching to direct/resident execution discards the command cache. Closing,
reducing the budget below the retained footprint, and dispatch failures release
retained commands and data. Device buffer adoption transfers matching data
allocations only; the new executor builds its own commands and pipelines. A
nonfinite result detected after GPU execution prevents publication; immutable
commands may remain cached until close, with writable data reset on every replay.

See [indirect replay evidence](execution-plan-evidence/metal-indirect/README.md)
for exact-result tests, cold/warm measurements and current limitations.
