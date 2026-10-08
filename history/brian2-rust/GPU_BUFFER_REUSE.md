# GPU allocations between Device runs

Metal and CUDA can now transfer compatible resident allocations from one
completed Device activation to the next. This is opt-in:

```python
import brian2 as b
import brian2_rust

b.set_device("rust_standalone", engine="metal", numeric_mode="float32",
             event_delivery="sparse", gpu_buffer_reuse=True)
# Construct groups, synapses, monitors and Network here.
net.run(10 * b.ms)
net.run(10 * b.ms)
b.get_device().close_gpu()
```

Use `engine="cuda"` for NVIDIA. The existing `cuda_dag_execution` choice still
applies; explicit `direct` does not retain allocations. Metal selects bounded
`auto` buffer retention when this option is enabled. Default
`gpu_buffer_reuse=False` keeps the existing per-activation lifetime.

Every activation still exports and validates the current model, builds its own
plan. Kernel compilation can separately opt into [compilation reuse](GPU_COMPILATION_REUSE.md). Current host state and reconstructed pending
events are authoritative. This is allocation reuse, not GPU-resident continuation
state or a persistent kernel; this buffer option itself does not reuse kernels. Changes to clocks, code, parameters
and buffers cannot silently reuse the previous execution plan.

Only same-backend, same-device buffers transfer. Each slot is independently
matched by index, shape, dtype and strides. The retained total is bounded by the existing 512 MiB default
execution budget for GPU data buffers; host snapshots and pipelines are separate.
Incompatible slots use fresh allocations; removed slots are released. A changing
pending-event array no longer prevents reuse of fixed topology arrays. Old
unmatched allocations are released before growing the new set. CUDA's global
memory pool and Graph/chunk auxiliary storage have separate accounting.

All buffers written by the new plan are refreshed. Changed read-only input is
also uploaded, including bitwise differences such as signed zero. A buffer that
was writable in the previous plan is refreshed even if it becomes read-only and
the old and new initial snapshots match. Prior GPU mutation could otherwise
leave stale data. Unchanged, exclusively read-only arrays avoid another upload.
Returned host results and monitors are still fully materialized on every run.

Metal moves sole ownership of the compatible completed allocations between native
handles. CUDA synchronizes the old stream, discards Graphs and chunk captures,
moves buffers to the new stream and rebuilds captures with the new clock schedule.
Both use the freshly validated bindings and kernels.

`metal_runtime.activation_buffer_reuse` or
`cuda_runtime.activation_buffer_reuse` in each result summary records the
requested option and whether allocations were adopted. Lower-level runtime
counters record actual reuse, transfer bytes, newly allocated data bytes,
reused buffer count and reused data bytes on both backends.
`close_gpu()` releases the Device's retained executor while
keeping host state, clocks and monitors available for later continuation.
Repeated calls are safe. Reinitializing or activating the Device also releases
retained resources. Construction, execution or result-publication errors release
both the old and new executor owned by this transition.

The recurrent diagnostic now adopts buffers on every measured continuation,
retaining fixed arrays while replacing variable pending arrays. All result
comparisons pass. Two M3 rounds confirm the allocation/transfer reductions but
show different wall-time improvements; no general speedup is claimed. The prior
[all-or-nothing experiment](execution-plan-evidence/gpu-device-buffers/README.md)
and the new [per-buffer evidence](execution-plan-evidence/gpu-partial-buffers/README.md)
preserve both findings. Preparation, kernel compilation and GPU-resident
continuation state remain separate work.
