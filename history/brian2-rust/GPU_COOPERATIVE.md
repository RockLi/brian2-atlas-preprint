# CUDA cooperative DAG experiment

`cuda_dag_execution="cooperative"` is an explicit experimental strategy. The
default remains unchanged. It executes the canonical multi-clock DAG schedule
inside one CUDA kernel, with a cooperative grid barrier after each stage.

```python
set_device("rust_standalone", engine="cuda", numeric_mode="float32",
           event_delivery="sparse", cuda_dag_execution="cooperative")
```

Each stage retains the existing generated scalar lane function, buffer types,
ownership rules, integer ticks and floating-point compiler options. Threads
iterate over stage lanes using the global grid index and stride. Inactive lanes
and lanes that return early from a stage still reach the unconditional grid
barrier. Scalar faults remain checked before a result is published.

The runtime queries cooperative-launch support and the exact compiled kernel's
maximum resident blocks per multiprocessor. The initial policy caps the total
128-thread block count at the number of SMs, further limited by the largest
stage's lane count. This is a grid-size policy, not measured block placement. It launches with CuPy's cooperative driver path, which also checks the
occupancy limit. No software spin barrier or oversubscribed grid is used. See
[CuPy 13.6 launch implementation](https://github.com/cupy/cupy/blob/v13.6.0/cupy/cuda/function.pyx)
and [CUDA grid synchronization](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#grid-synchronization-cg).

The address table is constructed from this runtime's owned device buffers;
addresses are never accepted from an exported plan. Immutable arrays and schedule
tables remain resident between replays. Mutable arrays reset from the validated
snapshot. Existing append-only spike upload omissions and populated-prefix
readback remain in use, avoiding a full host arena pack/unpack. Switching to
another DAG strategy or closing the executor releases cooperative buffers and
its compiled module. Cross-activation cooperative residency is not implemented.

Admission is bounded to 64 stages, 65,536 stage executions and 2^31 lane visits,
plus the configured total working-buffer budget including metadata. These are
experimental admission bounds; they do not predict duration or prove that a
display GPU watchdog cannot interrupt execution. Runtime launch or schedule
errors fail the call rather than silently selecting another strategy.

`cooperative-plan.json` binds source and canonical schedule hashes to the base
physical plan, and records the compiled binary hash, actual block count, SM count
and occupancy limit. Per-run diagnostics report one kernel launch, the number of
logical stages/barriers, transfers, buffer reuse and timings. A single launch is
not itself evidence of a speedup: grid synchronization, occupancy and register
pressure can dominate execution.

The accompanying native tests exercise multi-block scan/sparse delayed mutable
synapses, reductions, random and TimedArray inputs, multiple clocks, typed
results/monitors, numerical faults, repeated reset and Device store/restore.
The benchmark prospectively declares quiet, low-activity and dense delayed-STDP
cases, with separate retained executors, one warmup and five randomized paired
rounds. Compilation is prohibited in measured replays. Every full state, trace,
weight and spike result must pass both the independent f32 reference and compiled
CPU f32 control; f64 compatibility is reported separately. This experiment does
not update external Brian2CUDA/GeNN rankings.

Both bounded Modal jobs have terminated. Native correctness passed, but all three
paired cases were slower than default chunked graphs. Keep this mode experimental;
see [results and audit](execution-plan-evidence/cooperative/README.md). Full accepted-model conformance and a
general optimal launch policy remain open.


## Explicit internal geometry experiments

After preparing an experimental cooperative executor, its runtime provides
`configure_grid(blocks)` for bounded experiments. The value must be a Python
integer in `1..(SM count * exact-kernel active blocks per SM)`; invalid requests
raise without changing the previous setting. Closed runtimes reject changes.
The run path validates geometry again and uses that immutable launch binding for
both dispatch and reporting, rather than relying on CuPy's implicit grid clamp.

Every replay records a `b2-cooperative-launch-v0` binding with its plan and cubin
hashes, block/thread counts, shared-memory size, SM count and occupancy bound.
Its canonical hash distinguishes different configurations of the same cubin.
The initial `cooperative-plan.json` remains the prepared default geometry;
per-replay launch bindings record tuning changes. Actual SM placement is not
instrumented, and a block count equal to the SM count does not prove one block
was placed on every SM.

The geometry sweep compares seven predeclared block counts with default chunked
execution, retaining one cooperative function and the same device addresses
across all configurations. Configuration and validation occur outside the
full reset-to-result interval; one warmup and five randomized measured rounds
follow bootstrap. Results are tuning evidence, not automatic policy promotion.

The completed L4/A100 sweep passes full f32 checks, but none of the seven tested
block counts beats default chunked graphs in the three cases. See the
[full geometry results and audit](execution-plan-evidence/geometry/README.md).
