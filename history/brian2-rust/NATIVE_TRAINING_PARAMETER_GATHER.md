# Runtime canonical parameter-bank reads

Dynamic training now reads a canonical optimizer bank through a runtime int32
selector. Values remain in the optimizer bank, so the next training step sees
updated parameters while carried neuron/synapse state stays intact. CPU, Metal
and the shared CUDA source implement this operation. NVIDIA execution remains
unverified. The original local acceptance is recorded in
`mpi-evidence/training-parameter-gather-final-20261004/README.md`; later source
versions and their separate evidence are described below.

## Native and expression contract

The new SSA nodes are `parameter_gather` and `integer_parameter_gather`, each
with `bank` and `index`. `index` names an earlier integer SSA node, not a literal
bank offset. The existing fixed `mapped_parameter` nodes keep their meaning.
An evaluated selector must lie inside its bank. Skipped lazy branches and
inactive detached events do not read it. Required invalid reads fail atomically.

`ParameterBank(bank, dtype="float")` from `training_equations` binds a callable
expression name. For example, a typed dynamic transform can use
`v = .8*v + gain(route(pick))`, with `gain=ParameterBank(0)` and
`route=ParameterBank(1, "integer")`. Integer banks must contain entirely declared,
frozen int32 slots. Boolean reads apply the existing detached Boolean cast.
Float gathers cannot reinterpret integer storage. Bounds and storage types are
validated before use, including negative selectors and mixed banks.

Reverse mode reconstructs the selector from the saved pre-action context and
fixed request parameters, then adds its adjoint to the selected canonical slot.
Repeated selections accumulate; other slots receive no contribution. Integer
routing has zero adjoint. Existing masks, optimizer transactions, fixed counter
noise, full BPTT and TBPTT boundaries retain their contracts. Results report the
detached-index-routing gradient scope even without indirect state addressing.

GPU opcodes 48 and 49 encode the selector node, exact bank length and flattened
bank offset. Integer payloads retain their bits, including values outside float32
integer precision and bit patterns representing NaNs as floats. The device
checks the decoded int32 before accessing the bank. Forward and reverse execute
on the owning GPU lane in local MPI. Native loaders require v5r6, so old v5r5
libraries fail explicitly. Existing metadata/tape bounds include the new nodes;
the saved context already contains the required selectors.

## Automatic Brian conversion

Canonical bank ownership is retained across mutable parameter links, default
index aliases and explicit `Variables.add_reference` references to selected
storage. The base neuron conversion leaves private deferred references until
the dynamic phase has allocated all physical selector storage. The dynamic phase
resolves those references in neuron integration, resets, thresholds and synaptic
event paths. No unresolved node is serialized into executable native programs.

Parameter selectors use separate context reads of physical state before user
statements. Therefore `pick = ...; v += peer` uses Brian's previously loaded
`peer`, even though later state writeback can use the updated `pick`. This timing
is verified against compiled Cython, including reset and synaptic event code.

`runtime_parameter_layout` records canonical bank ownership and selector address
descriptors. For runtime descriptors, intermediate tables address integer state;
the final table contains bank offsets. Fixed portions can use frozen integer
routing banks. Such banks are detached metadata, never copies of trainable
parameter values. External mutable selector sources are rejected.

The v5 action programs contain actual execution. Any retained v4 rectangular
shape program that needed action-specific extra selector slots becomes a typed
identity placeholder, recorded in `v4_shape_placeholders`. These placeholders
are not used to execute the dynamic model. Threshold margin provenance retains
the pre-resolution expression program; `runtime_parameter_layout` describes
the deferred references used to construct actual action programs.

## Evidence and remaining boundary

The pre-nested-index-repair frozen acceptance completed with **507 passed / 183
NVIDIA-only skipped / 0 failed**, 690 unique cases across 12 full modules and
221 source/dependency files. Both orchestration processes exited 0; the exact
source/runtime and disjoint results are verified in
`mpi-evidence/training-parameter-gather-final-20261004/combined-verification.json`.

The native tests use independent recurrences and finite differences for all
floating initial states and selected/unselected parameter slots, fixed noise,
full/TBPTT, repeated/nested integer gathers, precise int32 payloads, lazy bounds,
batch-specific selectors, masks, optimizer updates, carry/checkpoint and non-root
2/8-process rollback. Frontend tests additionally compile real Brian Cython
Euler/RK4 trajectories, compare typed payloads and reject external mutable state.
Continuation is checked against an independent recurrence using each step's
current optimizer weights, clock and noise sequence.

The original multi-level explicit reference attempt exposed missing index
variables and unsafe Cython read order, including a SIGBUS reference run. Those
failures remain diagnostic evidence. A subsequent repair now collects transitive
indices, loads them in dependency order across Cython/C++/NumPy, rejects read
cycles and preserves template-owned self-index metadata. Explicit two-level
references through fixed integer banks or selected mutable integer state have
passed final CPU/Metal and existing-frontend regression: 133 passed, 54 NVIDIA
skipped, 0 failed across four complete modules, with 233 frozen files and parent
exit 0. Separate existing Brian NumPy/Cython subsets each passed 47 cases. See
`mpi-evidence/training-nested-index-final-20261004/`. This is a separate source
version from the earlier parameter-gather acceptance. The high-level Brian
`linked_var(index=...)` constructor restriction remains unchanged; arbitrary
reference graphs are not claimed as verified.

Subsequent v5r7 work implements state-dependent refractory rules and validates
joint neuron/synapse SDEs, plasticity, summed input and mutable delay; see
`NATIVE_TRAINING_STATE_REFRACTORY.md` and `NATIVE_TRAINING_STOCHASTIC_COMBINATIONS.md`.
Nested synaptic selectors with shared mutable routes and nonidentity bank maps
are described in `NATIVE_TRAINING_NESTED_SYNAPSE.md`, including the delayed-event
context-order repair and a separate final acceptance record. These later records
must not be pooled with the older source version's test totals.

Actual contiguous Subgroup endpoints on selected parents are now implemented;
see `NATIVE_TRAINING_SUBGROUP.md` for parent storage, relative coordinates,
synaptic-only constant banks, and the separate final acceptance record.
Arbitrary asynchronous continuous integration, external callbacks,
int64/unsigned and bitwise expressions remain unfinished.
No cloud job or cross-host validation was performed in these local phases.
