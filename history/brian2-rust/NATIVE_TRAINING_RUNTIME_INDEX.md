# Runtime linked indices and conditional writes

The native action engine implements runtime state addressing on CPU, Metal
and local MPI. CUDA shares the translated device implementation, with NVCC and
NVIDIA hardware acceptance still pending. The Brian dynamic frontend now lowers
mutable links to selected physical state for neuron integration, reset, threshold
preparation and synaptic transforms. Runtime optimizer-bank gathers are described
in `NATIVE_TRAINING_PARAMETER_GATHER.md`; complete shared/summed/delay/stochastic-integrator
combinations remain unfinished. Scope and local
acceptance for the frontend are recorded below.

An optional `action.indirect` object contains `reads` and `writes` maps. Map keys
are context/output slot numbers, serialized as JSON object keys. Omitted entries
keep the existing fixed addresses; the original `reads`/`writes` cells also
specify the required storage type.

A read entry contains an integer state cell `index` and one or more `tables` of
physical state indices. The first table is indexed by the cell's current value.
For nested lookup, the selected intermediate cell must itself contain an int32
index into the next table. The last table resolves the physical read address.
All selected final cells must have the placeholder's floating/int32/Boolean
storage kind. Each action snapshots its logical reads before computing outputs.

A write entry contains the same tables and an index source:

- `{"kind":"read","slot":2}` uses a snapshotted integer context value.
- `{"kind":"output","slot":2}` uses the action's computed integer output.

All values and write destinations are resolved before stores are applied. Stores
follow the action's output order; when destinations collide, the last output wins.
This allows a Brian reset to read a linked variable through its old index,
change the index locally, and write the computed value through the new index.
The frontend emits Brian logical writeback order and retains integer locals
whose computed values select another output, even when their own stored value
is subsequently overwritten by a fixed alias.

The native tape retains actual read/write addresses and old destination values.
Reverse mode differentiates only surviving writes and scatters read adjoints to
their actual physical cells. Index computation and integer choices have zero
adjoint. Results identify the convention as `full-bptt-detached-index-routing`
or `tbptt-detach-boundaries-and-index-routing`; combined delay/index plans state
both detached decisions. Full/TBPTT and fixed counter-noise semantics are retained.

Negative, out-of-range, mistyped or empty lookup tables fail before memory access.
Masked or inactive detached events do not access invalid runtime indices. A
surrogate derivative through an inactive event requires a defined counterfactual
transform; an invalid counterfactual address with an active adjoint fails explicitly.
Any error leaves the trainer's committed state, optimizer, clock and RNG unchanged.

Migration ownership uses all possible destinations. Managed delay queue cells
cannot be accessed through index metadata, and delay rebuilding includes cloned
address tables in its admission budget. Index table depth is bounded to 16, each
table to one million entries, with the existing action/context/tape budgets.

`tests/test_training_indirect.py` contains an independent sequential recurrence,
finite differences for every floating initial cell and parameter, fixed-noise
full/TBPTT checks, nested and batch-specific indices, last-writer collisions,
checkpoint/carry, bounds/type/budget errors, non-root MPI rollback, and a real
compiled Brian Cython trace. Final evidence is recorded separately after the
frozen regression completes.

Runtime optimizer-bank reads now use separate typed gathers; see
`NATIVE_TRAINING_PARAMETER_GATHER.md` for the implemented and verified boundary.
Remaining work includes complete combinations with linked/shared/summed/refractory/delay
state. Mutable links to refractory
state now support read-only paths and conditional writes, including the shared
condition-name/index rules described below. Cross-host work is deferred.


## Historical CPU core acceptance

The frozen main run plus a disjoint candidate-ownership extension passed
**327 tests / 106 NVIDIA-only skipped / 0 failed**, with 433 unique identities
across 10 modules and 198 source/dependency hashes. The two focused new modules
contribute 44 passes and one CUDA skip. Evidence and the aggregate verifier are
in `mpi-evidence/training-runtime-index-core-final-20261003/README.md`.
No implementation or executable changed between these two runs. Existing Metal
and local GPU/MPI regressions passed. That historical runtime explicitly rejected
new indexed GPU actions; the v5r5 implementation described below replaces that
restriction. Automatic Brian conversion remains to be implemented.


## Device addressing (v5r5)

Metal and CUDA kernels now gather runtime reads, resolve destinations before
stores, retain old target values, and scatter VJPs through taped addresses.
Colliding stores use the final writer. Integer/Boolean cells keep exact typed
payloads, while index selection has zero adjoint. Inactive detached actions do
not require valid counterfactual addresses; required undefined surrogate
counterfactuals fail atomically.

The owner computes indexed actions on device in MPI mode. It distributes both
output values and selected destinations; replicated apply phases save the same
old values and reverse through the same taped addresses. Maximum action width
uses 128 forward payload entries (64 values plus 64 addresses) and 129 reverse
entries, with a separate error slot. Address metadata and the larger tape are
included in admission bounds. Old v5r4 libraries fail the v5r5 symbol check.

`tests/test_training_indirect_gpu.py` covers fixed-noise finite differences,
full/TBPTT, nested and batch-dependent indexing, carry/checkpoint, collisions,
int32/Boolean payloads, candidate-mask migration, bounds/rollback, tape budgets,
and 64-slot collectives. GPU MPI tests use 2/8 processes on one local Metal GPU;
they do not establish multiple physical GPUs or cross-host execution.


### Frozen v5r5 acceptance

The 11-module local regression passed **407 tests / 230 NVIDIA-only skipped /
0 failed**, with 637 unique identities and 199 source/dependency hashes. The new
GPU addressing module contributes 51 passes and 50 CUDA skips; actual Metal,
local 2/8-process MPI and independent numerical VJPs are included. Full evidence:
`mpi-evidence/training-runtime-index-gpu-final-20261003/README.md`.
The native runtime hash is
`e982a79af601199b1913520bbf388b249f0f2f1a6a322e93d1511ebe5902d903`.
CUDA source translation passed; NVCC and NVIDIA execution remain pending.


## Automatic Brian mutable-state links

`lower_brian_dynamic_training` now retains mutable int32 selectors as physical
state and attaches native addressing metadata. Source allocation finishes before
links are resolved, so object naming does not determine canonical ownership.
Constant portions of index mappings are cached; runtime portions remain address
tables. Provenance and action-table sizes are included in frontend admission.

For neuron integration/reset, logical context slots remain distinct. Only indices
actually needed by selected output expressions or write destinations are resolved.
A constant reset can therefore replace an invalid old selector before writing a
constant through its new value, without accessing an unused old linked cell.
Threshold expressions first compute into a fixed scratch cell through an ordinary
indexed action, preserving the existing threshold preparation/owner invariant.

Synaptic event transforms preserve Brian's sorted logical writeback order.
With runtime addressing, fixed aliases cannot be discarded before index outputs
are resolved: an overwritten integer local can still choose a floating target.
The native indexed evaluator handles both fixed and dynamic collisions using its
last-writer tape. Fixed-only actions continue to require unique write targets.
Migration registration enumerates all possible destinations, and read-only aliases
retain the lifetime of all candidate source cells.

For a mutable linked name, `neuron_state_layout` or `dynamic_state_layout` contains
typed inert placeholders, not the currently selected values. The corresponding
`provenance.runtime_index_layout[object][name]` rows describe the actual view:
start with the integer value at `index`, select an entry in each `tables` row,
read the next integer for intermediate rows, and read the final physical state
cell after the last row. `root_name` is lowering provenance for the logical index
local. Existing fixed-link layout entries still name canonical physical cells.
`inactive_alias_slots` identifies unused placeholders; their adjoints are zero.

`tests/test_training_indirect_frontend.py` constructs real Brian networks and
converts them automatically. It checks Euler/RK2/RK4 trajectories, threshold
reads, overwritten integer aliases, fixed-noise full/TBPTT finite differences for
every float initial cell and parameter, batch/carry/checkpoint, non-root rollback,
and synaptic index mutation/write collisions. This is distinct from earlier
handwritten native IR tests. The latest frozen frontend evidence belongs in
`mpi-evidence/training-runtime-index-frontend-final-sharded-20261003/`.


### Frozen frontend acceptance

The final frontend regression passed **394 tests / 127 NVIDIA-only skipped /
0 failed** across 9 modules, with 521 unique identities and 200 frozen source
hashes. The new frontend module contributes 64 passes and 25 skips. Native hash:
`1a33608cf36640b4ae53f50c45023744f397d1b02c852bcce6f7a15a19c647c8`.

The initial run hit disk exhaustion after completing the new module. That exact
module was retained; the remaining full collections ran in small processes to
release compilation caches. The verifier checks retained bytes, complete shard
identities/results, source hashes and successful terminal states. Full record:
`mpi-evidence/training-runtime-index-frontend-final-sharded-20261003/README.md`.
No NVIDIA or cross-host acceptance was added.


## Refractory conditions through runtime links

The converter resolves Brian's conditional variables by name, in sorted identifier
order. When several referenced variables share `not_refractory`, the condition
array follows Brian's name-resolution rule and its index follows the last
referenced conditional variable, including read-only variables. It snapshots
the flag before statements, while final sorted writeback can use a modified
integer local. An explicit assignment to the same condition name updates its
logical Boolean value for subsequent guarded statements in that event; distinct
flag aliases retain their separate Cython locals. Flags introduced only by code
generation are bound to physical storage even when absent from `Synapses.variables`.
Brian's scalar/vector admission rules still apply to these assignments. The transform compiler allows
Boolean guard mutation only through explicit opt-in. A failed guard therefore
does not necessarily suppress physical
writeback: the old logical value can be copied to a newly selected destination.

The same index rule applies to the neuron updater's activity-flag output and
threshold's activity-flag read. The updater computes its fresh flag from its own
refractory counter, then stores through the resolved index. The threshold reads
the resolved flag but clears its own physical flag on a spike, matching Brian's
Cython template. Indexed threshold flags use a detached Boolean preparation cell.
Flag arrays are initialized from the real Brian snapshot, including entries
left untouched by duplicate index maps. Standard fixed refractory timing and
stop-gradient Boolean/index decisions remain unchanged.

Provenance exposes `refractory_activity_layout`, `refractory_condition_indices`
and `conditional_write_semantics`. The new regression module is
`tests/test_training_runtime_refractory.py` and
`tests/test_training_implicit_refractory.py`: actual Cython trajectories and flag
arrays, independent Euler/conditional-write recurrences with every float initial
state and parameter checked by finite differences, plus device/MPI continuation,
mask migration, checkpoints, delayed paths and rollback.

The latest frozen acceptance passed **327 tests / 90 NVIDIA-only skipped /
0 failed**, with 417 unique identities across four full modules and 213 frozen
source/dependency/check-script hashes. All shards and the parent exited 0;
the strict verifier passed. Native SHA-256:
`69bd30927f5d993ab8d15a05955a5c55704e1e5c257dc837b5e8634dd1928c57`.
Evidence: `mpi-evidence/training-runtime-refractory-final-r3-20261004/README.md`.

The preceding ten-module r2 baseline passed 666 tests with 185 NVIDIA skips before
the implicit-condition binding fix. Its preserved sources and artifacts are
audited separately; its totals are not pooled into the latest acceptance.
No NVIDIA hardware acceptance was added. Earlier directories remain historical
snapshots; remaining local interfaces are listed in the latest evidence's `NEXT.md`.
