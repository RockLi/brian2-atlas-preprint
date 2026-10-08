# Lane-local delayed pathway history fusion

The default Metal/CUDA DAG planner now folds eligible mutable pathway-history
copies into a preceding population kernel. This extends the existing immutable
sparse-enqueue fusion pass. No execution mode or precision option changes.

The proof requires the same clock, population and full population range; only
ordinary lane-local population producers (including already fused history or
sparse-enqueue producers) qualify. An intervening dispatch may be crossed only
when whole-buffer read/write sets commute. Mutable bindings count as both reads
and writes. Different clocks, subgroup ranges, linked canonical population
kernels, type aliases and the 30-data-binding limit block fusion. The history
copy reads this lane's event flag and writes this lane's ring slot. Event offsets
remain unchanged for named events. Every remaining dispatch retains its global
stage barrier and canonical order.

The transformation keeps logical node identities, typed buffers, arithmetic,
event order, delay tick values and pending restoration. Physical plan/kernel
hashes change where fusion applies, naturally invalidating predecessor compiled
programs and graphs. Native-only function injection remains after planning.

For the delayed recurrent STDP fixture, six dispatches become four per tick:
population plus pre/post history copies, target-owned pre delivery, edge-owned
post delivery, and reset. The two history stages commute with intervening
read-only access to event flags. Population and edge kernels still launch enough
workgroups to cover their full domain. No grid-wide in-kernel synchronization is
introduced. This differs from the explicit single-workgroup experiment.

Tests compare complete results with history fusion disabled and repeat execution,
including CPU-f32 workers, real Metal/CUDA, early pathways, named events,
subgroups, source-state hazards, empty projections and existing delay/plasticity
regressions. The pass's `pathway_history=False` keyword is an internal audit and
benchmark switch, not a new public Device setting. See
[history-fusion evidence](execution-plan-evidence/gpu-history/README.md) for
matched replay timings and limitations.
