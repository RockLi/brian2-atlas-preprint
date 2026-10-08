# Requested-leaf activity in native VJPs

Native CPU reverse and the scalar/vector/dynamic Metal interpreters now prune derivative
paths that cannot reach a requested continuous leaf. This fixes models such as
`k=poisson(scale+sqrt(r)); v=amp*k` with scale=r=0 and r detached: forward is
finite, the active rate derivative through scale is one, and the irrelevant
sqrt derivative must not poison the boundary gradient. Positive-rate likelihood
scores and ordinary pathwise VJPs use the same rule.

Activity follows actual visited branches, selected min/max operands and the
existing surrogate contract. Parameter masks apply to canonical bank slots,
including mapped, gathered, neuron and timed parameters. Dynamic state leaves
use recorded physical addresses and detach flags, including indirect reads;
placeholder addresses cannot determine activity. Optimizer `trainable=False`
still preserves requested parameter gradients. No epsilon or replacement finite
derivative is introduced. Active singular derivatives remain explicit errors.
Forward domain checks are unchanged; pruning cannot make an invalid forward
expression admissible.

The CPU helper uses fixed visited/activity arrays rather than per-node JSON
allocation. Dynamic CPU admission reserves an additional 256 bytes of serial
VJP scratch. The GPU reuses its existing private activity arrays within the
16 KiB/lane allowance. Both unit rate probes and ordinary dynamic VJPs share
that activity calculation. Dynamic GPU gradient/training calls require
`b2_train_vjp_activity_v1 == 1`; this capability is scoped to the v5 dynamic
interpreter. Evaluation layouts and the v5r9 entry point are retained.

Legacy scalar/vector SSA gradient/training calls separately require
`b2_train_static_vjp_activity_v1 == 1`. The host sets bit 2 of legacy operation
word m[9] and appends canonical masks after constants, clock and noise staging;
legacy mask base is m[13]-m[5]. Evaluation and built-in/no-SSA reverse keep their
old capability contract. Static GPU admission reserves 16 KiB per lane of
interpreter scratch plus host/device mask copies.

The shared VJP helper distinguishes mode 1 (true v5 physical addresses/detach,
v5 masks) from mode 2 (legacy v4 local-state leaves, appended masks). Legacy
standard math and lazy expressions that delegate to the dynamic evaluator use
mode 2; ordinary v5 pathwise and likelihood-score reverse explicitly use mode 1.
The direct scalar/vector interpreters use the same canonical mask rule.

The formerly failing scalar/vector GPU `v+scale+sqrt(frozen)` counterexamples
now return finite gradients on actual Metal and agree with native CPU. The
historical failing records remain in the previous phase; the current successful
probe is in `mpi-evidence/training-static-vjp-activity-20261005/`.
CUDA shares the shader implementation but lacks NVIDIA runtime acceptance.

Use the isolated runner recorded in
[the current activity evidence directory](mpi-evidence/training-static-vjp-activity-20261005/README.md).
Tests compare equivalent finite constant-branch models across positive/zero-rate
Poisson, pathwise updates, normal/uniform noise amplitudes and physical indirect
reads; active singular cases remain rollback controls. Complete affected modules,
source hashes and process terminals are recorded there. The whole stochastic
and dynamic-synapse goal remains active.
