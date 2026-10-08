# FlyWire integration review — 2026-09-06

Reviewed `codex/flywire-benchmark` at `4031b077` against the development branch
`gate0/minimal-rust-backend` at `455ad242` (frozen B2IR v1). The integration keeps
the v1 protocol, typed arrays, expression refractory, named event streams,
linked-variable declarations and native Function declarations. It does not
restore the benchmark branch's obsolete probe-v25 schema.

## Findings addressed

1. **External payload integrity:** the independent Rust CSR loader previously
   checked dimensions, offsets, targets and finite values, but not the declared
   SHA-256. A valid numeric edit could therefore escape integrity validation.
   The loader now verifies the digest using bounded buffers on the same file
   handle before validation/materialization. A regression changes a finite
   weight and verifies rejection by the independent runner.
2. **Typed-instance integration:** the old benchmark writer encoded every array
   as float64. The merged streaming writer preserves the main branch's
   float32/float64/integer/boolean byte encodings. External CSR columns are
   explicitly restricted to float64 constant per-edge parameters, in both the
   Python exporter and Rust validator; a float32 mapping fails clearly.
3. **Code generation and protocol conflicts:** linked-index constants, native C
   ABI declarations and v1 envelope verification remain intact while large
   external arrays use the streaming native Reader. Original v1 canonical
   vectors and migrations continue to pass unchanged.

The remaining scientific/performance limitations are documented in
[FLYWIRE_EI_RESULTS.md](FLYWIRE_EI_RESULTS.md). Its three-host performance numbers
are historical measurements of their recorded build commits, not new performance
claims for the merged compiler.

## Validation of committed integration

- Full Python suite: **189 passed, 121 subtests passed**.
- Rust release build and `cargo test --release --locked --offline`: passed.
- Full graph (139,255 neurons, 15,091,983 weighted edges), EI sensory condition,
  1,000 ms: **240,860 biological spikes**. Final states, spike IDs/times/counts,
  refractory states and recorded voltage/conductance trajectories compare with
  zero error against the previously recorded original Brian2 C++ output.
- Replays of the new full-graph artifact at 1, 4 and 8 threads: zero error against
  its own initial output. This is an integration check, not a timing baseline.
- Logs, full-graph build metadata and replay measurements are retained in
  `output/flywire-merge-validation/` in the primary working directory.

Existing uncommitted Brunel/exact-indegree development is kept outside this
merge commit. Its combination with the merge passed **193 tests and 121
subtests** in an independent worktree. Five overlapping files required explicit
conflict resolution; the prepared overlay preserves all 16 modified files and
both existing untracked files. Native replay RSS peaked at 283.8 MiB in the
integration check. No remote push or deployment is part of this merge.
