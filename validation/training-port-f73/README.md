# Training increment from `f73dcb20c`

This directory records the incremental training port into Atlas `dev`. Product
code, tests and user-facing training documentation live in `brian2-atlas`.
Development history and acceptance records live here. Release tags and frozen
paper measurements are unchanged.

## Completed acceptance

All final gates passed for Atlas commit `63b11d7421cb9d5db342784a0d95fa2bdd8b3ba1`.

| Gate | Passed | Scope / exclusions |
| --- | ---: | --- |
| Rust library | 51 | Native unit tests |
| CPU increment | 1,214 | 2,065 conditional hardware/MPI/compiled-target skips, individually recorded |
| Metal | 323 | Representative function/backend/rank selections |
| Metal/MPI supplement | 4 | Includes fresh-process checkpoint restore and failure atomicity |
| CPU MPI | 176 | 97 parameterized/module cases plus 79 fixed-two-rank function cases |
| Modal NVIDIA L4 | 333 | CUDA, same-host MPI and C ABI checks; zero failures/skips |
| Installed platform wheel | passed | CPU/AOT simulation, CPU/Metal training, bundled executables and fresh-process restore |
| Public Modal image | passed | Actual editable Atlas installation and native executable resolution |

The final L4 upload contains 923 files verified against the frozen Atlas commit.
The wheel's 20 changed Python/runtime resources match the source bytes. There is
no claim of a full device parameter Cartesian product, cross-host acceptance,
multiple physical GPUs, or a new measurement of paper benchmarks.

Initial local runs were interrupted before completion and are not counted.
One subsequent MPI case failed because the host could not execute the native
binary from the external volume; its isolated retry and both complete MPI gates
passed using the identical binary content, with the final MPI gates executing
from the local temporary filesystem. No numerical implementation repair was
needed. The failed combined receipt remains in `results/mpi-resumed/`.
Cloud setup attempts exposed missing upload dependencies and a copy-layer
ordering error; these were corrected before accepted numerical execution.
An uncached CUDA run was stopped to avoid recompiling the same library in every
module. The final run retained the full selected test list. Attempt records are
kept alongside the completed result and are not counted as successful gates.
Public execution logs redact local paths and normalize trailing whitespace;
original and public SHA-256 values are both retained.

## Source scope

[Comparison](comparison.json) classifies all 474 paths from legacy commit
`f73dcb20ca8c18b0b9f0427f119e33e67873742b` against Atlas base
`ae649a244e28d7d4b3eb5df4e35215e14c1378c7`. 262 files were already present
(including two with equivalent Python ASTs); 106 product paths were ported;
12 Atlas adaptations were retained. The port preserves the public Atlas API,
AtlasIR naming, wheel/source runtime lookup, licensing, upstream parser tests
and summed-endpoint cache optimization. Two mixed files were merged by hunk.
Uncommitted development after the selected legacy commit is excluded.

[Port file mapping](port-files.json) retains original Git blob identities and
initial port hashes. Current acceptance binds the tested product files to their
actual hashes; it does not reuse historical executions as current passes.
The Modal source-image helper additionally includes Atlas build metadata,
backend packages and licensing inputs needed by Atlas installation.

[Historical import](history-import.json) maps seven changed/new documents and
51 evidence/control files into `history/brian2-rust/`. Public copies redact
local user and volume paths and retain both original and public SHA-256 values.
Their old audit programs depend on the separately retained original local
archive; they are historical records, not portable current test entrypoints.

## Reproduce current acceptance

Use an Atlas checkout containing this port and a Python environment with its
Cython extensions and test dependencies built, including SciPy for the independent Poisson oracle. Select both binaries from that
checkout. Output directories must be outside tracked source:

```sh
cargo +1.98.1 build --release --locked --manifest-path "$ATLAS/brian2-rust/Cargo.toml"
export B2_RUNNER="$ATLAS/brian2-rust/target/release/b2-runner"
export B2_TRAIN_RUNNER="$ATLAS/brian2-rust/target/release/b2-train"
python validation/training-port-f73/run_gate.py --atlas "$ATLAS" --mode cpu --output "$RESULTS/cpu"
python validation/training-port-f73/run_gate.py --atlas "$ATLAS" --mode metal --output "$RESULTS/metal"
python validation/training-port-f73/run_gate.py --atlas "$ATLAS" --mode mpi --output "$RESULTS/mpi"
```

CPU runs all non-GPU parameterizations in the listed modules; MPI-disabled cases
are reported as skips. GPU and MPI gates are bounded representative selections
by function, backend and rank count, up to two ranks. Their selected-test lists
are the exact scope, not a full GPU Cartesian-product claim. Metal requires a
real Apple GPU; MPI requires `mpicc`/`mpiexec`. Hardware skips are not passes.

The explicit cloud command uploads the selected source tree and uses a single
NVIDIA L4. It can incur cloud charges and does not deploy a persistent service:

```sh
python validation/training-port-f73/modal_gate.py --atlas "$ATLAS" --output "$RESULTS/cuda"
```

The cloud gate records uploaded source hashes, CUDA/compiler/MPI identity,
selected tests and JUnit results. Same-host MPI ranks share logical device 0.
Cross-host and multiple-physical-GPU execution are outside this acceptance.
`verify_modal_image.py` separately checks the public example's Atlas image
installation on a CPU container.

Local logs and caches are retained outside Git. Only sanitized reports, source
identities and test selections belong here. Interrupted or setup-failed attempts
must remain distinguished from completed acceptance.

## Acceptance harness revisions

`run_gate_initial.py` retains the exact first CPU/Metal harness. The device
revision `run_gate_device.py` recognizes the `gpu_backend` fixture and avoids
matching the substring in `compile` as an MPI module. `run_gate.py` additionally
includes tests whose bodies explicitly call `mpi(2)`; their source-derived
function list is in `fixed-mpi-functions.json`. `--mode mpi-fixed` runs just
that supplementary subset; normal `--mode mpi` includes both parameterized and
fixed-rank functions. Exact completed selections and harness hashes are recorded
with each result. GPU/MPI selection is representative by function and rank,
not exhaustive over every argument combination.

The code under acceptance is Atlas commit
[`63b11d7421cb9d5db342784a0d95fa2bdd8b3ba1`](https://github.com/RockLi/brian2-atlas/commit/63b11d7421cb9d5db342784a0d95fa2bdd8b3ba1).
See [source identity](source-identity.json), [acceptance summary](acceptance.json)
and [per-gate records](results/).

The final CUDA gate compiles the unchanged CUDA library once and records its
host source, generated kernel and binary hashes. Each test lifetime receives a
hash-checked copy of that real library, extending the same reuse approach used
by the module fixtures. GPU execution, MPI transport, device buffers, numerical
comparisons and checkpoint checks are unchanged. This avoids recompiling the
identical CUDA source in every module; it does not substitute a mock backend.
