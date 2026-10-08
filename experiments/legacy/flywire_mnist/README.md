# FlyWire MNIST development implementation

Current handoff: [final experiment and delivery report](../../FLYWIRE_MNIST_FINAL_REPORT.md). The optimized CPU full-test result is 87.14%; historical weak-input results below remain unchanged.

Full-dataset runner and completed CNN comparison: [FULL_MNIST.md](FULL_MNIST.md).

Current optimization results and remaining acceptance gaps:
[acceptance status](../../validation/FLYWIRE_MNIST_ACCEPTANCE_STATUS.md).
The historical small-sample evidence below is the first implementation milestone;
it does not describe the later input/readout validation campaigns.

CPU is the required f64 baseline. Native Metal and CUDA use explicit f32;
MPI is an explicit, isolated CPU subprocess adapter. This first implementation
trains a linear readout of a **fixed** conductance-LIF connectome. It does not
train internal synapses or establish that a living fruit fly recognizes digits.

Development branch: `codex/flywire-mnist`, based on `f014e616d` from the latest
GPU/execution-plan worktree at the start of development. The original worktree
and the independently advancing MPI branch are not modified.

## Run

Use a Python environment with the repository's Brian2 dependencies, NumPy,
SciPy and pytest. Build the extensions in this checkout (an editable install of
another worktree is insufficient):

```sh
python setup.py build_ext --inplace
cargo build --release --locked --manifest-path brian2-rust/Cargo.toml
export PYTHONPATH="$PWD/brian2-rust/python:$PWD/brian2-rust/experiments:$PWD"
python -m flywire_mnist prepare --data /tmp/flywire-mnist-data
python -m flywire_mnist run \
  --data /tmp/flywire-mnist-data \
  --graph /path/to/flywire-v783-ei-sensory \
  --output /tmp/flywire-mnist-cpu-run \
  --backend cpu --train-count 100 --validation-count 50
```

`--graph` accepts the existing full EI data directory (`connectome.b2csr`,
`annotations.npz`, `manifest.json`) with pinned hashes, or a bundled provenance
bearing induced circuit JSON. **All bundled circuits currently contain only two
ALPN inputs**: for diagnostic runs supply `--config` with `{"fanout": 2}`.
Their output explicitly identifies the induced subgraph; they are not whole-brain
evidence. The full graph uses fanout 4, 512 selected KC and all 96 MBON.

GPU: use `--backend metal` or `--backend cuda`. Metal requires an accessible Apple
GPU; CUDA requires the existing backend's NVIDIA toolchain/runtime. There is no
silent CPU fallback. GPU uses sparse bitset recurrent delivery, and Metal uses
resident DAG buffers. `--max-gpu-mib` bounds execution buffers (default 512 MiB;
the first full-graph smoke uses 1024 MiB). It is **not a process RSS limit**;
host preparation, snapshots and predecessor executors also consume memory.

MPI: use `--backend mpi --mpi-source /path/to/mpi-checkout --ranks 2`.
That checkout needs its own built runner and Brian2 extensions. The worker imports
only the explicit MPI checkout, validates and compiles its own B2IR project, and
launches real ranks with the existing checked MPI launcher. The adapter records
commit/source hashes and rejects source changes during a run. The current MPI
API binds immutable instances, so this initial adapter recompiles per sample;
it is a correctness/capacity integration, not an efficient batch implementation.

Outputs must be new directories. By default per-sample runtime files are removed
after feature extraction, while features, identifiers and small runtime summaries
are retained. `--keep-runtime` retains large execution files for diagnosis.
An incomplete run never receives `complete-development-run` status. This initial
development command has no automatic resume; the separate full-dataset runner
adds resumable shards and frozen-input CPU execution.

## Scientific and execution contract

- Full graph: v783 EI, 139,255 neurons and 15,091,983 weighted directed edges.
  The existing voltage, excitation, inhibition, threshold, refractory period,
  recurrent weights and 1.8 ms delay are preserved. `--no-recurrence` explicitly
  sets transmission to zero as an ablation.
- Artificial 784-channel Bernoulli-per-tick input projects only to annotated
  ALPN, with amplitude divided by fanout. This is not a natural visual pathway.
  Encoding is label-independent and keyed by original sample ID. Background,
  projection and sample streams are independent and deterministic.
- Every sample starts from the same initial state at tick zero. CPU reuses a
  compiled binary through `run_compatible_instance`, including source, immutable
  layer and plan validation. The AOT loader now reads frozen spike schedule
  lengths from Instance, so changing input event counts does not require a new
  executable. GPU uses fresh sample state and predecessor compilation reuse.
- Four observational state counters increment only on a spike in each stimulus
  window. They never feed back into voltage or synaptic dynamics. Final selected
  counts form 2,432 features on the full graph. Tests compare counters against
  recorded spike ticks and compare the dynamics with counters removed.
- Official MNIST training data is deterministically split 50,000 fit / 10,000
  validation. Development sample counts select prefixes of these disjoint sets.
  Downloads include checksummed official test files, but `run` never decodes or
  evaluates them. Small prefixes need not contain all ten classes.
- Standardization is fitted only on fit data. A ten-output ridge readout selects
  alpha from 0.1/1/10/100 using validation. The report is therefore **development
  validation after selection**, not a final test result. Pixel-only and projected
  input readouts receive the same split and alpha budget; shuffled fit labels are
  a sanity control. Saved readouts must be paired with the experiment manifest's
  graph, encoding, readout ordering and backend/precision.

## Verification

```sh
python -m pytest brian2-rust/tests/test_flywire_mnist.py \
  brian2-rust/tests/test_artifact.py -q
FLYWIRE_MNIST_TEST_METAL=1 python -m pytest \
  brian2-rust/tests/test_flywire_mnist.py -q -k metal
FLYWIRE_MNIST_MPI_SOURCE=/path/to/mpi-checkout python -m pytest \
  brian2-rust/tests/test_flywire_mnist.py -q -k mpi
```

CPU tests cover actual A→B→A replay, exact Rust-reference state/spike comparison,
counter noninterference, malformed-instance rejection, split isolation and
label-free encoding. Metal tests compare fresh/reused execution with a compiled
CPU-f32 mirror. MPI tests compare two-rank final states with CPU-f64. Platform
tests skip only when their explicit environment switches are absent.

## Remaining research gates

The first [recorded smoke evidence](evidence/summary.json) uses the full graph,
10 fit images and 10 validation images on both CPU and Metal. Both FlyWire
readouts score 1/10 on that tiny validation set (pixel baseline 3/10); several
classes are absent from these prefixes. This is **not evidence of useful digit
recognition**, nor a negative scientific verdict on the connectome. Median
end-to-end sample times were about 5.86 s CPU and 16.49 s Metal, with concurrent
diagnostics affecting timing. One full-graph 2,432-dimensional Metal feature
vector exactly matches the compiled CPU-f32 control, despite differing from
CPU-f64. The observed precision sensitivity needs a larger, dedicated study.

Related verification passed 32 tests and 21 subtests with actual Metal and MPI
enabled. Final CLI runs additionally exercised artifact cleanup and compilation
reuse on the 240-cell diagnostic circuit. CUDA has not been run on NVIDIA
hardware in this milestone. Raw output paths and source identities are retained
in the evidence; the committed feature archives contain no original images.

The implementation is the first executable milestone, not completion of
`FLYWIRE_MNIST_GPU_PLAN.md`. Remaining gates include the six-config input
calibration, larger training/validation runs, statistics-preserving random
rewiring, matched no-recurrence runs, repeated seeds, full f32/f64 sensitivity
analysis, locked official-test evaluation, CUDA hardware validation, resource
admission/resume, MPI multi-node scale validation, and a live digit drawing UI.
Full-graph CPU and GPU smoke results must not be used as claims about accuracy
or the value of biological topology.
