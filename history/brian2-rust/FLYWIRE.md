# FlyWire empirical weighted-connectome benchmark

For transmitter signs, conductance LIF and paired sensory stimulation, see
[the EI benchmark](FLYWIRE_EI.md) and [cross-host EI results](FLYWIRE_EI_RESULTS.md).
For the separate fixed whole-brain network with a small explicit plastic
KC→MBON projection, see the [CPU learning study](experiments/flywire_learning/README.md).

This benchmark runs **all 139,255 proofread FlyWire v783 neurons** and every
positive directed connection, with no five-contact threshold. It compares Rust
AOT with Brian2 C++ standalone using the same file, edge order, initial state,
float64 LIF equations, monitors and requested thread counts.

The official table has 16,847,997 neuron-pair/neuropil rows. Summing contacts
across neuropils yields **15,091,983 directed weighted edges representing
54,492,922 biological synaptic contacts**. These are different quantities. The
portable CSR is 182,217,884 bytes. Its SHA-256 is
`4f0a4a31332ba489d796fefc7228c7471d0d0ca0939805ffe7369bfca1e12682`.

## Scientific scope and source

This is an execution benchmark on empirical connectivity, not a fitted fruit-fly
brain simulation. Each biological contact contributes a positive weight of
0.0001 to homogeneous LIF neurons (tau=20 ms, refractory=2 ms, fixed delay=0.5 ms,
dt=0.1 ms). Initial voltages and tonic drives use NumPy's seed 783. Neurotransmitter
predictions, synapse coordinates, morphology and behaviour are not modelled.
Aggregation is appropriate to this linear, static, equal-delay contact model;
it does not establish equivalence for individual-contact plasticity or delays.

Data: [FlyWire Consortium, v783.0, Zenodo 10676866](https://zenodo.org/records/10676866),
licensed **CC BY 4.0**. Cite the dataset and
[Dorkenwald et al., Neuronal wiring diagram of an adult brain (2024)](https://www.nature.com/articles/s41586-024-07558-y).
The source record also credits synapse detection, cleft segmentation and
neurotransmitter prediction work; follow those citations when using those data.
Raw data and generated outputs are not committed to this repository.

## Run from repository root

Install the optional Feather reader (`pyarrow`; no pandas dependency) and build
the current Rust runner first. The formal benchmark enforces the existing suite's
native Rust >=1.98 / LLVM >=22 toolchain requirement.

```sh
uv pip install --python .venv/bin/python pyarrow
mkdir -p brian2-rust/output/flywire-data
curl -L --fail https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content -o brian2-rust/output/flywire-data/proofread_connections_783.feather
curl -L --fail https://zenodo.org/api/records/10676866/files/proofread_root_ids_783.npy/content -o brian2-rust/output/flywire-data/proofread_root_ids_783.npy
cargo build --release --locked --manifest-path brian2-rust/Cargo.toml

MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/flywire_import.py \
  --connections brian2-rust/output/flywire-data/proofread_connections_783.feather \
  --neurons brian2-rust/output/flywire-data/proofread_root_ids_783.npy \
  --output brian2-rust/output/flywire-v783

MPLCONFIGDIR=.cache/matplotlib .venv/bin/python brian2-rust/examples/flywire_benchmark.py \
  --graph brian2-rust/output/flywire-v783 \
  --output brian2-rust/output/flywire-lif-comparison \
  --duration-ms 1000 --levels 1,4 --repeats 5 \
  --rustc /absolute/path/to/native/rustc
```

Output directories must be new. On macOS the C++ multi-thread baseline uses the
existing Apple clang/Homebrew libomp wrapper; pass `--cxx` to select another
compiler. The measurement script currently requires POSIX `wait4` for per-process
native peak RSS. `--duration-ms 100` is a shorter full-graph smoke benchmark.

The importer verifies the two official MD5 checksums before conversion. It reads
Feather record batches from a seekable file, makes a disk-backed source bucket
sort, and aggregates one source's outgoing rows at a time. Root IDs remain exact
uint64 values; signed Arrow IDs are explicitly cast before `searchsorted` to
avoid mixed-integer float64 promotion. Isolated proofread neurons are retained.
Input hashes, contact conservation, degree statistics, preprocessing time and
process peak RSS are saved in the import manifest. `--allow-custom-data` is for
fixtures or other data and explicitly disables the official identity check.

## Measurement contract

- Full final voltage/refractory arrays, 16-neuron trajectories, and every spike
  index/time/count are compared for both backends and every warm/measured replay.
- Native binaries are warmed before interleaved replay; backend order alternates
  and thread-level order rotates. The network must produce spikes.
- Simulation + recording time excludes import, compilation, process startup and
  binary output. Native process wall time includes startup/loading/output.
- Native replay RSS comes from `wait4` in a fresh, small stdlib-only supervisor.
  This prevents Linux fork/exec rusage from including the large benchmark
  parent's inherited RSS before exec. Native wall time excludes supervisor startup.
  First-build Python
  RSS is reported separately, as are preprocessing and end-to-end first runs.
- Reports include raw samples, spread, toolchain, graph hash and source identity.
  A spread above 15% signals a noisy measurement, not a stable release baseline.
  Build times are single observations, not repeated timing estimates.
- C++ input/output materialisation is part of its existing standalone path.
  A C++ OOM is not assumed; if both backends fit, both are measured normally.

## Generic binary topology API

```python
import brian2_rust
S = Synapses(G, G, 'multiplicity : 1 (constant)',
             on_pre='v_post += 0.0001*multiplicity', delay=0.5*ms,
             clock=G.clock)
brian2_rust.connect_binary_csr(S, 'connectome.b2csr',
                              parameters={'multiplicity': 0})
```

`B2CSR001` is a little-endian numeric format: four u64 header values
(source count, target count, edge count, parameter column count), followed by
source_count+1 u64 offsets, edge_count u32 target IDs and column-major f64
parameter arrays. Names and dimensions come from the Brian model. Files are
validated for shape, offsets, target bounds and finite parameter values. The
Device checks the attachment SHA-256 before execution; Per-edge Brian arrays (`S.i`, `S.j`, and parameters) remain deferred; use
`binary_topology.csr_arrays` for explicit read access. AOT embeds the arrays in
`instance.bin`, so native replay needs no original CSR or Python. The standalone
reference runner reads the original CSR path and validates its numeric content;
its SHA-256 field is provenance, not an independently enforced digest check.

B2IR schema is v25, following the upstream v24 custom-schedule contract.

The API supports immutable per-edge parameters, pre pathways and uniform pathway
delays. With `build_on_run=True`, repeated `Network.run` calls and `store`/`restore`
preserve undelivered source events across run boundaries. The binary topology,
pathway delays and parameters must remain unchanged across continuation or a
restored checkpoint; changes are rejected before execution. This does not add
mutable per-edge plastic state or procedural per-edge delay initializers to CSR.
Use a separate, disjoint explicit projection for local plasticity, as in the
learning study.

Queued multi-run builds (`build_on_run=False`) still reject binary topology
continuation. Other procedural topology types still support one run per
activation. At most 100 million
file-backed edges are accepted per file. The population ceiling is now one million,
subject to the existing array/tick/recording budgets. Explicit JSON topology retains
its existing smaller budgets. The native AOT instance reader streams external-array
models into their typed vectors instead of retaining a second complete file buffer.

Tests: `test_binary_topology.py` covers exact large IDs, isolated nodes, neuropil
aggregation, malformed files, full FlyWire neuron count, and NumPy/reference/AOT
conformance. Existing Rust regression tests remain required.

Measured full-graph results: [2026-09-05 Apple M3 run](FLYWIRE_RESULTS.md).

To remeasure existing compiled artifacts without rebuilding, run
`python brian2-rust/examples/flywire_replay.py --builds /path/to/original-output --output /path/to/new-replays`.
Keep the original thread placement environment (including Linux NUMA and OpenMP
binding) when replaying. The report records build and measurement commits separately.

Cross-host measurements: [Mac Studio and Linux, 2026-09-06](FLYWIRE_CROSS_HOST_RESULTS.md).
