# FlyWire transmitter-informed whole-brain benchmark

This extends the [topology engineering benchmark](FLYWIRE.md) with curated
transmitter signs, a conductance LIF model, frozen background input and a paired
olfactory stimulation experiment. It is a biologically informed reduced model,
**not experimentally validated whole-brain physiology**. The original positive
LIF and this model are separate workloads; their spike counts and timings should
not be interpreted as an isolated E/I ablation.

## Data and transmitter decisions

Use the complete v783 graph: 139,255 biological neurons, 15,091,983 directed
weighted edges, representing 54,492,922 biological contacts. There is no
five-contact threshold. The official source is
[Zenodo v783.0](https://zenodo.org/records/10676866), CC BY 4.0.

Annotations are pinned to the paper-matched
[v2.1.0 release](https://github.com/flyconnectome/flywire_annotations/tree/ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0),
not a changing live annotation table. Its SHA-256 is
`30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6`.
The importer checks the official connection MD5, annotation SHA-256, original
CSR SHA-256, and an exact uint64 root-ID match for all neurons.

Six official probability columns (`gaba_avg`, `ach_avg`, `glut_avg`, `oct_avg`,
`ser_avg`, `da_avg`) are averaged per source neuron, weighted by contact count.
Argmax determines the provisional transmitter. This is not a majority vote over
individual contacts. The 1,732 rows with missing scores (30,806 contacts) do not
contribute to that average; their connections remain in the graph.

Unambiguous curated fast-transmitter evidence in `known_nt` takes precedence:
acetylcholine is excitatory; GABA, glutamate and histamine are inhibitory.
Conflicting curated excitatory/inhibitory evidence gives zero fast action.
Without curated fast evidence, the EM prediction is used; predicted monoamines
and unknown predictions give zero fast action. Curated peptides/monoamines do
not by themselves specify a fast conductance in this implementation. This is
not a receptor-resolved interpretation of glutamate or neuromodulation.

There are 8,684,795 positive, 6,058,491 negative and 348,697 zero-fast-action edges.
All remain allocated and traversed, including zero edges. Signed contact counts
are stored once as float64; the inhibitory multiplier belongs to the model.
The 59,391 curated decisions are the number using curated evidence, not the
number whose sign changed relative to the classifier. `neurons.csv` preserves
raw and annotation predictions, known transmitters, decision source and IDs.
The signed graph SHA-256 is
`b84a19b5c6d899181e6ade3a914eba89d53cb3a4cc36789702785b4cfa35ec38`.

## Dynamics and stimulus

The equations, in SI units, are:

```text
dv/dt  = (-(v + 52 mV) - ge*v - gi*(v + 70 mV)) / 20 ms
 dge/dt = -ge / 5 ms
 dgi/dt = -gi / 5 ms
```

Voltage integration pauses during the 2.2 ms refractory period; conductances
continue decaying. Threshold is -45 mV, reset -52 mV, Euler dt 0.1 ms, and
recurrent delay 1.8 ms. Both conductances are positive, relative to leak.
Excitatory reversal is 0 mV and inhibitory reversal -70 mV. A positive contact
adds `0.275 / 52` to `ge`; a negative contact adds `4 * 0.275 / 52` to `gi`.
The CLI's `*_weight_mv` values are reference amplitudes converted to dimensionless
conductance, not literal voltage jumps. Fourfold inhibitory conductance does
not mean fourfold inhibitory current: driving forces differ.

[Shiu et al. (2024)](https://www.nature.com/articles/s41586-024-07763-9) provides
precedent for reduced whole-brain LIF modelling and some membrane/time constants.
That paper uses a different release, current-based synapses and calibration
experiment. The reversal potentials, fourfold conductance ratio, olfactory
stimulus and background calibration here are modelling choices, not a replication
or a universal experimentally measured E/I ratio.

The background has 512 independent frozen Bernoulli-per-bin approximations to
300 Hz Poisson processes, randomly mapped over all biological neurons, with
reference amplitude 3.5 mV. Neurons sharing a channel share input. This reduces
input dimensionality and may affect correlations. The 1–5 Hz mean target is an
engineering calibration under background drive, **not spontaneous activity in
a zero-input brain**. Initial voltages are -52 +/- 0.8 mV (seed 783).

A separate frozen 80 Hz schedule drives each of 68 annotated `ORN_DM1` neurons
between 300 and 700 ms, with reference amplitude 40 mV. This is a synthetic
DM1-selective drive, not a fitted chemical odor. Two `DM1_lPN` neurons are selected
by annotation (root IDs 720575940619071005 and 720575940630770042). Readouts also
include all 685 ALPNs, 5,177 Kenyon cells, 96 MBONs and 2,869 CX neurons. The 580
external spike-generator units are not extra biological neurons.

Four conditions share the same initial state, background and stimulus schedules:

| Condition | Sensory input amplitude | ORN outgoing transmission |
| --- | --- | --- |
| rest | zero | enabled |
| odor | enabled | enabled |
| cut_rest | zero | disabled |
| cut | enabled | disabled |

Cutting transmission only in the stimulated condition would confound stimulus
response with baseline changes. Both cut conditions are necessary. All
non-sensory spike IDs and times must be identical between `cut` and `cut_rest`.
The experiment supports ORN-dependent downstream response in this model; it does
not establish that every response travels exclusively through one anatomical
pathway. First positive 1 ms bins are descriptive, not fitted causal latencies.

## Reproduce

First build the Rust runner and create the original full CSR as in FLYWIRE.md.
Download [`Supplemental_file1_neuron_annotations.tsv`](https://raw.githubusercontent.com/flyconnectome/flywire_annotations/ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0/supplemental_files/Supplemental_file1_neuron_annotations.tsv)
(save as `annotations-v2.1.0.tsv`); the importer enforces its checksum.
Run from repository root with the project's Python environment:

```sh
python brian2-rust/examples/flywire_ei_import.py \
  --graph brian2-rust/output/flywire-v783 \
  --connections brian2-rust/output/flywire-data/proofread_connections_783.feather \
  --annotations brian2-rust/output/flywire-data/annotations-v2.1.0.tsv \
  --output brian2-rust/output/flywire-v783-ei-sensory

python brian2-rust/examples/flywire_ei_benchmark.py \
  --graph brian2-rust/output/flywire-v783-ei-sensory \
  --output brian2-rust/output/flywire-full-lif-ei \
  --levels 1,4,8 --repeats 3 --rustc /absolute/path/to/native/rustc

python brian2-rust/examples/flywire_ei_profile.py \
  --graph brian2-rust/output/flywire-v783-ei-sensory \
  --build brian2-rust/output/flywire-full-lif-ei/aot-odor-t1 \
  --output brian2-rust/output/flywire-full-lif-ei/profile
```

Use new output directories. `--reuse-builds` instead validates existing model,
condition, graph and input metadata and repeats measurements without recompiling;
keep an earlier report before replacing it. Large C++ replay dumps are removed
immediately after numerical comparison to bound scratch disk use. Build
artifacts and their reference snapshots remain available.

For 27 use native ARM Rust (its default Rust may be x86 under Rosetta), and
`--cxx apple-clang-libomp`. For 23 use native Rust 1.98.1 and `/usr/bin/g++` 13.3;
run both backends under `numactl --physcpubind=0-15 --membind=0` with
`OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1`.
Do not change production services. Same-host comparisons are the performance
contract; different Python versions, host loads and OS scheduling limit direct
cross-host rankings.

## Validation and interpretation

Every condition compares full final voltage/conductance/refractory/transmission
arrays, all spike IDs/times/counts, and voltage/conductance trajectories at 20
recorded neurons between Rust and original Brian2 C++ standalone. Continuous
values permit rtol 1e-12 / atol 1e-14 across backends; replay against each backend's
own output requires zero error. Measurements warm each thread/backend pair and
interleave rotating thread levels and alternating backend order.

Native time includes simulation and recording, excluding import, compilation,
startup and output. Native RSS is measured by a fresh small supervisor to avoid
Linux inherited-parent RSS. Python/frontend and compiler peaks are separate and
are not subject to the 500 MiB **native process** target. The low-RSS claim does
not mean the entire preprocessing/build pipeline fits in 500 MiB.

Mean rate is evaluated after the first 100 ms. Voltage bounds check all final
neurons and recorded trajectories, not all neurons at every time step. The 5 ms
participation statistic measures unique active neurons per bin. These checks
cannot establish absence of a clinical seizure or physiological realism. Inspect
rate tails too: the calibrated mean does not guarantee physiological individual
rates. Three deterministic replays test reproducibility, not independent-seed
biological robustness. One second is a benchmark duration, not a long-term
stability experiment.

The existing native emitter already balances target owners by combined incoming
degree. Profile before adding hub splitting: target splitting can alter floating
addition order and determinism. The observed per-owner event estimate counts
outgoing recurrent edges over the recorded source spike train; it includes events
scheduled beyond the last time step and is not an exact delivered-event counter.
Profiled timings include instrumentation and are excluded from benchmark medians.
Serial and parallel paths have different threshold fusion; superlinear self
speedup must not be interpreted as pure core scaling. The `threshold_seconds`
phase moves into `neuron_state_seconds` in the fused parallel path.

See [measured cross-host EI results](FLYWIRE_EI_RESULTS.md) for acceptance outcomes,
source identities, timing spreads, model limitations and artifact locations.

Generate the two descriptive figures from a collected aggregate report:

```sh
python brian2-rust/examples/flywire_ei_figures.py \
  --graph brian2-rust/output/flywire-v783-ei-sensory \
  --snapshots brian2-rust/output/flywire-full-lif-ei \
  --report brian2-rust/output/flywire-ei-cross-host/report.json \
  --output brian2-rust/output/flywire-full-lif-ei/figures
```
