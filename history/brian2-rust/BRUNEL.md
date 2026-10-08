# Brunel (2000) balanced network

This example reproduces model A and the four Figure 8 simulation points from
Nicolas Brunel, *Dynamics of Sparsely Connected Networks of Excitatory and
Inhibitory Spiking Neurons*, Journal of Computational Neuroscience 8, 183–208
(2000), DOI 10.1023/A:1008925309027. It also implements the broad-delay AR
experiment described by Figure 4.

## Scientific model

The full network contains 10,000 excitatory and 2,500 inhibitory neurons.
Every target receives exactly 1,000 distinct excitatory and 250 distinct
inhibitory recurrent inputs, for 15,625,000 recurrent synapses. Every neuron
also receives 1,000 independent external Poisson inputs. The membrane time
constant is 20 ms, threshold 20 mV, reset 10 mV, absolute refractory period
2 ms, excitatory PSP 0.1 mV, and simulation step 0.1 ms.

| Name | State | `g` | `nu_ext / nu_thr` | Delay |
| --- | --- | ---: | ---: | --- |
| `sr` | synchronous regular | 3.0 | 2.0 | fixed 1.5 ms |
| `si_fast` | synchronous irregular, fast | 6.0 | 4.0 | fixed 1.5 ms |
| `ai` | asynchronous irregular | 5.0 | 2.0 | fixed 1.5 ms |
| `si_slow` | synchronous irregular, slow | 4.5 | 0.9 | fixed 1.5 ms |
| `ar` | asynchronous regular | 3.0 | 2.0 | uniform 0–3 ms |

The AR point holds `g` and external drive at the SR point and broadens only
the recurrent delays. This directly demonstrates the paper's result that a
broad delay distribution stabilizes asynchronous activity in the low-`g`
region.

## Single-point runner

From the repository root:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python \
  brian2-rust/examples/brunel_device.py \
  --backend rust --regime si_fast --threads 4 \
  --output brian2-rust/output/brunel-si-fast
```

The output contains `result.json` with parameters, timings and statistics,
plus `activity.npz` with spikes, voltage traces, population rate and spectrum.
`--backend cpp` and `--backend numpy` provide independent Brian2 baselines.
`--network-scale` is for engineering smoke tests; scientific runs use 1.

Rust constructs the two projections with `connect_fixed_indegree`, without
materializing 15.625 million source/target pairs in Python. `Uniform` provides
the deterministic per-edge 0–3 ms delay initializer for AR. C++/NumPy use
Brian's `sample(N_pre, size=...)` generator and `rand()*3*ms` delay assignment.
Backends therefore share the statistical ensemble but use different random
streams.

Render the fixed-delay SR versus broad-delay AR experiment with:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python \
  brian2-rust/examples/brunel_figures.py \
  --delay-comparison brian2-rust/output/brunel-sr \
                     brian2-rust/output/brunel-ar \
  --output brian2-rust/output/brunel-delay-comparison.png
```

## Artifact-reused sweep

`g` and `nu_ext / nu_thr` are instance scalars. One native definition can
therefore execute a complete fixed-delay phase grid without invoking rustc for
each point:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python \
  brian2-rust/examples/brunel_sweep.py \
  --artifact brian2-rust/output/brunel-ai \
  --g-min 1 --g-max 8 --g-count 20 \
  --eta-min 0.5 --eta-max 4 --eta-count 20 --threads 4 \
  --output brian2-rust/output/brunel-phase-sweep

MPLCONFIGDIR=.cache/matplotlib .venv/bin/python \
  brian2-rust/examples/brunel_figures.py \
  --sweep brian2-rust/output/brunel-phase-sweep \
  --output brian2-rust/output/brunel-phase-sweep.png
```

A 20 by 20 smoke grid at `network_scale=0.01` took 5.24 s. The completed
full-scale 20 by 20 grid took 2459.65 s (40 min 59.65 s) on this host. Benchmark
reports must separate frontend construction, compilation, initialization,
simulation, result loading and analysis.
