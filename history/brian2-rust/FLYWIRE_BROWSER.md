# FlyWire browser circuit

Neural Lab includes **240, 1,024 and 4,096-cell DM1 olfactory induced subgraphs**, running the
shared Rust reference executor as WASM in a Worker. It is extracted from the
existing transmitter-informed FlyWire benchmark, not a synthetic random graph.
The default 240-cell circuit retains all **6,660 directed weighted edges**,
representing **66,815 biological contacts**. A weighted edge aggregates contacts;
it does not model each contact as an independent synaptic state.

## Data and attribution

FlyWire Consortium / Dorkenwald et al. (2024), *Neuronal wiring diagram of an adult
brain*, [paper](https://www.nature.com/articles/s41586-024-07558-y).
The [official v783 release](https://zenodo.org/records/10676866) is CC BY 4.0.
Transmitter decisions use the pinned annotation release described in
[FLYWIRE_EI.md](FLYWIRE_EI.md). The subset carries data and annotation provenance,
original root IDs as exact decimal strings, original row indices, cell classes,
transmitter decision sources, contact multiplicities and signed weights in
[`wasm/flywire-circuit.json`](wasm/flywire-circuit.json).

`tools/flywire_circuit.py` verifies both full CSR hashes, identical graph indexing,
root IDs and annotation/sign consistency before extracting this asset. Its
canonical node/edge SHA-256 is
`88b73db2e575fdc7adae8d5caf05b6dac5b3efdc587ab8c99d46da8b18bd40fa`.

Selection is deterministic, with root-ID ordering breaking contact-count ties:

| Group | Cells | Selection |
| --- | ---: | --- |
| DM1 olfactory receptor neurons | 68 | All annotated ORN_DM1 cells |
| Antennal-lobe local neurons | 32 | Strongest total input from DM1 ORNs and PNs |
| DM1 projection neurons | 2 | Both annotated DM1_lPN cells |
| Kenyon cells | 96 | Strongest input from the two DM1 PNs |
| Lateral-horn cells | 32 | Strongest PN input among LHLN and LHCENT |
| Mushroom-body output neurons | 10 | Strongest input from selected KCs |

Every induced edge is retained without a contact threshold, including zero-fast-
action edges. Connections to all omitted neurons are absent. This is a deliberately
selected, small olfactory circuit, not a representative whole-brain sample.

Regenerate from already imported full data:

```sh
python brian2-rust/tools/flywire_circuit.py \
  --graph /path/to/flywire-v783-ei-sensory \
  --original /path/to/flywire-v783-csr-verified \
  --output brian2-rust/wasm/flywire-circuit.json
```

Normal `tools/build_wasm.py` uses the included subset asset to export
`flywire-template.json`; it does not require the large full-brain files or an
online data service. File-backed CSR is converted to explicit browser topology
only for this subset. The WASM executor's prohibition on external CSR paths is
unchanged.

## Dynamics and controls

The conductance LIF equations, signs, reference amplitudes, 20 ms membrane time
constant, 5 ms conductance decay, 2.2 ms refractory interval and 1.8 ms recurrent
delay follow `examples/flywire_device.py`. Voltage is represented numerically in
mV. Threshold/reset are −45/−52 mV. A positive contact adds
`recurrent_weight_mv / 52` to ge; a negative contact adds the same amount times
the inhibitory multiplier to gi. The UI defaults remain 0.275 ref. mV per contact
and a multiplier of 4. Curated or inferred transmitter signs are reduced modelling
choices and do not specify receptor-resolved neurotransmission.

The default is 300 ms at fixed Euler dt 0.1 ms. The scale selector loads an exact
240, 1,024 or 4,096-cell graph; dt remains fixed to preserve the delay/refractory contract. Duration, seed, stimulus window,
input rates/amplitudes, recurrent strength and inhibitory multiplier are editable.
The default stimulus drives all 68 DM1 ORNs at 80 Hz from 80–220 ms with a 40 ref.
mV amplitude. This is synthetic DM1-selective drive, not a fitted odor identity.

Unlike the full-brain benchmark's shared frozen background channels, this browser
subset uses an independent Bernoulli-per-tick input per neuron, with a 300 Hz
background rate and 3.5 ref. mV amplitude. Each tick makes both random draws for
every neuron in every condition, even when stimulus amplitude is zero. The same
seed therefore supplies matched stochastic input for intervention comparisons.
Initialization uses uniform voltage offsets ±0.8 mV, ge equal to the background
mean conductance, and gi = 0. No compensation is added for missing neurons.

| Preset | Sensory amplitude | ORN outgoing transmission |
| --- | ---: | --- |
| Stimulate DM1 | 40 ref. mV | On |
| Background only | 0 | On |
| Block sensory output | 40 ref. mV | Off |
| Cut baseline | 0 | Off |

These demonstrate propagation and an intervention within this reduced circuit.
They do not validate whole-brain physiology or animal behavior.

## Display and verification

The network layout groups cells schematically; it uses no anatomical coordinates.
For readability it shows at most 96 evenly selected cells per group and the
600 strongest edges between those displayed cells. All cells and induced edges
are simulated; counts in the drawing disclose the display subset.
Replay and the time slider display actual recorded spikes from the last 10 ms.
Clicking a cell displays its exact root ID and spike count, and selects one of two
recorded state probes from that group. Six group-rate cards summarize the full run.
Voltage and ge/gi traces, raster, frequency curve and distribution use real WASM
output. The shaded interval marks the stimulus window even in zero-amplitude
baseline controls. Model changes and edits still require **Run experiment**.

`wasm/check-flywire.mjs` verifies all four matched conditions, including identical
non-sensory spike streams for `cut` and `cut_rest`, and byte-identical results with
17- versus 128-tick execution batches. `tests/test_flywire_browser.py` checks asset
identity, complete WASM/native results, and an independent Brian NumPy comparison
using a deterministic initial sensory-conductance pulse (random amplitudes are
zero in that numerical comparison). This tests synaptic signs, recurrence,
propagation and scheduling without assuming that Brian NumPy shares the portable
executor's RNG algorithm.

## Expanded olfactory circuits

| Cells | Weighted edges | Biological contacts |
| ---: | ---: | ---: |
| 240 | 6,660 | 66,815 |
| 1,024 | 54,609 | 458,573 |
| 4,096 | 284,105 | 901,832 |

Starting from the original 240 cells, add the unselected cell with the largest
sum of incoming biological contacts from already selected cells, update the
scores, and repeat. Eligible classes are ALLN, Kenyon_Cell, LHLN, LHCENT and MBON;
root ID breaks ties. The original 68 DM1 sensory cells and both DM1 PNs remain.
Selections are nested and every induced edge is retained, including zero-fast
edges. The expanded assets retain the same source hashes and per-cell provenance.
They remain selected olfactory subsets with omitted inputs, not whole-brain models.
Different sizes change both recurrent inputs and RNG assignment; use matched
stimulus/cut comparisons at the same size rather than treating sizes as identical
physiological conditions.

Regenerate by adding `--count 1024` or `--count 4096` to the extraction command,
with output `wasm/flywire-circuit-1024.json` or `wasm/flywire-circuit-4096.json`.
The normal WASM build exports all three templates from the included assets.
The UI fetches only the chosen circuit on Run; large assets are not preloaded for
classic models. Count/template mismatch is rejected before simulation.
