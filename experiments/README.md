# Research experiment collections

The legacy FlyWire learning, MNIST and vision collections are preserved in `legacy/`, including models, scripts, protocols, recorded environments, provenance and small results. The import map is `../migration/experiment-collection-import.json`. Every file is byte-identical to its captured source.

These are historical research entry points. Their original relative imports, data paths and backend assumptions are retained. Source bindings, archival restoration and the remaining historical limits are documented in the [source guide](../archives/HISTORICAL_SOURCES.md). Python syntax checks pass; this import does not claim portable or scientifically accepted execution. Use each collection's recorded input/source identities and archived code for its historical results. Do not rerun a historical campaign against the current Atlas `dev` and label it as the same experiment.

- [FlyWire learning](legacy/flywire_learning/README.md)
- [FlyWire MNIST](legacy/flywire_mnist/README.md)
- [FlyWire vision](legacy/flywire_vision/README.md)

The [PD14 reproduction entry point](reproduction/pd14/README.md) builds the unchanged research program against an exact Atlas Git dependency and executes a bounded deterministic simulation. It passed on macOS arm64. This new run is distinct from the historical full-scale experiments. Large execution histories and historical engine snapshots remain in the checksummed archives indexed under `../archives/`.

The [historical research programs](legacy/brian2-rust/README.md) also preserve contextual dendritic, multi-area model and Litwin-Kumar scripts, analysis tools, tests and adjacent small records. Their exact mapping is in `../migration/research-family-import.json`.
