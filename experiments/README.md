# Research experiment collections

The legacy FlyWire learning, MNIST and vision collections are preserved in `legacy/`, including models, scripts, protocols, recorded environments, provenance and small results. The import map is `../migration/experiment-collection-import.json`. Every file is byte-identical to its captured source.

These are historical research entry points. Their original relative imports, data paths and backend assumptions are being mapped to this repository and the independently versioned Atlas engine. Python syntax checks pass; this import does not claim portable or scientifically accepted execution. Use each collection's recorded input/source identities and archived code for its historical results. Do not rerun a historical campaign against the current Atlas `dev` and label it as the same experiment.

- [FlyWire learning](legacy/flywire_learning/README.md)
- [FlyWire MNIST](legacy/flywire_mnist/README.md)
- [FlyWire vision](legacy/flywire_vision/README.md)

A new end-to-end reproduction entry point pinned to an Atlas commit will be added after the product backend port is validated. Large execution histories and historical engine snapshots remain in the checksummed archives indexed under `../archives/`.
