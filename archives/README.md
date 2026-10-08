# Retained research archives

`index.json` records the two migration archives, byte sizes, SHA-256 checksums and their current location on T7. They are local preservation copies; a public archival download has not yet been published.

`catalog.jsonl` maps each retained research/evaluation/PDF file to its archive member and content hash. For an original symbolic link, the entry points to the separately captured target bytes. This makes the record independent of the original link target remaining available. Historical source trees inside evidence folders stay in these archives rather than becoming a second editable engine in this repository.

The snapshot includes committed, uncommitted, untracked and selected ignored materials captured on 2026-10-08. Its Git HEAD identifies the source checkout at capture time, not the engine used by every experiment. Use each experiment's retained source manifest and runtime identity for scientific attribution. Concurrent source changes will be recorded separately.

Archive checksums establish byte identity. They do not establish scientific acceptance, completed simulation validation or public availability. The manuscript's retained inputs under `paper/data/` are already in Git; the much larger execution history is indexed here. The [historical source guide](HISTORICAL_SOURCES.md) records verified source bindings and restoration, including remaining limitations. The [pinned PD14 workflow](../experiments/reproduction/pd14/README.md) provides a new bounded reproduction against Atlas.

## Complete initial-capture disposition

[`source-disposition.jsonl.gz`](source-disposition.jsonl.gz) assigns all 59,309 inventoried paths to the committed product, committed preprint materials, retained archives, preserved upstream references, or explicitly excluded generated files. The [summary and checksums](../migration/captured-source-disposition.json) record the exact target commits and scope. Read it with Python’s `gzip.open(path, "rt")`; each line is one JSON record keyed by its original `source` path.

An excluded generated-file record does not mean its bytes were archived. Archive locations remain local to the recorded T7 capture. Concurrent source increments after the initial capture require their own manifests and final audit; this table does not declare the entire migration complete.

## Restore archived evidence

Use [`restore_evidence.py`](restore_evidence.py) to select original source paths, verify archive and individual file hashes, and restore into a new directory. See [the restoration guide](HISTORICAL_SOURCES.md) for an example and validation scope.
