# Historical research programs

This tree preserves contextual dendritic, multi-area model and Litwin-Kumar experimental programs, analysis tools and their tests. The original `examples/`, `tools/` and `tests/` relationships are retained. See `../../../migration/research-family-import.json` for each source path, content hash and module description.

No editable Brian2 or `brian2_rust` engine is included here. Historical result identities come from the original records and the archived engine captures indexed in `../../../archives/`. The capture HEAD plus file hashes identify these imported scripts; they are not a claim about which engine produced every result.

These programs retain historical machine paths, launch destinations and optional dependency assumptions. Their syntax and import-copy integrity have been checked, but they are not a turnkey reproduction command. Do not run a campaign launcher without configuring its explicit inputs and execution destination. Do not substitute the current Atlas branch for a recorded historical engine revision. A [new PD14 reproduction](../../reproduction/pd14/README.md) uses a fixed Atlas commit and has passed its documented bounded end-to-end run.

Some product tests use model-building helpers from these research families. Their required fixtures have been copied into the product tree with explicit provenance; the engine never depends on this historical tree at import time.

Additional performance and comparator campaigns, remote-run scripts, and experiment-specific regression checks are indexed in `../../../migration/comparison-program-import.json`. Their local script-helper dependencies are preserved explicitly, including shared frozen model builders. The `src/bin/pd14.rs` file is an experimental consumer of the Atlas library; it is not an engine implementation. The [separate pinned Cargo reproduction package](../../reproduction/pd14/README.md) is available. Other historical campaign paths and environments retain their original assumptions; see the [source restoration guide](../../../archives/HISTORICAL_SOURCES.md) for verified preservation scope and limitations.
