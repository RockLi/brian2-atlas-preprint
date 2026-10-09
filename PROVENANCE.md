# Capture provenance

Public copies are path-redacted exports. Original hashes in historical records
identify the privately preserved originals; use the root
`migration/public-path-redaction.json` and `migration/public-commit-map.json`
to verify public copies and resolve source revisions.


The initial manuscript import was captured from the development checkout of `RockLi/brian2` on 2026-10-08. Its HEAD was `81eb571d78292a0e24a8e79980ea0d5c26e8a48c`, and the checkout included additional uncommitted work.

`migration/paper-import.json` records the original path, new path, byte count, and SHA-256 for each captured file. The source manuscript lived under `docs/preprint/`, which was ignored by the original repository's root Git rules; copying only committed files would have omitted it.

The capture HEAD is not the source identity of every experiment. The retained JSON evidence and source manifests under `paper/data/` continue to identify the actual source snapshots, runners, configurations, and outputs used by each cohort. Those historical identities are preserved during migration.

Regenerating figures may change SVG metadata or rendering hashes. The import manifest records the captured input state; regenerated output hashes and numerical consistency checks belong to `paper/validation/build_report.json` and the migration validation records.

Large original execution archives remain separate from this manuscript import. Their locations and checksums are recorded in [archives/index.json](archives/index.json). [Historical Git and source restoration](archives/HISTORICAL_SOURCES.md) documents the verified Git bundle, reviewed-source overlay, evidence restoration command, and remaining historical FlyWire limitations. These archives are locally retained on T7; public downloads have not been published.
