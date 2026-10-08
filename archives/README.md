# Retained research archives

`index.json` records the two migration archives, byte sizes, SHA-256 checksums and their current location on T7. They are local preservation copies; a public archival download has not yet been published.

`catalog.jsonl` maps each retained research/evaluation/PDF file to its archive member and content hash. For an original symbolic link, the entry points to the separately captured target bytes. This makes the record independent of the original link target remaining available. Historical source trees inside evidence folders stay in these archives rather than becoming a second editable engine in this repository.

The snapshot includes committed, uncommitted, untracked and selected ignored materials captured on 2026-10-08. Its Git HEAD identifies the source checkout at capture time, not the engine used by every experiment. Use each experiment's retained source manifest and runtime identity for scientific attribution. Concurrent source changes will be recorded separately.

Archive checksums establish byte identity. They do not establish scientific acceptance, completed simulation validation or public availability. The manuscript's retained inputs under `paper/data/` are already in Git; the much larger execution history is indexed here. Portable experiment entry points and source-version bindings remain under migration.
