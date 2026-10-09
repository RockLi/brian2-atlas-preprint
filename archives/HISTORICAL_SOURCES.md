# Historical sources and restoration

Public copies are path-redacted exports. Original hashes in historical records
identify the privately preserved originals; use the root
`migration/public-path-redaction.json` and `migration/public-commit-map.json`
to verify public copies and resolve source revisions.


The maintained implementations are the latest migrated versions. Historical results keep their actual commit and content identities. A newer source file must not be used to claim exact reproduction of an older result.

## Committed engine history

[`historical-git.json`](historical-git.json) identifies the local T7 Git bundle (2,356,102,319 bytes), its SHA-256 and nine experiment revisions. The bundle was restored into a separate mirror, `git fsck --full` passed, and each recorded revision resolved to its expected commit and tree. Capture did not alter the original checkout. This covers reachable Git history; uncommitted contents require separate snapshots.

After obtaining the bundle at the indexed local location, verify its SHA-256 against the index and restore it in a new directory:

```sh
git clone --mirror /path/to/legacy-history.bundle legacy-history.git
git --git-dir=legacy-history.git fsck --full
git --git-dir=legacy-history.git worktree add --detach historical-source RECORDED_COMMIT
```

Replace `RECORDED_COMMIT` with the experiment's actual revision, not Atlas `dev`. The archive is locally available on T7; no public download has been published.

## Reviewed source snapshot

[`review-source-overlay.json`](review-source-overlay.json) records an immutable tar archive containing all 626 files declared by `paper/data/v3/review_snapshot.json`. Each content hash was verified, then the overlay was applied over the recorded base commit in an independent checkout and all 626 hashes were checked again. Three uncommitted historical contents were recovered from existing Git objects and frozen in this archive.

To restore, check out the indexed base commit into a new directory, verify the overlay SHA-256 against this index, and extract the verified archive over that checkout. Verify each listed file against the index before use. This restores the declared reviewed scope; it does not establish the contents of unlisted historical working-tree files or rerun scientific experiments.

## Selected evidence files

From the preprint repository root, list a scope before restoring it:

```sh
python archives/restore_evidence.py --prefix brian2-rust/mpi-evidence/training-20261002 --list
python archives/restore_evidence.py --prefix brian2-rust/mpi-evidence/training-20261002 --archives /path/to/20261008T063409Z --output /new/path/restored-training
```

The tool checks the catalog, full archive hashes and each extracted file. It refuses an existing output directory, preserves original source-relative paths, and materializes captured symbolic-link targets as regular files. The actual 22-file restoration passed; [`historical-evidence-restoration.json`](../migration/historical-evidence-restoration.json) records that verification. Restoring evidence does not execute experiments.

## Old FlyWire MNIST development runs

The CPU-f64 and Metal-f32 manifests refer to six unique intermediate contents across four paths (`__main__.py`, `backends/__init__.py`, `graph.py`, and `model.py`). These contents were not found in the captured archives, available Git objects, registered worktrees, or the targeted T7 FlyWire/cleanup backup search. The latter inventoried 104,878 files and inspected 12 relevant archives without read errors. This was a targeted search, not an exhaustive scan of the disk.

All four latest source files are preserved in `experiments/legacy/flywire_mnist/` and were verified byte-for-byte against the current source checkout and committed preprint contents. See [`flywire-latest-and-backups.json`](../migration/flywire-latest-and-backups.json) for hashes and search scope. The latest code remains the maintained version. Existing historical results, manifests, source hashes and backups remain retained; exact source restoration of those two intermediate runs is incomplete. This pre-existing limitation does not block migration of the latest implementation.

[`historical-source-bindings.json`](../migration/historical-source-bindings.json) records per-file bindings for the four audited manifests. FlyWire learning's nine declared files and the reviewed snapshot's 626 declared files all resolve; the two old MNIST manifests retain the explicit unresolved intermediate contents above. Other historical records preserve their original identities; this audit is not a claim of universal historical reproducibility.

For a newly validated run using a fixed Atlas revision, use the [PD14 reproduction workflow](../experiments/reproduction/pd14/README.md). Its README states the verified scale and numerical checks.
