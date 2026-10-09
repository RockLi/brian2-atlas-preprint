# Brian2 Atlas preprint materials

This repository contains the manuscript, retained experimental evidence, figure generators, and reproduction workflows for **Brian2 Atlas**.

The product implementation is maintained in [RockLi/brian2-atlas](https://github.com/RockLi/brian2-atlas). Experimental results retain their original source identities. The current Atlas development branch must not be substituted for an experiment's recorded revision.

The [experiment-code coverage audit](experiments/COVERAGE.md) maps every final-paper study to its drivers and identifies historical build/input limitations.

## Frozen release pair

The `biorxiv-v1` annotated tag identifies this evidence archive and pairs with
[Brian2 Atlas `v0.1.0`](https://github.com/RockLi/brian2-atlas/tree/v0.1.0).
`main` holds the frozen archive; ongoing work belongs on `dev`.
The paper-version label does not assert that a bioRxiv submission or DOI exists.

## Manuscript and figures

- [Manuscript source](paper/MANUSCRIPT.md)
- [Portable reading version](paper/MANUSCRIPT.html)
- [Supplementary methods and evidence](paper/SUPPLEMENTARY.md)
- [Dated writing-stage evidence matrix](paper/EVIDENCE_MATRIX.md)
- [Current manuscript material index](paper/README.md)

The captured figure-building environment uses Python 3.14.4. Install the plotting dependencies and rebuild from retained data:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r paper/requirements-lock.txt
python paper/scripts/build_manuscript.py
```

The complete dependency versions are pinned in `paper/requirements-lock.txt`; `paper/requirements.txt` lists the direct dependencies. A clean local rebuild reproduced the outputs at the revision recorded in the [initial build validation](migration/clean-manuscript-build.json). Current structure and figure hashes are in the [build report](paper/validation/build_report.json).

The generator defaults to Matplotlib’s bundled fonts, avoiding host font-discovery differences. It writes the figures, `paper/MANUSCRIPT.html`, and `paper/validation/build_report.json`. Its checks cover retained-file hashes, figure/table structure, and consistency of the reported numbers. They do not constitute new simulations or hardware measurements.

The manuscript CI runs this same retained-evidence build and saves its generated artifacts. It checks the reporting pipeline; simulation and hardware validation are separate.

Historical evidence collectors require explicit source locations; see [collection inputs and scope](paper/COLLECTION.md). Rebuilding the manuscript does not require running these collectors.

## Migration and reproduction

The manuscript materials have been imported with file-level provenance. A [pinned PD14 build-to-simulation workflow](experiments/reproduction/pd14/README.md) has passed end-to-end validation against Atlas commit `b769c21004a89e2a6f3a14521f23012db654aadd`. Other legacy research entry points retain their historical execution assumptions. The [historical source guide](archives/HISTORICAL_SOURCES.md) records verified source restoration and exact limitations; the [final cutoff map](migration/migration-cutoff-disposition.json) accounts for the user-approved migration scope.

[Research experiment collections](experiments/README.md) preserve the legacy FlyWire learning, MNIST and vision scripts and their recorded small results. Use the recorded historical sources and environments for those collections; the pinned PD14 command is the separately validated portable reproduction workflow.

[Historical development records](history/README.md) preserve the original experiment reports, implementation notes and intermediate contracts with source hashes. Their version-specific claims remain historical.

[Research archive catalog](archives/README.md) maps retained execution and evaluation files to checksummed local archives, including resolved external evidence links.

[Capture provenance](PROVENANCE.md) distinguishes the import snapshot from the individual experimental source versions. [The import manifest](migration/paper-import.json) records every captured manuscript file and its original hash.

The upstream Brian2 author and license files are retained with the imported material. Manuscript authorship is stated in the manuscript itself.

The product migration cutoff is the third frozen increment (`20261008T161008Z`), explicitly selected by the user. Later development is recorded as a separate future port. The [migration report](https://github.com/RockLi/brian2-atlas/blob/dev/migration/FINAL_REPORT.md) summarizes the accepted product, distribution checks, historical limits and remaining publication tasks.

[Cutoff paper and reproduction integrity](migration/final-paper-reproduction-integrity.json) preserves checksums at the migration cutoff. Subsequent manuscript edits are checked in the [current PDF validation](paper/validation/current_commit_pdf_qa.json).

The [current paper validation](paper/data/release_validation/acceptance.json) distinguishes the maintained implementation and public API revision from the CUDA and PD14 qualification revisions. The 709-case CUDA follow-up and repeated PD14 workflow are retained at `b769c21004a89e2a6f3a14521f23012db654aadd`; changing the manuscript implementation citation does not relabel those measurements. English and Chinese reading versions are checked together.

## License, citation and scope

Original code is licensed under [Apache-2.0](LICENSE). Original manuscript,
figures and research data by Xinjun Li are licensed under
[CC BY 4.0](LICENSES/CC-BY-4.0.txt). Imported Brian2 code and third-party
datasets, model sources and images retain their own terms. See
[LICENSE_SCOPE.md](LICENSE_SCOPE.md) for the boundaries and
[CITATION.cff](CITATION.cff) for citation metadata. Record the exact commit used;
no paper DOI is asserted by these metadata.

## Public evidence exports

Host-specific paths have been replaced with synthetic locations in the published
history. Original evidence is privately preserved with its original checksums.
The [redaction map](migration/public-path-redaction.json) binds original content
hashes to the public export hashes; a redacted export is not byte-identical to
the original measurement artifact. Configure actual input/output locations when
running historical scripts. The [commit map](migration/public-commit-map.json)
resolves original commit identities to the public history.
