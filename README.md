# Brian2 Atlas preprint materials

This repository contains the manuscript, retained experimental evidence, figure generators, and reproduction workflows for **Brian2 Atlas**.

The product implementation is maintained in [RockLi/brian2-atlas](https://github.com/RockLi/brian2-atlas). Experimental results retain their original source identities. The current Atlas development branch must not be substituted for an experiment's recorded revision.

## Manuscript and figures

- [Manuscript source](paper/MANUSCRIPT.md)
- [Portable reading version](paper/MANUSCRIPT.html)
- [Supplementary methods and evidence](paper/SUPPLEMENTARY.md)
- [Evidence and experiment matrix](paper/EVIDENCE_MATRIX.md)
- [Original manuscript material index](paper/README.md)

The captured figure-building environment uses Python 3.14.4. Install the plotting dependencies and rebuild from retained data:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r paper/requirements-lock.txt
python paper/scripts/build_manuscript.py
```

The complete dependency versions are pinned in `paper/requirements-lock.txt`; `paper/requirements.txt` lists the direct dependencies. A clean local rebuild reproduced all tracked manuscript outputs byte-for-byte ([validation record](migration/clean-manuscript-build.json)).

The generator defaults to Matplotlib’s bundled fonts, avoiding host font-discovery differences. It writes the figures, `paper/MANUSCRIPT.html`, and `paper/validation/build_report.json`. Its checks cover retained-file hashes, figure/table structure, and consistency of the reported numbers. They do not constitute new simulations or hardware measurements.

The manuscript CI runs this same retained-evidence build and saves its generated artifacts. It checks the reporting pipeline; simulation and hardware validation are separate.

Historical evidence collectors require explicit source locations; see [collection inputs and scope](paper/COLLECTION.md). Rebuilding the manuscript does not require running these collectors.

## Migration and reproduction status

The manuscript materials have been imported with file-level provenance. A [pinned PD14 build-to-simulation workflow](experiments/reproduction/pd14/README.md) has passed end-to-end validation against Atlas commit `6677a5bafd3b703ab56b6ed176e9aad70f4638cd`. Other legacy research entry points retain their historical execution assumptions. The [historical source guide](archives/HISTORICAL_SOURCES.md) records verified source restoration and exact limitations; final product-increment and distribution reconciliation remain in progress.

[Research experiment collections](experiments/README.md) preserve the legacy FlyWire learning, MNIST and vision scripts and their recorded small results. Use the recorded historical sources and environments for those collections; the pinned PD14 command is the separately validated portable reproduction workflow.

[Historical development records](history/README.md) preserve the original experiment reports, implementation notes and intermediate contracts with source hashes. Their version-specific claims remain historical.

[Research archive catalog](archives/README.md) maps retained execution and evaluation files to checksummed local archives, including resolved external evidence links.

[Capture provenance](PROVENANCE.md) distinguishes the import snapshot from the individual experimental source versions. [The import manifest](migration/paper-import.json) records every captured manuscript file and its original hash.

The upstream Brian2 author and license files are retained with the imported material. Manuscript authorship is stated in the manuscript itself.
