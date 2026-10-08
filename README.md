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
python -m pip install -r paper/requirements.txt
python paper/scripts/build_manuscript.py
```

The generator defaults to Matplotlib’s bundled fonts, avoiding host font-discovery differences. It writes the figures, `paper/MANUSCRIPT.html`, and `paper/validation/build_report.json`. Its checks cover retained-file hashes, figure/table structure, and consistency of the reported numbers. They do not constitute new simulations or hardware measurements.

## Migration and reproduction status

The manuscript materials have been imported with file-level provenance. Experimental scripts, archive indexing, portable collection paths, and the pinned Atlas checkout are being integrated. A complete reproduction workflow will be documented after its end-to-end validation.

[Capture provenance](PROVENANCE.md) distinguishes the import snapshot from the individual experimental source versions. [The import manifest](migration/paper-import.json) records every captured manuscript file and its original hash.

The upstream Brian2 author and license files are retained with the imported material. Manuscript authorship is stated in the manuscript itself.
