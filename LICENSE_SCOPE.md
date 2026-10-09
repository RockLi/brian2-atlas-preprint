# License and repository scope

This repository is the Brian2 Atlas research compendium: manuscript, figures,
original experimental code, reported data and reproduction records. The maintained
engine is in [brian2-atlas](https://github.com/RockLi/brian2-atlas).

| Material | Applicable terms |
| --- | --- |
| Original experimental, analysis, figure-generation and reproduction code by Xinjun Li | [Apache-2.0](LICENSE) |
| Original manuscript text, translations, author-created figures/tables and original research data authored by Xinjun Li | [CC BY 4.0](LICENSES/CC-BY-4.0.txt); attribute Xinjun Li, the work title, this repository and the version/commit, and indicate changes |
| Imported Brian2 source and Brian2-derived code | CeCILL-2.1 and preserved component notices in [Brian2-LICENSE](LICENSES/Brian2-LICENSE) |
| Third-party model/data inputs, photographs, screenshots and other incorporated works | Their source-specific terms and attribution; they are excluded from the blanket original-content grant |

The Apache and CC BY grants apply only to rights held by the author. File-level
and source-specific third-party notices take precedence for those materials.
In particular, original screenshot pixels, external datasets and archived source
bundles are not relicensed merely by being copied into this repository. Frozen
archives retain their internal notices; existing grants for older revisions remain
intact. Dataset provenance is recorded in the supplement and evidence manifests.

The manuscript and its original figures/data are attributed to Xinjun Li (2026),
*A Unified Intermediate Representation and Execution Architecture for Heterogeneous
and Distributed Neural Simulation*. See [CITATION.cff](CITATION.cff). No DOI,
publication venue or public release date is asserted before one is assigned.

The [experiment coverage map](experiments/COVERAGE.md) identifies study drivers.
Each measurement retains its recorded implementation and input identity; the
current engine commit is not substituted for an earlier measurement's source.
Large external arrays are not included, and an archive catalog entry alone does
not establish a durable public download. The compendium is not a release package
for the engine, and paper rendering does not rerun simulations.
