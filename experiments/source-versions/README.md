# Hash-pinned paper source versions

`referenced-sources.tar.gz` retains 95 exact source contents that were previously only in local Git history, local archives or remote experiment directories. Members are named by SHA-256; [index.json](index.json) maps each to its original source path and paper evidence record. Four JavaScript assets were reconstructed using the recorded asset-version substitution and verified against their original full hashes.

This includes capacity model generators, generated MPI programs and reviewed frontend/training sources. It is a collection of referenced source contents, not a standalone build tree or a replacement for Atlas dev. The training evaluation has its complete captured snapshot separately under `../legacy/training-evaluation/`.

Three assets of the historical Figure 7 Neural Lab build remain unresolved: `index.html`, `lab.css`, and generated `pkg/b2_runner.js`. The screenshot, exported experiment and source hashes remain preserved. Current browser source exists in Atlas; it is not declared byte-identical to that old build. These are interface-capture limitations, not missing numerical benchmark drivers.

For extraction, read the member's content from the tar archive, verify its SHA-256, and write it into a new directory according to the selected record's source path. Multiple experiments can refer to different contents at the same path; do not combine them into a fabricated common source tree.
