# Preprint v2 slides

The public deck is the [22-slide English PDF](atlas-preprint-v2-slides.pdf) for
[Xinjun Li, preprint v2](https://doi.org/10.5281/zenodo.23273257), affiliated with
Next Brain. It includes MPI distribution and heterogeneous rank placement, with
the qualification boundaries retained.

The PDF embeds 1920 × 1080 slide images, with five clickable resource links on
slide 20. Body text is rasterized. Keep the filename when updating the deck.

## Public and local materials

Maintain presentations here on `dev`, manuscript sources in `paper/`, and
reproduction material in `experiments/`. Release tags remain frozen.

- Public: `atlas-preprint-v2-slides.pdf`, English generation source in `src/`,
  original English SVG inputs in `assets/en/`, and the build/verification tools.
- Local only: `local/atlas-preprint-v2-slides.pptx` and
  `local/atlas-preprint-v2-slides-zh-CN.pdf`. The entire `local/` directory is ignored.
- Chinese authoring material, where available locally: `local/src/zh-CN.mjs`,
  `local/src/translations.json` and `local/assets/zh-CN/`. These are not published.

The local editable English PowerPoint retains native text, six tables, seven
charts and speaker notes. Do not commit or attach it to a public release. Keep
local backups of `local/`; Git and a fresh clone cannot restore ignored files.
The older workspace copies remain retained separately.

Edit `src/en.mjs` for English content. When maintaining the private Chinese
version, synchronize its source and translations in `local/src/`. See
[figure provenance and hashes](assets/SOURCES.json). Rasterized/cropped figure
PNGs are generated locally from the SVG inputs, including the embedded browser
workbench screenshot. Builds do not retrieve assets or run simulations.

## Runtime requirements

The verified environment uses macOS, Arial (plus Arial Unicode MS for local
Chinese builds), Node 24.19.0, Python 3.12.14 and Codex bundle `26.1007.11041`.
[Runtime versions](runtime.json) and [Python dependencies](requirements.txt) are
recorded. Fonts are not redistributed. Different renderers/fonts can change pixels.

The renderer uses `@oai/artifact-tool` **2.8.89**, `sharp` **0.35.5**, and the
presentations skill's finalization utilities from the Codex bundle. The artifact
package is a private bundled dependency, not a public npm installation. Readers
without that runtime can use the public PDF, but cannot run this exact renderer
from public package dependencies alone. Runtime code and fonts are not vendored.

In Codex, obtain executable/package paths from `load_workspace_dependencies`
and the presentations skill directory from the installed skill entry. Configure:

```sh
export SLIDES_NODE=/path/to/runtime/node/bin/node
export ARTIFACT_NODE_MODULES=/path/to/runtime/node/node_modules
export PRESENTATIONS_SKILL_DIR=/path/to/presentations/skills/presentations
export SLIDES_PYTHON=/path/to/runtime/python/bin/python3
```

Use that Python environment, or one with `python -m pip install -r slides/requirements.txt`.
From the repository root, the default command rebuilds English only:

```sh
"$SLIDES_PYTHON" slides/build.py
```

It writes the English PDF to `slides/` and the editable PPTX to `slides/local/`.
A public clone needs no Chinese source files. If the private Chinese inputs are
present, rebuild both languages with `--language all`; the Chinese PDF is also
written to `slides/local/`. Missing private inputs fail before rendering starts.

Explicit toolchain options are `--node`, `--node-modules`, and `--skill-dir`.
The interpreter running `build.py` also runs the finalizer and PDF assembler.
Sources are resolved relative to the script, independently of the current directory.

Each run uses a fresh ignored `slides/.build/run-*` directory. It validates the
English PPTX and assembles the requested PDFs before replacing finished outputs.
Partial/cached-render environment flags are not inherited. To compare a fresh
build without replacing retained files:

```sh
"$SLIDES_PYTHON" slides/build.py --output-dir slides/.build/rebuilt
"$SLIDES_PYTHON" slides/verify.py slides/.build/rebuilt \
  --reference slides --report slides/.build/comparison.json
```

`--output-dir` puts all requested outputs in that override directory. Use an ignored
local directory. Verification always requires the English PDF, and also checks
any available local PPTX/Chinese PDF. It searches `local/` for private references;
use `--require-all` to require both PDFs and the PPTX during a bilingual local audit.

Strict comparison checks PDF RGB pixels/links and PPTX text, geometry, chart data,
notes and embedded media. It does not require byte-identical ZIP metadata. The
original migration passed all 44 pages in the matching sandboxed rendering
context; unrestricted rendering also passed content checks but showed small
rasterization differences. GPU availability and antialiasing can affect pixels.
Review any differences; after intentional content changes, run structural checks
without the old `--reference` and visually review the affected slides.

## Git and licensing

Commit the English PDF, English source, necessary SVG inputs and documentation.
Never commit `local/`, PPTX files, Chinese slide PDFs, `.build/`, render PNGs,
validation receipts, runtime dependencies, virtual environments or bytecode.
Ignore rules cover private materials and build caches; do not bypass them with
`git add -f`. The build never modifies branches or release tags.

Original generation code is Apache-2.0. Original slide text and figures by
Xinjun Li are CC BY 4.0. Incorporated works retain their source-specific terms;
see [repository license scope](../LICENSE_SCOPE.md). Cite the preprint DOI and the
actual repository commit when adapting these materials.
