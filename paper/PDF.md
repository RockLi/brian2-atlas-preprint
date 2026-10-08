# Build the manuscript PDF

Run from the repository root. The HTML and retained figure data are self-contained; no simulation or historical evidence collector is required.

```sh
python -m pip install -r paper/requirements-pdf-lock.txt
python paper/scripts/build_manuscript.py
python paper/scripts/export_pdf.py
```

Chromium or Google Chrome must be installed separately. The exporter looks on PATH and in the standard macOS Chrome location; override with `--chromium /path/to/chrome` or the `CHROMIUM` environment variable. Use `--output /path/to/document.pdf` to change the destination. `--timeout` controls the bounded browser export wait.

The default result is `output/pdf/brian2-atlas-preprint.pdf`. Intermediate HTML, the supplementary figure PDF and the disposable browser profile stay in `tmp/pdfs/` inside this checkout. Place the checkout and Python environment on the external storage used for reproduction; set pip/uv caches there as well. Generated PDFs and temporary files are ignored by Git.

The export preserves the manuscript print styles, retained dendritic and resource supplements, and the three B2IR visualization panels. Their embedded input hashes remain checked. Rebuilding these pages does not rerun scientific experiments or establish new backend qualification.

Historical PDF versions are captured in the archive catalog under `../archives/`; they are not regenerated or relabeled by this command. The archive index records local availability and does not claim a public download that has not been published.

Relative evidence links in the PDF resolve to retained files under `paper/` in the checkout used for export, rather than the temporary print directory. Keep that checkout available for local evidence navigation. Public PDF distribution will require publication-ready code/data links as a separate release step.
