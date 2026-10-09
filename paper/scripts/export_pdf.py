"""Print the portable manuscript to PDF with local Chromium and no network assets."""
from pathlib import Path
import argparse
import os
import shutil
import hashlib
import json
import re
import subprocess
import tempfile
from pypdf import PdfReader, PdfWriter
import pypdfium2 as pdfium
from pdf_supplements import append_atlasir_figure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--chromium', type=Path, default=os.environ.get('CHROMIUM'),
                    help='Chromium/Chrome executable; defaults to CHROMIUM or local discovery')
parser.add_argument('--output', type=Path, help='Output PDF path')
parser.add_argument('--timeout', type=float, default=300,
                    help='Maximum Chromium export time in seconds (default: 300)')
args = parser.parse_args()
if args.timeout <= 0:
    parser.error('--timeout must be positive')
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'paper/MANUSCRIPT.html'
OUTPUT = (args.output or ROOT / 'output/pdf/brian2-atlas-preprint.pdf').resolve()
WORK = ROOT / 'tmp/pdfs'
candidates = [args.chromium] if args.chromium else [
    shutil.which('chromium'), shutil.which('chromium-browser'),
    shutil.which('google-chrome'), shutil.which('google-chrome-stable'),
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
]
chromium = next((Path(p).resolve() for p in candidates
                 if p and Path(p).is_file() and os.access(p, os.X_OK)), None)
if chromium is None:
    parser.error('Chromium/Chrome is required; supply --chromium or CHROMIUM')
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
html = SOURCE.read_text()
# Relative evidence links belong to paper/, even though the print HTML is temporary.
assert '<meta charset="utf-8">' in html
html = html.replace('<meta charset="utf-8">', '<meta charset="utf-8"><base href="' + SOURCE.parent.as_uri() + '/">', 1)
html = re.sub(r'<title>.*?</title>', '<title>A Unified Intermediate Representation and Execution Architecture for Heterogeneous and Distributed Neural Simulation</title>', html)
html = html.replace('<h2>References</h2>', '<section class="references"><h2>References</h2>')
html = html.replace('<h2>Appendix A.', '</section><h2 class="compatibility-appendix">Appendix A.', 1)
html = html.replace('<h3>5.5 Browser portability and interactive execution</h3>', '<h3 class="browser-section">5.5 Browser portability and interactive execution</h3>')
html, cohort_table_count = re.subn(
    r'(<p><strong>Table 2 \|.*?</p>\s*)<table>',
    r'\1<table class="cohort-table">', html, flags=re.S)
assert cohort_table_count == 1, cohort_table_count
html, dendritic_table_count = re.subn(
    r'(<p><strong>Table 5 \|.*?</p>\s*)<table>',
    r'\1<table class="dendritic-table">', html, flags=re.S)
assert dendritic_table_count == 1
html, count = re.subn(
    r'<p>(<img[^>]+>)</p>\s*<p>(<strong>Figure \d+.*?</p>)',
    r'<figure>\1<figcaption>\2</figcaption></figure>', html, flags=re.S)
assert count == 10, count
html = html.replace('</p></figcaption>', '</figcaption>')
html = re.sub(r'<figure>(<img[^>]+alt="Figure 7\.[^>]+>)', r'<figure class="browser-figure">\1', html)
dendritic_section = ROOT / 'paper/data/common_source/section.html'
html = html.replace('</main>', dendritic_section.read_text().replace('<h2>S13.', '<h2 class="dendritic-supplement">S13.', 1) + '</main>', 1)
resource_section = ROOT / 'paper/data/full_scale/section.html'
if resource_section.exists():
    resource_validation = json.loads((ROOT / 'paper/validation/full_scale_resource_analysis.json').read_text())
    assert resource_validation['status'] == 'passed'
    assert hashlib.sha256(resource_section.read_bytes()).hexdigest() == resource_validation['section_sha256']
    html = html.replace('</main>', resource_section.read_text() + '</main>', 1)
css = '''
@page {size:A4; margin:19mm 18mm 19mm;
  @bottom-right {content:counter(page);font:8pt Arial;color:#67737c;}
}
@media print {
  body {font:10.5pt/1.43 Georgia,serif;color:#202830;background:#fff;}
  main {max-width:none;margin:0;padding:0;box-shadow:none;}
  h1 {font-size:23pt;line-height:1.18;margin:0 0 14pt;}
  h2 {font-size:14pt;line-height:1.25;margin:19pt 0 9pt;padding-top:9pt;break-after:avoid;}
  h3 {font-size:11.5pt;line-height:1.3;margin:14pt 0 6pt;break-after:avoid;}
  h4 {font:700 10.5pt/1.3 Arial,sans-serif;margin:12pt 0 6pt;break-after:avoid;}
  p {margin:0 0 8pt;orphans:3;widows:3;}
  p:has(>strong:first-child) {font-size:10.5pt;}
  main>p:has(>em:only-child) {font:9pt/1.4 Arial,sans-serif;color:#64707a;margin-bottom:14pt;}
  main>p.author-block {font:13pt/1.45 Arial,sans-serif;color:#202830;margin:0 0 14pt;}
  .author-block span {font-size:10pt;color:#64707a;}
  a {color:#245c85;text-decoration:none;overflow-wrap:anywhere;}
  figure {margin:14pt 0;break-inside:avoid;}
  figure img {display:block;max-width:100%;width:100%;height:auto;max-height:160mm;object-fit:contain;margin:0 auto 8pt;}
  figcaption {font:9pt/1.35 Arial,sans-serif;}
  figure.browser-figure img {max-height:192mm;}
  figure.browser-figure {break-after:page;}
  table {font:8pt/1.35 Arial,sans-serif;display:table;overflow:visible;width:100%;table-layout:fixed;margin:8pt 0 14pt;break-inside:avoid;}
  table.cohort-table {break-inside:auto;}
  table.dendritic-table th:first-child,table.dendritic-table td:first-child {width:38%;}
  table.dendritic-table th:not(:first-child),table.dendritic-table td:not(:first-child) {width:12.4%;}
  table.resource-component-table {break-inside:auto;}
  h2.conditional-resource-section {break-before:page;}
  h2.dendritic-supplement {break-before:page;}
  th,td {padding:6pt 5pt;overflow-wrap:anywhere;}
  th {background:#edf3f6;}
  tr {break-inside:avoid;}
  thead {display:table-header-group;}
  p:has(+ table) {break-after:avoid;font:9pt/1.35 Arial,sans-serif;}
  code {font-size:9pt;overflow-wrap:anywhere;}
  h2.compatibility-appendix {break-before:page;}
  h2.compatibility-appendix + p + table th:first-child,
  h2.compatibility-appendix + p + table td:first-child {width:42%;text-align:left;}
  h2.compatibility-appendix + p + table th:not(:first-child),
  h2.compatibility-appendix + p + table td:not(:first-child) {width:14.5%;text-align:center;}
  h2.compatibility-appendix + p + table th,
  h2.compatibility-appendix + p + table td {padding:4pt 3pt;}
  .references {font-size:9pt;line-height:1.35;break-inside:avoid;}
  .references p {margin-bottom:6pt;break-inside:avoid;}
  .references ol {padding-left:19pt;margin:0 0 6pt;}
  * {-webkit-print-color-adjust:exact;print-color-adjust:exact;}
}
'''
html = html.replace('</style>', '</style><style>' + css + '</style>', 1)
print_html = WORK / 'manuscript-print.html'
print_html.write_text(html)
with tempfile.TemporaryDirectory(prefix='brian2-pdf-chrome-', dir=WORK) as profile:
    pending = Path(profile) / 'manuscript.pdf'
    log = Path(profile) / 'chrome.log'
    with log.open('wb') as handle:
        process = subprocess.Popen([
        str(chromium),
        '--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
        '--disable-background-networking', '--disable-extensions',
        '--no-pdf-header-footer', '--allow-file-access-from-files',
        '--user-data-dir=' + profile, '--print-to-pdf=' + str(pending),
        print_html.as_uri(),
        ], stdout=subprocess.DEVNULL, stderr=handle)
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    stderr = log.read_bytes()
    if not pending.is_file() or b'%%EOF' not in pending.read_bytes()[-1024:]:
        raise RuntimeError('Chrome did not finish the PDF: ' + stderr.decode(errors='replace')[-2000:])
    OUTPUT.write_bytes(pending.read_bytes())
assert OUTPUT.is_file() and OUTPUT.stat().st_size > 10000
reader = PdfReader(OUTPUT)
writer = PdfWriter()
writer.clone_document_from_reader(reader)
text_document = pdfium.PdfDocument(OUTPUT)
page_texts = [page.get_textpage().get_text_range() for page in text_document]
text_document.close()
for i, text in enumerate(page_texts):
    if 'S13. Additional dendritic CPU evidence' in text:
        writer.add_outline_item('Supplement S13 | Dendritic CPU validation and cache ablation', i)
if resource_section.exists():
    resource_pages = [i for i, text in enumerate(page_texts)
                      if 'S21. Conditional resource analysis' in text]
    assert len(resource_pages) == 1, resource_pages
    writer.add_outline_item('Supplement S21 | Conditional full-reference resources', resource_pages[0])
append_atlasir_figure(writer)
writer.add_metadata({'/Author': 'Xinjun Li', '/Subject': 'brian2-atlas; heterogeneous and distributed neural simulation'})
with OUTPUT.open('wb') as handle:
    writer.write(handle)
print(OUTPUT)
