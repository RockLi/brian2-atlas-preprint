from pathlib import Path
from reportlab.pdfgen import canvas
from pypdf import PdfReader

import argparse

parser = argparse.ArgumentParser(description='Assemble a 22-page slide PDF from fresh renders.')
parser.add_argument('render', type=Path)
parser.add_argument('out', type=Path)
parser.add_argument('--language', choices=['en', 'zh-CN'], default='en')
args = parser.parse_args()
render, out = args.render, args.out
expected = [render / f'slide-{n:02d}.png' for n in range(1, 23)]
if sorted(render.glob('slide-*.png')) != expected:
    raise SystemExit('Expected exactly 22 fresh slide renders.')
out.parent.mkdir(parents=True, exist_ok=True)
pdf = canvas.Canvas(str(out), pagesize=(960, 540), pageCompression=1, invariant=1)
pdf.setTitle('Brian2 Atlas — 预印本 v2 中文演示文稿' if args.language == 'zh-CN' else 'Brian2 Atlas — Preprint v2 Slides')
pdf.setAuthor('Xinjun Li · Next Brain')
pdf.setSubject('22 页中文演示文稿；基于英文预印本 v2；CC BY 4.0' if args.language == 'zh-CN' else '22 slides, based on preprint v2. CC BY 4.0')
links = [
    ('https://doi.org/10.5281/zenodo.23273257', (258,225,890,270)),
    ('https://github.com/RockLi/brian2-atlas', (258,187,890,219)),
    ('https://doi.org/10.5281/zenodo.23269098', (258,149,890,184)),
    ('https://github.com/RockLi/brian2-atlas-preprint', (258,107,925,136)),
    ('https://doi.org/10.5281/zenodo.23269145', (258,69,925,104)),
]
for n in range(1,23):
    pdf.drawImage(str(render / f'slide-{n:02d}.png'), 0, 0, width=960, height=540)
    if n == 20:
        for url, box in links:
            pdf.linkURL(url, box, relative=0, thickness=0)
    pdf.showPage()
pdf.save()
reader = PdfReader(out)
assert len(reader.pages) == 22
assert all(tuple(float(v) for v in page.mediabox) == (0,0,960,540) for page in reader.pages)
assert len(reader.pages[19]['/Annots']) == 5
print(f'{out}\n22 pages, 16:9, 5 clickable resource links; {out.stat().st_size:,} bytes')
