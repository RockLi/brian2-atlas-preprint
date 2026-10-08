"""Append the retained online B2IR figure to the manuscript export."""
from pathlib import Path
import json
import subprocess
import sys
from pypdf import PdfReader


def append_b2ir_figure(writer):
    root = Path(__file__).resolve().parents[3]
    if not (root / 'docs/preprint/data/b2ir_visualization/capture.json').exists():
        return
    offset = len(writer.pages)
    output = root / 'tmp/pdfs/b2ir-supplement-integrated.pdf'
    subprocess.run([
        sys.executable, str(Path(__file__).with_name('build_b2ir_supplement.py')),
        '--output', str(output), '--page-offset', str(offset),
    ], check=True, capture_output=True)
    reader = PdfReader(output)
    record = json.loads((root / 'docs/preprint/data/b2ir_visualization/capture.json').read_text())
    assert len(reader.pages) == len(record['panels'])
    writer.append(reader, import_outline=False)
    writer.add_outline_item('Supplementary Figure S1 | Online B2IR inspection', offset)
