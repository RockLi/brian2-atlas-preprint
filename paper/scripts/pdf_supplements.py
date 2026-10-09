"""Append the retained online AtlasIR figure to the manuscript export."""
from pathlib import Path
import json
import subprocess
import sys
from pypdf import PdfReader


def append_atlasir_figure(writer):
    root = Path(__file__).resolve().parents[2]
    if not (root / 'paper/data/b2ir_visualization/capture.json').exists():
        return
    offset = len(writer.pages)
    output = root / 'tmp/pdfs/atlasir-supplement-integrated.pdf'
    subprocess.run([
        sys.executable, str(Path(__file__).with_name('build_atlasir_supplement.py')),
        '--output', str(output), '--page-offset', str(offset),
    ], check=True, capture_output=True)
    reader = PdfReader(output)
    record = json.loads((root / 'paper/data/b2ir_visualization/capture.json').read_text())
    assert len(reader.pages) == len(record['panels'])
    writer.append(reader, import_outline=False)
    writer.add_outline_item('Supplementary Figure S1 | Online AtlasIR inspection', offset)

# Preserve existing export callers.
append_b2ir_figure = append_atlasir_figure
