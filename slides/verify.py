#!/usr/bin/env python3
"""Check deliverables and optionally compare a rebuild with retained originals."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from pypdf import PdfReader

NAMES = ('atlas-preprint-v2-slides.pptx', 'atlas-preprint-v2-slides.pdf',
         'atlas-preprint-v2-slides-zh-CN.pdf')
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
C = '{http://schemas.openxmlformats.org/drawingml/2006/chart}'
LINKS = {
    'https://doi.org/10.5281/zenodo.23273257',
    'https://github.com/RockLi/brian2-atlas',
    'https://doi.org/10.5281/zenodo.23269098',
    'https://github.com/RockLi/brian2-atlas-preprint',
    'https://doi.org/10.5281/zenodo.23269145',
}


def pdf_signature(path):
    doc = PdfReader(path)
    assert len(doc.pages) == 22, f'{path.name}: page count'
    assert doc.metadata.author == 'Xinjun Li · Next Brain'
    pixels = []
    for page in doc.pages:
        assert tuple(map(float, page.mediabox)) == (0, 0, 960, 540)
        assert len(page.images) == 1
        image = page.images[0].image.convert('RGB')
        assert image.size == (1920, 1080)
        pixels.append(hashlib.sha256(image.tobytes()).hexdigest())
    links = {str(a.get_object()['/A']['/URI']) for a in doc.pages[19]['/Annots']}
    assert links == LINKS
    return {'page_rgb_sha256': pixels, 'links': sorted(links)}


def pptx_signature(path):
    with ZipFile(path) as package:
        assert package.testzip() is None
        names = package.namelist()
        slides = sorted((n for n in names if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)),
                        key=lambda n: int(re.search(r'(\d+)\.xml', n)[1]))
        assert len(slides) == 22
        signature = {'slides': [], 'charts': [], 'notes': [], 'media': []}
        table_owners, chart_owners = [], []
        for i, name in enumerate(slides, 1):
            xml = ET.fromstring(package.read(name))
            table_count = len(xml.findall(f'.//{A}tbl'))
            chart_count = len(xml.findall(f'.//{C}chart'))
            if table_count:
                table_owners.append(i)
            if chart_count:
                chart_owners.append(i)
            signature['slides'].append({
                'text': [n.text for n in xml.iter(A + 't')],
                'transforms': [ET.tostring(n).decode() for n in xml.iter(A + 'xfrm')],
                'tables': table_count, 'charts': chart_count,
            })
        assert table_owners == [8, 10, 12, 14, 18, 22]
        assert chart_owners == [7, 9, 11, 13, 15, 16, 21]
        for name in sorted(names):
            if re.fullmatch(r'ppt/(?:slides/)?charts/chart\d+\.xml', name):
                xml = ET.fromstring(package.read(name))
                signature['charts'].append([n.text for n in xml.iter()
                                            if n.tag in (C + 'v', A + 't')])
            if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml', name):
                xml = ET.fromstring(package.read(name))
                signature['notes'].append([n.text for n in xml.iter(A + 't')])
            if name.startswith('ppt/media/'):
                signature['media'].append(hashlib.sha256(package.read(name)).hexdigest())
        assert len(signature['charts']) == 7
        assert len(signature['notes']) == 22
        assert any('Next Brain' in (t or '') for t in signature['slides'][0]['text'])
        assert not any('B2IR' in (t or '') for s in signature['slides'] for t in s['text'])
        signature['media'].sort()
        return signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--require-all', action='store_true', help='Require both PDFs and the local PPTX.')
    args = parser.parse_args()
    report = {'slide_count': 22, 'checks': {}}
    for name in NAMES:
        path = args.directory / name
        if not path.is_file():
            path = args.directory / 'local' / name
        if not path.is_file():
            assert name != NAMES[1] and not args.require_all, f'Missing required output: {name}'
            continue
        signature = (pptx_signature if name.endswith('.pptx') else pdf_signature)(path)
        check = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'structure': 'passed'}
        if args.reference:
            reference = args.reference / name
            if not reference.is_file():
                reference = args.reference / 'local' / name
            if reference.is_file():
                old = (pptx_signature if name.endswith('.pptx') else pdf_signature)(reference)
                assert signature == old, f'{name}: differs from reference'
                check['reference_comparison'] = ('slide text, geometry, charts, notes and embedded media match'
                                                  if name.endswith('.pptx') else 'all 22 pages have identical RGB pixels and links')
            else:
                assert name != NAMES[1] and not args.require_all, f'Missing required reference: {name}'
                check['reference_comparison'] = 'no local-only reference supplied'
        report['checks'][name] = check
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
