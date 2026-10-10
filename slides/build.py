#!/usr/bin/env python3
"""Rebuild English slides, optionally including local Chinese materials."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / 'local'
NAMES = ('atlas-preprint-v2-slides.pptx', 'atlas-preprint-v2-slides.pdf',
         'atlas-preprint-v2-slides-zh-CN.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', default=os.environ.get('SLIDES_NODE', 'node'))
    parser.add_argument('--node-modules', type=Path,
                        default=os.environ.get('ARTIFACT_NODE_MODULES'))
    parser.add_argument('--skill-dir', type=Path,
                        default=os.environ.get('PRESENTATIONS_SKILL_DIR'))
    parser.add_argument('--output-dir', type=Path, help='Override all output locations, e.g. an ignored verification directory.')
    parser.add_argument('--language', choices=['en', 'all'], default='en')
    args = parser.parse_args()
    languages = ('en', 'zh-CN') if args.language == 'all' else ('en',)
    if args.language == 'all':
        for name in ('src/zh-CN.mjs', 'src/translations.json',
                     'assets/zh-CN/fig1.svg', 'assets/zh-CN/fig9.svg'):
            if not (LOCAL / name).is_file():
                parser.error(f'Missing local-only input: local/{name}; a public clone supports --language en.')
    if not args.node_modules or not args.skill_dir:
        parser.error('Supply --node-modules and --skill-dir (see README.md).')
    modules, skill = args.node_modules.resolve(), args.skill_dir.resolve()
    versions = json.loads((ROOT / 'runtime.json').read_text())
    for name, version in versions['node_packages'].items():
        package = modules / name / 'package.json'
        if not package.is_file() or json.loads(package.read_text())['version'] != version:
            parser.error(f'Requires {name} {version} in --node-modules.')
    for entry in (ROOT / 'requirements.txt').read_text().splitlines():
        name, expected = entry.split('==')
        if importlib.metadata.version(name) != expected:
            parser.error(f'Requires Python package {entry}.')
    for name in ('artifact_tool_utils.mjs', 'inspect_presentation_package_integrity.py',
                 'inspect_presentation_layout_geometry.py'):
        if not (skill / 'container_tools' / name).is_file():
            parser.error(f'Missing presentations skill component: {name}')
    node = shutil.which(args.node)
    if not node:
        parser.error(f'Node executable not found: {args.node}')
    cache = ROOT / '.build'
    cache.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='run-', dir=cache))
    output = args.output_dir.resolve() if args.output_dir else None
    env = os.environ.copy()
    for name in ('SKIP_RENDER', 'RENDER_ONLY', 'FINALIZE'):
        env.pop(name, None)
    env.update(SLIDES_ROOT=str(ROOT), PRESENTATIONS_SKILL_DIR=str(skill),
               SLIDES_PYTHON=sys.executable, RUNTIME_NODE_MODULES=str(modules),
               RUNTIME_NODE=node, RUNTIME_PYTHON=sys.executable)
    products = work / 'products'
    products.mkdir()
    for language in languages:
        local = work / language
        local.mkdir()
        (local / 'node_modules').symlink_to(modules, target_is_directory=True)
        source = ROOT if language == 'en' else LOCAL
        shutil.copyfile(source / 'src' / f'{language}.mjs', local / 'build.mjs')
        shutil.copytree(source / 'assets' / language, local / 'assets')
        if language == 'zh-CN':
            shutil.copyfile(ROOT / 'assets/en/fig7.svg', local / 'assets/fig7.svg')
            shutil.copyfile(LOCAL / 'src/translations.json', local / 'translations.json')
        subprocess.run([node, str(local / 'build.mjs')], cwd=local, env=env, check=True)
        name = NAMES[1] if language == 'en' else NAMES[2]
        subprocess.run([sys.executable, str(ROOT / 'src/assemble_pdf.py'),
                        str(local / 'render'), str(products / name),
                        '--language', language], check=True)
    shutil.copyfile(work / 'en/final/validated.pptx', products / NAMES[0])
    subprocess.run([sys.executable, str(ROOT / 'verify.py'), str(products),
                    *(['--require-all'] if args.language == 'all' else [])], check=True)
    # English PDF is public. Editable PPTX and Chinese PDF are always local by default.
    for name in NAMES:
        if not (products / name).is_file():
            continue
        destination = output if output is not None else (ROOT if name == NAMES[1] else LOCAL)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(products / name, destination / name)
        print(f'Output: {destination / name}', flush=True)
    print(f'Ignored intermediate files: {work}', flush=True)


if __name__ == '__main__':
    main()
