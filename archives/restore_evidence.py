"""Restore selected historical evidence with catalog and per-file hash checks."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in value:
        raise ValueError(f'Unsafe archive/catalog path: {value!r}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True, help='Original source directory prefix, as listed in catalog.jsonl')
    parser.add_argument('--archives', type=Path, help='Directory containing the checksummed tar archives')
    parser.add_argument('--output', type=Path, help='New output directory; original source paths are retained below it')
    parser.add_argument('--list', action='store_true', help='List selected records without extracting files')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    index = json.loads((root / 'index.json').read_text())
    catalog = root / index['catalog']['path']
    if sha256(catalog) != index['catalog']['sha256']:
        raise ValueError('Catalog checksum differs from index.json')
    prefix = str(relative_path(args.prefix.rstrip('/'))) + '/'
    rows = [row for line in catalog.read_text().splitlines()
            if (row := json.loads(line))['source'].startswith(prefix)]
    if not rows:
        parser.error('No catalog records match the selected prefix')
    if args.list:
        print(json.dumps({'prefix': prefix, 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
                          'sources': [r['source'] for r in rows]}, indent=2))
        return
    if args.archives is None or args.output is None:
        parser.error('--archives and --output are required unless --list is selected')
    directory = args.archives.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('Output already exists; use a new directory')
    archive_info = {row['path']: row for row in index['archives']}
    used = {}
    for name in sorted({row['archive'] for row in rows}):
        relative_path(name)
        expected = archive_info[name]
        path = directory / name
        actual = sha256(path)
        if path.stat().st_size != expected['bytes'] or actual != expected['sha256']:
            raise ValueError(f'Archive checksum/size mismatch: {name}')
        used[name] = {'sha256': actual, 'bytes': expected['bytes']}
    output.mkdir(parents=True, exist_ok=False)
    restored = []
    for name in used:
        with tarfile.open(directory / name, 'r:') as archive:
            for row in rows:
                if row['archive'] != name:
                    continue
                source = relative_path(row['source'])
                relative_path(row['member'])
                member = archive.getmember(row['member'])
                if not member.isfile() or member.size != row['bytes']:
                    raise ValueError(f'Expected a captured regular file: {row["source"]}')
                target = output.joinpath(*source.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.extractfile(member) as src, target.open('xb') as dst:
                    while chunk := src.read(8 * 1024 * 1024):
                        digest.update(chunk)
                        dst.write(chunk)
                if digest.hexdigest() != row['sha256']:
                    raise ValueError(f'Content checksum mismatch: {row["source"]}')
                target.chmod(member.mode & 0o777)
                restored.append({'source': row['source'], 'sha256': digest.hexdigest(),
                                 'bytes': row['bytes'], 'original_kind': row['original_kind'],
                                 'restored_kind': 'regular-file'})
    report = {'schema': 'atlas-historical-evidence-restore-v1', 'status': 'verified',
              'prefix': prefix, 'catalog_sha256': index['catalog']['sha256'],
              'archives': used, 'files': restored,
              'scope': 'Byte-verified restoration of the selected original evidence paths. Original symlink targets are restored as regular files. This does not execute an experiment or substitute an Atlas revision for its historical source identity.'}
    (output / 'restore-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'verified', 'files': len(restored), 'manifest': str(output / 'restore-manifest.json')}))


if __name__ == '__main__':
    main()
