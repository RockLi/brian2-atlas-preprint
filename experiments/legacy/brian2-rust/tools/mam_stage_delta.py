"""Bounded, hash-verified block deltas for already staged MPI artifacts."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shutil

BLOCK = 4096
MAX_FILE = 512 * 2**20
MAX_JSON = 64 * 2**20


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(root, name):
    path = root / name
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    assert path.resolve().is_relative_to(root.resolve())
    return path


def build(old, new, output):
    manifest = json.loads((new / 'mpi/manifest.json').read_text())
    names = ['mpi/' + name for name in manifest['files']]
    names += ['mpi/manifest.json', 'mpi/build.json', 'mpi/b2-mpi',
              'preparation.json', 'admission.json', 'placement.json']
    records = {}
    for name in names:
        prior, target = inside(old, name), inside(new, name)
        assert target.stat().st_size <= MAX_FILE
        changes = []
        with target.open('rb') as current:
            baseline = prior.open('rb') if prior.is_file() else None
            try:
                offset = 0
                while block := current.read(BLOCK):
                    if baseline is None or block != baseline.read(BLOCK):
                        changes.append([offset, base64.b64encode(block).decode()])
                    offset += len(block)
            finally:
                if baseline is not None:
                    baseline.close()
        records[name] = dict(baseline_sha256=sha(prior) if prior.is_file() else None,
                             sha256=sha(target), size=target.stat().st_size,
                             mode=target.stat().st_mode & 0o777, changes=changes)
    document = dict(schema='b2-mpi-block-delta-v1', block_bytes=BLOCK, files=records)
    encoded = json.dumps(document).encode()
    assert len(encoded) <= MAX_JSON
    with output.open('xb') as stream:
        stream.write(gzip.compress(encoded, mtime=0))
    assert output.stat().st_size <= 16 * 2**20
    # Reconstruct on the build host first; this proves the transport recipe.
    verification = output.with_name(output.name + '.verified-tree')
    apply(old, output, verification)
    return dict(path=str(output), bytes=output.stat().st_size, sha256=sha(output),
                uncompressed_bytes=len(encoded), files=len(records),
                changed_blocks=sum(len(r['changes']) for r in records.values()),
                reconstructed_tree=str(verification))


def apply(old, delta, output):
    with gzip.open(delta, 'rb') as stream:
        encoded = stream.read(MAX_JSON + 1)
    assert len(encoded) <= MAX_JSON
    document = json.loads(encoded)
    assert document['schema'] == 'b2-mpi-block-delta-v1' and document['block_bytes'] == BLOCK
    assert 1 <= len(document['files']) <= 64
    output.mkdir(exist_ok=False)
    for name, record in document['files'].items():
        prior, target = inside(old, name), inside(output, name)
        assert type(record['size']) is int and 0 <= record['size'] <= MAX_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        if record['baseline_sha256'] is not None:
            assert sha(prior) == record['baseline_sha256'], ('baseline mismatch', name)
            shutil.copyfile(prior, target)
        else:
            target.touch(exist_ok=False)
        with target.open('r+b') as stream:
            seen = set()
            for offset, value in record['changes']:
                block = base64.b64decode(value, validate=True)
                assert type(offset) is int and offset >= 0 and offset % BLOCK == 0 and offset not in seen
                assert 0 < len(block) <= BLOCK and offset + len(block) <= record['size']
                seen.add(offset)
                stream.seek(offset)
                stream.write(block)
            stream.truncate(record['size'])
        target.chmod(record['mode'])
        assert sha(target) == record['sha256'], ('reconstruction mismatch', name)
    manifest = json.loads((output / 'mpi/manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert sha(inside(output / 'mpi', name)) == expected
    build = json.loads((output / 'mpi/build.json').read_text())
    assert sha(output / 'mpi/b2-mpi') == build['executable_sha256']
    result = dict(executable_sha256=build['executable_sha256'],
                  verified_files=len(manifest['files']), delta_sha256=sha(delta))
    (output / 'delta-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'apply'])
    parser.add_argument('--old', required=True, type=Path)
    parser.add_argument('--new', type=Path)
    parser.add_argument('--delta', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build(args.old, args.new, args.delta) if args.mode == 'build' else apply(args.old, args.delta, args.output)
    print(json.dumps(result, indent=2))
