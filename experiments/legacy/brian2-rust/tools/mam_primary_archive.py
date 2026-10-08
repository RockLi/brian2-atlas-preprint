"""Stream and independently verify one complete primary archive on brick2.

Run only through the separately admitted archive launcher. Cache release keeps
file pages bounded; a pending archive is never evidence of successful backup.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

from mam_primary_analysis_pipeline import BASE, OUTPUT, AUDIT_OUTPUT, read, sha, check
from mam_primary_resources import LABEL, MODEL

ARCHIVE = BASE/'mam-primary-100500ms-verified-v1.tar.gz'
REPORT = BASE/'mam-primary-100500ms-backup-v1.json'
CATALOG = BASE/'mam-primary-100500ms-backup-catalog-v1.json'
MAX_BYTES = 224*2**30
CACHE_WINDOW = 64*2**20
BLOCK = 2**20


def attempt_paths(version=1):
    check(type(version) is int and version in (1, 2), 'unsupported archive attempt')
    return dict(source=BASE/f'primary-archive-v{version}-source',
                guard=BASE/f'guards/primary-archive-v{version}.json',
                archive=BASE/f'mam-primary-100500ms-verified-v{version}.tar.gz',
                report=BASE/f'mam-primary-100500ms-backup-v{version}.json',
                catalog=BASE/f'mam-primary-100500ms-backup-catalog-v{version}.json',
                unit=f'b2mpi-archive-primary-v{version}',
                directory=f'primary-archive-v{version}')


def member_identity(path, base):
    """Preserve catalogued AppleDouble bytes under explicit safe aliases.

    Existing source manifests can contain T7-created metadata sidecars. Never
    relax the archive writer's name rules or silently omit manifest entries.
    """
    relative = path.relative_to(base)
    name = relative.as_posix()
    if not any(part.startswith('._') for part in relative.parts):
        return name, {}
    check(not any(part.startswith('._') for part in relative.parts[:-1])
          and relative.name.startswith('._') and len(relative.name) > 2,
          'unexpected metadata path')
    sibling = path.with_name(path.name[2:])
    check(path.is_file() and not path.is_symlink()
          and sibling.is_file() and not sibling.is_symlink()
          and 26 <= path.stat().st_size <= 64*2**10,
          'invalid AppleDouble sidecar')
    with path.open('rb') as stream:
        check(stream.read(8) == bytes.fromhex('0005160700020000'),
              'invalid AppleDouble header')
    alias = 'appledouble-metadata/'+hashlib.sha256(name.encode()).hexdigest()+'.bin'
    return alias, dict(original_relative_path=name, metadata_format='AppleDouble v2')


class CachedStream:
    """Sequential bounded I/O, hashing exactly the bytes consumed/produced."""
    def __init__(self, stream, *, writing=False, limit=MAX_BYTES, window=CACHE_WINDOW):
        self.stream = stream
        self.writing, self.limit, self.window = writing, limit, window
        self.count = self.released = 0
        self.digest = hashlib.sha256()

    def tell(self):
        return self.count

    def _record(self, data):
        self.count += len(data)
        check(self.count <= self.limit, 'stream byte budget exceeded')
        self.digest.update(data)
        if self.count-self.released >= self.window:
            self.release()

    def read(self, size):
        check(0 <= size <= BLOCK, 'unbounded archive source read')
        data = self.stream.read(size)
        self._record(data)
        return data

    def write(self, data):
        check(self.count+len(data) <= self.limit, 'archive byte budget exceeded')
        count = self.stream.write(data)
        check(count == len(data), 'short archive write')
        self._record(data)
        return count

    def release(self):
        if self.writing:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        if self.count > self.released:
            os.posix_fadvise(self.stream.fileno(), self.released,
                             self.count-self.released, os.POSIX_FADV_DONTNEED)
            self.released = self.count


def write_verified_archive(files, expected, destination, *, limit=MAX_BYTES):
    """No overwrite; hash during copy, then stream every decompressed member.

    The second pass also hashes the compressed file and drains its gzip trailer
    from the raw stream. It does not hold a whole member or archive in memory.
    Only a fully checked, fsynced archive receives the final name.
    """
    check(hasattr(os,'posix_fadvise'), 'Linux cache release is required')
    check(set(files) == set(expected) and 0 < len(files) <= 1024, 'archive coverage')
    check(sum(row['bytes'] for row in expected.values()) <= limit, 'source byte budget')
    for name in files:
        p = Path(name)
        check(name == p.as_posix() and not p.is_absolute() and '..' not in p.parts
              and not any(part.startswith('._') for part in p.parts), 'unsafe archive name')
    pending = destination.with_name(destination.name+'.pending')
    check(not destination.exists() and not pending.exists(), 'archive attempt already exists')
    with pending.open('xb') as raw:
        writer = CachedStream(raw, writing=True, limit=limit)
        with gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) as compressed, \
                tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(files):
                path, row = files[name], expected[name]
                check(path.stat().st_size == row['bytes'], 'source size changed')
                # Only explicitly selected regular bytes; never archive links,
                # uid ownership, xattrs, devices or an implicit directory walk.
                info = tarfile.TarInfo(name)
                info.size = row['bytes']; info.mode = 0o600; info.mtime = 0
                with path.open('rb') as source:
                    reader = CachedStream(source, limit=row['bytes'])
                    archive.addfile(info, reader)
                    check(source.read(1) == b'' and reader.count == row['bytes']
                          and reader.digest.hexdigest() == row['sha256'], 'source changed during archive')
                    reader.release()
        writer.release()
        compressed_bytes, compressed_sha = writer.count, writer.digest.hexdigest()
    with pending.open('rb') as raw:
        reader = CachedStream(raw, limit=limit)
        seen = set()
        with gzip.GzipFile(fileobj=reader,mode='rb') as compressed:
            with tarfile.open(fileobj=compressed,mode='r|') as archive:
                for member in archive:
                    check(member.name in expected and member.name not in seen
                          and member.isfile() and member.size == expected[member.name]['bytes'], 'archive member differs')
                    seen.add(member.name)
                    digest, count = hashlib.sha256(), 0
                    with archive.extractfile(member) as stream:
                        while block := stream.read(BLOCK):
                            digest.update(block); count += len(block)
                    check(count == member.size and digest.hexdigest() == expected[member.name]['sha256'],
                          'archived member hash mismatch')
            # Force gzip CRC/trailer validation even though tar ends before gzip.
            while compressed.read(BLOCK):
                pass
        while reader.read(BLOCK):
            pass
        reader.release()
        check(seen == set(expected) and reader.count == compressed_bytes
              and reader.digest.hexdigest() == compressed_sha, 'archive file changed during verification')
    # Atomic publication without overwriting; fsync the directory so the final
    # name is durable. A failed attempt keeps its pending bytes for inspection.
    os.link(pending, destination)
    pending.unlink()
    directory = os.open(destination.parent, os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return dict(bytes=compressed_bytes,sha256=compressed_sha,
                verified_members=len(seen),uncompressed_bytes=sum(r['bytes'] for r in expected.values()))


def inventory(control):
    """Bind raw and scientific artifacts to their prior completed audits."""
    from mam_launch_native_primary import rust_gate
    # The staged control tree has the same narrow paths as the existing gate.
    check(rust_gate(control/'evidence', control/'resources') is not None, 'primary output/resource gates missing')
    raw_audit = read(AUDIT_OUTPUT)
    check(sha(AUDIT_OUTPUT) == sha(control/'evidence/primary-output-audit/full-report.json'), 'raw audit changed')
    pipeline = read(OUTPUT/'report.json')
    check(pipeline['analysis_complete'] is True
          and pipeline['output_audit_sha256'] == sha(AUDIT_OUTPUT)
          and sha(OUTPUT/'report.json') == sha(control/'analysis-report.json'), 'analysis pipeline differs')
    package = read(control/'package.json')
    hosts = read(control/'postrun-hosts.json')
    from mam_primary_resources import NODES
    check([r['host'] for r in hosts] == NODES
          and all(r['package_catalog_sha256'] == sha(control/'package.json')
                  and r['all_deployed_files_verified'] is True for r in hosts), 'post-run host artifact coverage')
    check(set(pipeline['catalogs']) == {'activity','cell','correlation','series'}, 'analysis stage coverage')
    check(sha(BASE/'guards/primary-postrun-v1.json') ==
          sha(control/'evidence/primary-output-audit/full-guard.json'), 'analysis guard changed')
    files, expected = {}, {}
    def add(path, digest=None, size=None):
        check(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(BASE.resolve()),
              'archive source is not a regular brick2 file')
        name, metadata = member_identity(path, BASE)
        check(name not in files, 'duplicate archive source')
        files[name] = path
        expected[name] = dict(bytes=path.stat().st_size if size is None else size,
                              sha256=sha(path) if digest is None else digest, **metadata)
    run = BASE/'primary-host-v1/runs'/LABEL
    check({p.name for p in run.iterdir()} == {'results.bin','events.bin','summary.json','mpi-runtime.json'},
          'unexpected/incomplete primary recording tree')
    for name, key in [('results.bin','results_bytes'),('events.bin','events_bytes')]:
        add(run/name, raw_audit['dump_sha256'][name], raw_audit['dump_bytes'][key])
    for name in ['summary.json','mpi-runtime.json']:
        check((run/name).stat().st_size < 32*2**20, 'oversized metadata')
        add(run/name)
    for name, row in package['catalog'].items():
        add(BASE/LABEL/name,row['sha256'],row['bytes'])
    add(BASE/LABEL/'model.json',MODEL)
    add(BASE/'primary-host-v1/guard.py',package['guard_sha256'],package['guard_bytes'])
    # The deployed MPI directory deliberately points to the existing home
    # runtime. Include only its three explicitly pinned files as regular bytes;
    # never follow an arbitrary link or sweep that home directory.
    runtime=Path('/atlas-home/0003/workspace/brian2-mpi-linux-20260907/mpi')
    check(set(package['mpi_runtime'])=={'bin/mpiexec.hydra','bin/hydra_pmi_proxy','lib/libmpi.so.12'},
          'unexpected runtime archive coverage')
    for name,digest in package['mpi_runtime'].items():
        deployed=BASE/'primary-host-v1/mpi'/name
        resolved=deployed.resolve(strict=True)
        check(resolved.is_relative_to(runtime.resolve()) and resolved.is_file(),'MPI runtime target changed')
        key=str(deployed.relative_to(BASE));check(key not in files,'duplicate runtime archive member')
        files[key]=resolved
        expected[key]=dict(bytes=resolved.stat().st_size,sha256=digest)
    for stage, digest in pipeline['catalogs'].items():
        check(stage in ['activity','cell','correlation','series'], 'unexpected analysis stage')
        directory = OUTPUT/stage
        check(sha(directory/'catalog.json') == digest, 'analysis catalog changed')
        catalog = read(directory/'catalog.json')
        check({p.name for p in directory.iterdir()} == set(catalog)|{'catalog.json'}, 'untracked analysis file/scratch')
        for name, row in catalog.items():
            check(Path(name).name == name, 'unsafe analysis file name')
            add(directory/name,row['sha256'],row['bytes'])
        add(directory/'catalog.json',digest)
    for path in OUTPUT.iterdir():
        if path.is_file():
            check(path.stat().st_size < 32*2**20, 'oversized pipeline log/report')
            add(path)
    source = BASE/'primary-postrun-v1-source'
    check(sha(source/'catalog.json') == pipeline['source_catalog_sha256'], 'analysis source catalog changed')
    for name,row in read(source/'catalog.json').items(): add(source/name,row['sha256'],row['bytes'])
    add(source/'catalog.json')
    add(AUDIT_OUTPUT)
    add(BASE/'guards/primary-postrun-v1.json')
    # Exact staged control/source inventory, not a recursive sweep of runtime
    # files. This includes terminal records, all-host hashes and archive code.
    staged = read(control.parent/'catalog.json')
    for name,row in staged.items(): add(control.parent/name,row['sha256'],row['bytes'])
    add(control.parent/'catalog.json')
    check(len(files) <= 1024 and sum(r['bytes'] for r in expected.values()) < MAX_BYTES, 'archive budget')
    return files, expected


def run(control, attempt_version=1):
    started = time.monotonic()
    paths = attempt_paths(attempt_version)
    archive, report, catalog = (paths[n] for n in ('archive', 'report', 'catalog'))
    check(Path('/data/brick2').is_mount() and control.resolve().is_relative_to(BASE.resolve()), 'brick2 required')
    check(control == paths['source']/'control', 'archive attempt source differs')
    check(not report.exists() and not catalog.exists() and not archive.exists(), 'archive already attempted/completed')
    files, expected = inventory(control)
    free = os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize
    check(free > 1280*2**30+MAX_BYTES, 'archive plus monitored reserve does not fit')
    with catalog.open('x') as stream:stream.write(json.dumps(expected,indent=2)+'\n')
    result = write_verified_archive(files, expected, archive)
    result.update(schema='b2-mam-primary-archive-v1',archive=str(archive),archive_verified=True,
                  attempt_version=attempt_version,
                  catalog_sha256=sha(catalog),elapsed_seconds=time.monotonic()-started,
                  scientific_acceptance=False,performance_cost_acceptance=False,
                  independent_device_backup=False,local_t7_raw_copy=False,
                  scope='Full primary recording, deployed input sources, complete observation artifacts and audited '
                        'controls archived and rehashed on brick2. Same-filesystem preservation, not independent '
                        'device redundancy or scientific equivalence; later reference comparisons remain open.')
    with report.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--attempt-version',type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    print(json.dumps(run(args.control,args.attempt_version)))
