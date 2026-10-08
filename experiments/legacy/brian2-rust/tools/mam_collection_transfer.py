"""Bounded catalog-based private-cluster collection without a source archive.

Run under the existing resource/time guard. Catalogs travel via Teleport;
file bytes stream through one private socket. A success receipt is required
before consumers may use an output tree. Failure preserves partial evidence.
"""
import argparse
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import socket
import stat
import time

from mam_direct_transfer_v3 import BLOCK, admit_disk, frame, read_frame

DEFAULT_BYTES = 64 * 2**30
MAX_FILE_BYTES = 256 * 2**30
MAX_TOTAL_BYTES = 512 * 2**30
DEFAULT_RESERVE = 128 * 2**30
SCHEMA = 'mam-direct-collection-v1'
CACHE_WINDOW = 64 * 2**20


def release_cache(stream, start, writing):
    """Keep completed file pages from accumulating inside the Linux cgroup."""
    if writing:
        stream.flush()
        os.fsync(stream.fileno())
    end = stream.tell()
    if end > start and hasattr(os, 'posix_fadvise'):
        os.posix_fadvise(stream.fileno(), start, end-start, os.POSIX_FADV_DONTNEED)
    return end


def validate(catalog, max_file_bytes=DEFAULT_BYTES, max_total_bytes=DEFAULT_BYTES):
    for value, cap in [(max_file_bytes, MAX_FILE_BYTES), (max_total_bytes, MAX_TOTAL_BYTES)]:
        if type(value) is not int or not 0 < value <= cap:
            raise ValueError('invalid collection size limit')
    if catalog.get('schema') != SCHEMA or not isinstance(catalog.get('files'), list):
        raise ValueError('invalid collection catalog')
    files = catalog['files']
    if not 0 < len(files) <= 512:
        raise ValueError('invalid collection file count')
    paths = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {'path', 'bytes', 'sha256'}:
            raise ValueError('invalid file catalog entry')
        name = item['path']
        if not isinstance(name, str) or not name or len(name.encode()) > 1024 or '\\' in name:
            raise ValueError('invalid relative file path')
        p = PurePosixPath(name)
        if not p.parts or str(p) != name or p.is_absolute() or '..' in p.parts or p.parts[0] == '.collection-parts' or '\0' in name:
            raise ValueError('unsafe or reserved collection path')
        if type(item['bytes']) is not int or not 0 <= item['bytes'] <= max_file_bytes:
            raise ValueError('file exceeds admitted limit')
        digest = item['sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid file SHA-256')
        paths.append(name)
    if paths != sorted(set(paths)):
        raise ValueError('catalog paths must be unique and sorted')
    names = set(paths)
    if any(str(parent) in names for name in paths for parent in PurePosixPath(name).parents):
        raise ValueError('file/directory path collision')
    total = sum(item['bytes'] for item in files)
    if total > max_total_bytes:
        raise ValueError('collection exceeds admitted total limit')
    canonical = json.dumps(catalog, sort_keys=True, separators=(',', ':')).encode()
    if len(canonical) > 2**20:
        raise ValueError('catalog exceeds control-plane size limit')
    return total, hashlib.sha256(canonical).hexdigest()


def source_path(root, name):
    root = root.resolve(strict=True)
    p = root / name
    if p.is_symlink() or not p.resolve(strict=True).is_relative_to(root):
        raise ValueError('source file escapes root or is a symlink')
    for parent in p.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('source parent is a symlink')
    if not p.is_file():
        raise ValueError('source is not a regular file')
    return p


def receive(bind, peer, catalog, output, ready, receipt,
            max_file_bytes=DEFAULT_BYTES, max_total_bytes=DEFAULT_BYTES,
            reserve_bytes=DEFAULT_RESERVE):
    total, digest = validate(catalog, max_file_bytes, max_total_bytes)
    if not ipaddress.ip_address(bind).is_private or not ipaddress.ip_address(peer).is_private:
        raise ValueError('private addresses required')
    if any(p.exists() or p.is_symlink() for p in [output, ready, receipt]):
        raise FileExistsError('collection destination/control already exists')
    free_before = admit_disk(output.parent, total, reserve_bytes)
    started = time.monotonic()
    token = secrets.token_hex(32)
    with socket.socket() as listener:
        listener.bind((bind, 0))
        listener.listen(1)
        listener.settimeout(90)
        control = dict(schema=SCHEMA, address=bind, port=listener.getsockname()[1],
                       token=token, catalog_sha256=digest, files=len(catalog['files']),
                       bytes=total, max_file_bytes=max_file_bytes,
                       max_total_bytes=max_total_bytes, reserve_bytes=reserve_bytes)
        temporary = ready.with_name(ready.name + '.writing')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(control, stream)
        os.link(temporary, ready)
        temporary.unlink()
        connection, address = listener.accept()
        with connection:
            connection.settimeout(30)
            header = read_frame(connection)
            if address[0] != peer or not hmac.compare_digest(header['token'], token) or header['catalog_sha256'] != digest:
                raise ValueError('peer, capability or catalog mismatch')
            admit_disk(output.parent, total, reserve_bytes)
            output.mkdir()
            parts = output / '.collection-parts'
            parts.mkdir()
            for index, item in enumerate(catalog['files']):
                if read_frame(connection) != dict(index=index, **item):
                    raise ValueError('file header does not match catalog order')
                target = output / item['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.parent.resolve().is_relative_to(output.resolve()):
                    raise ValueError('destination parent escapes collection')
                partial = parts / str(index)
                remaining = item['bytes']
                file_hash = hashlib.sha256()
                with partial.open('xb') as stream:
                    cache_start = 0
                    while remaining:
                        block = connection.recv(min(BLOCK, remaining,
                                                    CACHE_WINDOW - (stream.tell() - cache_start)))
                        if not block:
                            raise EOFError('file payload truncated')
                        admit_disk(output.parent, len(block), reserve_bytes)
                        stream.write(block)
                        file_hash.update(block)
                        remaining -= len(block)
                        if stream.tell() - cache_start >= CACHE_WINDOW:
                            cache_start = release_cache(stream, cache_start, writing=True)
                    release_cache(stream, cache_start, writing=True)
                if file_hash.hexdigest() != item['sha256']:
                    raise ValueError('file payload hash mismatch')
                os.link(partial, target)
                partial.unlink()
            if connection.recv(1) != b'':
                raise ValueError('unexpected trailing collection payload')
            parts.rmdir()
            result = dict(schema=SCHEMA, complete=True, files=len(catalog['files']),
                          bytes=total, catalog_sha256=digest, output=str(output),
                          wall_seconds=time.monotonic()-started, peer=peer,
                          max_file_bytes=max_file_bytes, max_total_bytes=max_total_bytes,
                          reserve_bytes=reserve_bytes, free_bytes_before=free_before,
                          cache_window_bytes=CACHE_WINDOW,
                          file_cache_release_supported=hasattr(os, 'posix_fadvise'),
                          source_archive_created=False)
            with receipt.open('x') as stream:
                json.dump(result, stream, indent=2)
                stream.write('\n')
            frame(connection, result)
    return result


def send(root, catalog, ready):
    total, digest = validate(catalog, ready['max_file_bytes'], ready['max_total_bytes'])
    if ready['schema'] != SCHEMA or digest != ready['catalog_sha256'] or total != ready['bytes'] or len(catalog['files']) != ready['files']:
        raise ValueError('sender catalog differs from admitted catalog')
    started = time.monotonic()
    with socket.create_connection((ready['address'], ready['port']), timeout=30) as connection:
        connection.settimeout(30)
        frame(connection, dict(token=ready['token'], catalog_sha256=digest))
        for index, item in enumerate(catalog['files']):
            p = source_path(root, item['path'])
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                st = os.fstat(stream.fileno())
                if not stat.S_ISREG(st.st_mode) or st.st_size != item['bytes']:
                    raise ValueError('source type or size changed')
                frame(connection, dict(index=index, **item))
                file_hash = hashlib.sha256()
                remaining = item['bytes']
                cache_start = 0
                while remaining:
                    block = stream.read(min(BLOCK, remaining,
                                            CACHE_WINDOW - (stream.tell() - cache_start)))
                    if not block:
                        raise EOFError('source truncated during transfer')
                    connection.sendall(block)
                    file_hash.update(block)
                    remaining -= len(block)
                    if stream.tell() - cache_start >= CACHE_WINDOW:
                        cache_start = release_cache(stream, cache_start, writing=False)
                release_cache(stream, cache_start, writing=False)
                if stream.read(1) or file_hash.hexdigest() != item['sha256']:
                    raise ValueError('source content changed')
        connection.shutdown(socket.SHUT_WR)
        result = read_frame(connection)
        if not result['complete'] or result['catalog_sha256'] != digest or result['bytes'] != total or result['files'] != len(catalog['files']):
            raise ValueError('invalid collection completion receipt')
    return dict(schema=SCHEMA, complete=True, files=result['files'], bytes=total,
                catalog_sha256=digest, wall_seconds=time.monotonic()-started,
                cache_window_bytes=CACHE_WINDOW,
                file_cache_release_supported=hasattr(os, 'posix_fadvise'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['send', 'receive'])
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--ready', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--bind')
    parser.add_argument('--peer')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--max-file-bytes', type=int, default=DEFAULT_BYTES)
    parser.add_argument('--max-total-bytes', type=int, default=DEFAULT_BYTES)
    parser.add_argument('--reserve-bytes', type=int, default=DEFAULT_RESERVE)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    if args.mode == 'send':
        result = send(args.root, catalog, json.loads(args.ready.read_text()))
    else:
        result = receive(args.bind, args.peer, catalog, args.output, args.ready, args.receipt,
                         args.max_file_bytes, args.max_total_bytes, args.reserve_bytes)
    print(json.dumps(result, indent=2))
