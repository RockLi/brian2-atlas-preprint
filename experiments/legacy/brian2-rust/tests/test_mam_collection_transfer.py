"""Collection-level admission and real socket failures, including partial trees."""
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
import mam_collection_transfer as transfer


def catalog(data):
    return dict(schema=transfer.SCHEMA, files=[dict(path=n, bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for n, v in sorted(data.items())])


@pytest.mark.parametrize('path', ['', '.', '../escape', '/absolute', 'a//b', 'a/./b', 'a/../b', 'a/', '.collection-parts/0', 'a\\b', 'a\0b'])
def test_catalog_rejects_unsafe_paths(path):
    with pytest.raises(ValueError):
        transfer.validate(catalog({path: b'x'}))


def test_catalog_caps_and_namespace():
    c = catalog({'a': b'x', 'b': b'y'})
    for item in c['files']:
        item['bytes'] = 100 * 2**30
    with pytest.raises(ValueError):
        transfer.validate(c)
    assert transfer.validate(c, 128 * 2**30, 200 * 2**30)[0] == 200 * 2**30
    with pytest.raises(ValueError):
        transfer.validate(c, 128 * 2**30, 200 * 2**30 - 1)
    for kwargs in [dict(max_file_bytes=True), dict(max_file_bytes=257 * 2**30), dict(max_total_bytes=513 * 2**30)]:
        with pytest.raises(ValueError):
            transfer.validate(c, **kwargs)
    for bad in [catalog({'a': b'x', 'a/b': b'y'}), catalog({}), catalog({str(i): b'' for i in range(513)})]:
        with pytest.raises(ValueError):
            transfer.validate(bad)
    c = catalog({'a': b'x'})
    c['files'].append(dict(c['files'][0]))
    with pytest.raises(ValueError):
        transfer.validate(c)


def test_source_symlink_is_not_followed(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (tmp_path / 'outside').write_bytes(b'x')
    (source / 'link').symlink_to(tmp_path / 'outside')
    with pytest.raises(ValueError):
        transfer.source_path(source, 'link')


def test_initial_disk_rejection_leaves_no_tree(tmp_path, monkeypatch):
    class Volume:
        f_bavail, f_frsize = 0, 1
    monkeypatch.setattr(transfer, 'admit_disk', lambda *a: (_ for _ in ()).throw(ValueError('space')))
    with pytest.raises(ValueError):
        transfer.receive('127.0.0.1', '127.0.0.1', catalog({'a': b'x'}),
                         tmp_path / 'out', tmp_path / 'ready', tmp_path / 'receipt')
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('failure', [None, 'wrong_order', 'hash', 'truncated', 'extra', 'source_change', 'disk_drop', 'competing_receipt'])
def test_collection_socket_roundtrip(tmp_path, failure, monkeypatch):
    # Exercise cache release inside a file, not only at its final flush.
    monkeypatch.setattr(transfer, 'CACHE_WINDOW', 2**20)
    release = transfer.release_cache
    releases = []
    def observed_release(stream, start, writing):
        end = release(stream, start, writing)
        releases.append((writing, start, end))
        return end
    monkeypatch.setattr(transfer, 'release_cache', observed_release)
    data = {'a.txt': b'header', 'nested/data.bin': bytes(range(256)) * 8192 + b'end', 'z-empty': b''}
    c = catalog(data)
    source = tmp_path / 'source'
    source.mkdir()
    for name, raw in data.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    output, ready, receipt = [tmp_path / n for n in ['output', 'ready', 'receipt']]
    if failure == 'disk_drop':
        original = transfer.admit_disk
        calls = 0
        def remaining_space(*args):
            nonlocal calls
            calls += 1
            if calls >= 4:
                raise ValueError('concurrent writer exhausted reserve')
            return original(*args)
        monkeypatch.setattr(transfer, 'admit_disk', remaining_space)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(transfer.receive, '127.0.0.1', '127.0.0.1', c,
                             output, ready, receipt, 3 * 2**20, 4 * 2**20, 0)
        deadline = time.monotonic() + 5
        while not ready.exists():
            if future.done():
                future.result()
            assert time.monotonic() < deadline
            time.sleep(.01)
        control = json.loads(ready.read_text())
        assert ready.stat().st_mode & 0o777 == 0o600
        if failure is None:
            sent = transfer.send(source, c, control)
            result = future.result(timeout=5)
            assert sent['complete'] and result['complete'] and result['files'] == 3
            assert json.loads(receipt.read_text())['catalog_sha256'] == transfer.validate(c)[1]
            assert sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()) == sorted(data)
            assert not (output / '.collection-parts').exists()
            for name, raw in data.items():
                assert (output / name).read_bytes() == raw == (source / name).read_bytes()
            assert not list(tmp_path.rglob('*.tar*'))
            assert all(end - start <= 2**20 for _, start, end in releases)
            assert any(writing and start >= 2**20 for writing, start, end in releases)
            assert any(not writing and start >= 2**20 for writing, start, end in releases)
        elif failure == 'source_change':
            (source / 'nested/data.bin').write_bytes(b'X' * len(data['nested/data.bin']))
            with pytest.raises(ValueError):
                transfer.send(source, c, control)
            with pytest.raises(ValueError):
                future.result(timeout=5)
            assert not receipt.exists() and not (output / 'nested/data.bin').exists()
        else:
            if failure == 'competing_receipt':
                receipt.write_text('preserve')
            with socket.create_connection((control['address'], control['port']), timeout=5) as s:
                transfer.frame(s, dict(token=control['token'], catalog_sha256=control['catalog_sha256']))
                try:
                    for i, item in enumerate(c['files']):
                        transfer.frame(s, dict(index=(i + 1 if failure == 'wrong_order' else i), **item))
                        raw = data[item['path']]
                        if i == 1 and failure == 'hash':
                            raw = b'X' * len(raw)
                        if i == 1 and failure == 'truncated':
                            s.sendall(raw[:-1])
                            break
                        s.sendall(raw)
                    if failure == 'extra':
                        s.sendall(b'extra')
                    s.shutdown(socket.SHUT_WR)
                except (BrokenPipeError, ConnectionResetError):
                    assert failure in ['wrong_order', 'hash', 'disk_drop']
                with pytest.raises((ValueError, EOFError, FileExistsError)):
                    future.result(timeout=5)
            if failure == 'competing_receipt':
                assert receipt.read_text() == 'preserve'
            else:
                assert not receipt.exists()
            for name, raw in data.items():
                assert (source / name).read_bytes() == raw
