"""Version 3: direct single-file transfer with explicit disk and size admission.

Run under the existing cgroup guard. The receiver binds one explicit private
address, accepts only the declared peer and a one-use random capability, and
closes after one file. Integrity expectations travel through Teleport control.
"""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import time

# Transfer the existing raw file directly; do not create a source tar.
# Larger ceilings require explicit admission. External cgroup, timeout and
# RLIMIT_FSIZE guards remain mandatory; this protocol does not reserve space.
DEFAULT_MAX_BYTES = 64 * 2**30
MAX_BYTES = 256 * 2**30
DEFAULT_RESERVE_BYTES = 128 * 2**30
BLOCK = 2**20


def exact(connection, count):
    chunks = bytearray()
    while len(chunks) < count:
        part = connection.recv(count-len(chunks))
        if not part:
            raise EOFError('truncated transfer')
        chunks.extend(part)
    return bytes(chunks)


def frame(connection, value):
    data = json.dumps(value).encode()
    assert len(data) <= 4096
    connection.sendall(struct.pack('!I', len(data)) + data)


def read_frame(connection):
    length = struct.unpack('!I', exact(connection, 4))[0]
    assert 0 < length <= 4096
    return json.loads(exact(connection, length))


def validate_expected(expected, max_bytes=DEFAULT_MAX_BYTES):
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError("transfer limit must be explicit and at most 256 GiB")
    if type(expected["bytes"]) is not int or not 0 < expected["bytes"] <= max_bytes:
        raise ValueError("file exceeds admitted transfer limit")
    assert len(expected['sha256']) == 64
    bytes.fromhex(expected['sha256'])


def admit_disk(directory, size, reserve_bytes):
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("invalid free-space reserve")
    volume = os.statvfs(directory)
    available = volume.f_bavail * volume.f_frsize
    if available < size + reserve_bytes:
        raise ValueError("file plus reserve exceeds available destination space")
    return available


def receive(bind, peer, expected, output, ready, receipt,
            max_bytes=DEFAULT_MAX_BYTES, reserve_bytes=DEFAULT_RESERVE_BYTES):
    validate_expected(expected, max_bytes)
    free_before = admit_disk(output.parent, expected['bytes'], reserve_bytes)
    import ipaddress
    assert ipaddress.ip_address(bind).is_private and ipaddress.ip_address(peer).is_private
    assert not output.exists() and not ready.exists() and not receipt.exists()
    partial = output.with_name(output.name + '.partial')
    assert not partial.exists()
    token = secrets.token_hex(32)
    started = time.monotonic()
    with socket.socket() as listener:
        listener.bind((bind, 0))
        listener.listen(1)
        listener.settimeout(90)
        data = dict(address=bind, port=listener.getsockname()[1], token=token,
                    bytes=expected['bytes'], sha256=expected['sha256'],
                    max_bytes=max_bytes, reserve_bytes=reserve_bytes)
        # The token is never printed. Retain the protected file for the
        # Teleport-authenticated controller; it expires when this socket closes.
        temporary = ready.with_name(ready.name + '.writing')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream)
        temporary.rename(ready)
        connection, address = listener.accept()
        with connection:
            connection.settimeout(30)
            assert address[0] == peer, 'unexpected peer'
            header = read_frame(connection)
            assert hmac.compare_digest(header['token'], token), 'invalid capability'
            assert header['bytes'] == expected['bytes'] and header['sha256'] == expected['sha256']
            admit_disk(output.parent, expected['bytes'], reserve_bytes)
            digest = hashlib.sha256()
            remaining = expected['bytes']
            with partial.open('xb') as stream:
                while remaining:
                    block = connection.recv(min(BLOCK, remaining))
                    if not block:
                        raise EOFError('payload ended early')
                    admit_disk(output.parent, len(block), reserve_bytes)
                    stream.write(block)
                    digest.update(block)
                    remaining -= len(block)
                stream.flush()
                os.fsync(stream.fileno())
            assert connection.recv(1) == b'', 'unexpected extra payload'
            assert digest.hexdigest() == expected['sha256'], 'payload hash mismatch'
            assert partial.stat().st_size == expected['bytes']
            # Atomically publish without overwriting a competing destination.
            os.link(partial, output)
            partial.unlink()
            result = dict(bytes=expected['bytes'], sha256=digest.hexdigest(),
                          wall_seconds=time.monotonic()-started, peer=peer,
                          path=str(output), complete=True, protocol_version=3,
                          max_bytes=max_bytes, reserve_bytes=reserve_bytes,
                          free_bytes_before=free_before,
                          source_archive_created=False)
            receipt.write_text(json.dumps(result, indent=2) + '\n')
            frame(connection, result)
    return result


def send(source, ready):
    validate_expected(ready, ready.get('max_bytes', DEFAULT_MAX_BYTES))
    assert source.stat().st_size == ready['bytes']
    started = time.monotonic()
    with socket.create_connection((ready['address'], ready['port']), timeout=30) as connection:
        connection.settimeout(30)
        frame(connection, {key: ready[key] for key in ['token', 'bytes', 'sha256']})
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            while block := stream.read(BLOCK):
                digest.update(block)
                connection.sendall(block)
        connection.shutdown(socket.SHUT_WR)
        result = read_frame(connection)
        assert digest.hexdigest() == ready['sha256'] == result['sha256']
        assert result['bytes'] == ready['bytes'] and result['complete']
    return dict(bytes=result['bytes'], sha256=result['sha256'],
                wall_seconds=time.monotonic()-started, complete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['receive', 'send'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--ready', required=True, type=Path)
    parser.add_argument('--bind')
    parser.add_argument('--peer')
    parser.add_argument('--expected', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--max-bytes', type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument('--reserve-bytes', type=int, default=DEFAULT_RESERVE_BYTES)
    args = parser.parse_args()
    if args.mode == 'receive':
        result = receive(args.bind, args.peer, json.loads(args.expected.read_text()),
                         args.output, args.ready, args.receipt, args.max_bytes, args.reserve_bytes)
    else:
        result = send(args.source, json.loads(args.ready.read_text()))
    print(json.dumps(result, indent=2))
