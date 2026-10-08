"""Boundaries and real-socket integrity for archive-free collection."""
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import time

import pytest


spec = importlib.util.spec_from_file_location(
    "mam_transfer_v3", Path(__file__).parents[1] / "tools/mam_direct_transfer_v3.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def expected(size, data=b"fixture"):
    return dict(bytes=size, sha256=hashlib.sha256(data).hexdigest())


def test_large_file_requires_explicit_bounded_admission():
    item = expected(100 * 2**30)
    with pytest.raises(ValueError):
        transfer.validate_expected(item)
    transfer.validate_expected(item, 100 * 2**30)
    for limit in [True, 0, -1, 256 * 2**30 + 1]:
        with pytest.raises(ValueError):
            transfer.validate_expected(item, limit)
    with pytest.raises(ValueError):
        transfer.validate_expected(expected(100 * 2**30 + 1), 100 * 2**30)


def test_disk_admission_includes_retained_reserve(tmp_path, monkeypatch):
    class Volume:
        f_bavail = 100
        f_frsize = 1
    monkeypatch.setattr(transfer.os, "statvfs", lambda _: Volume())
    assert transfer.admit_disk(tmp_path, 60, 40) == 100
    with pytest.raises(ValueError):
        transfer.admit_disk(tmp_path, 61, 40)
    with pytest.raises(ValueError):
        transfer.admit_disk(tmp_path, 1, -1)


def test_disk_rejection_precedes_listener_and_output_creation(tmp_path, monkeypatch):
    class Volume:
        f_bavail = 100
        f_frsize = 1
    monkeypatch.setattr(transfer.os, "statvfs", lambda _: Volume())
    with pytest.raises(ValueError):
        transfer.receive("127.0.0.1", "127.0.0.1", expected(61),
                         tmp_path / "out", tmp_path / "ready", tmp_path / "receipt",
                         max_bytes=61, reserve_bytes=40)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", [None, "hash", "truncated", "extra", "competing_output", "disk_drop", "token"])
def test_direct_socket_transfer(tmp_path, failure, monkeypatch):
    data = bytes(range(256)) * 8192 + b"tail"
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    output, ready, receipt = [tmp_path / n for n in ["output.bin", "ready.json", "receipt.json"]]
    item = expected(len(data), data)
    if failure == "hash":
        item["sha256"] = "0" * 64
    if failure == "disk_drop":
        calls = 0
        class Volume:
            f_bavail = 0
            f_frsize = 1
        def disk_space(_):
            nonlocal calls
            calls += 1
            result = Volume()
            result.f_bavail = len(data) * 4 if calls <= 3 else 0
            return result
        monkeypatch.setattr(transfer.os, "statvfs", disk_space)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        receiver = pool.submit(transfer.receive, "127.0.0.1", "127.0.0.1", item,
                               output, ready, receipt, len(data), 0)
        deadline = time.monotonic() + 5
        while not ready.exists():
            if receiver.done():
                receiver.result()
            assert time.monotonic() < deadline
            time.sleep(.01)
        control = json.loads(ready.read_text())
        assert ready.stat().st_mode & 0o777 == 0o600
        if failure is None:
            sent = transfer.send(source, control)
            result = receiver.result(timeout=5)
            assert sent["complete"] and result["protocol_version"] == 3
            assert not result["source_archive_created"]
            assert output.read_bytes() == source.read_bytes() == data
            assert not output.with_name(output.name + ".partial").exists()
            assert json.loads(receipt.read_text())["sha256"] == item["sha256"]
            assert not list(tmp_path.glob("*.tar*"))
        else:
            if failure == "competing_output":
                output.write_bytes(b"preserve me")
            with socket.create_connection((control["address"], control["port"]), timeout=5) as connection:
                header = {k: control[k] for k in ["token", "bytes", "sha256"]}
                if failure == "token":
                    header["token"] = "0" * 64
                transfer.frame(connection, header)
                payload = data[:-1] if failure == "truncated" else data
                try:
                    if failure != "token":
                        connection.sendall(payload + (b"extra" if failure == "extra" else b""))
                    connection.shutdown(socket.SHUT_WR)
                except (BrokenPipeError, ConnectionResetError):
                    assert failure == "disk_drop"
                with pytest.raises((AssertionError, EOFError, FileExistsError, ValueError)):
                    receiver.result(timeout=5)
            assert not receipt.exists()
            if failure == "competing_output":
                assert output.read_bytes() == b"preserve me"
            else:
                assert not output.exists()
            assert source.read_bytes() == data
