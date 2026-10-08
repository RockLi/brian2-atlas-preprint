import hashlib
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from mam_benchmark_event_io import DurableEventWriter, SYNC_BYTES


class TraceIO:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def sync_data(self, fd):
        self.calls.append(('sync',))
        if self.fail == 'sync':
            raise OSError('injected sync failure')

    def release(self, fd, offset, length):
        self.calls.append(('release', offset, length))
        if self.fail == 'release':
            raise OSError('injected cache release failure')

    def sync_directory(self, path):
        self.calls.append(('directory',))
        if self.fail == 'directory':
            raise OSError('injected directory sync failure')


def test_bytes_sync_boundaries_and_tail(tmp_path):
    data = struct.pack('<II', 1005000, 4129923)*(SYNC_BYTES//8+3)
    io = TraceIO();path = tmp_path/'events.bin'
    writer = DurableEventWriter(path, 2*SYNC_BYTES, io=io)
    writer.append(data[:16]);writer.append(memoryview(data)[16:])
    assert writer.durable_bytes == SYNC_BYTES
    result = writer.finish()
    assert path.read_bytes() == data
    assert result['sha256'] == hashlib.sha256(data).hexdigest()
    assert result['records'] == len(data)//8 and result['durable_bytes'] == len(data)
    assert io.calls == [('sync',), ('release', 0, SYNC_BYTES),
                        ('sync',), ('release', SYNC_BYTES, 24), ('directory',)]
    with pytest.raises(RuntimeError):writer.append(data[:8])
    with pytest.raises(RuntimeError):writer.finish()


@pytest.mark.parametrize('failure', ['sync', 'release', 'directory'])
def test_io_failure_never_yields_success_or_deletes_prefix(tmp_path, failure):
    path=tmp_path/'events.bin';writer=DurableEventWriter(path, SYNC_BYTES, io=TraceIO(failure))
    writer.append(struct.pack('<II', 5000, 17))
    with pytest.raises(OSError):writer.finish()
    assert writer.poisoned and not writer.finished and path.exists()
    with pytest.raises(RuntimeError):writer.finish()
    writer.close()
    assert path.read_bytes() == struct.pack('<II', 5000, 17)


@pytest.mark.parametrize('bad', [b'x', bytes(16)])
def test_invalid_input_or_budget_refused_before_additional_write(tmp_path, bad):
    path=tmp_path/'events.bin';writer=DurableEventWriter(path, 8, io=TraceIO())
    writer.append(b'12345678')
    with pytest.raises(ValueError):writer.append(bad)
    with pytest.raises(RuntimeError):writer.finish()
    writer.close()
    assert path.read_bytes() == b'12345678'


def test_existing_file_is_never_replaced(tmp_path):
    path=tmp_path/'events.bin';path.write_bytes(b'original')
    with pytest.raises(FileExistsError):DurableEventWriter(path, 8, io=TraceIO())
    assert path.read_bytes() == b'original'


def test_zero_events_still_persist_an_empty_stream(tmp_path):
    io=TraceIO();path=tmp_path/'events.bin';writer=DurableEventWriter(path, 8, io=io)
    r=writer.finish()
    assert r['bytes']==r['records']==0 and r['sha256']==hashlib.sha256(b'').hexdigest()
    assert path.read_bytes()==b'' and io.calls==[('sync',),('release',0,0),('directory',)]


def test_short_write_is_poisoned_and_retained(tmp_path):
    path=tmp_path/'events.bin';writer=DurableEventWriter(path, 16, io=TraceIO())
    original=writer.stream
    class Short:
        closed=False
        def write(self,data):original.write(data[:3]);return 3
        def close(self):original.close();self.closed=True
    writer.stream=Short()
    with pytest.raises(OSError, match='Short'):writer.append(b'12345678')
    with pytest.raises(RuntimeError):writer.finish()
    writer.close()
    assert path.read_bytes()==b'123'
