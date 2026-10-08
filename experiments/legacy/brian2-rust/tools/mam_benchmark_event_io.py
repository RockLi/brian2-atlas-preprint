"""Bounded Linux event output for a future explicitly configured MAM benchmark.

Eight-byte records are passed through unchanged. This module does not initialize
NEST, change a model, or make old non-durable outputs durable retroactively.
"""
import hashlib
import os
from pathlib import Path
import struct
import sys
import time

BUFFER_BYTES = 65536
SYNC_BYTES = 1048576


class LinuxIO:
    def __init__(self):
        if (sys.platform != 'linux' or struct.calcsize('P') != 8
                or not hasattr(os, 'fdatasync') or not hasattr(os, 'posix_fadvise')):
            raise RuntimeError('Durable event benchmark requires Linux64 fdatasync/fadvise')

    def sync_data(self, fd):
        os.fdatasync(fd)

    def release(self, fd, offset, length):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)

    def sync_directory(self, path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class DurableEventWriter:
    """One new stream, <=1 MiB between data syncs, final directory sync.

    A failure poisons the writer. The incomplete file is retained, close() can
    release its descriptor, and finish() cannot produce a success receipt.
    IO dependency injection is for fault tests, not exposed as a benchmark CLI.
    Timings are local writer timings; append/finish include their nested sync
    counters and must not be added to those counters a second time.
    """
    def __init__(self, path, maximum_bytes, *, io=None):
        if (type(maximum_bytes) is not int or not 0 < maximum_bytes <= 512*2**30
                or maximum_bytes % 8):
            raise ValueError('Event byte ceiling must be an aligned integer in 8..512 GiB')
        self.path = Path(path)
        self.io = LinuxIO() if io is None else io
        self.maximum_bytes = maximum_bytes
        self.written = self.durable_bytes = self.released_bytes = 0
        self.poisoned = self.finished = False
        self.digest = hashlib.sha256()
        self.timings = dict(append_wall_seconds=0., finish_wall_seconds=0.,
                            write_seconds=0., data_sync_seconds=0.,
                            cache_release_seconds=0., directory_sync_seconds=0.)
        self.data_sync_calls = self.cache_release_calls = 0
        self.stream = self.path.open('xb', buffering=BUFFER_BYTES)

    def _require_live(self):
        if self.poisoned or self.finished or self.stream.closed:
            raise RuntimeError('Event writer is poisoned, finished, or closed')

    def _sync(self):
        start = time.perf_counter()
        self.stream.flush()
        self.io.sync_data(self.stream.fileno())
        self.timings['data_sync_seconds'] += time.perf_counter()-start
        self.data_sync_calls += 1
        self.durable_bytes = self.written
        start = time.perf_counter()
        self.io.release(self.stream.fileno(), self.released_bytes,
                        self.written-self.released_bytes)
        self.timings['cache_release_seconds'] += time.perf_counter()-start
        self.cache_release_calls += 1
        self.released_bytes = self.written

    def append(self, records):
        self._require_live()
        start = time.perf_counter()
        try:
            data = memoryview(records).cast('B')
            if len(data) % 8:
                raise ValueError('Incomplete eight-byte event record')
            if len(data) > self.maximum_bytes-self.written:
                raise ValueError('Event budget exceeded before write')
            offset = 0
            while offset < len(data):
                size = min(len(data)-offset, BUFFER_BYTES,
                           SYNC_BYTES-(self.written-self.durable_bytes))
                tick = time.perf_counter()
                count = self.stream.write(data[offset:offset+size])
                self.timings['write_seconds'] += time.perf_counter()-tick
                if type(count) is not int or count <= 0 or count > size:
                    raise OSError('Invalid event write count')
                self.digest.update(data[offset:offset+count])
                self.written += count
                offset += count
                if count != size:
                    raise OSError('Short event write; partial evidence retained')
                if self.written-self.durable_bytes == SYNC_BYTES:
                    self._sync()
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.timings['append_wall_seconds'] += time.perf_counter()-start

    def finish(self):
        self._require_live()
        start = time.perf_counter()
        try:
            if self.written != self.durable_bytes or self.data_sync_calls == 0:
                self._sync()
            if os.fstat(self.stream.fileno()).st_size != self.written:
                raise OSError('Event file size differs from written bytes')
            self.stream.close()
            tick = time.perf_counter()
            self.io.sync_directory(self.path.parent)
            self.timings['directory_sync_seconds'] += time.perf_counter()-tick
            self.finished = True
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.timings['finish_wall_seconds'] += time.perf_counter()-start
        return dict(schema='b2-mam-durable-event-stream-v1', complete=True,
                    path=str(self.path), bytes=self.written, records=self.written//8,
                    sha256=self.digest.hexdigest(), durable_bytes=self.durable_bytes,
                    maximum_bytes=self.maximum_bytes, buffer_bytes=BUFFER_BYTES,
                    data_sync_interval_bytes=SYNC_BYTES, data_sync_calls=self.data_sync_calls,
                    cache_release_calls=self.cache_release_calls, directory_synced=True,
                    io_backend=type(self.io).__name__, timings=dict(self.timings),
                    scope='Local event-stream durability receipt, not simulator or scientific acceptance')

    def close(self):
        """Release an incomplete descriptor; never deletes or declares success."""
        if not self.finished:
            self.poisoned = True
        self.stream.close()
