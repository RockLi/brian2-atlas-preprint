"""Bounded event adapters for the complete retained-data correlation analysis."""
from contextlib import contextmanager
import hashlib
import mmap
import os
import numpy as np
from mam_paper_correlation import CandidateHistogram, summarize_histogram_bounded, MAX_STREAM_EVENTS, PAPER_END_TICKS

CHUNK = 131072
CACHE_CHUNK = 64 * 2**20


def require_cache_release():
    if not all(hasattr(os, name) for name in ('posix_fadvise', 'POSIX_FADV_DONTNEED', 'fdatasync')) or not hasattr(mmap, 'MADV_DONTNEED'):
        raise RuntimeError('bounded file-cache mode requires Linux/POSIX cache advice')


def file_sha(path):
    require_cache_release()
    h = hashlib.sha256()
    with path.open('rb') as f:
        offset = 0
        while block := f.read(2**20):
            h.update(block)
            if f.tell() - offset >= CACHE_CHUNK:
                os.posix_fadvise(f.fileno(), offset, f.tell()-offset, os.POSIX_FADV_DONTNEED)
                offset = f.tell()
        os.posix_fadvise(f.fileno(), offset, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


@contextmanager
def mapped_cache(mapping, path):
    """Release processed views of an existing mapping; never close its mmap."""
    require_cache_release()
    if not isinstance(mapping, np.memmap) or mapping.dtype != np.dtype('u1') or mapping.ndim != 1:
        raise ValueError('requires the result byte mapping')
    if not os.path.samefile(mapping.filename, path):
        raise ValueError('cache file does not match mapping')
    fd = os.open(path, os.O_RDONLY)
    def release(values):
        if not len(values):
            return
        start = values.ctypes.data-mapping.ctypes.data
        end = start+(len(values)-1)*values.strides[0]+values.dtype.itemsize
        if not 0 <= start < end <= mapping.size or values.strides[0] <= 0:
            raise ValueError('event view outside mapping')
        lo, hi = start//mmap.PAGESIZE*mmap.PAGESIZE, end//mmap.PAGESIZE*mmap.PAGESIZE
        if hi > lo:
            mapping._mmap.madvise(mmap.MADV_DONTNEED, lo, hi-lo)
            os.posix_fadvise(fd, lo, hi-lo, os.POSIX_FADV_DONTNEED)
    try:
        yield release
    finally:
        try:
            mapping._mmap.madvise(mmap.MADV_DONTNEED)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def rust_blocks(population, release):
    ticks, cells = population['spike_ticks'], population['indices']
    if ticks.shape != cells.shape:
        raise ValueError('mismatched Rust event arrays')
    for start in range(0, len(ticks), CHUNK):
        raw, ids = ticks[start:start+CHUNK], cells[start:start+CHUNK]
        try:
            yield raw.astype(np.int64)+1, ids
        finally:
            release(raw)
            release(ids)


def native_blocks(path, dtype):
    require_cache_release()
    if path.stat().st_size % dtype.itemsize:
        raise ValueError('partial native event record')
    with path.open('rb') as f:
        while True:
            offset = f.tell()
            block = np.fromfile(f, dtype=dtype, count=CHUNK)
            if not len(block):
                break
            try:
                yield block['tick'], block['cell']
            finally:
                os.posix_fadvise(f.fileno(), offset, f.tell()-offset, os.POSIX_FADV_DONTNEED)


def summarize_population(blocks, *, neurons, end_tick, expected_raw, frozen):
    """Two bounded passes preserve the lowest recorded ID, including warmup."""
    if type(expected_raw) is not int or not 0 <= expected_raw <= MAX_STREAM_EVENTS:
        raise ValueError('population exceeds bounded event budget')
    if type(neurons) is not int or neurons <= 0 or type(end_tick) is not int or end_tick not in PAPER_END_TICKS:
        raise ValueError('invalid population size or declared duration')
    first, total = None, 0
    for ticks, cells in blocks():
        if (ticks.ndim != 1 or cells.shape != ticks.shape or len(ticks) > CHUNK
                or ticks.dtype.kind not in 'iu' or cells.dtype.kind not in 'iu'
                or np.any(ticks < 0) or np.any(ticks > end_tick)
                or np.any(cells < 0) or np.any(cells >= neurons)):
            raise ValueError('invalid population event block')
        total += len(ticks)
        if total > expected_raw:
            raise ValueError('population event count exceeded')
        if len(cells):
            value = int(cells.min())
            first = value if first is None else min(first, value)
    if total != expected_raw:
        raise ValueError('population event count mismatch')
    observed = np.zeros((end_tick-5000)//10, dtype=np.int64)
    if first is None:
        np.testing.assert_array_equal(observed, frozen)
        return dict(available=False, mean_pairwise_correlation=None, selected_cells=0,
                    unavailable_reason='Official wrapper accesses ids[0] on an empty population.'), {}, observed
    counter = CandidateHistogram(first, end_tick=end_tick)
    minimum = None
    for ticks, cells in blocks():
        counter.add(ticks, cells)
        if np.any(cells >= neurons):
            raise ValueError('cell outside population')
        if len(cells):
            value = int(cells.min())
            minimum = value if minimum is None else min(minimum, value)
        keep = (ticks >= 5000) & (ticks < end_tick)
        np.add.at(observed, (ticks[keep].astype(np.int64)-5000)//10, 1)
    if counter.raw_events != expected_raw or minimum != first:
        raise ValueError('event source changed between passes')
    np.testing.assert_array_equal(observed, frozen)
    report, ids, selected = summarize_histogram_bounded(counter.counts, first)
    samples = dict(selected_ids=ids, selected_spikes=selected.sum(axis=1, dtype=np.int64),
                   selected_histogram_sum=selected.sum(axis=0, dtype=np.int64))
    return report, samples, observed
