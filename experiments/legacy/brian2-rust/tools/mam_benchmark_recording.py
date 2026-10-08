"""Versioned NEST benchmark recording lifecycle; does not construct a model."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from mam_benchmark_event_io import DurableEventWriter, LinuxIO, BUFFER_BYTES

DT = .1
EVENT_DTYPE = np.dtype([('tick', '<u4'), ('cell', '<u4')])


def durable_json(path, value):
    """Exclusive, bounded receipt. Successful process termination is also required."""
    io = LinuxIO()
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    if len(raw) > 8*2**20:
        raise ValueError('Benchmark receipt exceeds 8 MiB')
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        io.sync_data(stream.fileno())
    io.sync_directory(path.parent)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def verify_event_file(path, expected_bytes, expected_sha):
    """Read back all event bytes with bounded buffers and advisory cache release."""
    io = LinuxIO()
    digest = hashlib.sha256()
    count = 0
    with Path(path).open('rb', buffering=0) as stream:
        if os.fstat(stream.fileno()).st_size != expected_bytes:
            raise OSError('Event verification size mismatch')
        while data := stream.read(BUFFER_BYTES):
            digest.update(data)
            io.release(stream.fileno(), count, len(data))
            count += len(data)
    if count != expected_bytes or digest.hexdigest() != expected_sha:
        raise OSError('Event verification content mismatch')
    return digest.hexdigest()


def record_chunks(nest, recorder, args, rank, cell_count, memory_snapshot):
    """Same chunk/event semantics as the reference, with explicit durable finish.

    Callback injection is for lifecycle tests only. A successful receipt describes
    this rank's event output, not all ranks, final state output, or equivalence.
    """
    started = time.perf_counter()
    chunks, memory, total_spikes, elapsed_ms = [], {}, 0, 0.
    progress_path = args.output / f'rank{rank}.progress.jsonl'
    event_path = args.output / f'rank{rank}.events.bin'
    writer = DurableEventWriter(event_path, args.max_spikes_per_rank*8)
    io = LinuxIO()
    try:
        with progress_path.open('x') as progress:
            while elapsed_ms < args.duration_ms:
                duration = min(args.chunk_ms, args.duration_ms-elapsed_ms)
                t0 = time.perf_counter()
                nest.Simulate(duration)
                t1 = time.perf_counter()
                count = recorder.get('n_events')
                if count > args.max_chunk_spikes or total_spikes+count > args.max_spikes_per_rank:
                    raise RuntimeError('Explicit spike recording budget exceeded before array extraction')
                events = recorder.get('events')
                ticks = np.rint(np.asarray(events['times']) / DT).astype(np.int64)
                ids = np.asarray(events['senders'], dtype=np.int64)-1
                if not (len(ticks) == len(ids) == count
                        and np.all((ticks >= 0) & (ticks <= round((elapsed_ms+duration)/DT)))
                        and np.all((ids >= 0) & (ids < cell_count))):
                    raise ValueError('Invalid recorded event tick or cell')
                records = np.empty(count, dtype=EVENT_DTYPE)
                records['tick'], records['cell'] = ticks, ids
                writer.append(records)
                recorder.set(n_events=0)
                if recorder.get('n_events') != 0:
                    raise RuntimeError('NEST recorder did not reset')
                total_spikes += count
                elapsed_ms += duration
                t2 = time.perf_counter()
                row = dict(end_ms=elapsed_ms, spikes=int(count),
                           simulation_seconds=t1-t0, output_seconds=t2-t1)
                memory[f'after_{elapsed_ms:g}ms'] = memory_snapshot()
                entry = dict(event='chunk_complete', rank=rank, **row,
                             total_spikes=total_spikes, event_bytes=writer.written,
                             durable_event_bytes=writer.durable_bytes,
                             memory=memory[f'after_{elapsed_ms:g}ms'])
                progress.write(json.dumps(entry, allow_nan=False)+'\n')
                progress.flush()
                print(json.dumps(entry, allow_nan=False), flush=True)
                row['monitor_progress_seconds'] = time.perf_counter()-t2
                chunks.append(row)
            finish_started = time.perf_counter()
            receipt = writer.finish()
            progress.flush()
            io.sync_data(progress.fileno())
        io.sync_directory(args.output)
        finish_seconds = time.perf_counter()-finish_started
        verify_started = time.perf_counter()
        verified_sha = verify_event_file(event_path, receipt['bytes'], receipt['sha256'])
        verification_seconds = time.perf_counter()-verify_started
    finally:
        writer.close()
    if receipt['records'] != total_spikes:
        raise OSError('Event receipt count mismatch')
    return dict(chunks=chunks, phase_memory=memory, local_spikes=total_spikes,
                event_bytes=receipt['bytes'], event_sha256=verified_sha,
                durable_event_receipt=receipt, event_readback_verified=True,
                simulation_seconds=sum(x['simulation_seconds'] for x in chunks),
                event_output_seconds=sum(x['output_seconds'] for x in chunks),
                monitor_progress_seconds=sum(x['monitor_progress_seconds'] for x in chunks),
                recording_finish_seconds=finish_seconds,
                event_verification_seconds=verification_seconds,
                recording_wall_seconds=time.perf_counter()-started,
                timing_scope='Rank-local; writer timings are nested in output/finish; external job wall is authoritative')
