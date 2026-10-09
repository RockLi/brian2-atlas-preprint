#!/usr/bin/env python3
"""Prepare preregistered MNIST/SHD artifacts on the authorized remote host.

No training, test scoring, test-label decoding or full-dataset dense encoding.
SHD test access is restricted to spikes/times to establish the common extent.
Every invocation records source identity, downloads, failures and file hashes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

MIB = 1024**2
GIB = 1024**3
MNIST_FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}
SHD_FILES = {
    'shd_train.h5.gz': 'd47c9825dee33347913e8ce0f2be08b0',
    'shd_test.h5.gz': '3062a80ec0c5719404d5b02e166543b1',
}
SOURCES = {
    'mnist_origin': 'http://yann.lecun.com/exdb/mnist/',
    'mnist_download_mirror': 'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'mnist_mirror_authority': 'https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/mnist.py',
    'mnist_mirror_reason': 'The original lecun.org/.com pages timed out during 2026-10-04 source audit; this mirror and original-file MD5 identities are published by official torchvision.',
    'shd_origin': 'https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/',
    'shd_download': 'https://zenkelab.org/datasets/',
    'shd_md5': 'https://zenkelab.org/datasets/md5sums.txt',
    'shd_timestamp_unit_authority': 'https://raw.githubusercontent.com/fzenke/spytorch/master/notebooks/SpyTorchTutorial4.ipynb',
    'shd_timestamp_unit': 'seconds; official tutorial uses time_step=1e-3 and max_time~1.4',
    'shd_license': 'CC BY 4.0; Cramer, Stradmann, Schemmel, Zenke, IEEE TNNLS 2022, DOI10.1109/TNNLS.2020.3044364',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def immutable_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f'Would replace a different artifact: {path}; use a new revision directory')
        return
    temp = path.with_name(path.name + '.new')
    if temp.exists():
        raise FileExistsError(f'Incomplete prior write preserved: {temp}')
    temp.write_bytes(content)
    os.replace(temp, path)


def immutable_json(path, value):
    immutable_bytes(path, json_bytes(value))


def immutable_npy(path, array):
    import io
    import numpy as np
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    immutable_bytes(path, buffer.getvalue())


class Context:
    def __init__(self, root, run, free_floor_gib):
        self.root, self.run = root, run
        self.free_floor = int(free_floor_gib * GIB)
        self.downloads = []
        self.started = time.monotonic()
        self.log_path = run / 'events.jsonl'

    def event(self, kind, **data):
        entry = {'at_utc': utc(), 'event': kind, **data}
        with self.log_path.open('a') as out:
            out.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    def check_resources(self):
        free = shutil.disk_usage(self.root).free
        if free < self.free_floor:
            raise RuntimeError(f'resource_pending: free disk {free} < reserved floor {self.free_floor}')
        if time.monotonic() - self.started > 3600:
            raise TimeoutError('Data preparation invocation exceeded preregistered3600s cap')

    def fetch(self, filename, url, md5=None, *, target_dir='raw', max_bytes=2*GIB):
        self.check_resources()
        dest = self.root / target_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if md5 and file_hash(dest, 'md5') != md5:
                raise ValueError(f'Existing raw artifact MD5 mismatch, preserved: {dest}')
            identity = {'path': str(dest.relative_to(self.root)), 'url': url, 'bytes': dest.stat().st_size,
                        'sha256': file_hash(dest), 'md5': file_hash(dest, 'md5'), 'expected_md5': md5, 'reused': True}
            self.downloads.append(identity)
            self.event('raw_reused', **identity)
            return dest
        last = None
        # Partial files and both failures remain in the run ledger, never erased.
        for attempt in range(1, 3):
            partial = dest.with_name(dest.name + f'.{self.run.name}.attempt{attempt}.part')
            before = time.monotonic()
            received = 0
            try:
                # System curl uses the host's trusted TLS transport. Never disable
                # certificate verification or permit a redirect to plain HTTP.
                # Failed runs' partials, HTTP headers and stderr remain evidence.
                if not url.startswith('https://'):
                    raise ValueError('Dataset downloads require HTTPS')
                curl = shutil.which('curl')
                if curl is None:
                    raise RuntimeError('curl is required for the audited remote transport')
                header_path = self.run / f'{filename}.attempt{attempt}.headers'
                error_path = self.run / f'{filename}.attempt{attempt}.stderr'
                partial.touch(exist_ok=False)
                command = [curl, '--fail', '--location', '--silent', '--show-error',
                           '--proto', '=https', '--proto-redir', '=https',
                           '--connect-timeout', '30', '--max-time', '900',
                           '--max-filesize', str(max_bytes), '--dump-header', str(header_path),
                           '--output', str(partial), '--write-out', '%{url_effective}\n%{http_code}\n',
                           '--user-agent', 'AtlasTrainingEvaluation/1.0 (reproducible dataset preparation)', url]
                with error_path.open('xb') as errors:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)
                    try:
                        while process.poll() is None:
                            received = partial.stat().st_size
                            self.check_resources()
                            if received > max_bytes or time.monotonic() - before > 910:
                                raise TimeoutError('Per-file size or900s download cap reached')
                            time.sleep(.25)
                        output, _ = process.communicate(timeout=5)
                    except BaseException:
                        process.kill()
                        process.wait(timeout=5)
                        raise
                received = partial.stat().st_size
                if process.returncode != 0:
                    raise RuntimeError(f'curl failed with exit {process.returncode}: {error_path.read_text()}')
                final_url, http_code = output.strip().splitlines()[-2:]
                if not final_url.startswith('https://') or http_code != '200' or received > max_bytes:
                    raise ValueError('Unexpected final download URL, HTTP status or size')
                headers = {'raw_headers_path': str(header_path.relative_to(self.root)),
                           'raw_headers_sha256': file_hash(header_path), 'http_code': http_code}
                actual_md5 = file_hash(partial, 'md5')
                if md5 and actual_md5 != md5:
                    raise ValueError(f'MD5 mismatch: expected={md5}, actual={actual_md5}')
                os.replace(partial, dest)
                identity = {'path': str(dest.relative_to(self.root)), 'url': url, 'final_url': final_url,
                            'bytes': received, 'sha256': file_hash(dest), 'md5': actual_md5, 'expected_md5': md5,
                            'headers': headers, 'transport': 'curl; TLS verification enabled; HTTPS-only redirects',
                            'elapsed_s': time.monotonic()-before, 'attempt': attempt, 'reused': False}
                self.downloads.append(identity)
                self.event('download_completed', **identity)
                return dest
            except Exception as error:
                last = error
                self.event('download_failed', url=url, attempt=attempt, bytes_received=received,
                           partial_path=str(partial.relative_to(self.root)), error=str(error), traceback=traceback.format_exc())
        raise RuntimeError(f'Two source download attempts failed: {url}') from last

    def gunzip(self, archive, max_bytes=4*GIB):
        destination = archive.with_suffix('')
        if destination.exists():
            # Compare uncompressed byte identity to the archive on every reuse.
            digest = hashlib.sha256()
            with gzip.open(archive, 'rb') as src:
                for block in iter(lambda: src.read(MIB), b''):
                    digest.update(block)
            if file_hash(destination) != digest.hexdigest():
                raise ValueError(f'Uncompressed source differs from archive: {destination}')
            return destination
        partial = destination.with_name(destination.name + f'.{self.run.name}.part')
        n = 0
        with gzip.open(archive, 'rb') as src, partial.open('xb') as out:
            for block in iter(lambda: src.read(MIB), b''):
                n += len(block)
                if n > max_bytes:
                    raise ValueError('Decompressed source exceeds4GiB per-file cap')
                out.write(block)
                self.check_resources()
        os.replace(partial, destination)
        self.event('decompressed', path=str(destination.relative_to(self.root)), bytes=n, sha256=file_hash(destination))
        return destination


def read_idx(path, expected_shape):
    import numpy as np
    with Path(path).open('rb') as source:
        prefix = source.read(4)
        if prefix != bytes([0, 0, 8, len(expected_shape)]):
            raise ValueError(f'Incorrect uint8IDX header in{path}: {prefix!r}')
        shape = struct.unpack('>' + 'I'*len(expected_shape), source.read(4*len(expected_shape)))
    if tuple(shape) != tuple(expected_shape):
        raise ValueError(f'Unexpected source shape: {shape} != {expected_shape}')
    offset = 4*(1+len(shape))
    if Path(path).stat().st_size != offset + math.prod(shape):
        raise ValueError(f'Unexpected IDX data size: {path}')
    return np.memmap(path, dtype=np.uint8, mode='r', offset=offset, shape=shape)


def prepare_mnist(ctx):
    import numpy as np
    ctx.event('mnist_start', test_labels_policy='download/hash sealed bytes only; no decoding')
    raw = {}
    for filename, expected in MNIST_FILES.items():
        archive = ctx.fetch(filename, SOURCES['mnist_download_mirror'] + filename, expected, target_dir='mnist/raw', max_bytes=64*MIB)
        raw[filename[:-3]] = ctx.gunzip(archive)
    images = read_idx(raw['train-images-idx3-ubyte'], (60000,28,28))
    labels = read_idx(raw['train-labels-idx1-ubyte'], (60000,))
    # Test images and labels remain byte-hashed originals; neither decoded here.
    rng = np.random.Generator(np.random.PCG64(20261004))
    validation = []
    counts = {}
    for label in range(10):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < 500:
            raise ValueError(f'MNIST class{label} has fewer than500 training samples')
        chosen = rng.choice(candidates, size=500, replace=False)
        validation.extend(chosen.tolist())
        counts[str(label)] = {'official_train':int(len(candidates)), 'train':int(len(candidates)-500), 'validation':500}
    val_ids = np.asarray(sorted(validation), dtype='<i8')
    train_ids = np.setdiff1d(np.arange(60000, dtype='<i8'), val_ids)
    assert len(train_ids) == 55000 and len(val_ids) == 5000
    assert len(np.intersect1d(train_ids, val_ids)) == 0
    processed = ctx.root / 'mnist/processed'
    immutable_npy(processed/'train_indices.npy', train_ids)
    immutable_npy(processed/'validation_indices.npy', val_ids)
    immutable_npy(processed/'test_indices.npy', np.arange(10000, dtype='<i8'))
    config = {'dataset':'MNIST','revision':'mnist-preprocess-r1','B':32,'T':100,'input_channels':784,
              'image_storage':'original uint8IDX memmap, 28x28 row-major flatten','input_encoding':'x[t,pixel]=pixel_uint8/255.0 repeated across100ticks; deterministic analogue-current input',
              'precision':'normalize after converting uint8 to requested numeric profile; no fullT dense dataset cache',
              'split_algorithm':'PCG64(seed20261004), class order0..9, choose500 without replacement per class; indices sorted ascending',
              'train_samples':55000,'validation_samples':5000,'test_samples':10000,'per_class_counts':counts,
              'test_labels_accessed':False,'test_images_accessed':False,'test_policy':'Original test bytes sealed; decode only for final selected-checkpoint evaluation',
              'raw_sources':{name:{'path':str(path.relative_to(ctx.root)),'sha256':file_hash(path)} for name,path in raw.items()},
              'library_versions':{'numpy':np.__version__}}
    immutable_json(processed/'preprocess.json',config)
    ctx.event('mnist_completed', train=55000, validation=5000, test=10000)
    return config


def count_bins(times, units, T, channels=700, dt_s=.001):
    """Sparse lossless binning: flat(t,channel) IDs and uint64 multiplicities."""
    import numpy as np
    times = np.asarray(times, dtype=np.float64)
    raw_units = np.asarray(units)
    if times.ndim != 1 or raw_units.ndim != 1 or len(times) != len(raw_units):
        raise ValueError('Event times/units must be matching one-dimensional arrays')
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError('Non-finite/negative event time')
    units = raw_units.astype(np.int64)
    if not np.array_equal(raw_units, units) or np.any(units < 0) or np.any(units >= channels):
        raise ValueError('Non-integral or out-of-range event channel')
    bins = np.floor(times/dt_s).astype(np.int64)
    if np.any(bins < 0) or np.any(bins >= T):
        raise ValueError('Event outside frozen full time window; never truncate')
    flat, count = np.unique(bins*channels+units, return_counts=True)
    count = count.astype('<u8')
    if int(count.sum(dtype=np.uint64)) != len(times):
        raise ValueError('Event conservation violated')
    return flat.astype('<i8'), count


def pick_speaker_split(labels, speaker):
    import numpy as np
    if labels.ndim != 1 or speaker.ndim != 1 or labels.shape != speaker.shape:
        raise ValueError('Train label/speaker arrays do not align')
    unique = np.unique(speaker)
    if len(unique) > 20:
        raise ValueError('Unexpected>20speakers; exhaustive frozen subset search would exceed declared scope')
    classes = set(range(20))
    if set(np.unique(labels).tolist()) != classes:
        raise ValueError('Official SHD train does not cover exactly labels0..19')
    best = None
    feasible = 0
    for length in range(1, len(unique)):
        for selected in itertools.combinations(unique.tolist(), length):
            mask = np.isin(speaker, selected)
            if set(np.unique(labels[mask]).tolist()) != classes or set(np.unique(labels[~mask]).tolist()) != classes:
                continue
            feasible += 1
            # Integer distance equivalent to abs(nval - .2*N), no FP tie bias.
            key = (abs(5*int(mask.sum()) - len(labels)), tuple(selected))
            if best is None or key < best[0]:
                best = (key, selected, mask)
    if best is None:
        raise ValueError('No train-only speaker subset preserves every class; protocol revision required, test labels remain sealed')
    _, selected, mask = best
    return np.flatnonzero(~mask).astype('<i8'), np.flatnonzero(mask).astype('<i8'), list(selected), feasible


def encoded_input_size(B, T, channels, nonzero_counts, float_values=False):
    """Exact compact JSON size for rectangular integer count tensor; no tensor allocated."""
    scalar_length = B*T*channels*(3 if float_values else 1)  #0.0 vs0
    for count in nonzero_counts:
        token = json.dumps(float(count) if float_values else int(count), separators=(',',':'), allow_nan=False)
        scalar_length += len(token)-(3 if float_values else 1)
    commas = B*T*(channels-1) + B*(T-1) + B-1
    brackets = 2*(B*T+B+1)
    return scalar_length+commas+brackets


def count_bin_contract_check():
    """Independent small exact boundaries/duplicates and JSON byte formula check."""
    import numpy as np
    times = np.array([0., .001, .001, np.nextafter(.001,0), .002, .002])
    units = np.array([0,1,1,1,0,0])
    flat, counts = count_bins(times,units,T=3,channels=2)
    expected = np.array([[1,1],[0,2],[2,0]],dtype=np.uint16)
    actual = np.zeros((3,2),dtype=np.uint16)
    actual.ravel()[flat] = counts
    if not np.array_equal(actual,expected) or int(actual.sum()) != len(times):
        raise AssertionError('Synthetic count-bin boundary oracle failed')
    for dtype in [np.uint16,np.float64]:
        dense = actual.astype(dtype)[None,:,:]
        expected_size = len(json.dumps(dense.tolist(),separators=(',',':'),allow_nan=False).encode())
        calculated = encoded_input_size(1,3,2,counts,float_values=dtype==np.float64)
        if expected_size != calculated:
            raise AssertionError('JSON analytic byte count differs from real serialization')
    return {'passed':True,'duplicates_preserved':True,'left_closed_right_open_boundaries':True,'exact_JSON_length_formula_verified':True}


def prepare_shd(ctx):
    import h5py
    import numpy as np
    ctx.event('shd_start', test_access='spikes/times only; no labels, units, speaker or score access')
    published_md5 = ctx.fetch('shd-md5sums.txt',SOURCES['shd_md5'],target_dir='provenance',max_bytes=MIB)
    hashes = {parts[1]:parts[0] for line in published_md5.read_text().splitlines() if len(parts:=line.split())==2}
    for name, expected in SHD_FILES.items():
        if hashes.get(name) != expected:
            raise ValueError(f'Published SHD identity changed for{name}; preserve and review before executing')
    raw = {}
    for name, expected in SHD_FILES.items():
        raw[name[:-3]] = ctx.gunzip(ctx.fetch(name,SOURCES['shd_download']+name,expected,target_dir='shd/raw'))
    extent = {}
    test_access_audit = {'allowed_dataset':['spikes/times'],'accessed_datasets':[],'forbidden_data_accessed':False,'labels_accessed':False,'units_accessed':False,'speaker_accessed':False,'purpose':'Global maximum timestamp only; class labels and scores sealed'}
    for split in ['train','test']:
        with h5py.File(raw[f'shd_{split}.h5'],'r') as source:
            times_dataset = source['spikes/times']
            expected_samples = 8156 if split=='train' else 2264
            if len(times_dataset) != expected_samples:
                raise ValueError(f'Unexpected SHD{split}sample count:{len(times_dataset)}')
            if split=='test':
                test_access_audit['accessed_datasets'].append('spikes/times')
            maximum = 0.
            for i in range(len(times_dataset)):
                times = np.asarray(times_dataset[i],dtype=np.float64)
                if not np.all(np.isfinite(times)) or np.any(times<0):
                    raise ValueError(f'Invalid timestamp in{split} sample{i}')
                if len(times):
                    maximum = max(maximum,float(times.max()))
                if i%1000==0:
                    ctx.check_resources()
            extent[split]={'samples':expected_samples,'maximum_timestamp_seconds':maximum}
    maximum = max(item['maximum_timestamp_seconds'] for item in extent.values())
    dt = .001
    T = int(math.floor(maximum/dt))+1
    if T <= 0 or T > 100000:
        raise ValueError(f'Unexpected SHD full-windowT={T}; review source units before allocating')
    processed = ctx.root/'shd/processed'
    with h5py.File(raw['shd_train.h5'],'r') as source:
        labels = np.asarray(source['labels'],dtype=np.int64)
        speaker_raw = np.asarray(source['extra/speaker'])
        # SHD official speaker IDs are integers; reject rather than silently recode.
        speaker = speaker_raw.astype(np.int64)
        if not np.array_equal(speaker,speaker_raw):
            raise ValueError('Unexpected nonintegral train speaker IDs; freeze mapping before proceeding')
        train_ids,val_ids,val_speakers,feasible = pick_speaker_split(labels,speaker)
        immutable_npy(processed/'train_indices.npy',train_ids)
        immutable_npy(processed/'validation_indices.npy',val_ids)
        immutable_npy(processed/'test_indices.npy',np.arange(2264,dtype='<i8'))
        immutable_npy(processed/'official_train_labels.npy',labels.astype('<i8'))
        immutable_npy(processed/'official_train_speakers.npy',speaker.astype('<i8'))
        # Predetermined first32 ascending training IDs, independent of activity.
        batch_ids = train_ids[:32]
        if len(batch_ids)!=32:
            raise ValueError('Not enough training samples for frozenB32')
        immutable_npy(processed/'admission_B32_sample_indices.npy',batch_ids)
        selected = set(batch_ids.tolist())
        batch_flat,batch_counts,batch_offsets = [],[],[0]
        conservation = []
        histogram = {}
        total_events = total_cells = max_count = 0
        encoding_digest = hashlib.sha256()
        for i in range(len(labels)):
            times = source['spikes/times'][i]
            units = source['spikes/units'][i]
            flat,counts = count_bins(times,units,T,dt_s=dt)
            count = len(times)
            largest = int(counts.max(initial=0))
            total_events += count;total_cells += len(flat);max_count=max(max_count,largest)
            conservation.append([i,count,int(counts.sum(dtype=np.uint64)),len(flat),largest])
            for value,n in zip(*np.unique(counts,return_counts=True)):
                histogram[str(int(value))]=histogram.get(str(int(value)),0)+int(n)
            # Typed little-endian sparse bytes with unambiguous per-sample framing.
            encoding_digest.update(struct.pack('<QQ',i,len(flat)))
            encoding_digest.update(flat.tobytes());encoding_digest.update(counts.tobytes())
            if i in selected:
                batch_flat.extend(flat.tolist());batch_counts.extend(counts.tolist());batch_offsets.append(len(batch_flat))
            if i%1000==0:
                ctx.check_resources();ctx.event('shd_count_progress',processed_samples=i,total_samples=len(labels))
        if max_count > np.iinfo(np.uint32).max:
            raise ValueError('Event counts exceeduint32; no silent clipping allowed')
        count_dtype='uint16' if max_count<=np.iinfo(np.uint16).max else 'uint32'
        immutable_npy(processed/'count_conservation.npy',np.asarray(conservation,dtype='<i8'))
        immutable_npy(processed/'admission_B32_flat_bins.npy',np.asarray(batch_flat,dtype='<i8'))
        immutable_npy(processed/'admission_B32_counts.npy',np.asarray(batch_counts,dtype='<u2' if count_dtype=='uint16' else '<u4'))
        immutable_npy(processed/'admission_B32_offsets.npy',np.asarray(batch_offsets,dtype='<i8'))
        split_detail={'rule':'exhaustive nonempty proper train-speaker subsets, all20classes in both parts; minimizeabs(5*nval-N); tie lexicographic numeric speaker tuple',
                      'train_sample_count':int(len(train_ids)),'validation_sample_count':int(len(val_ids)),'official_train_samples':8156,'official_test_samples':2264,
                      'validation_speakers':list(map(int,val_speakers)),'train_speakers':sorted(set(speaker[train_ids].tolist())),'feasible_subsets':feasible,
                      'per_class_train':np.bincount(labels[train_ids],minlength=20).tolist(),'per_class_validation':np.bincount(labels[val_ids],minlength=20).tolist(),
                      'disjoint_speaker_sets':not bool(set(speaker[train_ids])&set(speaker[val_ids])),'test_labels_accessed':False}
        immutable_json(processed/'split.json',split_detail)
        array_bytes=encoded_input_size(32,T,700,batch_counts)
        float_bytes=encoded_input_size(32,T,700,batch_counts,float_values=True)
        admission={'workload':'A2','global_batch':32,'T':T,'input_channels':700,'dt_seconds':dt,'shape':[32,T,700],
                   'sample_ids':batch_ids.tolist(),'selection':'first32ascending training indices, frozen before fitting, not chosen for activity',
                   'storage':'sparse flat-bin/count arrays plus per-sample offsets; no dense full library',
                   'integer_JSON_inputs_array_bytes_exact':array_bytes,'float_JSON_inputs_array_bytes_exact':float_bytes,
                   'method':'exact compactJSON punctuation/scalar-count formula; independently checked against actual serialization on boundary/duplicate fixture',
                   'native_JSON_limit_bytes':64*MIB,'input_alone_rejects_integer_JSON':array_bytes>64*MIB,'input_alone_rejects_float_JSON':float_bytes>64*MIB,
                   'whole_request_bytes':None,'whole_request_pending':'model,weights,optimizerstate,labels,metadata required; inputs array alone is a lower bound, not complete request',
                   'dense_bytes_uint16':32*T*700*2,'dense_bytes_FP32':32*T*700*4,'dense_bytes_FP64':32*T*700*8,
                   'batch_event_count':sum(batch_counts),'microbatch_enabled':False,'original_monolithic_case_retained':True}
        immutable_json(processed/'admission_B32.json',admission)
    config={'dataset':'SHD','revision':'shd-count-bin-r1','dt_seconds':dt,'T':T,'channels':700,'classes':20,
            'time_unit':'seconds, official data stored timestamps cast toFP64 before floor(t/.001)','window_rule':'T=floor(maximum_timestamp_seconds_across_official_train_and_test / dt_seconds)+1; no test labels read','extent':extent,
            'encoding':'left-closed/right-open count bins; flatbin=tick*700+channel; np.unique withcounts; no clipping, truncation, binary conversion or reversedchannels',
            'count_dtype':count_dtype,'test_count_dtype_policy':'Train-proven dtype only; test count overflow must be checked at final evaluation and fail rather than truncate. No test units read during preparation.',
            'train_count_bin_max':max_count,'train_total_raw_events':total_events,'train_total_binned_event_sum':total_events,
            'train_nonzero_bin_count':total_cells,'train_duplicate_events_preserved':total_events-total_cells,
            'train_nonzero_count_histogram':histogram,'train_sparse_encoding_sha256':encoding_digest.hexdigest(),
            'train_sparse_hash_encoding':'for each officialtrain sample:i:uint64LE,ncells:uint64LE,flatbin:int64LE[ncells],counts:uint64LE[ncells]',
            'conservation_columns':['official_train_sample_id','original_events','sum_of_counts','nonzero_bins','maximum_count'],
            'count_bin_contract_check':count_bin_contract_check(),'test_access_audit':test_access_audit,
            'raw_sources':{name:{'path':str(path.relative_to(ctx.root)),'sha256':file_hash(path)} for name,path in raw.items()},
            'library_versions':{'numpy':np.__version__,'h5py':h5py.__version__},'performance_run':False}
    immutable_json(processed/'preprocess.json',config)
    immutable_json(processed/'test-access-audit.json',test_access_audit)
    ctx.event('shd_completed',T=T,train_samples=split_detail['train_sample_count'],validation_samples=split_detail['validation_sample_count'],test_samples=2264,
              conservation_passed=True,test_labels_accessed=False,max_count=max_count,integer_input_bytes=array_bytes,float_input_bytes=float_bytes)
    return config


def inventory(root):
    return [{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':file_hash(p)}
            for p in sorted(root.rglob('*')) if p.is_file() and 'preparation-runs' not in p.parts and '.part' not in p.name and p.name!='data-manifest.json']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--datasets',choices=['all','mnist','shd'],default='all')
    parser.add_argument('--expected-host',default='rock-mac-studio-1.local')
    parser.add_argument('--allow-download',action='store_true',help='Explicit network/download opt-in; run only on authorized remote host')
    parser.add_argument('--free-disk-floor-gib',type=float,default=50)
    args=parser.parse_args()
    if not args.allow_download:
        parser.error('Pass --allow-download only on the authorized remote execution host')
    host=socket.gethostname()
    if host.split('.')[0]!=args.expected_host.split('.')[0]:
        raise RuntimeError(f'Host guard refused data processing on{host}; expected{args.expected_host}')
    root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
    run=root/'preparation-runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run.mkdir(parents=True,exist_ok=False)
    ctx=Context(root,run,args.free_disk_floor_gib)
    report={'started_at_utc':utc(),'script_sha256':file_hash(__file__),'host':host,'platform':platform.platform(),'python':sys.version,
            'argv':sys.argv,'sources':SOURCES,'performance_run':False,'test_label_accessed':False,'datasets':{},'execution_status':'started'}
    try:
        ctx.check_resources()
        immutable_json(root/'provenance/source-audit.json',SOURCES)
        # The failed r1 script is retained. Each revision has its own immutable
        # identity and each invocation records exactly which revision ran.
        script_relative = Path('provenance/scripts') / (report['script_sha256'] + '.py')
        immutable_bytes(root/script_relative,Path(__file__).read_bytes())
        immutable_bytes(run/'prepare_data.executed.py',Path(__file__).read_bytes())
        report['executed_script_path'] = str(script_relative)
        report['curl_version'] = subprocess.check_output(['curl','--version'],text=True).splitlines()[0]
        report['contract_self_check']=count_bin_contract_check()
        report['dataset_failures'] = {}
        for name,prepare in [('mnist',prepare_mnist),('shd',prepare_shd)]:
            if args.datasets not in ['all',name]:
                continue
            try:
                report['datasets'][name]=prepare(ctx)
            except Exception as error:
                report['dataset_failures'][name]={'error':str(error),'traceback':traceback.format_exc()}
                ctx.event('dataset_failed',dataset=name,error=str(error))
        # Reference text failures cannot prevent independent raw acquisition and
        # count-bin validation. They remain explicit unresolved provenance items.
        report['reference_failures'] = {}
        references = [('torchvision-mnist-source.py','mnist_mirror_authority'),
                      ('SpyTorchTutorial4.ipynb','shd_timestamp_unit_authority')]
        for filename,key in references:
            try:
                reference = ctx.fetch(filename,SOURCES[key],target_dir='provenance',max_bytes=16*MIB)
                if key == 'mnist_mirror_authority':
                    source_text = reference.read_text()
                    if SOURCES['mnist_download_mirror'] not in source_text or any(md5 not in source_text for md5 in MNIST_FILES.values()):
                        raise ValueError('Official torchvision source does not corroborate frozen mirror/MD5 identities')
            except Exception as error:
                report['reference_failures'][key]={'error':str(error),'traceback':traceback.format_exc()}
                ctx.event('reference_failed',reference=key,error=str(error))
        if report['dataset_failures']:
            raise RuntimeError('One or more dataset preparation stages failed; completed independent datasets are retained')
        manifest={'revision':'atlas-data-r1','script_sha256':report['script_sha256'],'files':inventory(root),'test_labels_accessed':False,
                  'shape_only_test_timestamp_use':True,'performance_run':False,'stage':'preprocessing artifacts only, not completed training',
                  'reference_failures':report['reference_failures']}
        immutable_json(root/'data-manifest.json',manifest)
        report['data_manifest_sha256']=file_hash(root/'data-manifest.json')
        report['execution_status']='completed' if not report['reference_failures'] else 'data_ready_provenance_pending'
    except BaseException as error:
        report.update(execution_status='failed',error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
        ctx.event('preparation_failed',error=str(error),error_type=type(error).__name__)
    finally:
        report['finished_at_utc']=utc();report['elapsed_s']=time.monotonic()-ctx.started;report['downloads']=ctx.downloads
        (run/'report.json').write_bytes(json_bytes(report))
        print(json.dumps({'execution_status':report['execution_status'],'report':str(run/'report.json'),'elapsed_s':report['elapsed_s']},ensure_ascii=False),flush=True)
    return 0 if report['execution_status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
