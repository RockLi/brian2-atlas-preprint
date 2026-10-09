#!/usr/bin/env python3
"""Read-only stdlib A1-v4 evidence audit; never imports a model or test labels.

Full mode is remote-only. Report-only mode does not verify fixture payloads,
full batch logs or checkpoint counters. --self-test uses tiny synthetic data.
Existing reports, once-only ledgers and checkpoints are never modified.
"""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import socket
import statistics
import struct
import tempfile
import zipfile

SEEDS = [11, 23, 37, 51, 71]
VIEWS = [('atlas', 'atlas', False), ('sj-layerwise', 'spikingjelly_frontier', False),
         ('snn-layerwise', 'snntorch_fp64', False), ('sj-compile', 'spikingjelly_frontier', True),
         ('snn-compile', 'snntorch_fp64', True)]
CONTRACT = 'protocol/A1-execution-addendum-arm64-long-r1.json'
CAP = 1800
COUNTS = {'train': 55000, 'validation': 5000, 'test': 10000}
BATCHES = 1719
CORE = ['tools/run_a1_queue_arm64_r1.py', 'tools/run_a1_mnist_arm64_long_r1.py', 'tools/prepare_data.py',
        'tools/run_recurrent_queue.py', 'tools/run_qualification_followup_queue.py',
        'tools/generate_dense_large.py', 'tools/generate_recurrent_cases.py',
        'adapters/torch_adapter.py', 'adapters/atlas_adapter.py', 'adapters/oracle.py', CONTRACT,
        'protocol/recurrent-fixture-r1.json', 'protocol/finite-engine-cases.json',
        'runtime/arm64-r2/b2-train', 'environment/cpu-lock.txt', 'environment/hardware.json',
        'snapshot/brian2-rust/python/brian2_rust/training.py']
SOURCE_KEYS = ['tools/run_a1_mnist_arm64_long_r1.py', 'adapters/torch_adapter.py', 'adapters/atlas_adapter.py', 'adapters/oracle.py']


def not_sealed(path):
    # The audit has no reason to open either held-out label or image payload.
    # Enforce this even for a malformed/untrusted manifest or path.
    names = [Path(path).name.lower(), Path(path).resolve().name.lower()]
    if any('t10k-' in name or 'test-label' in name or 'test-image' in name for name in names):
        raise ValueError('validator deliberately does not open held-out payloads')
    return Path(path)


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key: '+key)
        result[key] = value
    return result


def loads(text):
    def invalid(value):
        raise ValueError('nonfinite JSON token: '+value)
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid)


def read(path):
    return loads(not_sealed(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with not_sealed(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def rooted(root, relative):
    path = (root/relative).resolve()
    path.relative_to(root.resolve())
    return not_sealed(path)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def near(a, b):
    return number(a) and number(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def plan():
    return [dict(view=name, engine=engine, compiled=compiled, seed=seed)
            for offset, seed in enumerate(SEEDS) for name, engine, compiled in VIEWS[offset:]+VIEWS[:offset]]


class Audit:
    def __init__(self, context=''):
        self.context, self.errors, self.limitations = context, [], []
    def check(self, condition, message):
        if not condition:
            self.errors.append(message)
        return bool(condition)
    def note(self, message):
        if message not in self.limitations:
            self.limitations.append(message)
    def json(self, path, required=False, interrupted=False):
        if not Path(path).exists():
            if required:
                self.errors.append('missing '+str(path))
            return None
        try:
            value = read(path)
            if not isinstance(value, (dict, list)):
                raise ValueError('report root is not an object or list')
            return value
        except (OSError, ValueError) as error:
            (self.limitations if interrupted else self.errors).append(f'{path}: unreadable JSON ({error})')
            return None
    def hash(self, path, expected, required=True):
        if not Path(path).is_file():
            if required:
                self.errors.append('missing hashed file '+str(path))
            return False
        try:
            actual = sha(path)
            return self.check(isinstance(expected, str) and actual == expected, 'SHA256 mismatch: '+str(path))
        except (OSError, ValueError) as error:
            self.errors.append(f'cannot hash {path}: {error}')
            return False


def command_flags(command, audit):
    result = {}
    if not isinstance(command, list) or not all(isinstance(s, str) for s in command):
        audit.check(False, 'command must be a string array')
        return result
    i = 0
    while i < len(command):
        token = command[i]
        if token.startswith('--'):
            audit.check(token not in result, 'duplicate command flag '+token)
            if i+1 < len(command) and not command[i+1].startswith('--'):
                result[token] = command[i+1]
                i += 1
            else:
                result[token] = True
        i += 1
    return result


def q0_relative(view):
    return 'evidence/arm64-r2/q0/atlas/report.json' if view == 'atlas' else f'evidence/remote-qualify-v1/q0-{view}/report.json'


def command_identity(command, slot, directory, root, audit, worker=False, physical_root=True):
    flags = command_flags(command, audit)
    for key, expected in [('--engine', slot['engine']), ('--seed', str(slot['seed']))]:
        audit.check(flags.get(key) == expected, 'command identity differs: '+key)
    audit.check(('--compile' in flags) == slot['compiled'], 'command compile view differs')
    audit.check(flags.get('--allow-run') is True, 'run command lacks allow-run guard')
    audit.check(('--worker' in flags) == worker, 'supervisor/worker command role differs')
    recorded_root = root if physical_root else Path(flags.get('--root', str(root)))
    if not physical_root:
        audit.note('Report-only command paths checked relative to their recorded remote root, not against the local copied root')
    recorded_output = recorded_root/directory.relative_to(root)
    for flag, want in [('--root', recorded_root), ('--contract', recorded_root/CONTRACT), ('--output', recorded_output),
                       ('--q0-report', recorded_root/q0_relative(slot['view']))]:
        try:
            audit.check(Path(flags.get(flag, '')).resolve() == want.resolve(), 'command path differs: '+flag)
        except (TypeError, OSError):
            audit.check(False, 'invalid command path '+flag)
    audit.check(any(Path(part).name == 'run_a1_mnist_arm64_long_r1.py' for part in command), 'command is not the A1-v4 runner')
    return flags


def npy_header(stream):
    if stream.read(6) != b'\x93NUMPY':
        raise ValueError('invalid NPY magic')
    version = stream.read(2)
    if version == b'\x01\x00':
        size = struct.unpack('<H', stream.read(2))[0]
    elif version in (b'\x02\x00', b'\x03\x00'):
        size = struct.unpack('<I', stream.read(4))[0]
    else:
        raise ValueError('unsupported NPY version')
    if size > 65536:
        raise ValueError('oversized NPY header')
    header = ast.literal_eval(stream.read(size).decode('utf-8' if version[0] == 3 else 'latin1'))
    if set(header) != {'descr', 'fortran_order', 'shape'} or header['fortran_order'] is not False:
        raise ValueError('only standard C-order NPY payloads accepted')
    return header


def index_array(path, count):
    with not_sealed(path).open('rb') as stream:
        header = npy_header(stream)
        if header != dict(descr='<i8', fortran_order=False, shape=(count,)):
            raise ValueError('index array dtype/shape mismatch')
        data = stream.read(count*8+1)
        if len(data) != count*8:
            raise ValueError('index payload byte length mismatch')
    return [value[0] for value in struct.iter_unpack('<q', data)]


def check_shared(root, freeze, audit, full, required):
    path = root/'fixtures/a1-arm64-long-r1/manifest.json'
    manifest = audit.json(path, required=required)
    if manifest is None:
        return None
    audit.check(manifest.get('schema') == 'a1-shared-arrays-arm64-long-r1', 'wrong shared-array schema')
    for key, expected in [('contract_sha256', freeze['files'].get(CONTRACT)),
                          ('generator_sha256', freeze['files'].get('tools/run_a1_mnist_arm64_long_r1.py')),
                          ('seeds', SEEDS), ('original_train_samples', 55000), ('epochs', 10),
                          ('train_batches_per_epoch', 1719), ('tail_batch', 24)]:
        audit.check(manifest.get(key) == expected, 'shared manifest differs: '+key)
    expected_files = {f'seed-{seed}.npz' for seed in SEEDS}
    audit.check(set(manifest.get('files', {})) == expected_files, 'shared arrays require all five seeds')
    expected_split = {'train_indices.npy', 'validation_indices.npy', 'preprocess.json'}
    audit.check(set(manifest.get('split_sha256', {})) == expected_split, 'shared split hash keys differ')
    if not full:
        audit.note('Report-only mode: NPZ bytes, split arrays and raw train source hashes are not verified')
        return manifest
    processed = root/'data/mnist/processed'
    for name, expected in manifest.get('split_sha256', {}).items():
        audit.hash(rooted(processed, name), expected)
    metadata = audit.json(processed/'preprocess.json', required=True) or {}
    for key, value in [('train_samples', 55000), ('validation_samples', 5000), ('test_samples', 10000), ('B', 32), ('T', 100)]:
        audit.check(metadata.get(key) == value, 'preprocessing contract differs: '+key)
    for name in ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'):
        row = metadata.get('raw_sources', {}).get(name, {})
        try:
            audit.hash(rooted(root/'data', row['path']), row['sha256'])
        except (KeyError, ValueError) as error:
            audit.check(False, 'invalid raw train source: '+str(error))
    audit.note('Held-out image/label payloads are never opened or hashed; their byte identity is only a recorded worker/source claim')
    try:
        train = index_array(processed/'train_indices.npy', 55000)
        validation = index_array(processed/'validation_indices.npy', 5000)
        audit.check(train == sorted(set(train)) and validation == sorted(set(validation)), 'split indices must be sorted unique')
        audit.check(sorted(train+validation) == list(range(60000)), 'train/validation split is not disjoint full official train set')
        for seed in SEEDS:
            filename = f'seed-{seed}.npz'
            info = manifest['files'][filename]
            artifact = root/'fixtures/a1-arm64-long-r1'/filename
            audit.hash(artifact, info['sha256'])
            audit.check(artifact.stat().st_size == info['bytes'], filename+' byte size differs')
            with zipfile.ZipFile(artifact) as archive:
                audit.check(sorted(archive.namelist()) == ['bank_0.npy', 'bank_1.npy', 'epoch_indices.npy'], filename+' unexpected NPZ members')
                for index, count in enumerate((784*128, 128*10)):
                    with archive.open(f'bank_{index}.npy') as stream:
                        header = npy_header(stream)
                        audit.check(header == dict(descr='<f8', fortran_order=False, shape=(count,)), 'initial bank shape/dtype differs')
                        data = stream.read(count*8+1)
                        audit.check(len(data) == count*8, 'initial bank payload length differs')
                        audit.check(hashlib.sha256(data).hexdigest() == info['weights_sha256'][index], 'initial bank raw hash differs')
                        audit.check(all(math.isfinite(v[0]) for v in struct.iter_unpack('<d', data)), 'nonfinite initial weight')
                with archive.open('epoch_indices.npy') as stream:
                    audit.check(npy_header(stream) == dict(descr='<i8', fortran_order=False, shape=(10, 55000)), 'epoch order shape/dtype differs')
                    audit.check(len(info['epoch_order_sha256']) == 10, 'ten archived epoch order hashes required')
                    for epoch in range(10):
                        data = stream.read(55000*8)
                        audit.check(len(data) == 55000*8, 'incomplete archived epoch order')
                        audit.check(hashlib.sha256(data).hexdigest() == info['epoch_order_sha256'][epoch], 'epoch order raw SHA256 differs')
                        audit.check(sorted(v[0] for v in struct.iter_unpack('<q', data)) == train, 'epoch is not an exact permutation of frozen train split')
                    audit.check(stream.read(1) == b'', 'extra epoch payload bytes')
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile, struct.error) as error:
        audit.check(False, 'shared fixture validation failed: '+str(error))
    return manifest


def check_batches(path, audit, interrupted=False, counts=None):
    """Stream scalar records only; no predictions or labels exist in this log."""
    counts = COUNTS if counts is None else counts
    groups, current, previous_time, rows = {}, None, -1., 0
    first_after = []
    allowed = {'phase', 'epoch', 'offset', 'samples', 'elapsed_s', 'elapsed_total_s', 'public_api_s'}
    with not_sealed(path).open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                row = loads(line)
            except ValueError as error:
                if interrupted and not line.endswith('\n') and stream.read(1) == '':
                    audit.note('Interrupted final batch-log fragment preserved and excluded')
                    break
                audit.check(False, 'malformed batch log: '+str(error))
                break
            rows += 1
            phase_, epoch = row.get('phase'), row.get('epoch')
            if not audit.check(phase_ in counts and integer(epoch) and 1 <= epoch <= 10, 'invalid batch phase/epoch'):
                break
            key = (phase_, epoch)
            if key != current:
                audit.check(key not in groups, 'batch phase group appears more than once')
                if current is None:
                    audit.check(key == ('train', 1), 'first batch must start epoch1 training')
                else:
                    old = groups[current]
                    audit.check(old['samples'] == counts[current[0]], 'phase transitioned before its full denominator')
                    legal = ((current[0] == 'train' and key == ('validation', current[1])) or
                             (current[0] == 'validation' and (key == ('train', current[1]+1) or (phase_ == 'test' and epoch <= current[1]))))
                    audit.check(legal, 'illegal batch phase/epoch transition')
                groups[key] = dict(samples=0, batches=0, correct=0, weighted_loss=0., first_time=row.get('elapsed_total_s'), last_time=None, batch_seconds=0.)
                current = key
                first_after.append(dict(phase=phase_, epoch=epoch, elapsed_total_s=row.get('elapsed_total_s')))
            group = groups[key]
            n, offset = row.get('samples'), row.get('offset')
            audit.check(integer(offset) and offset == group['samples'], 'batch offsets are not consecutive')
            audit.check(integer(n) and n == min(32, counts[phase_]-group['samples']) and n > 0, 'batch size/tail or phase denominator differs')
            t, duration = row.get('elapsed_total_s'), row.get('elapsed_s')
            audit.check(number(t) and t >= previous_time, 'batch total clock is not finite/monotone')
            audit.check(number(duration) and duration >= 0 and number(t) and duration <= t, 'invalid batch duration')
            audit.check(row.get('public_api_s') is None or (number(row['public_api_s']) and 0 <= row['public_api_s'] <= duration+1e-6), 'public API duration outside batch wall')
            if phase_ == 'test':
                audit.check(set(row) <= allowed, 'partial test metric/prediction field published in batch log')
            else:
                audit.check(set(row) <= allowed|{'loss', 'correct'}, 'unexpected training/validation batch fields')
                hits, loss = row.get('correct'), row.get('loss')
                audit.check(integer(hits) and integer(n) and 0 <= hits <= n, 'invalid batch correct count')
                audit.check(number(loss) and loss >= 0, 'invalid batch loss')
                if integer(hits): group['correct'] += hits
                if number(loss) and integer(n): group['weighted_loss'] += loss*n
            if not integer(n) or not number(t) or not number(duration):
                break
            group['samples'] += n
            group['batches'] += 1
            group['batch_seconds'] += duration
            group['last_time'] = t
            previous_time = t
    return dict(groups=groups, rows=rows, first_phase_batches=first_after, last_time=previous_time)


def aggregate(value, denominator, audit, name):
    if not isinstance(value, dict):
        audit.check(False, name+' missing aggregate')
        return False
    valid = audit.check(value.get('samples') == denominator, name+' denominator differs')
    correct = value.get('correct')
    valid &= audit.check(integer(correct) and 0 <= correct <= denominator, name+' invalid integer correct count')
    valid &= audit.check(near(value.get('accuracy'), correct/denominator if integer(correct) else None), name+' accuracy differs from aggregate correct/samples')
    valid &= audit.check(number(value.get('loss')) and value['loss'] >= 0, name+' invalid aggregate loss')
    valid &= audit.check(number(value.get('elapsed_s')) and value['elapsed_s'] >= 0, name+' invalid aggregate duration')
    return bool(valid)


def best_epoch(epochs):
    # max keeps the first index on integer ties; no test score is consulted.
    return max(epochs, key=lambda row: row['validation']['correct']) if epochs else None


def tensor_metadata(storage, offset, size, stride, *unused):
    if not isinstance(storage, dict) or storage.get('kind') != 'storage':
        raise ValueError('unexpected tensor storage reference')
    if not integer(offset) or not isinstance(size, tuple) or not isinstance(stride, tuple):
        raise ValueError('unexpected tensor metadata')
    return dict(kind='tensor', storage=storage, offset=offset, size=size, stride=stride)


class MetadataOnlyUnpickler(pickle.Unpickler):
    """Only inert tensor descriptors and OrderedDict; never torch.load/pickle globals.

    Tensor weight/gradient payloads remain inside the ZIP and are not decoded.
    Only scalar Adam step storage is read by checkpoint_counter().
    """
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2', '_rebuild_tensor_v3'):
            return tensor_metadata
        if module == 'torch' and name in ('DoubleStorage', 'FloatStorage', 'LongStorage', 'IntStorage', 'ByteStorage'):
            return ('storage_dtype', name)
        raise ValueError('unsupported pickle global rejected without execution: '+module+'.'+name)
    def persistent_load(self, identity):
        if not isinstance(identity, tuple) or len(identity) != 5 or identity[0] != 'storage':
            raise ValueError('unexpected persistent storage ID')
        _, dtype, key, location, count = identity
        if (not isinstance(dtype, tuple) or dtype[0] != 'storage_dtype' or not isinstance(key, str)
                or not key.isdigit() or location != 'cpu' or not integer(count) or count < 0):
            raise ValueError('unexpected storage metadata')
        return dict(kind='storage', dtype=dtype[1], key=key, count=count)


def checkpoint_counter(path, engine):
    if engine == 'atlas':
        envelope = read(path)
        if envelope.get('schema') != 'b2-native-training-checkpoint-v1':
            raise ValueError('unknown native checkpoint schema')
        payload_bytes = envelope['payload'].encode()
        if hashlib.sha256(payload_bytes).hexdigest() != envelope['sha256']:
            raise ValueError('native checkpoint internal payload hash differs')
        payload = loads(envelope['payload'])
        return dict(counters=[payload['state']['step']], runtime_sha256=payload['runtime_sha256'],
                    plan=payload['plan'], evidence='native checksummed JSON state.step')
    with zipfile.ZipFile(not_sealed(path)) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError('duplicate Torch ZIP members')
        candidates = [name for name in members if name.endswith('/data.pkl')]
        if len(candidates) != 1 or archive.getinfo(candidates[0]).file_size > 16*1024**2:
            raise ValueError('unsupported Torch checkpoint metadata container')
        prefix = candidates[0][:-len('data.pkl')]
        byteorder = archive.read(prefix+'byteorder').decode() if prefix+'byteorder' in members else 'little'
        if byteorder not in ('little', 'big'):
            raise ValueError('unexpected Torch storage byteorder')
        metadata = MetadataOnlyUnpickler(io.BytesIO(archive.read(candidates[0]))).load()
        state = metadata['optimizer']['state']
        if set(state) != {0, 1}:
            raise ValueError('expected exactly two optimizer parameter states')
        counters = []
        for key in (0, 1):
            tensor = state[key]['step']
            if tensor['kind'] != 'tensor' or tensor['size'] != () or tensor['offset'] != 0:
                raise ValueError('Adam counter is not a scalar tensor')
            storage = tensor['storage']
            if storage['count'] != 1:
                raise ValueError('Adam step storage must hold exactly one scalar')
            formats = {'DoubleStorage': 'd', 'FloatStorage': 'f', 'LongStorage': 'q', 'IntStorage': 'i'}
            fmt = ('<' if byteorder == 'little' else '>')+formats[storage['dtype']]
            data = archive.read(prefix+'data/'+storage['key'])
            if len(data) != struct.calcsize(fmt):
                raise ValueError('Adam scalar storage length differs')
            value = struct.unpack(fmt, data)[0]
            if not number(value) or int(value) != value:
                raise ValueError('Adam counter is nonfinite/nonintegral')
            counters.append(int(value))
        groups = metadata['optimizer'].get('param_groups', [])
        return dict(counters=counters, optimizer_groups=groups, weight_metadata=metadata.get('weights'),
                    evidence='restricted metadata-only Torch ZIP parser plus two scalar step storages')


def check_shape_qualification(path, engine, audit, required):
    report = audit.json(path, required=required)
    if report is None:
        return False
    cases = report.get('cases', {})
    expected = {'B32', 'tail_B24', 'evaluate_B8', 'evaluate_B16', 'no_gradient_evaluate_B2_exact_threshold_and_neighbors'}
    passed = report.get('passed') is True and set(cases) == expected
    for name, steps in [('B32', 3), ('tail_B24', 1)]:
        checks = cases.get(name, {}).get('checks', {})
        keys = {f'step{i}_{field}' for i in range(1, steps+1) for field in ['loss', 'logits', 'counter']}
        keys |= {f'step{i}_{field}_{bank}' for i in range(1, steps+1) for field in ['gradients', 'weights', 'first', 'second'] for bank in (0, 1)}
        passed &= set(checks) == keys and cases.get(name, {}).get('passed') is True
        passed &= all(check.get('passed') is True for check in checks.values())
    for name in ('evaluate_B8', 'evaluate_B16'):
        checks = cases.get(name, {}).get('checks', {})
        passed &= set(checks) == {'loss', 'logits'} and cases.get(name, {}).get('passed') is True
        passed &= all(check.get('passed') is True for check in checks.values())
    boundary = cases.get('no_gradient_evaluate_B2_exact_threshold_and_neighbors', {})
    checks = boundary.get('checks', {})
    keys = {'loss', 'logits', 'explicit_zero_and_neighbors'}
    keys |= {'actual_native_first_tick_spikes'} if engine == 'atlas' else {
        'actual_cell_exact_zero_margin', 'actual_callable_T1_loss', 'actual_callable_T1_logits', 'actual_callable_T1_exact_threshold'}
    passed &= keys <= set(checks) and boundary.get('passed') is True and boundary.get('grad_enabled') is False
    passed &= all(check.get('passed') is True for check in checks.values())
    margins = [value for row in boundary.get('canonical_actual_FP64_margin', []) for value in row]
    passed &= bool(margins) and all(number(v) for v in margins) and any(v == 0 for v in margins) and any(v < 0 for v in margins) and any(v > 0 for v in margins)
    if engine != 'atlas':
        probes = boundary.get('actual_cell_margin_probes', [])
        passed &= len(probes) == 2
        for probe in probes:
            passed &= probe.get('passed') is True and probe.get('training_mode') is True and probe.get('grad_enabled') is False
            margin = [v for row in probe.get('actual_margin', []) for v in row]
            spikes = [v for row in probe.get('actual_spikes', []) for v in row]
            direct, expected_margin = probe.get('actual_exact_margin_spikes'), probe.get('actual_exact_margin_input')
            passed &= len(margin) == 3 and len(spikes) == 3 and any(v == 0 for v in margin)
            passed &= all(s == int(m > 0) for s, m in zip(spikes, margin))
            passed &= expected_margin == [-2.**-20, 0., 2.**-20] and direct == [0., 0., 1.]
        probe = boundary.get('actual_callable_first_tick_probe', {})
        passed &= probe.get('sizes') == [784, 128, 10] and probe.get('B') == 2 and probe.get('T') == 1
        passed &= probe.get('actual_logits') == probe.get('expected_strict_threshold_logits')
    if required:
        audit.check(passed, 'formal training/test lacks complete actual-shape and boundary qualification')
    elif report.get('passed') is True:
        audit.check(passed, 'shape qualification claims pass but required checks/observations are missing')
    return bool(passed)


def check_prior_q0(root, view, engine, compiled, freeze, audit, full):
    path = root/q0_relative(view)
    report = audit.json(path, required=True)
    if report is None:
        return None
    rel = str(path.relative_to(root))
    audit.hash(path, freeze['files'].get(rel))
    audit.check(report.get('engine') == engine and report.get('dense_qualification_status') == 'passed', 'prior Q0 engine/status differs')
    expected_cases = {'base_negative_count_input', 'batch_duplicate', 'initial_threshold_boundary', 'single_sample_no_carry'}
    audit.check(set(report.get('cases', {})) == expected_cases, 'prior Q0 case denominator differs')
    for name, row in report.get('cases', {}).items():
        audit.check(row.get('passed') is True and bool(row.get('checks')) and all(c.get('passed') is True for c in row['checks'].values()), 'prior Q0 does not record passing '+name)
        raw_path = path.parent/(name+'.json')
        artifact = audit.json(raw_path, required=True)
        if artifact is None:
            continue
        audit.hash(raw_path, freeze['files'].get(str(raw_path.relative_to(root))))
        audit.check(artifact.get('checks') == row.get('checks'), 'prior Q0 raw/report checks differ')
        raw = artifact.get('raw', {})
        if engine == 'atlas':
            audit.check(raw.get('runner_sha256') == freeze['files'].get('runtime/arm64-r2/b2-train'), 'Atlas Q0 runtime differs')
            audit.check(raw.get('training_api_sha256') == freeze['files'].get('snapshot/brian2-rust/python/brian2_rust/training.py'), 'Atlas Q0 API differs')
        else:
            audit.check(raw.get('engine') == engine and raw.get('compile') is compiled, 'Torch Q0 engine/compile differs')
    if full:
        for rel, expected in report.get('identities', {}).items():
            audit.hash(rooted(root, rel), expected)
    return sha(path)


def check_epochs(data, batches, directory, slot, root, freeze, audit, full):
    epochs = data.get('epochs', [])
    audit.check(isinstance(epochs, list) and [row.get('epoch') for row in epochs] == list(range(1, len(epochs)+1)) and len(epochs) <= 10, 'completed epochs must be consecutive 1..10')
    clocks, counter_evidence, improved, current_best = [], [], {}, -1
    qualification = data.get('qualification_elapsed_s')
    for row in epochs:
        e = row['epoch']
        for name, count in [('train', 55000), ('validation', 5000)]:
            aggregate(row.get(name), count, audit, f'epoch{e} {name}')
            if full and batches is not None:
                observed = batches['groups'].get((name, e), {})
                audit.check(observed.get('samples') == count, f'epoch{e} lacks full {name} batch records')
                audit.check(observed.get('batches') == (count+31)//32, f'epoch{e} wrong {name} batch count')
                audit.check(observed.get('correct') == row[name]['correct'], f'epoch{e} {name} aggregate correct differs from batch log')
                audit.check(near(observed.get('weighted_loss', -1)/count, row[name]['loss']), f'epoch{e} {name} aggregate loss differs from batch log')
                audit.check(observed.get('batch_seconds', math.inf) <= row[name]['elapsed_s']+1e-6, f'epoch{e} {name} wall excludes batch computation')
        t, remaining, wall = row.get('elapsed_total_s'), row.get('remaining_s'), row.get('epoch_wall_s')
        post = CAP-remaining if number(remaining) else None
        audit.check(number(t) and number(post) and number(qualification) and qualification <= t <= post, f'epoch{e} inconsistent qualification/validation/checkpoint clock')
        audit.check(number(wall) and wall >= row['train']['elapsed_s']+row['validation']['elapsed_s'], f'epoch{e} epoch wall omits train/validation')
        reserve = 2.5*row['validation']['elapsed_s']+2
        audit.check(near(row.get('test_reserve_estimate_s'), reserve) and near(row.get('continue_threshold_s'), 1.1*wall+reserve), f'epoch{e} changed fixed continuation rule')
        if clocks:
            audit.check(post >= clocks[-1]['post_checkpoint_observation_s'], 'epoch clock regressed')
            prior = epochs[e-2]
            audit.check(prior['remaining_s'] > prior['continue_threshold_s'], 'started another completed epoch despite reserve stop rule')
        if full and batches is not None:
            last = batches['groups'].get(('validation', e), {}).get('last_time')
            audit.check(number(last) and last <= t+1e-6, f'epoch{e} committed before final validation batch')
        candidates = []
        if full and batches:
            candidates += [v['elapsed_total_s'] for v in batches['first_phase_batches'] if number(v['elapsed_total_s']) and number(post) and v['elapsed_total_s'] >= post and (v['phase'] == 'test' or v['epoch'] > e)]
        # result.json's timestamp is after the epoch progress write completed;
        # an epoch_complete progress.json timestamp itself is before that write.
        if data.get('_from_result') and number(data.get('elapsed_total_s')) and number(post) and data['elapsed_total_s'] >= post:
            candidates.append(data['elapsed_total_s'])
        marker_path = directory/'test-started.json'
        if marker_path.is_file():
            marker = audit.json(marker_path) or {}
            if number(marker.get('elapsed_total_s')) and number(post) and marker['elapsed_total_s'] >= post:
                candidates.append(marker['elapsed_total_s'])
        upper = min(candidates) if candidates else None
        clocks.append(dict(epoch=e, validation_correct=row['validation']['correct'], post_checkpoint_observation_s=post,
                           all_epoch_overhead_upper_bound_s=upper, upper_bound_available=upper is not None))
        evidence = dict(epoch=e, expected_counter=e*BATCHES, logged_train_batches=e*BATCHES if full and batches else None,
                        independently_recorded_counter=None, basis='frozen runner guards counter; no per-epoch counter value in batches/result')
        if row['validation']['correct'] > current_best:
            current_best = row['validation']['correct']
            suffix = '.json' if slot['engine'] == 'atlas' else '.pt'
            path = directory/f'best-epoch-{e}{suffix}'
            improved[e] = path.name
            if full:
                audit.check(path.is_file(), f'improved epoch{e} checkpoint missing')
                if path.is_file():
                    try:
                        counter = checkpoint_counter(path, slot['engine'])
                        evidence['independently_recorded_counter'] = counter['counters']
                        evidence['basis'] = counter['evidence']
                        evidence['checkpoint_sha256_observed'] = sha(path)
                        evidence['hash_bound_to_selected_record'] = (data.get('selection') or {}).get('epoch') == e
                        audit.check(all(value == e*BATCHES for value in counter['counters']), f'epoch{e} checkpoint Adam counter differs')
                        if slot['engine'] == 'atlas':
                            audit.check(counter['runtime_sha256'] == freeze['files'].get('runtime/arm64-r2/b2-train'), 'native checkpoint runtime differs')
                            for key, expected in [('sizes',[784,128,10]), ('seed',slot['seed']), ('backend','cpu'), ('beta',[.95,.95]), ('threshold',[1.,1.]), ('logit_scale',5.)]:
                                audit.check(counter['plan'].get(key)==expected, 'native checkpoint plan differs: '+key)
                        else:
                            groups = counter['optimizer_groups']
                            audit.check(len(groups)==1 and groups[0].get('params')==[0,1], 'Torch checkpoint optimizer bank identities differ')
                            if len(groups)==1:
                                for key, expected in [('lr',.001),('betas',(.9,.999)),('eps',1e-8),('weight_decay',0)]:
                                    audit.check(groups[0].get(key)==expected, 'Torch checkpoint optimizer differs: '+key)
                    except (OSError, ValueError, KeyError, TypeError, pickle.UnpicklingError, zipfile.BadZipFile, struct.error, EOFError) as error:
                        if slot['engine'] == 'atlas':
                            audit.check(False, f'epoch{e} native checkpoint integrity/counter parse failed: {error}')
                        else:
                            audit.note(f'epoch{e} checkpoint counter unavailable to restricted stdlib parser: {error}')
        counter_evidence.append(evidence)
    selected = data.get('selection')
    if epochs:
        best = best_epoch(epochs)
        audit.check(isinstance(selected, dict), 'complete epochs require selected checkpoint')
        if isinstance(selected, dict):
            audit.check(selected.get('epoch') == best['epoch'] and selected.get('validation_correct') == best['validation']['correct'], 'selection is not maximum integer correct with earliest tie')
            audit.check(near(selected.get('validation_accuracy'), best['validation']['accuracy']), 'selected validation accuracy differs')
            audit.check(selected.get('checkpoint') == improved.get(best['epoch']), 'selected checkpoint name differs from eligible epoch')
            if full and selected.get('checkpoint') == improved.get(best['epoch']):
                audit.hash(directory/selected['checkpoint'], selected.get('sha256'))
    else:
        audit.check(selected is None, 'partial/no epoch cannot select a checkpoint')
    if any(row['independently_recorded_counter'] is None for row in counter_evidence):
        audit.note('Some epoch Adam counters have no independent numeric checkpoint record; batch count plus frozen runtime assertion is weaker evidence')
    return dict(epoch_clocks=clocks, counters=counter_evidence,
                all_epoch_counters_independently_recorded=all(row['independently_recorded_counter'] is not None for row in counter_evidence) if epochs else None)


def check_test(directory, slot, root, contract_hash, data, progress, summary, batches, audit, cap_proved, full):
    view = slot['engine']+('-compile' if slot['compiled'] else '-eager')
    ledger_path = root/'evidence/a1-arm64-long-r1-test-once'/f"{view}-seed-{slot['seed']}-{contract_hash}.json"
    marker = audit.json(directory/'test-started.json')
    ledger = audit.json(ledger_path)
    ledger_directory = Path(ledger.get('run_directory', '')) if ledger is not None else None
    owned = ledger_directory is not None and (ledger_directory.resolve() == directory.resolve() if full else ledger_directory.parts[-3:] == directory.parts[-3:])
    stale_rejected = (ledger is not None and not owned and data.get('status') == 'unqualified'
                      and 'already claimed' in data.get('error', '') and marker is None)
    if stale_rejected:
        audit.note('An earlier durable ledger prevented a repeated formal test; no new test score is eligible')
    if marker is not None:
        audit.check(owned and marker == ledger, 'local test marker does not equal the owned durable ledger')
    if owned:
        for key, expected in [('attempt', 1), ('seed', slot['seed']), ('engine', slot['engine']),
                              ('compile', slot['compiled']), ('contract_sha256', contract_hash), ('selected', data.get('selection'))]:
            audit.check(ledger.get(key) == expected, 'test ledger identity differs: '+key)
        audit.check(data.get('selection') is not None, 'test ledger created without a selected complete epoch')
        audit.check(number(ledger.get('elapsed_total_s')), 'test ledger has invalid clock')
        if number(ledger.get('elapsed_total_s')) and ledger['elapsed_total_s'] > CAP:
            audit.note('Test-attempt marker was timestamped after the cap; no strict-cap success is inferred from that attempt')
    elif ledger is not None and not stale_rejected:
        audit.check(False, 'ledger belongs to a different run; current test identity unproved')
    attempted = marker is not None or ledger is not None
    if summary:
        audit.check(summary.get('test_attempted') is attempted, 'supervisor test-attempt field differs from durable files')
    tests = [group for (phase_, _), group in (batches or {}).get('groups', {}).items() if phase_ == 'test']
    audit.check(len(tests) <= 1, 'multiple test groups recorded')
    if tests or data.get('test_labels_decoded') is True:
        audit.check(owned and marker is not None, 'test consumption lacks exclusive durable marker')
    if tests and owned:
        audit.check(ledger['elapsed_total_s'] <= tests[0]['first_time'], 'test batch precedes test ledger')
    if data.get('selection') is None:
        audit.check(not tests and data.get('test_labels_decoded') is not True, 'held-out test opened without complete-epoch selection')
    for label, source in [('result/progress', data), ('latest progress', progress or {}), ('supervisor', summary or {})]:
        for field in ('test_accuracy', 'test_loss', 'test_correct'):
            audit.check(field not in source, label+' publishes an unstructured test metric')
        if 'test' in source:
            aggregate(source['test'], 10000, audit, label+' held-out')
            audit.check(source.get('test_status') == 'completed_once', label+' publishes score before completed_once')
    aggregate_complete = data.get('test_status') == 'completed_once' and data.get('status') == 'completed' and 'test' in data
    if data.get('status') == 'completed':
        audit.check(aggregate_complete and marker is not None and owned, 'completed worker lacks complete once-only test evidence')
        if batches is not None:
            audit.check(len(tests) == 1 and tests[0]['samples'] == 10000 and tests[0]['batches'] == 313, 'completed test lacks all 10000/313 batches including tail16')
            test_keys = [key for key in batches['groups'] if key[0] == 'test']
            audit.check(bool(test_keys) and test_keys[0][1] == data['selection']['epoch'], 'test epoch differs from selected checkpoint')
        if summary and not summary.get('hard_cap_triggered'):
            audit.check(summary.get('test') == data.get('test'), 'supervisor/result held-out aggregates differ')
    eligible = bool(full and aggregate_complete and cap_proved and not audit.errors)
    return dict(attempt_recorded=attempted, ledger_owned_by_run=owned, complete_aggregate_recorded=aggregate_complete,
                score_eligible_within_strict_cap=eligible,
                held_out_test=data.get('test') if eligible else None,
                recorded_full_aggregate=data.get('test') if aggregate_complete else None,
                limitation='Only the complete aggregate correct/samples and loss arithmetic can be checked. No predictions or test labels are read; accuracy itself is not independently recomputed.')


def quality_rows(clocks, effective, cap_proved):
    result = {}
    for percent, target in [(90, 4500), (95, 4750)]:
        first = next((row for row in clocks if row['validation_correct'] >= target), None)
        upper = first.get('all_epoch_overhead_upper_bound_s') if first else None
        result[str(percent)] = dict(target_validation_correct=target, denominator=5000,
            target_status='target_reached_in_recorded_complete_epoch' if first else ('quality_not_reached_observed' if clocks else 'not_observed_no_complete_epoch'),
            observation_censored=effective != 'completed', epoch=first['epoch'] if first else None,
            post_checkpoint_observation_s=first['post_checkpoint_observation_s'] if first else None,
            time_to_target_upper_bound_s=upper,
            strict_all_overhead_time_available=number(upper) and upper <= CAP,
            completed_seed=effective == 'completed' and cap_proved,
            timing_note='Upper bound is the earliest later recorded event after epoch progress serialization; includes cold imports, shape qualification, input construction, train/validation/checkpoint and logging overhead, and may include part of the next operation. It is not an exact crossing time.')
    return result


def validate_slot(root, run, slot, external, freeze, shared, qhash, full):
    name = f"{slot['view']}-seed-{slot['seed']}"
    directory = run/name
    audit = Audit(name)
    outer_reason = external.get('termination_reason')
    interrupted = outer_reason in ('timeout', 'resource_limit') or external.get('exit_code', 0) < 0
    summary = audit.json(directory/'terminal.json', interrupted=interrupted)
    interrupted |= bool(summary and summary.get('hard_cap_triggered'))
    result = audit.json(directory/'result.json', interrupted=interrupted)
    progress = audit.json(directory/'progress.json', interrupted=interrupted)
    data = dict(result if result is not None else progress or {})
    data['_from_result'] = result is not None
    effective = (outer_reason if outer_reason in ('timeout', 'resource_limit') else
                 summary.get('status') if summary else data.get('status') if data else
                 'worker_result_missing' if external or directory.exists() else 'not_launched')
    row = dict(**slot, slot=name, effective_status=effective, outer_termination_reason=outer_reason,
               outer_exit_code=external.get('exit_code'), worker_status=data.get('status'), errors=audit.errors,
               limitations=audit.limitations, elapsed_supervisor_s=(summary or {}).get('elapsed_total_s'),
               completed_epochs=len(data.get('epochs', [])), cap_s=CAP,
               scientific_completion=effective == 'completed', identity_verified=False,
               strict_cap_proved=False, quality=quality_rows([], effective, False), test={})
    if not external and not directory.exists():
        audit.note('Scheduled denominator retained; slot has not produced launch evidence')
        return row
    if external:
        audit.check(external.get('name') == name, 'outer launch name differs')
        audit.check(external.get('timeout_s') == CAP+30 and external.get('rss_guard_bytes') == 64*1024**3, 'outer cap/RSS guard differs')
        command_identity(external.get('command', []), slot, directory, root, audit, physical_root=full)
        if external.get('remaining_owned_processes'):
            audit.note('Outer supervisor recorded surviving descendants; cleanup is not proved and scientific success is withheld')
        if external.get('child_terminal_sha256'):
            audit.hash(directory/'terminal.json', external['child_terminal_sha256'])
        if summary and external.get('child_status') is not None:
            audit.check(external['child_status'] == summary.get('status'), 'queue child status differs from inner supervisor')
            audit.check(external.get('completed_epochs') == summary.get('completed_epochs') and external.get('test_status') == summary.get('test_status'), 'queue child summary fields differ')
    launch = audit.json(directory/'launch.json', required=bool(summary or data), interrupted=interrupted)
    if launch:
        flags = command_identity(launch.get('command', []), slot, directory, root, audit, worker=True, physical_root=full)
        audit.check(launch.get('contract_sha256') == freeze['files'].get(CONTRACT), 'inner launch contract identity differs')
        audit.check(launch.get('script_sha256') == freeze['files'].get('tools/run_a1_mnist_arm64_long_r1.py'), 'inner launch source identity differs')
        audit.check(launch.get('total_wall_cap_s') == CAP, 'inner launch strict cap differs')
        try:
            started, deadline = float(flags['--started']), float(flags['--deadline'])
            audit.check(near(started, launch.get('started_monotonic')) and near(deadline-started, CAP), 'worker deadline/start differs from supervisor cap')
        except (KeyError, ValueError, TypeError):
            audit.check(False, 'worker command lacks valid start/deadline')
        audit.check(all(value == '1' for value in launch.get('thread_environment', {}).values()) and len(launch.get('thread_environment', {})) == 4, 'requested compute thread environment differs')
    if summary:
        audit.check(summary.get('worker_result_present') is (directory/'result.json').exists(), 'worker-result-presence summary differs')
        audit.check(summary.get('completed_epochs') == len(data.get('epochs', [])), 'supervisor completed epoch count differs')
        audit.check(summary.get('selected_checkpoint') == data.get('selection'), 'supervisor selected checkpoint differs')
        if summary.get('owned_descendants_terminated', {}).get('remaining_non_zombie'):
            audit.check(summary.get('status') == 'cleanup_not_verified_do_not_launch_next_case', 'remaining descendants not reflected in supervisor status')
            audit.note('Inner supervisor recorded surviving descendants; cleanup is not proved')
        if summary.get('owned_descendants_terminated', {}).get('inspection_error') or summary.get('descendant_inspection_errors'):
            audit.check(summary.get('status') == 'cleanup_not_verified_do_not_launch_next_case', 'cleanup errors not reflected in effective status')
        if summary.get('hard_cap_triggered'):
            audit.check(summary.get('status') in ('censored_at_cap', 'cleanup_not_verified_do_not_launch_next_case'), 'hard cap not reflected in supervisor status')
            audit.check('test' not in summary, 'hard-cap supervisor must suppress final test score')
        elif data:
            audit.check(summary.get('status') in (data.get('status'), 'cleanup_not_verified_do_not_launch_next_case'), 'supervisor status differs from worker')
    if not data:
        audit.note('No complete worker report; recorded timeout/resource/software outcome retained without fabricated samples or score')
        return row
    for source_name, source in [('worker', data), ('progress', progress or {})]:
        if not source: continue
        for key, expected in [('schema', 'a1-mnist-run-arm64-long-r1'), ('engine', slot['engine']), ('compile', slot['compiled']), ('seed', slot['seed']), ('total_wall_cap_s', CAP)]:
            audit.check(source.get(key) == expected, source_name+' identity differs: '+key)
        for rel in SOURCE_KEYS:
            audit.check(source.get('source_hashes', {}).get(rel) == freeze['files'].get(rel), source_name+' source hash differs: '+rel)
    if result and progress:
        audit.check(progress.get('epochs', []) == result.get('epochs', [])[:len(progress.get('epochs', []))], 'progress epochs not a prefix of result')
        audit.check(number(progress.get('elapsed_total_s')) and number(result.get('elapsed_total_s')) and progress['elapsed_total_s'] <= result['elapsed_total_s'], 'result clock precedes latest progress')
    needs_arrays = bool(data.get('common_arrays_sha256') or data.get('epochs') or data.get('A1_shape_qualification') == 'passed')
    if needs_arrays:
        audit.check(shared is not None, 'worker used arrays without shared manifest')
        if shared:
            audit.check(data.get('contract_sha256') == freeze['files'].get(CONTRACT), 'worker contract hash differs')
            audit.check(data.get('common_arrays_sha256') == shared['files'][f"seed-{slot['seed']}.npz"]['sha256'], 'worker initial/shuffle arrays differ')
            audit.check(data.get('shared_manifest_sha256') == sha(root/'fixtures/a1-arm64-long-r1/manifest.json'), 'worker shared manifest differs')
        audit.check(data.get('q0_report_sha256') == qhash, 'worker prior Q0 differs')
        for name, expected in data.get('environment_lock_sha256', {}).items():
            if full: audit.hash(rooted(root/'environment', name), expected)
        audit.check(data.get('environment_lock_sha256', {}).get('cpu-lock.txt') == freeze['files'].get('environment/cpu-lock.txt'), 'worker CPU environment lock differs')
    if slot['engine'] == 'atlas' and (needs_arrays or data.get('runtime_sha256')):
        audit.check(data['runtime_sha256'] == freeze['files'].get('runtime/arm64-r2/b2-train'), 'worker Atlas runtime differs')
        audit.check(data.get('training_api_sha256') == freeze['files'].get('snapshot/brian2-rust/python/brian2_rust/training.py'), 'worker Atlas API differs')
    batches = None
    if full and (directory/'batches.jsonl').is_file():
        batches = check_batches(directory/'batches.jsonl', audit, interrupted=interrupted)
    training_started = bool(data.get('epochs') or (batches and batches['rows']))
    row['actual_shape_qualification_passed'] = check_shape_qualification(directory/'a1-shape-qualification.json', slot['engine'], audit, required=training_started or effective == 'completed')
    if training_started:
        audit.check(data.get('A1_shape_qualification') == 'passed' and number(data.get('qualification_elapsed_s')), 'formal training lacks recorded qualification completion')
    if full and (data.get('epochs') or effective == 'completed'):
        audit.check(batches is not None, 'completed epochs lack batch log')
    epochs = check_epochs(data, batches, directory, slot, root, freeze, audit, full)
    row.update(epochs)
    if batches:
        row['batch_log'] = dict(rows=batches['rows'], last_time_s=batches['last_time'],
            groups=[dict(phase=key[0], epoch=key[1], samples=value['samples'], batches=value['batches']) for key, value in batches['groups'].items()])
        partial_epochs = {epoch for phase_, epoch in batches['groups'] if phase_ != 'test' and epoch > len(data.get('epochs', []))}
        audit.check(partial_epochs <= {len(data.get('epochs', []))+1}, 'uncommitted batch log spans multiple epochs')
    wall = (summary or {}).get('elapsed_total_s')
    worker_wall = data.get('elapsed_total_s')
    cap_proved = bool(effective == 'completed' and summary and summary.get('exit_code') == 0 and not summary.get('hard_cap_triggered')
                      and not external.get('remaining_owned_processes') and not summary.get('owned_descendants_terminated', {}).get('remaining_non_zombie')
                      and number(wall) and wall <= CAP and number(worker_wall) and worker_wall <= CAP
                      and (batches is None or batches['last_time'] <= CAP))
    row['strict_cap_proved'] = cap_proved
    if effective == 'completed':
        audit.check(data.get('status') == 'completed' and result is not None, 'completed supervisor lacks completed final worker report')
        audit.check(external.get('exit_code', 0) == 0 and (summary or {}).get('exit_code') == 0, 'completed slot has nonzero exit')
        if not cap_proved:
            audit.note('Complete aggregate retained as recorded but strict end-to-end 1800s proof failed; no eligible test score or paired timing. Teardown/poll latency is not classified as evidence corruption or given a grace budget.')
    row['quality'] = quality_rows(epochs['epoch_clocks'], effective, cap_proved)
    row['test'] = check_test(directory, slot, root, freeze['files'].get(CONTRACT), data, progress, summary, batches, audit, cap_proved, full)
    row['identity_verified'] = full and not audit.errors and needs_arrays
    return row


def summarize(rows, evidence_ok, full):
    by_key = {(row['view'], row['seed']): row for row in rows}
    result = {}
    for name, _, _ in VIEWS:
        current = [by_key[(name, seed)] for seed in SEEDS]
        summary = dict(formal_seed_count=5, recorded_seed_count=len(current),
            outcomes={str(row['seed']): row['effective_status'] for row in current},
            completed_seed_count=sum(row['effective_status'] == 'completed' for row in current),
            strict_cap_proved_count=sum(row['strict_cap_proved'] for row in current),
            per_seed=[dict(seed=row['seed'], execution=row['effective_status'], quality=row['quality'], test=row.get('test', {})) for row in current],
            target_pairs={})
        tests = [row.get('test', {}) for row in current]
        all_tests = bool(full and evidence_ok and all(test.get('score_eligible_within_strict_cap') for test in tests))
        summary['complete_five_seed_test_summary'] = all_tests
        if all_tests:
            values = [test['held_out_test']['accuracy'] for test in tests]
            summary['test_accuracy_mean'] = statistics.mean(values)
            summary['test_accuracy_sample_sd'] = statistics.stdev(values)
            summary['test_accuracy_scope'] = 'Five complete reported aggregates; predictions/labels and correctness not independently recomputed'
        for target in ('90', '95'):
            pairs = []
            for seed in SEEDS:
                left, right = by_key[('atlas', seed)], by_key[(name, seed)]
                lq, rq = left['quality'][target], right['quality'][target]
                eligible = bool(full and evidence_ok and not left['errors'] and not right['errors'] and
                    lq['completed_seed'] and rq['completed_seed'] and lq['strict_all_overhead_time_available'] and rq['strict_all_overhead_time_available'])
                pairs.append(dict(seed=seed, eligible=eligible, atlas_status=left['effective_status'], view_status=right['effective_status'],
                    atlas_target_status=lq['target_status'], view_target_status=rq['target_status'],
                    atlas_time_upper_bound_s=lq['time_to_target_upper_bound_s'], view_time_upper_bound_s=rq['time_to_target_upper_bound_s']))
            complete = len(pairs) == 5 and all(pair['eligible'] for pair in pairs)
            pair_summary = dict(complete_five_seed_pairing=complete, eligible_n=sum(pair['eligible'] for pair in pairs), denominator=5,
                seeds=pairs, no_subset_summary=True,
                scope='Conservative all-overhead upper bounds, not exact crossing times; do not interpret their ratio as an exact speedup or engine leadership.')
            if complete:
                pair_summary['median_atlas_upper_bound_s'] = statistics.median(p['atlas_time_upper_bound_s'] for p in pairs)
                pair_summary['median_view_upper_bound_s'] = statistics.median(p['view_time_upper_bound_s'] for p in pairs)
                pair_summary['median_paired_upper_bound_difference_s'] = statistics.median(p['atlas_time_upper_bound_s']-p['view_time_upper_bound_s'] for p in pairs)
            summary['target_pairs'][target] = pair_summary
        result[name] = summary
    return result


def validate(root, run, mode):
    full = mode == 'full'
    if full and socket.gethostname().split('.')[0] != 'rock-mac-studio-1':
        raise ValueError('Full evidence reads are restricted to authorized100.90.28.27')
    audit = Audit('queue')
    out = dict(schema='a1-arm64-r1-evidence-validation-v1', mode=mode, root=str(root), run=str(run),
        validator_sha256=sha(__file__), performance_run=False, errors=audit.errors, limitations=audit.limitations, rows=[])
    freeze = audit.json(run/'freeze.json', required=True)
    if not isinstance(freeze, dict) or not isinstance(freeze.get('files'), dict):
        audit.check(False, 'queue freeze lacks file identities')
        out['status'] = 'evidence_inconsistent'
        return out
    audit.check(freeze.get('schema') == 'a1-queue-arm64-r1' and freeze.get('finite_slots') == 25, 'queue freeze schema/denominator differs')
    audit.check(freeze.get('order') == plan(), 'queue order must be exact rotated five views times five seeds')
    for key, expected in [('per_worker_total_wall_s', 1800), ('per_slot_outer_supervision_s', 1830), ('preparation_cap_s', 300), ('stage_execution_cap_s', 48600)]:
        audit.check(freeze.get(key) == expected, 'queue freeze cap differs: '+key)
    audit.check(set(CORE) <= set(freeze['files']), 'queue freeze lacks required source/runtime/environment files')
    if full:
        for rel, expected in freeze['files'].items():
            try:
                audit.hash(rooted(root, rel), expected)
            except (OSError, ValueError) as error:
                audit.check(False, 'invalid frozen identity path: '+str(error))
    else:
        audit.note('Report-only mode: live source/runtime bytes, full batch logs, common array payloads and checkpoint counters are not verified; no paired result is eligible')
    contract = audit.json(root/CONTRACT, required=True) or {}
    audit.hash(root/CONTRACT, freeze['files'].get(CONTRACT))
    for key, expected in [('status', 'frozen_pretraining'), ('formal_seeds', SEEDS), ('sizes', [784, 128, 10]),
                          ('T', 100), ('global_batch', 32), ('train_samples', 55000), ('validation_samples', 5000),
                          ('test_samples', 10000), ('max_epochs', 10), ('total_wall_cap_seconds_per_seed', CAP),
                          ('time_to_quality_targets', [.9, .95]), ('implementation_sha256', freeze['files'].get('tools/run_a1_mnist_arm64_long_r1.py'))]:
        audit.check(contract.get(key) == expected, 'frozen A1 contract differs: '+key)
    terminal = audit.json(run/'terminal.json')
    progress = audit.json(run/'progress.json')
    if progress is not None:
        audit.check(isinstance(progress, list), 'queue progress must be a record array')
    records = terminal.get('records', []) if isinstance(terminal, dict) else progress if isinstance(progress, list) else []
    audit.check(isinstance(records, list), 'queue terminal records must be an array')
    if not isinstance(records, list): records = []
    if isinstance(progress, list) and terminal:
        audit.check(progress == records[:len(progress)], 'queue progress is not a prefix of terminal records')
    if terminal:
        status = terminal.get('status')
        out['queue_status'] = status
        if status == 'finite_a1_queue_exited':
            audit.check(len(records) == 25 and terminal.get('finite_slots') == 25, 'finite queue must retain all 25 outcomes')
        elif status == 'coordinator_stopped':
            audit.check(terminal.get('finite_slots') == 25 and terminal.get('unexecuted_slots') == plan()[len(records):], 'stopped queue discarded or changed unexecuted slots')
        elif status in ('prerequisite_pending', 'prerequisite_incomplete'):
            audit.check(not records, 'prerequisite failure cannot have formal records')
        else:
            audit.check(False, 'unrecognized A1 queue terminal status')
    else:
        out['queue_status'] = 'no_terminal_snapshot'
        audit.note('Queue has no final terminal snapshot; scheduled/pending slots remain explicit and five-seed summaries are withheld')
    barrier = audit.json(run/'barrier.json', required=bool(records))
    if barrier:
        audit.check(barrier.get('status') == 'released', 'queue prerequisite barrier not released')
        prior_path = root/'evidence/extended-followup-arm64-r5/terminal.json'
        if full:
            audit.hash(prior_path, barrier.get('prerequisite_sha256'))
            prior = audit.json(prior_path, required=True) or {}
            audit.check(prior.get('status') == 'finite_extension_queue_exited', 'A1 prerequisite terminal differs')
    preparation = audit.json(run/'preparation.json', required=bool(records))
    prepared = bool(preparation and preparation.get('exit_code') == 0 and preparation.get('termination_reason') == 'exited')
    if preparation:
        audit.check(preparation.get('name') == 'shared-arrays' and preparation.get('timeout_s') == 300, 'shared array preparation identity/cap differs')
        audit.check(not preparation.get('remaining_owned_processes'), 'preparation left owned processes')
        flags = command_flags(preparation.get('command', []), audit)
        audit.check(flags.get('--prepare-shared') is True and '--allow-run' not in flags, 'preparation command is not array-only')
    if records:
        audit.check(prepared, 'formal slots launched without completed shared-array preparation')
    shared = check_shared(root, freeze, audit, full, required=True) if prepared or records else None
    if prepared:
        prep_log = run/'shared-arrays.log'
        if prep_log.is_file():
            lines = []
            for line in prep_log.read_text().splitlines():
                try:
                    value = loads(line)
                    if isinstance(value, dict) and value.get('status') == 'prepared': lines.append(value)
                except ValueError: pass
            audit.check(len(lines) == 1 and lines[0].get('sha256') == sha(root/'fixtures/a1-arm64-long-r1/manifest.json'), 'preparation stdout does not bind the shared manifest')
        else:
            audit.check(False, 'preparation stdout manifest receipt missing')
    prior_hashes = {}
    for name, engine, compiled in VIEWS:
        try:
            prior_hashes[name] = check_prior_q0(root, name, engine, compiled, freeze, audit, full)
        except (OSError, ValueError, KeyError, TypeError) as error:
            audit.check(False, f'{name} prior Q0 validation failed: {error}')
    indexed = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            audit.check(False, 'non-object launch record')
            continue
        identity = (record.get('view'), record.get('seed'))
        audit.check(identity not in indexed, 'duplicate formal view/seed launch')
        indexed[identity] = record
        if index < 25:
            audit.check(all(record.get(k) == v for k, v in plan()[index].items()), 'launch record order/identity differs')
        else:
            audit.check(False, 'more than 25 formal slot records')
    for slot in plan():
        key = (slot['view'], slot['seed'])
        name = f"{slot['view']}-seed-{slot['seed']}"
        direct = audit.json(run/(name+'-terminal.json'))
        external = indexed.get(key, direct or {})
        if direct and key in indexed:
            audit.check(all(indexed[key].get(k) == v for k, v in direct.items()), name+' outer launch receipt differs from queue record')
        try:
            row = validate_slot(root, run, slot, external, freeze, shared, prior_hashes.get(slot['view']), full)
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
            row = dict(**slot, slot=name, effective_status=external.get('termination_reason', 'unreadable'),
                errors=[f'evidence parsing/shape failure: {type(error).__name__}: {error}'], limitations=[],
                completed_epochs=0, strict_cap_proved=False, quality=quality_rows([], 'unreadable', False), test={})
        out['rows'].append(row)
        audit.errors.extend(name+': '+message for message in row['errors'])
    complete_ledger = terminal is not None and terminal.get('status') == 'finite_a1_queue_exited' and len(records) == 25
    out['denominator'] = dict(formal_slots=25, formal_seeds=SEEDS, views=[v[0] for v in VIEWS],
        recorded_outer_slots=len(records), retained_rows=len(out['rows']), complete_terminal_ledger=complete_ledger)
    out['views'] = summarize(out['rows'], not audit.errors and complete_ledger, full)
    out['all_25_scientifically_completed'] = all(row['effective_status'] == 'completed' for row in out['rows'])
    out['all_25_strict_cap_proved'] = all(row['strict_cap_proved'] for row in out['rows'])
    audit.note('A valid timeout, budget_rejected, unqualified, dependency failure or resource guard remains a scientific outcome, not automatically damaged evidence; no failed seed is removed')
    audit.note('Recorded qualification checks are audited but the numerical oracle is never rerun')
    audit.note('Counter evidence is independently numeric only at readable saved checkpoints. The frozen runner asserts counters at every full training epoch, but does not serialize every such value')
    audit.note('Strict cap eligibility conservatively requires inner supervisor total <=1800s, not the outer 1830s cleanup allowance. A late terminal/teardown can withhold a score without proving model computation exceeded the cap')
    audit.note('Exact all-overhead quality crossing time was not serialized: post-checkpoint observation is pre-progress-write; the validator reports a later observed upper bound and never substitutes training-only time')
    audit.note('Complete held-out correct/loss aggregates are source-bound recorded evidence only. No held-out payloads, labels, predictions or models are read or evaluated')
    out['status'] = 'evidence_inconsistent' if audit.errors else ('evidence_consistent_full_checks_with_limits' if full else 'report_checks_consistent_full_payload_unverified')
    return out


def self_test():
    """Tiny hand-made structures only; never opens an evaluation/data directory."""
    checks = {}
    counts = {'train': 33, 'validation': 1, 'test': 1}
    rows = [dict(phase='train', epoch=1, offset=0, samples=32, elapsed_s=.1, elapsed_total_s=1., public_api_s=None, loss=1., correct=20),
            dict(phase='train', epoch=1, offset=32, samples=1, elapsed_s=.1, elapsed_total_s=2., public_api_s=None, loss=.5, correct=1),
            dict(phase='validation', epoch=1, offset=0, samples=1, elapsed_s=.1, elapsed_total_s=3., public_api_s=None, loss=.4, correct=1),
            dict(phase='test', epoch=1, offset=0, samples=1, elapsed_s=.1, elapsed_total_s=4., public_api_s=None)]
    with tempfile.TemporaryDirectory(prefix='a1-validator-synthetic-') as temporary:
        folder = Path(temporary)
        def inspect(values, suffix='', interrupted=False):
            path = folder/'tiny.jsonl'
            path.write_text(''.join(json.dumps(v)+'\n' for v in values)+suffix)
            a = Audit()
            result = check_batches(path, a, interrupted=interrupted, counts=counts)
            return a, result
        a, value = inspect(rows)
        checks['scalar_batch_structure'] = not a.errors and value['groups'][('train', 1)]['samples'] == 33
        bad = [dict(v) for v in rows]; bad[1]['offset'] = 0
        a, _ = inspect(bad); checks['duplicate_offset_rejected'] = bool(a.errors)
        bad = [dict(v) for v in rows]; bad[-1]['correct'] = 1
        a, _ = inspect(bad); checks['partial_test_score_rejected'] = bool(a.errors)
        a, _ = inspect(rows[:1], '{"phase":', interrupted=True)
        checks['legal_interrupted_final_fragment'] = not a.errors and bool(a.limitations)
        a, _ = inspect(rows[:1], '{"phase":', interrupted=False)
        checks['uncensored_corrupt_fragment_rejected'] = bool(a.errors)
        epochs = [dict(epoch=1, validation={'correct': 4750}), dict(epoch=2, validation={'correct': 4750})]
        checks['earliest_integer_tie'] = best_epoch(epochs)['epoch'] == 1
        try:
            MetadataOnlyUnpickler(io.BytesIO(b'cos\nsystem\n.')).load()
            checks['arbitrary_pickle_global_blocked'] = False
        except ValueError:
            checks['arbitrary_pickle_global_blocked'] = True
        step = lambda key: dict(kind='tensor', storage=dict(kind='storage', dtype='DoubleStorage', key=key, count=1), offset=0, size=(), stride=())
        metadata = {'optimizer': {'state': {0: {'step': step('0')}, 1: {'step': step('1')}}}}
        archive = folder/'tiny.pt'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('archive/data.pkl', pickle.dumps(metadata, protocol=2))
            z.writestr('archive/byteorder', 'little')
            z.writestr('archive/data/0', struct.pack('<d', 1719.))
            z.writestr('archive/data/1', struct.pack('<d', 1719.))
        checks['stdlib_scalar_counter_decode'] = checkpoint_counter(archive, 'snntorch_fp64')['counters'] == [1719, 1719]
        synthetic = []
        for slot in plan():
            q = quality_rows([dict(epoch=1, validation_correct=4750, post_checkpoint_observation_s=2., all_epoch_overhead_upper_bound_s=3.)], 'completed', True)
            synthetic.append(dict(**slot, effective_status='completed', errors=[], strict_cap_proved=True, quality=q, test={}))
        summary = summarize(synthetic, True, True)
        checks['five_seed_pairs_only'] = summary['sj-layerwise']['target_pairs']['90']['complete_five_seed_pairing'] is True
        synthetic[1]['quality'] = quality_rows([], 'budget_rejected', False)
        synthetic[1]['effective_status'] = 'budget_rejected'
        summary = summarize(synthetic, True, True)
        pair = summary['sj-layerwise']['target_pairs']['90']
        checks['failure_seed_retained_no_subset_average'] = len(summary['sj-layerwise']['per_seed']) == 5 and not pair['complete_five_seed_pairing'] and 'median_view_upper_bound_s' not in pair
        try:
            not_sealed(folder/'t10k-labels-idx1-ubyte')
            checks['held_out_open_guard'] = False
        except ValueError:
            checks['held_out_open_guard'] = True
    with tempfile.TemporaryDirectory(prefix='a1-validator-receipts-') as temporary:
        root = Path(temporary)
        run = root/'evidence/a1-queue-arm64-r1'
        fixture = root/'fixtures/a1-arm64-long-r1'
        fixture.mkdir(parents=True)
        (fixture/'manifest.json').write_text('{}')
        frozen_files = {name: 'a'*64 for name in CORE}
        frozen = {'files': frozen_files}
        shared = {'files': {'seed-11.npz': {'sha256': 'b'*64}}}
        for status, timed_out in [('budget_rejected', False), ('censored_at_cap', True)]:
            slot = dict(view='atlas', engine='atlas', compiled=False, seed=11)
            directory = run/'atlas-seed-11'
            directory.mkdir(parents=True)
            command = ['python', str(root/'tools/run_a1_mnist_arm64_long_r1.py'), '--worker', '--allow-run',
                       '--root', str(root), '--contract', str(root/CONTRACT), '--engine', 'atlas', '--seed', '11',
                       '--output', str(directory), '--q0-report', str(root/q0_relative('atlas')),
                       '--started', '0.0', '--deadline', '1800.0']
            launch = dict(command=command, contract_sha256='a'*64, script_sha256='a'*64, started_monotonic=0.,
                          total_wall_cap_s=1800, thread_environment={key:'1' for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']})
            data = dict(schema='a1-mnist-run-arm64-long-r1', engine='atlas', compile=False, seed=11, status='started' if timed_out else status,
                        total_wall_cap_s=1800, source_hashes={key:'a'*64 for key in SOURCE_KEYS},
                        epochs=[], selection=None, phase='A1_shape_qualification', elapsed_total_s=1799. if timed_out else 15.,
                        test_status='sealed_no_selected_checkpoint', test_labels_decoded=False,
                        common_arrays_sha256='b'*64, shared_manifest_sha256=sha(fixture/'manifest.json'),
                        contract_sha256='a'*64, q0_report_sha256='c'*64, environment_lock_sha256={'cpu-lock.txt':'a'*64},
                        runtime_sha256='a'*64, training_api_sha256='a'*64)
            summary = dict(status=status, exit_code=-9 if timed_out else 1, hard_cap_triggered=timed_out,
                           elapsed_total_s=1800.2 if timed_out else 15.1, worker_result_present=not timed_out,
                           selected_checkpoint=None, completed_epochs=0, test_attempted=False, test_status='sealed_no_complete_epoch',
                           owned_descendants_terminated={'captured':[],'remaining_non_zombie':[]}, descendant_inspection_errors=[])
            for filename, value in [('launch.json',launch), ('terminal.json',summary), ('progress.json' if timed_out else 'result.json',data)]:
                (directory/filename).write_text(json.dumps(value))
            row = validate_slot(root, run, slot, {}, frozen, shared, 'c'*64, False)
            checks['legitimate_'+status+'_receipt_not_corruption'] = not row['errors'] and row['effective_status']==status and not row['test'].get('score_eligible_within_strict_cap')
            for file in directory.iterdir(): file.unlink()
            directory.rmdir()
    return dict(schema='a1-validator-synthetic-tests-v1', status='passed' if all(checks.values()) else 'failed',
                tests=checks, models_imported=False, real_evidence_read=False, held_out_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--run', type=Path, default=Path('evidence/a1-queue-arm64-r1'))
    parser.add_argument('--mode', choices=['report-only', 'full'], default='report-only')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        okay = result['status'] == 'passed'
    else:
        if args.root is None:
            parser.error('--root is required for evidence validation')
        if args.mode == 'full' and socket.gethostname().split('.')[0] != 'rock-mac-studio-1':
            parser.error('Full fixture/log/checkpoint validation is restricted to authorized 100.90.28.27')
        root = args.root.resolve()
        run = args.run.resolve() if args.run.is_absolute() else root/args.run
        try:
            result = validate(root, run, args.mode)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            result = dict(schema='a1-arm64-r1-evidence-validation-v1', status='evidence_inconsistent', errors=[f'{type(error).__name__}: {error}'])
        okay = not result.get('errors')
    content = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(content)
        print(json.dumps(dict(status=result['status'], output=str(args.output), sha256=sha(args.output))))
    else:
        print(content, end='')
    return 0 if okay else 1


if __name__ == '__main__':
    raise SystemExit(main())
