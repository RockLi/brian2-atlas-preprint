"""Validate full result identity and measure optional derived time storage."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def equal(left, right, *, omit_times=False):
    if isinstance(left, dict):
        expected = set(left) - ({'times', 'spike_times'} if omit_times else set())
        assert set(right) == expected
        for key in right:equal(left[key], right[key], omit_times=omit_times)
    elif isinstance(left, list):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):equal(x, y, omit_times=omit_times)
    elif isinstance(left, np.ndarray):
        assert left.shape == right.shape and left.dtype == right.dtype
        for start in range(0, len(left), 131072):
            assert np.array_equal(left[start:start + 131072], right[start:start + 131072])
    else:
        assert left == right


def time_storage(data):
    arrays = []
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ['times', 'spike_times']:
                    assert isinstance(item, np.ndarray)
                    arrays.append(item)
                else:visit(item)
        elif isinstance(value, list):
            for item in value:visit(item)
    visit(data)
    assert len({id(a) for a in arrays}) == len(arrays)
    return sum(a.nbytes for a in arrays), len(arrays)


def main(args):
    source = Path(__file__).resolve().parents[1] / 'python/brian2_rust/results.py'
    current = load_module(source, 'current_results')
    old = load_module(args.baseline_reader, 'old_results')
    baseline = json.loads((args.baseline / 'activity.json').read_text())
    assert sha(args.model) == baseline['model_sha256']
    for name, digest in baseline['result_sha256'].items():assert sha(args.results / name) == digest
    model = json.loads(args.model.read_text())
    started = time.perf_counter()
    if args.mode == 'compare':
        full = old.load_results(model, args.results)
        default = current.load_results(model, args.results)
        equal(full, default)
        del default
        gc.collect()
        lean = current.load_results(model, args.results, include_times=False)
        equal(full, lean, omit_times=True)
        full_bytes, count = time_storage(full)
        assert time_storage(lean) == (0, 0)
        report = dict(default_matches_pinned_reader_exactly=True,
                      tick_only_retained_data_exact=True, omitted_time_bytes=full_bytes,
                      omitted_time_arrays=count)
        metadata = lean['metadata']
    else:
        data = current.load_results(model, args.results, include_times=args.mode == 'default')
        total, count = time_storage(data)
        report = dict(derived_time_bytes=total, derived_time_arrays=count)
        metadata = data['metadata']
    report.update(schema='b2-mam-result-reader-modes-v1', mode=args.mode,
        model_sha256=sha(args.model), reader_sha256=sha(source), baseline_reader_sha256=sha(args.baseline_reader),
        spikes=metadata['spike_count'], elapsed_seconds=time.perf_counter() - started,
        scope='Postprocessing storage optimization with all result validation retained; no scientific or simulation-performance acceptance claim.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'results', 'baseline', 'baseline-reader', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=['default', 'lean', 'compare'], required=True)
    main(parser.parse_args())
