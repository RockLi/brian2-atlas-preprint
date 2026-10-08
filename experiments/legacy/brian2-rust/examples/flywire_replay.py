"""Repeat the FlyWire comparison using existing compiled artifacts."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

import numpy as np

from flywire_benchmark import replay
from performance_suite import benchmark_environment, inspect_rustc, validate_rustc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, out = args.builds.resolve(), args.output.resolve()
    report = json.loads((base/'report.json').read_text())
    toolchain = inspect_rustc(report['rustc']['resolved_path'])
    validate_rustc(toolchain)
    os.environ.update(benchmark_environment(toolchain))
    out.mkdir(parents=True, exist_ok=False)
    levels, repeats = report['threads'], report['repeats']
    with np.load(base/'aot-t1/snapshot.npz') as saved:
        expected = {name: saved[name] for name in saved.files}
    samples = {str(t): {'aot': [], 'cpp': []} for t in levels}
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        temporary = Path(temporary)
        for t in levels:
            for backend in ('aot', 'cpp'):
                folder = base/('aot-t1' if backend == 'aot' else f'cpp-t{t}')
                replay(backend, folder, temporary/f'warm-{backend}-{t}', t, expected,
                       cleanup=True)
        for i in range(repeats):
            order = levels[i % len(levels):] + levels[:i % len(levels)]
            for t in order:
                for backend in (('aot', 'cpp') if i % 2 == 0 else ('cpp', 'aot')):
                    print(f'[replay] {i+1}/{repeats} {backend} threads={t}', flush=True)
                    folder = base/('aot-t1' if backend == 'aot' else f'cpp-t{t}')
                    samples[str(t)][backend].append(replay(
                        backend, folder, temporary/f'{i}-{backend}-{t}', t, expected,
                        cleanup=True))
    summary = {}
    for t, backends in samples.items():
        row = summary[t] = {}
        for backend, values in backends.items():
            timings = [v['loop_seconds'] for v in values]
            median = statistics.median(timings)
            row[backend] = {'median_seconds': median,
                            'relative_spread': (max(timings)-min(timings))/median,
                            'native_peak_rss_bytes': max(v['peak_rss_bytes'] for v in values)}
        row['cpp_over_aot_speedup'] = row['cpp']['median_seconds']/row['aot']['median_seconds']
    source = Path(__file__).resolve().parents[1]
    report.update(samples=samples, summary=summary,
                  measurement_stable=all(v[b]['relative_spread'] <= .15
                                         for v in summary.values() for b in ('aot', 'cpp')),
                  reused_artifacts=str(base),
                  measurement_source_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
                  measurement_source_dirty=bool(subprocess.check_output(
                      ['git', 'status', '--porcelain'], cwd=source, text=True).strip()),
                  measurement_note='Fresh stdlib-only supervisor before native fork/exec; '
                  'five interleaved replays after warm-up; native wall time excludes supervisor startup.')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'stable': report['measurement_stable'], 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
