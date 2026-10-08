"""One sequential post-primary analysis, within one shared three-hour guard.

This payload must run under the admitted node23 systemd/resource guard. It
never starts a simulation and does not confer scientific acceptance.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from mam_primary_resources import LABEL, MODEL, audit as audit_resources

BASE = Path('/data/brick2/brian2-mpi-region-20260907')
TOTAL_SECONDS = 10800
OUTPUT = BASE/'primary-postrun-v1'
AUDIT_OUTPUT = BASE/'primary-output-full-v1-report.json'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    check(path.stat().st_size <= 32*2**20, 'oversized control file')
    return json.loads(path.read_text())


def sha(path):
    import hashlib
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resource_gate(directory):
    names = ['admission', 'launch', 'collected', 'runtime']
    required = [directory/(n+'.json') for n in names+['report', 'input-sha256']]
    if not all(p.exists() for p in required):
        return None
    control = {n: read(directory/(n+'.json')) for n in names}
    report = audit_resources(**control)
    check(report == read(directory/'report.json'), 'terminal resource decision differs')
    expected = read(directory/'input-sha256.json')
    check(set(expected) == set(names), 'resource hash coverage')
    check(all(sha(directory/(n+'.json')) == h for n, h in expected.items()),
          'terminal resource input changed')
    check(control['admission']['experiment_budget']['analysis_stage_wall_limit_seconds']
          == TOTAL_SECONDS, 'analysis budget differs from primary admission')
    return {p.name: sha(p) for p in required}


def stages(source, resource_directory, normalization):
    model = str(BASE/LABEL/'model.json')
    results = str(BASE/'primary-host-v1/runs'/LABEL)
    common = ['--model', model, '--results', results, '--bounded-memory']
    baseline = ['--baseline', str(OUTPUT/'activity')]
    definitions = [
        ('output-audit', 7200, 'mam_primary_output_audit.py',
         ['--resource-directory', str(resource_directory), '--output', str(AUDIT_OUTPUT)]),
        ('activity', 2400, 'analyze_multi_area_activity.py',
         common+['--spike-tick-offset', '1', '--end-tick', '1005000',
                 '--output', str(OUTPUT/'activity')]),
        ('cell', 1800, 'analyze_mam_paper_cell_metrics.py',
         common+baseline+['--output', str(OUTPUT/'cell')]),
        ('correlation', 2400, 'analyze_mam_paper_correlation.py',
         common+baseline+['--output', str(OUTPUT/'correlation')]),
        ('series', 1800, 'analyze_mam_paper_time_series.py',
         common+baseline+['--normalization', str(normalization),
                          '--output', str(OUTPUT/'series')]),
    ]
    return [(name, cap, [sys.executable, str(source/'tools'/tool), *args])
            for name, cap, tool, args in definitions]


def execute_sequence(plan, output, *, started, clock=time.monotonic,
                     runner=subprocess.run, validate=lambda name: None):
    """The deadline never resets, including validation and inter-stage work.

    The outer cgroup guard also bounds validation and this payload itself.
    An exclusive journal is an attempt marker: even a crash cannot auto-retry.
    """
    rows = []
    with (output/'stages.jsonl').open('x') as journal:
        def record(row):
            rows.append(row)
            journal.write(json.dumps(row, allow_nan=False)+'\n')
            journal.flush()
        for name, cap, command in plan:
            remaining = math.floor(TOTAL_SECONDS-(clock()-started))
            if remaining <= 0:
                record(dict(stage=name, state='budget_exhausted', remaining_seconds=remaining))
                raise TimeoutError('shared analysis budget exhausted before '+name)
            timeout = min(cap, remaining)
            begin = clock()
            record(dict(stage=name, state='started', timeout_seconds=timeout,
                        remaining_seconds=remaining, command=command))
            try:
                # No shell, daemon or MPI launcher. The outer control group
                # owns every child and terminates it at its hard deadline.
                with (output/(name+'.log')).open('x') as log:
                    result = runner(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=True)
                check(result.returncode == 0, 'nonzero analysis child status')
                validate(name)
                check(clock()-started < TOTAL_SECONDS, 'shared analysis deadline reached')
            except Exception as error:
                record(dict(stage=name, state='failed', seconds=clock()-begin,
                            error=type(error).__name__+': '+str(error)))
                raise
            record(dict(stage=name, state='complete', seconds=clock()-begin,
                        remaining_seconds=TOTAL_SECONDS-(clock()-started)))
    return rows


def validate_output(name):
    if name == 'output-audit':
        report = read(AUDIT_OUTPUT)
        check(report['internal_output_audit_passed'] is True
              and report['model_sha256'] == MODEL
              and report['retained_10500ms_prefix']['exact'] is True
              and report['retained_10500ms_prefix']['spikes'] == 633265154,
              'complete output/prefix gate failed')
        return
    directory = OUTPUT/name
    catalog = read(directory/'catalog.json')
    required = {
        'activity': {'activity.json', 'activity-arrays.npz', 'activity-overview.png', 'spike-rasters.png'},
        'cell': {'paper-cell-metrics.json', 'cell-metrics.npz'},
        'correlation': {'correlation.json', 'selection.npz'},
        'series': {'time-series.json', 'time-series.npz', 'spectrum.png'},
    }[name]
    check(required <= set(catalog), 'analysis catalog missing required output')
    for filename, row in catalog.items():
        check(Path(filename).name == filename and not filename.startswith('._'), 'unsafe catalog member')
        p = directory/filename
        check(p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes']
              and sha(p) == row['sha256'], 'analysis catalog differs')
    if name == 'activity':
        report = read(directory/'activity.json')
        raw = read(AUDIT_OUTPUT)
        check(report['model_sha256'] == MODEL and report['result_sha256'] == raw['dump_sha256']
              and report['window']['seconds'] == 100.0
              and report['window']['spike_tick_offset'] == 1
              and report['bounded_memory'] is True, 'primary activity is not bound to raw audit')


def run(source, resources, normalization):
    started = time.monotonic()
    source = source.resolve()
    check(source.is_relative_to(BASE.resolve()) and resources.resolve().is_relative_to(BASE.resolve()),
          'analysis inputs must reside on brick2')
    prerequisite = resource_gate(resources)
    check(prerequisite is not None, 'terminal resource inputs missing')
    check(not OUTPUT.exists() and not AUDIT_OUTPUT.exists(), 'primary analysis already attempted')
    OUTPUT.mkdir()
    manifest = read(source/'catalog.json')
    for name, row in manifest.items():
        p = source/name
        check(p.resolve().is_relative_to(source) and not p.is_symlink()
              and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'analysis source changed')
    check(normalization.resolve().is_relative_to(source), 'normalization must be pinned in source')
    rows = execute_sequence(stages(source, resources, normalization), OUTPUT,
                            started=started, validate=validate_output)
    report = dict(schema='b2-mam-primary-analysis-pipeline-v1', analysis_complete=True,
                  scientific_acceptance=False, performance_cost_acceptance=False,
                  shared_budget_seconds=TOTAL_SECONDS, elapsed_seconds=time.monotonic()-started,
                  resource_input_sha256=prerequisite, source_catalog_sha256=sha(source/'catalog.json'),
                  stages=rows, output_audit_sha256=sha(AUDIT_OUTPUT),
                  catalogs={n: sha(OUTPUT/n/'catalog.json') for n in ['activity','cell','correlation','series']},
                  scope='One full Rust observation analyzed. Archive integrity, paper/native comparisons, '
                        'inter-area analyses, independent seeds and tuned speed/cost acceptance remain open.')
    with (OUTPUT/'report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.resources, args.normalization)))
