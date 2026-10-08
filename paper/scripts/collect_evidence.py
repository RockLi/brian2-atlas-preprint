"""Extract already-recorded evidence; never launches simulation or remote work."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--gpu-root', type=Path, default=Path('/private/tmp/brian2-flywire-mnist/brian2-rust'))
parser.add_argument('--mpi-root', type=Path, default=Path('/private/tmp/brian2-mpi-cpu/brian2-rust'))
args = parser.parse_args()
CPU = ROOT.parents[1] / 'brian2-rust'
provenance = []

def read(path, kind='json'):
    raw = path.read_bytes()
    provenance.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    if path.suffix == '.gz': raw = gzip.decompress(raw)
    return json.loads(raw) if kind == 'json' else raw.decode()

mpi = read(args.mpi_root / 'mpi-evidence/rank-local/final/optimization-report.json')
assert mpi['complete']
mpi_rows = []
for version in ['before', 'after']:
    for ranks in [1, 2, 4]:
        runs = [r for r in mpi['runs'] if r['version'] == version and r['ranks'] == ranks and r['condition'] == 'rest' and not r['warmup']]
        assert len(runs) == 3
        samples = [r['summary']['timings']['simulation_and_recording_seconds'] for r in runs]
        mpi_rows.append({'version': version, 'ranks': ranks, 'samples_seconds': samples,
                         'median_seconds': statistics.median(samples),
                         'max_rank_rss_mib': max(v for r in runs for v in r['rank_peak_rss_bytes']) / 2**20})

primary = read(args.mpi_root / 'mpi-evidence/primary-run/terminal-resource-report.json')
raw_audit = read(args.mpi_root / 'mpi-evidence/primary-postrun/full-report.json')
assert primary['terminal_resource_audit_passed'] and raw_audit['internal_output_audit_passed']
ring = read(args.gpu_root / 'execution-plan-evidence/population-scale/audit-result.json')
assert ring['passed']
cuba = {host: read(args.gpu_root / f'execution-plan-evidence/precompiled-comparison/{host}/report.json.gz') for host in ['l4', 'a100']}
browser = read(args.gpu_root / 'validation/evidence/browser-local-20260910/wasm-check.json')
assert browser['status'] == 'passed' and len(browser['records']) == 13

# CPU numbers are parsed from the retained report; they are summary observations.
cpu_text = read(CPU / 'FLYWIRE_CROSS_HOST_RESULTS.md', 'text')
cpu_rows = []
for line in cpu_text.splitlines():
    fields = [x.strip() for x in line.strip('|').split('|')]
    if len(fields) != 8 or not fields[1].isdigit(): continue
    if not (fields[0].startswith('27 ·') or fields[0].startswith('23 ·')): continue
    number = lambda x: float(x.replace('†', '').strip())
    cpu_rows.append({'host': 'M1 Ultra' if fields[0].startswith('27') else 'EPYC 9454 × 2',
                     'threads': int(fields[1]), 'rust_seconds': number(fields[2]), 'cpp_seconds': number(fields[3]),
                     'rust_mib': number(fields[5]), 'cpp_mib': number(fields[6]),
                     'cpp_unstable': '†' in line, 'source_level': 'retained_report_summary'})
assert len(cpu_rows) == 8
read(CPU / 'PD14_LOCAL_16GB.md', 'text')
read(CPU / 'LITWIN_KUMAR_RESULTS.md', 'text')
read(args.gpu_root / 'EXECUTION_PLAN.md', 'text')
read(args.gpu_root / 'B2IR.md', 'text')
read(args.mpi_root / 'MPI.md', 'text')
read(args.gpu_root / 'WASM.md', 'text')
read(args.gpu_root / 'tests/test_synapses.py', 'text')

data = {'schema': 'b2-preprint-figure-data-v1', 'date': '2026-09-11',
        'provenance': provenance, 'mpi_ranklocal': mpi_rows,
        'mpi_primary': {'neurons': 4129924, 'recurrent_synapses': 24126516728, 'model_seconds': 100.5,
                        'nodes': 4, 'ranks': 32, 'wall_seconds': primary['launch_wall_seconds'],
                        'proxy_peak_bytes': [r['peak_bytes'] for r in primary['guards'] if r['role'].startswith('proxy')],
                        'raw_audit_passed_later': raw_audit['internal_output_audit_passed'],
                        'resource_report_raw_audit_flag_at_collection': primary['raw_output_audit_passed']},
        'cpu_flywire': cpu_rows,
        'gpu_ring': ring['results'],
        'gpu_cuba': {h: {'summary': d['summary'], 'source_manifest_sha256': d['source_manifest_sha256'],
                         'original_f64_gate_passed_all_samples': d['original_f64_gate_passed_all_samples']} for h,d in cuba.items()},
        'lifecycle': {'source_level': 'transcribed_retained_report_single_observations',
                      'labels': ['Rust full', 'Rust rolling', 'Brian C++'],
                      'native_gb': [1.082, .290, 2.327], 'summed_tree_gb': [4.961, 4.074, 2.767],
                      'recovery_processes': 3, 'model_seconds': 1000, 'exact_fields': 41, 'exact_segments': 11},
        'browser': {'records': browser['records'], 'protocol': browser['protocol'],
                    'max_score_error': max(r['score_max_abs_error'] for r in browser['records']),
                    'linear_memory_bytes': sorted(set(r['memory_bytes'] for r in browser['records']))},
        'semantics_example': {'kind': 'illustration_from_documented_regression', 'ticks': [0,1,2,3],
                              'canonical_x': [0,1,3,6], 'invalid_final_only_x': [0,0,0,0]}}
(ROOT / 'data/figure_data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
print('Extracted recorded MPI/GPU samples, CPU summaries, lifecycle summaries and browser checks.')
