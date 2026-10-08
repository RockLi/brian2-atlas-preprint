"""Collect small existing v3 records; no simulations, remote jobs or archive writes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import statistics
import subprocess
import xml.etree.ElementTree as ET

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
OUT = PAPER / 'data/v3'
OUT.mkdir(parents=True, exist_ok=True)
ARCHIVE = Path('/atlas-storage/0002/brian2-paper-reproduction/contextual-dendritic-gating')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

sources = []
def retain(path, name):
    path = Path(path)
    raw = path.read_bytes()
    (OUT / name).write_bytes(raw)
    sources.append({'source': str(path), 'retained': name, 'bytes': len(raw), 'sha256': digest(raw)})
    return json.loads(raw) if path.suffix == '.json' else raw

rows = []
for label, relative, name in [
    ('Single neuron, nonlinear NMDA', 'single-neuron-10s-benchmark-abba-v1/aggregate-v1.json', 'dendritic_nonlinear.json'),
    ('Single neuron, linear NMDA', 'single-neuron-linear-10s-benchmark-abba-v1/aggregate-v1.json', 'dendritic_linear.json'),
    ('Figure-3 complete topology, shortened protocol', 'paper-scale-100ms-remote-benchmark-v1/paper-scale-100ms-benchmark-abba-aggregate-v1.json', 'dendritic_network.json'),
]:
    d = retain(ARCHIVE / relative, name)
    assert d['valid'] and all(d['criteria'].values())
    assert d['order'] == 'cython-rust-rust-cython'
    m = d['primary_metric']
    ratio = m['cython']['median_seconds'] / m['rust_aot']['median_seconds']
    assert abs(ratio - m['rust_speedup_over_cython']) < 1e-12
    assert all(x['passed'] for x in d['scientific_state_comparisons'])
    for backend in ['cython', 'rust_aot']:
        if 'samples_seconds' in m[backend]:
            assert len(m[backend]['samples_seconds']) == m[backend]['count']
            assert abs(statistics.median(m[backend]['samples_seconds']) - m[backend]['median_seconds']) < 1e-12
    rows.append({'workload': label, 'model_time': '100 ms' if 'Figure-3' in label else '10 s',
                 'samples_per_backend': m['cython']['count'], 'cython_seconds': m['cython']['median_seconds'],
                 'rust_seconds': m['rust_aot']['median_seconds'], 'ratio': ratio,
                 'numerical_gate_from_archive': True, 'raw_arrays_reaudited_this_revision': False,
                 'aggregate': name, 'timing_scope': m['definition']})
retain(ARCHIVE / 'paper-scale-100ms-remote-benchmark-v1/evidence-audit-v1/contextual-fig3-100ms-benchmark-evidence-audit-v1.json', 'dendritic_network_prior_audit.json')

root = REPO / 'brian2-rust'
verification = retain(root / 'mpi-evidence/training-20261002/verification.json', 'training_delivery_verification.json')
tests = []
for name, label in [('mpi-final.xml', 'MPI STDP and queues'), ('native-final.xml', 'Initial native CPU/Metal training')]:
    raw = retain(root / 'mpi-evidence/training-20261002' / name, name)
    cases = ET.fromstring(raw).findall('.//testcase')
    counts = {'passed': 0, 'skipped': 0, 'failed': 0}
    for case in cases:
        state = 'skipped' if case.find('skipped') is not None else ('failed' if case.find('failure') is not None or case.find('error') is not None else 'passed')
        counts[state] += 1
    assert counts['failed'] == 0
    assert counts['passed'] == verification['tests'][name]['passed']
    tests.append({'scope': label, 'identities': len(cases), 'counts': counts, 'record': name,
                  'scope_limit': 'Historical delivery snapshot; no throughput, public-dataset accuracy or cross-host claim.'})
retain(root / 'mpi-evidence/training-20261002/README.md', 'training_delivery_notes.md')
for directory, file, name, scope in [
    ('training-metal-pipeline-full-20261006', 'acceptance.json', 'training_pipeline_acceptance.json', 'CPU/Metal and local MPI; original cache/weak/SDE snapshot'),
    ('training-indexed-synaptic-endpoints-20261006', 'cpu-acceptance.json', 'indexed_endpoints_cpu_acceptance.json', 'CPU and local MPI; indexed endpoint snapshot'),
    ('training-indexed-synaptic-endpoints-20261006', 'metal-acceptance.json', 'indexed_endpoints_metal_acceptance.json', 'CPU/Metal and local MPI; same indexed endpoint snapshot'),
    ('training-external-state-inputs-owner-20261006', 'cpu-acceptance.json', 'external_inputs_cpu_acceptance.json', 'CPU and local MPI; later external-input ownership snapshot'),
]:
    d = retain(root / 'mpi-evidence' / directory / file, name)
    assert d['terminal']['exit_code'] == 0
    assert sum(d['counts'].values()) == d['identities']
    tests.append({'scope': scope, 'identities': d['identities'], 'counts': d['counts'],
                  'hardware': d['hardware'], 'record': name, 'terminal': d['terminal'],
                  'scope_limit': 'Counts describe this frozen cohort only; overlapping suites are not added or inherited.'})
for directory, name in [
    ('training-metal-pipeline-full-20261006', 'training_pipeline_notes.md'),
    ('training-indexed-synaptic-endpoints-20261006', 'indexed_endpoints_notes.md'),
    ('training-external-state-inputs-owner-20261006', 'external_inputs_notes.md'),
]:
    retain(root / 'mpi-evidence' / directory / 'README.md', name)
retain(root / 'mpi-evidence/training-indexed-synaptic-endpoints-20261006/metal-audit-terminal.json', 'indexed_endpoints_metal_audit_terminal.json')
retain(root / 'mpi-evidence/training-metal-pipeline-full-20261006/terminal-receipts.json', 'training_pipeline_terminal_receipts.json')
retain(root / 'mpi-evidence/confirmation-seed1751-v8-science-identity-preparation/acceptance-ledger.md', 'mam_acceptance_ledger.md')

for relative, name in [
    ('COMPATIBILITY.md', 'compatibility_review.md'), ('MPI_TRAINING.md', 'mpi_training_review.md'),
    ('NATIVE_TRAINING.md', 'native_training_review.md'), ('NATIVE_TRAINING_DYNAMIC.md', 'dynamic_training_review.md'),
    ('NATIVE_TRAINING_V4_GPU.md', 'static_training_gpu_review.md'),
    ('NATIVE_TRAINING_CROSSHOST_GPU.md', 'crosshost_training_scope.md'),
    ('CONTEXTUAL_DENDRITIC.md', 'dendritic_review.md'),
]:
    retain(root / relative, name)
for relative, name in [
    ('evaluations/atlas-training-v1-20261004/reports/qualification-followup-r1/summary.zh.md', 'external_training_qualification.md'),
    ('evaluations/atlas-training-v1-20261004/reports/runtime-architecture-r1/impact.zh.md', 'external_training_architecture.md'),
]:
    retain(REPO / relative, name)

scope = ['brian2/codegen', 'brian2/core/functions.py', 'brian2/equations/unitcheck.py',
         'brian2-rust/src', 'brian2-rust/python', 'brian2-rust/tests',
         'brian2-rust/Cargo.toml', 'brian2-rust/Cargo.lock', 'brian2-rust/build.rs']
files = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', *scope], cwd=REPO).decode().split('\0')
manifest = []
for relative in sorted(set(files) - {''}):
    path = REPO / relative
    if path.is_file():
        raw = path.read_bytes()
        manifest.append({'path': relative, 'bytes': len(raw), 'sha256': digest(raw)})
diff = subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--', *scope], cwd=REPO)
status = subprocess.check_output(['git', 'status', '--short', '--', *scope], cwd=REPO).decode().splitlines()
review = {'schema': 'brian2-atlas-preprint-review-v3', 'generated_utc': datetime.now(timezone.utc).isoformat(),
          'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO).decode().strip(),
          'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip(),
          'source_scope': scope, 'status': status, 'tracked_diff_sha256': digest(diff),
          'source_files': manifest, 'scope_limit': 'Inspected local inputs, including uncommitted and untracked sources; not a qualified release or an executable source archive. Historical experiments retain their own source manifests.'}
save_json('review_snapshot.json', review)
evidence = {'schema': 'brian2-atlas-preprint-evidence-v3', 'generated_utc': review['generated_utc'],
            'sources': sources, 'dendritic_rows': rows, 'training_cohorts': tests,
            'mam': {'rust_seeds': [1729,1750,1751], 'nest_reference_seeds': [1729,1730,1731],
                    'descriptive_cohort_complete_from_ledger': True, 'scientific_equivalence_accepted': False,
                    'matched_performance_accepted': False, 'ledger': 'mam_acceptance_ledger.md'},
            'scope': 'Small retained source/report/acceptance records and arithmetic checks. No new simulation, hardware benchmark, scientific raw-array recomputation or remote job.'}
save_json('evidence.json', evidence)
print(json.dumps({'sources': len(sources), 'review_source_files': len(manifest), 'dendritic_workloads': len(rows), 'training_cohorts': len(tests)}, indent=2))
