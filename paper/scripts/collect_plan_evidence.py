"""Extract retained tuning reports, verifying their archive hashes; run no simulations."""
from pathlib import Path
from evidence_paths import legacy_repo, external_directory
import gzip
import hashlib
import json
import math
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
SOURCE = external_directory('PLAN_SOURCE')
ARCHIVE = external_directory('PLAN_ARCHIVE')
LEGACY = legacy_repo()
OUT = ROOT / 'data/plan-selection-reports'
OUT.mkdir(parents=True, exist_ok=True)
provenance = []
data = {'schema': 'b2-preprint-plan-selection-v2', 'provenance': provenance}

def source_file(path):
    raw = path.read_bytes()
    provenance.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    return raw

for cohort, folder in [('autotune', 'autotune-1e6954b2'), ('tuning-cache', 'tuning-cache-ef88d574')]:
    base = SOURCE / 'execution-plan-evidence' / cohort
    manifest = json.loads(source_file(base / 'manifest.json'))
    audit = json.loads(source_file(base / 'audit-result.json'))
    source_file(base / 'README.md')
    assert audit['passed']
    data[cohort] = {'base_revision': manifest['base'], 'retained_audit': audit, 'rows': []}
    for host in ['m3', 'l4', 'a100']:
        key = manifest['benchmarks'][host] + '/report.json'
        entry = manifest['artifacts'][key]
        archive = ARCHIVE / folder / 'evidence/blobs' / entry['blob']
        raw = gzip.decompress(archive.read_bytes())
        assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
        saved = OUT / f'{cohort}-{host}.json'
        saved.write_bytes(raw)
        provenance.append({'archive_path': str(archive), 'artifact_key': key, 'sha256': entry['sha256'],
                           'bytes': len(raw), 'saved': str(saved.relative_to(ROOT))})
        report = json.loads(raw)
        assert report['passed']
        for case in report['cases']:
            assert case['passed']
            row = {'host': host.upper(), 'case': case['name'], 'configuration': case['configuration'],
                   'expected_events': case['expected_events'], 'source': str(saved.relative_to(ROOT))}
            if cohort == 'autotune':
                tuning = case['tuning']; baseline = tuning['candidates']['baseline']['seconds']
                assert tuning['status'] == 'passed'
                eligible = [n for n,r in tuning['candidates'].items() if r['status'] == 'eligible'
                            and len(r['seconds']) == 3 and max(r['seconds']) < min(baseline)
                            and stats.median(r['seconds']) <= .95 * stats.median(baseline)]
                selected = min(eligible, key=lambda n: stats.median(tuning['candidates'][n]['seconds'])) if eligible else 'baseline'
                assert selected == tuning['selected']
                assert all(s['observable_sha256'] == tuning['reference_observable_sha256'] for s in tuning['samples'])
                later = [r['seconds'] for r in case['replay_samples']]
                assert len(later) == 5 and all(r['gates']['f32']['passed'] for r in case['replay_samples'])
                assert math.isclose(stats.median(later), case['selected_replay_median_seconds'])
                row.update(selected=selected, candidates=tuning['candidates'], samples=tuning['samples'],
                           calibration_seconds=case['tuning_wall_seconds'],
                           baseline_profiling_ms=1000*stats.median(baseline),
                           selected_profiling_ms=1000*stats.median(tuning['candidates'][selected]['seconds']),
                           later_replay_seconds=later, later_replay_median_ms=1000*stats.median(later),
                           f32_passed=case['selected_gates']['f32']['passed'],
                           f64_passed=case['selected_gates']['f64']['passed'])
            else:
                acts = case['activations']; assert len(acts) == 4
                assert acts[0]['tuning']['cache']['status'] == 'miss'
                assert all(a['tuning']['cache']['status'] == 'hit' for a in acts[1:])
                assert all(a['gates']['f32']['passed'] for a in acts)
                assert len({a['observable_sha256'] for a in acts}) == 1
                row.update(selected=acts[0]['tuning']['selected'], calibration_transport_seconds=acts[0]['wall_seconds'],
                           hit_seconds=[a['wall_seconds'] for a in acts[1:]],
                           hit_median_seconds=stats.median(a['wall_seconds'] for a in acts[1:]),
                           f32_passed=True, f64_passed=all(a['gates']['f64']['passed'] for a in acts))
            data[cohort]['rows'].append(row)

source_file(LEGACY/'brian2-rust/python/brian2_rust/planner.py')
for name in ['GPU_AUTOTUNE.md', 'EXECUTION_PLAN.md',
             'python/brian2_rust/plan.py', 'python/brian2_rust/gpu_autotune.py',
             'python/brian2_rust/gpu_tuning_cache.py']:
    source_file(SOURCE/name)
data['verification_scope'] = 'Six archived benchmark-report hashes, twelve final host/case summaries, selection rules, sample fingerprints and report-level numerical gates rechecked. Full raw arrays and historical regression suites were not rerun.'
(ROOT/'data/plan_selection.json').write_text(json.dumps(data, indent=2)+'\n')
print('Verified six archived reports and twelve host/case summaries.')
