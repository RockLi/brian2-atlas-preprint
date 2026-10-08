"""Retain existing small published-model records; never execute models or benchmarks."""
from pathlib import Path
from evidence_paths import legacy_repo, external_directory
from datetime import datetime, timezone
import hashlib
import json
import statistics

PAPER = Path(__file__).resolve().parents[1]
REPO = legacy_repo()
OUT = PAPER / 'data/published_models'
OUT.mkdir(parents=True, exist_ok=True)
NMDA = external_directory('NMDA_ROOT')
DENDRITIC = external_directory('DENDRITIC_ARCHIVE')
sources=[]
def retain(path,name):
    raw=Path(path).read_bytes()
    (OUT/name).write_bytes(raw)
    sources.append({'source':str(path),'retained':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    return json.loads(raw) if Path(path).suffix=='.json' else raw

c2560=retain(NMDA/'results/processed/cross_host_20260915.json','nmda_cpu2560.json')['primary_cpu_replays']['host23_r198_targetcpu_parallel_nmda_8thread']
c5120=retain(NMDA/'results/processed/scale5120_host23_performance.json','nmda_cpu5120.json')['compiled_region']
s10240=retain(NMDA/'results/processed/scale10240_cpu8_five_pair_validation.json','nmda_cpu10240.json')
c10240=s10240['host23_native_x86_64_pinned_cpus_0_7']['compiled_region']
c20480=retain(NMDA/'results/processed/scale20480_host23_cpu8_native_five_pairs.json','nmda_cpu20480.json')
rows=[]
for neurons,edges,nmda_edges,c,source in [
    (2560,11796480,5242880,c2560,'nmda_cpu2560.json'),
    (5120,47185920,20971520,c5120,'nmda_cpu5120.json'),
    (10240,188743680,83886080,c10240,'nmda_cpu10240.json'),
    (20480,754974720,335544320,{'cpp':c20480['original_brian2'],'rust':c20480['rust_cpu']},'nmda_cpu20480.json'),
]:
    def stats(d):
        if 'wall_seconds' in d:
            x=d['wall_seconds'];rss=d['peak_sampled_rss_bytes']['median']
            return x['median'],x['raw'],rss
        return d['median_seconds'],d['raw_seconds'],d['peak_sampled_rss_bytes']
    cpp,cpp_raw,cpp_rss=stats(c['cpp']);rust,rust_raw,rust_rss=stats(c['rust'])
    assert len(cpp_raw)==len(rust_raw) and len(cpp_raw)>=5
    assert statistics.median(cpp_raw)==cpp and statistics.median(rust_raw)==rust
    rows.append({'neurons':neurons,'synapses':edges,'nmda_synapses':nmda_edges,'model_time_seconds':1,
                 'dt_seconds':.0001,'threads_per_backend':8,'samples_per_backend':len(cpp_raw),
                 'cpp_seconds':cpp,'rust_seconds':rust,'ratio':cpp/rust,'cpp_samples':cpp_raw,'rust_samples':rust_raw,
                 'rss_increase_percent':100*(rust_rss/cpp_rss-1),'record':source,
                 'timing_scope':'Compiled native initialization, simulation, original monitoring and result dump; excludes Python preparation, IR/code generation, compilation and public-array backfill.',
                 'memory_scope':'Reported sampled native RSS; small-cohort peak summaries, large-cohort median peaks; not frontend-inclusive memory.'})
gate5120=retain(NMDA/'results/processed/shared_input_5120_seed971_numerical_audit.json','nmda_gate5120.json')
for g in [gate5120,s10240['scientific_gate']]:
    assert g['deterministic_core_screen_pass'] and not g['failed_requirements']
mpi=retain(NMDA/'results/processed/cpu_threads_vs_mpi_10240_20260919.json','nmda_cpu_vs_mpi.json')
cpu40=next(r for r in mpi['rows'] if r['threads']==40)
mpi_sec=mpi['mpi_one_node_40_rank']['simulation_seconds_median']
mpi_ratio=cpu40['simulation_seconds']['median']/mpi_sec
assert abs(mpi_ratio-mpi['mpi_speedup_over_same_40_cpus'])<1e-12
completion=retain(NMDA/'results/processed/completion_audit_20260921.json','nmda_completion_audit.json')
assert completion['complete'] and not completion['errors']
for name in ['nmda2025_external_validation_note.md','completion_audit.md','cpu_threads_vs_mpi_10240.md','mpi_multinode_20480.md']:
    retain(NMDA/'reports'/name,'nmda_'+name)
retain(NMDA/'SOURCE.md','nmda_source_identity.md')

ensembles=[]
for fig,path,name in [
    ('2','contextual-fig2-paper-env-ensemble-seeds0-9-v1/ensemble-v1.json','onasch_fig2_ensemble.json'),
    ('S1','contextual-s1-paper-env-ensemble-seeds0-9-v1/ensemble-v1.json','onasch_s1_ensemble.json'),
]:
    d=retain(DENDRITIC/path,name)
    assert d['complete_published_seed_ensemble'] and d['seeds']==list(range(10))
    assert d['grid_shape_per_seed']==[16,200] and d['timings_reported'] is False
    ensembles.append({'figure':fig,'seeds':10,'grid_cells_per_seed':3200,'total_cells':32000,
                      'pearson_ensemble_mean':d['pearson_ensemble_mean'],
                      'sign_mismatch_fraction':d['sign_mismatch_fraction_ensemble_mean'],'record':name,
                      'scope':'Scientific source-model rerun in the locked Brian paper environment; not 32,000 Atlas paired executions.'})
retain(REPO/'brian2-rust/FULL_PAPER_REPRODUCTION.md','onasch_full_reproduction_inventory.md')

result={'schema':'brian2-atlas-published-model-evidence-v1','generated_utc':datetime.now(timezone.utc).isoformat(),
        'nmda_source_commit':'68e6dd970cfc6bab26459fcb8c34ee0f16560d9e','sources':sources,
        'nmda_cpu_rows':rows,'nmda_mpi_same40':{'cpu_seconds':cpu40['simulation_seconds']['median'],
            'mpi_seconds':mpi_sec,'ratio':mpi_ratio,'samples_per_configuration':5,
            'scope':'Same 40 physical CPUs on one dual-socket host; simulation/recording interval, not the CPU8 compiled-region interval.'},
        'onasch_ensembles':ensembles,'source_environment_reproduction_is_atlas_execution':False,
        'complete_onasch_paper_reproduction_accepted':False,
        'scope':'Copied existing small records and checked reported medians, identities and acceptance fields. No models, raw scientific arrays, remote jobs or hardware benchmarks executed.'}
(OUT/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'retained_sources':len(sources),'nmda_ratios':[r['ratio'] for r in rows],
                  'mpi_same40_ratio':mpi_ratio,'onasch_correlations':[r['pearson_ensemble_mean'] for r in ensembles]},indent=2))
