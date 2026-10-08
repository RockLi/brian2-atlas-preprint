"""Audit complete primary outputs after successful terminal resource collection.

Run under a separately admitted Linux analysis guard on node23's data disk.
This is internal engineering validation; all scientific/comparative gates remain
open. Never call on a live simulation or treat a partial report as acceptance.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import struct
import time

import numpy as np
from mam_primary_resources import LABEL, MODEL, PLAN, audit as audit_resources
from mam_output_work_audit import audit_work, check, select_shared_counts

BASE = Path('/data/brick2/brian2-mpi-region-20260907')
OLD = 'full32-n1-k1-nest-semantics-v2-10500ms'
SHORT = 'full32-n1-k1-output32-o3-v1-2500ms'
OLD_MODEL = 'f61694955485e133b9517450bef92af03a56c0ad08fd261383b3fe3d1e4c7e1a'
SHORT_MODEL = '75ff4455b1345a2d0713cf88c0e655840333fab93d4407c61bf3fb692c651fcc'
OLD_HASHES = {'results.bin': '3accf1e5d4e5f39be0efd29d9ced6eb56d7c0d14d19d1ad5bec92de1a5c3f831',
              'events.bin': '0ed128acc29d3f8c23ce5464ccd0dae66d0a7f0d17702f5021c0fe0fe3698d9a'}
PARAMETERS = 'ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'
EXE = '8e0f21edf5e44bb193b1f6152cf560f702e07750bc0a171699f69b4a048352a5'


def read(path, cap=32*2**20):
    check(path.stat().st_size <= cap, 'oversized control/model file: '+str(path))
    return json.loads(path.read_text())


def run(resource_directory, output):
    # Reproduce the terminal resource decision and bind the runtime file to its
    # collected source hash before any raw file is opened.
    from mam_correlation_stream import file_sha, mapped_cache
    control = {n: read(resource_directory/(n+'.json')) for n in
               ['admission', 'launch', 'collected', 'runtime']}
    resource_report = audit_resources(**control)
    check(resource_report == read(resource_directory/'report.json'), 'resource report not reproducible')
    expected = read(resource_directory/'input-sha256.json')
    check(set(expected) == set(control), 'resource input hash coverage')
    for name, digest in expected.items():
        check(file_sha(resource_directory/(name+'.json')) == digest, 'resource input changed')
    check(output.resolve().is_relative_to(BASE.resolve()) and not output.exists(), 'output must be new on brick2')
    run_dir = BASE/'primary-host-v1/runs'/LABEL
    check(not (run_dir/'spike-spool').exists(), 'spool lifecycle not terminal')
    runtime_path = run_dir/'mpi-runtime.json'
    runtime_source = control['collected'][0]['source_files']['runs/'+LABEL+'/mpi-runtime.json']
    check(runtime_path.stat().st_size == runtime_source['bytes']
          and file_sha(runtime_path) == runtime_source['sha256'], 'runtime changed since terminal collection')
    runtime = read(runtime_path)
    check(runtime == control['runtime'], 'runtime differs from resource audit')
    started = time.perf_counter()
    sources = {}
    def pinned(path, digest, cap=512*2**20):
        check(path.stat().st_size <= cap and file_sha(path) == digest, 'pinned source mismatch: '+str(path))
        sources[str(path)] = digest
        return read(path, cap)
    primary = BASE/LABEL
    model = pinned(primary/'model.json', MODEL)
    short = pinned(BASE/SHORT/'model.json', SHORT_MODEL)
    from mam_extend_duration import audit_duration_delta, output_bytes
    delta = audit_duration_delta(short, model, 100.5)
    old_model = pinned(BASE/OLD/'model.json', OLD_MODEL)
    audit_duration_delta(short, old_model, 10.5)
    del short
    check(model['protocol']['layers']['instance'] ==
          '58222fc5a2185a10d68361caf5f2ed722ed8ceddc587907b88033be58a95b98e', 'instance identity')
    pinned(primary/'parameters.json', PARAMETERS)
    preparation = read(primary/'preparation.json')
    from brian2_rust.distributed import _verify_artifact
    manifest = _verify_artifact(primary/'mpi')
    check(manifest == preparation['manifest'] and manifest['plan_sha256'] == PLAN, 'primary manifest')
    deployed = BASE/'primary-host-v1'/LABEL/'mpi'
    check(_verify_artifact(deployed) == manifest, 'leader deployed artifact differs')
    build = read(primary/'mpi/build.json')
    check(read(deployed/'build.json') == build and build['opt_level'] == 3
          and build['panic'] == 'abort' and build['executable_sha256'] == EXE
          and file_sha(primary/'mpi/b2-mpi') == file_sha(deployed/'b2-mpi') == EXE, 'primary executable')
    owners = read(primary/'mpi/execution-plan.json')['population_owners']
    placement = read(primary/'placement.json')
    check(owners == placement['population_owners'] and len(owners) == 254
          and owners.count(None) == 1 and owners[88] is None, 'frozen placement')
    hist_path = BASE/'hotspot-sharing/exact-counts.jsonl'
    hist_sha = '2911d901cd8d82f51beaaec5122b7888ac0dfc618aad245ef58af70ccf36cbab'
    check(hist_path.stat().st_size == 17664 and file_sha(hist_path) == hist_sha, 'shared histogram changed')
    sources[str(hist_path)] = hist_sha
    hist = select_shared_counts(list(map(json.loads, hist_path.read_text().splitlines())), model, owners)
    baseline = pinned(BASE/('audit-'+OLD+'-report.json'),
                      '5c4a250f7df455b1557bb8cc6044bc76ce42de779a84eac8b580a4adb9e8fa46', 2**20)
    check(baseline['complete'] and baseline['independent_dump_reader_passed']
          and baseline['dump_sha256'] == OLD_HASHES, 'retained baseline proof')
    check(len(model['definition']['synapses']) == 8344
          and sum(p['count'] for p in model['definition']['populations']) == 4129924, 'full scale required')
    old_dir = BASE/('audit-'+OLD)/'node25/runs'/OLD
    limits = {'results.bin': 105290918564, 'events.bin': 103079217168}
    check(sum((run_dir/n).stat().st_size for n in limits) <= 208370135732, 'combined output budget')
    hashes = {}
    for name, limit in limits.items():
        check((run_dir/name).stat().st_size <= limit, 'output budget exceeded')
        hashes[name] = file_sha(run_dir/name)
        check((old_dir/name).stat().st_size == baseline['dump_bytes'][name]
              and file_sha(old_dir/name) == OLD_HASHES[name], 'retained raw baseline changed')
    from brian2_rust.results import load_results
    from mam_result_prefix import compare_populations_bounded
    with ExitStack() as stack:
        def load(definition, directory):
            data = load_results(definition, directory, include_times=False, release_file_cache=True)
            callbacks = []
            for key, name in [('_dump', 'results.bin'), ('_event_dump', 'events.bin')]:
                check(isinstance(data[key], np.memmap), 'independent binary mapping missing')
                stack.callback(data[key]._mmap.close)
                callbacks.append(stack.enter_context(mapped_cache(data[key], directory/name)))
            return data, callbacks
        loaded, callbacks = load(model, run_dir)
        summary = loaded['metadata']
        check(summary['mpi'] == runtime and summary['schema'] == 'b2-result-dump-v4'
              and summary['final_time_seconds'] == 100.5, 'complete primary metadata')
        work = audit_work(model, loaded, runtime, owners, hist,
                          release=callbacks[0], event_release=callbacks[1])
        check(work['exact_local_edges'] == baseline['exact_local_edges'] ==
              [r['exact_total_edges'] for r in placement['ranks']], 'local edge placement changed')
        check(sum(work['exact_local_edges']) == 24126516728
              and max(work['exact_local_edges']) <= 1024000000, 'full recurrent edge budget')
        sizes = output_bytes(model, summary['spike_count'],
                             sum(len(p['last_spikes']) for p in loaded['populations']), compact_spikes=True)
        check((run_dir/'results.bin').stat().st_size == sizes['results_bytes']
              and (run_dir/'events.bin').stat().st_size == sizes['events_bytes'], 'exact output byte formula')
        check(runtime['spike_history_storage'] == 'bounded-disk-spool'
              and runtime['spike_spool_bytes'] == 8*summary['spike_count'] <= 96*2**30
              and runtime['spike_spool_maximum_bytes'] == 96*2**30
              and runtime['spike_spool_maximum_population_bytes'] == 16*2**30
              and max(p['spikes']*8 for p in work['population_profile']) <= 16*2**30,
              'spool limits/counts')
        prior, old_callbacks = load(old_model, old_dir)
        prefix = compare_populations_bounded(prior['populations'], loaded['populations'], end_tick=105000,
                    old_release=old_callbacks[0], new_release=callbacks[0],
                    old_event_release=old_callbacks[1], new_event_release=callbacks[1])
        check(prefix['spikes'] == baseline['spikes'] == 633265154, 'complete 10.5 s prefix count')
    prebuilt = runtime['prebuilt_topology']
    check(prebuilt['columns'] == ['shared_projections', 'cache_limit_bytes', 'metadata_bytes',
          'peak_retained_bytes', 'remaining_bytes', 'fixed_projections', 'build_seconds_bits',
          'verify_seconds_bits'] and len(prebuilt['rank_records']) == 256, 'prebuilt schema')
    for rank, old in enumerate(baseline['prebuilt_topology']):
        row = prebuilt['rank_records'][rank*8:(rank+1)*8]
        check(row[:6] == [24, 134217728, old['metadata_bytes'], old['peak_retained_bytes'], 0, 8344],
              'prebuilt ownership/cache changed')
        check(all(np.isfinite(v := struct.unpack('<d', struct.pack('<Q', bits))[0]) and v > 0
                  for bits in row[6:]), 'prebuilt timing')
    check(runtime['rank_work_columns'] == ['owned_neurons', 'emitted_spikes', 'delivered_edges']
          and runtime['projection_report_collectives'] == 66
          and runtime['projection_report_packet_values'] == 640, 'work/report schema')
    stages = np.asarray(runtime['rank_stage_seconds']).reshape(32, 5)
    check(runtime['rank_stage_columns'] == ['initialization_local', 'initialization_wait',
          'simulation_local', 'simulation_wait', 'result_collection_and_reporting']
          and np.isfinite(stages).all() and (stages >= 0).all(), 'stage timing')
    report = dict(schema='b2-mam-primary-output-audit-v1', internal_output_audit_passed=True,
        terminal_resource_audit_passed=True, scientific_acceptance=False, nest_statistical_acceptance=False,
        performance_cost_acceptance=False, model_sha256=MODEL, plan_sha256=PLAN, executable_sha256=EXE,
        source_sha256=sources, duration_delta=delta, full_independent_readers_passed=True,
        work=work, retained_10500ms_prefix=prefix, dump_sha256=hashes, dump_bytes=sizes,
        spikes=summary['spike_count'], delivered_edges=summary['synaptic_events'],
        phase_seconds=dict(initialization=max(stages[:, 0]+stages[:, 1]),
                           simulation=max(stages[:, 2]+stages[:, 3]), collection=max(stages[:, 4])),
        elapsed_seconds=time.perf_counter()-started,
        scope='Complete internal raw-output/ownership audit and exact retained prefix. Leader artifact rehashed; all-host start hashes are original admission evidence. Post-run artifacts on other hosts, archive/collection integrity, scientific patterns and tuned cost/performance comparisons remain separate requirements.')
    with output.open('x') as f:
        f.write(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.resource_directory, args.output)
    print(json.dumps({k: result[k] for k in ['internal_output_audit_passed', 'spikes',
                                            'delivered_edges', 'elapsed_seconds']}))
