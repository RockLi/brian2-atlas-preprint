"""Endpoint-balanced motion pilot with frozen-decoder causal interventions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from .motion_challenge import iter_orbit, endpoint_hashes, balanced_orbit_seeds, movie
from .simulation import SimulationConfig
from .refinement import encode_variant, spatial_projection, linear_scores
from .sparse_refinement import fit_sparse
from .pilot import DIRECTIONS, event_features, fit, scores
from .direction_study import evaluate
from .run_experiment import save

TRAVEL = 2.0
ALPHAS = (.01, .1, 1.)
CONDITIONS = ('intact', 'cut_input', 'matched_cut', 'static_first', 'scrambled')


def motion_features(raw, baseline, projection):
    """External nonlinear motion computation; not intrinsic connectome decoding.

    Raw and positive residual activity are mapped to fixed 12x12 coordinates.
    Signed lagged spatial correlations pool across position and time. All
    operations are per clip; no label, phase metadata or batch statistic enters.
    """
    grids = ((np.asarray(raw).reshape(8, -1)-np.asarray(baseline).reshape(8, -1)) @ projection).reshape(8, 12, 12)
    values = []
    for data in (grids, np.maximum(grids, 0)):
        data = gaussian_filter(data.astype(float), sigma=(0, .65, .65), mode='wrap')
        for lag in (1, 2):
            a, b = data[:-lag], data[lag:]
            denominator = max(float(np.sqrt(np.sum(a*a)*np.sum(b*b))), 1e-12)
            for axis in (2, 1):
                for shift in (1, 2, 3):
                    values.append(float(np.sum(b*np.roll(a, shift, axis)-a*np.roll(b, shift, axis)))/denominator)
    return np.asarray(values)


def intervene(template, cells):
    """Disable selected cells' outgoing transmission, not their recorded spikes."""
    from brian2_rust.protocol import attach_protocol
    result = copy.deepcopy(template)
    ni = next(i for i, p in enumerate(result['definition']['populations']) if p['name'] == 'flywire_neurons')
    values = result['instance']['populations'][ni]['initial_state']['transmission']
    for cell in cells:
        if values[int(cell)] != struct.pack('>d', 1.).hex():
            raise ValueError('unexpected initial transmission')
        values[int(cell)] = struct.pack('>d', 0.).hex()
    attach_protocol(result)
    return result


def matched_cells(graph, inputs, readouts, seed=110926):
    """One declared positive-output control mask, approximate topology matching."""
    from brian2_rust.binary_topology import inspect_csr, csr_arrays
    offsets, targets, values = csr_arrays(inspect_csr(graph.csr))
    n = len(graph.root_ids)
    degree = np.diff(offsets).astype(float)
    indegree = np.bincount(targets, minlength=n).astype(float)
    matrix = csr_matrix((values[0], targets, offsets), shape=(n, n))
    strength = np.asarray(abs(matrix).sum(1)).ravel()
    negative = np.asarray((matrix < 0).sum(1)).ravel()
    eligible = (strength > 0) & (negative == 0)
    eligible[np.r_[inputs, readouts]] = False
    candidates = np.flatnonzero(eligible)
    descriptor = np.log1p(np.stack([degree, indegree, strength], 1))
    scale = descriptor[candidates].std(0)
    descriptor /= np.maximum(scale, 1e-8)
    tree = cKDTree(descriptor[candidates])
    rng = np.random.default_rng(seed)
    chosen, used, distances = [], set(), []
    for cell in rng.permutation(inputs):
        neighbors = min(128, len(candidates))
        while True:
            distance, indices = tree.query(descriptor[cell], k=neighbors)
            choices = [(float(d), int(candidates[i])) for d, i in zip(np.atleast_1d(distance), np.atleast_1d(indices)) if int(candidates[i]) not in used]
            if len(choices) >= 5 or neighbors == len(candidates):
                break
            neighbors = min(neighbors*2, len(candidates))
        if not choices: raise ValueError('insufficient unused topology matches')
        d, selected = choices[int(rng.integers(min(5, len(choices))))]
        chosen.append(selected);used.add(selected);distances.append(d)
    chosen = np.asarray(chosen, dtype=int)
    summary = {'seed': seed, 'cells_per_mask': len(chosen), 'candidate_cells': len(candidates),
               'matching': 'same count, positive-only outputs, nearest log out-degree/in-degree/absolute-contact-strength; exclude input and readout cells',
               'mean_standardized_log_distance': float(np.mean(distances)),
               'limitations': 'one approximate control mask; cell type, location and firing are not matched; not proof of pathway specificity'}
    for name, vector in (('out_degree', degree), ('in_degree', indegree), ('output_strength', strength)):
        summary[name] = {'target_mean': float(vector[inputs].mean()), 'control_mean': float(vector[chosen].mean()),
                         'target_sum': float(vector[inputs].sum()), 'control_sum': float(vector[chosen].sum())}
    return np.sort(chosen), summary


def paired_interval(a, b, rows):
    groups = sorted({r['group'] for r in rows})
    delta = np.asarray(a, dtype=float)-np.asarray(b, dtype=float)
    by_group = np.array([delta[[r['group'] == g for r in rows]].mean() for g in groups])
    bootstrap = np.random.default_rng(110926).choice(by_group, (10000, len(groups)), replace=True).mean(1)
    return {'difference': float(by_group.mean()), 'group_bootstrap_95_interval': np.quantile(bootstrap, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    from flywire_mnist.graph import load_graph
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = read(args.previous/'protocol.json')
    cfg = SimulationConfig(**parent['config'])
    channels = read(args.artifact/'channels.json')
    cells, inputs = np.asarray(parent['readout_indices']), np.asarray(parent['input_indices'])
    graph = load_graph(args.graph)
    control, matching = matched_cells(graph, inputs, cells)
    save(args.output/'matching.json', matching)
    template = read(args.artifact/'model.json')
    ni = next(i for i, p in enumerate(template['definition']['populations']) if p['name'] == 'flywire_neurons')
    neurons = template['definition']['populations'][ni]['count']
    spatial_path = Path(read(args.previous/'protocol.json')['parent_study'])/'spatial-map.npz'
    spatial_parent = read(spatial_path.parent/'protocol.json')
    if sha(spatial_path) != spatial_parent['spatial_map_sha256']:
        raise ValueError('spatial map changed')
    with np.load(spatial_path, allow_pickle=False) as m:
        mapping = dict(m)
    if not np.array_equal(mapping['cells'], cells):
        raise ValueError('spatial map cell order mismatch')
    projection = spatial_projection(mapping)
    pq = np.asarray([[r['p']-19, r['q']-17] for r in channels])
    input_xy = np.stack([np.sqrt(3)/2*(pq[:, 1]-pq[:, 0]), -(pq[:, 0]+pq[:, 1])/2], 1)/22
    input_projection = spatial_projection({'xy': input_xy, 'valid': np.ones(len(inputs), dtype=bool)})
    np.savez_compressed(args.output/'spatial-map.npz', **mapping, input_xy=input_xy)
    protocol = {'schema': 'flywire-causal-motion-v1', 'parent_study': str(args.previous),
                'parent_report_sha256': sha(args.previous/'report.json'), 'travel': TRAVEL,
                'stimulus_schema': 'flywire-periodic-motion-v1', 'clips_per_orbit': 16,
                'fit_groups': balanced_orbit_seeds(1000000, 8), 'validation_groups': balanced_orbit_seeds(1010000, 4),
                'test_groups': balanced_orbit_seeds(1020000, 8), 'directions': DIRECTIONS,
                'conditions': CONDITIONS, 'alphas': ALPHAS, 'linear_cells': (32, 128, 512),
                'config': parent['config'], 'input_mode': parent['input_mode'],
                'readout_indices': cells.tolist(), 'input_indices': inputs.tolist(), 'matched_cut_indices': control.tolist(),
                'graph_identity': graph.identity, 'spatial_map_sha256': sha(args.output/'spatial-map.npz'),
                'selection': 'fit-only transformations; validation chooses alpha/count; lock before all test conditions; no refit under ablation',
                'feature_scope': '100-500 ms; neural_linear raw counts; neural_motion external nonlinear space/time correlations of real downstream spikes; input_motion same calculation on actual encoded pulses',
                'causal_scope': 'cut all Mi1/Tm1 outgoing transmission; preserve external pulses and recorded cells; one approximate non-input/non-readout control mask; not a specific T4/T5 edge lesion',
                'static_reuse': 'identical static input schedules reuse exact frozen-run counts; one duplicate is rerun to check reset; all logical rows retain equal weight',
                'decision': 'report all models; paired whole-orbit bootstrap of dynamic-minus-static and matched-minus-target; positive lower bounds support effects within this artificial pilot only',
                'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in ('causal_motion.py','motion_challenge.py','refinement.py','simulation.py','pilot.py','sparse_refinement.py')}}
    balance = []
    for seed in protocol['fit_groups']+protocol['validation_groups']+protocol['test_groups']:
        hashes = endpoint_hashes(seed, TRAVEL)
        okay = all(all(v == list(by_kind.values())[0] for v in by_kind.values()) for by_kind in hashes.values())
        same_endpoints = all(np.array_equal(frames[0], frames[-1]) for _, frames in iter_orbit(seed, TRAVEL))
        if not okay or not same_endpoints:
            raise ValueError('endpoint shortcut balance failed')
        balance.append({'group': seed, 'endpoint_multisets_equal': okay, 'each_clip_first_equals_last': same_endpoints})
    save(args.output/'endpoint-audit.json', balance)
    save(args.output/'protocol.json', protocol)
    protocol_hash = sha(args.output/'protocol.json')
    print(json.dumps({'stage': 'protocol_locked', 'protocol_sha256': protocol_hash, 'fit':128, 'validation':64, 'test_per_condition':128}), flush=True)
    runners = {'intact': FrozenCPU(template, args.artifact/'compile/native', args.output/'intact', population='visual_input', threads=4)}
    if runners['intact'].base_hash != parent['base_sha256'] or runners['intact'].binary_hash != parent['binary_sha256']:
        raise ValueError('parent neural model changed')
    for name, mask in (('cut_input', inputs), ('matched_cut', control)):
        model = intervene(template, mask)
        # Verify the mutation is limited to outgoing transmission and its protocol hashes.
        restored = copy.deepcopy(model)
        restored['instance']['populations'][ni]['initial_state']['transmission'] = copy.deepcopy(template['instance']['populations'][ni]['initial_state']['transmission'])
        from brian2_rust.protocol import attach_protocol
        attach_protocol(restored)
        if restored != template:
            raise ValueError('intervention changed unrelated model fields')
        runners[name] = FrozenCPU(model, args.artifact/'compile/native', args.output/name, population='visual_input', threads=4)
    identities = {n: {'base_sha256': r.base_hash, 'binary_sha256': r.binary_hash} for n, r in runners.items()}
    save(args.output/'identities.json', identities)
    rows, labels = [], []
    features = {n: [] for n in ('neural_linear', 'neural_motion', 'input_motion')}
    native_runs, started = 0, time.perf_counter()

    def simulate(frames, name, key):
        nonlocal native_runs
        i, t, _ = encode_variant(frames, channels, cfg, protocol['input_mode'])
        result = runners[name].run(i, t, key)
        pop = result['populations'][ni]
        neural = event_features(pop['indices'], pop['spike_ticks'], cells, neurons)
        encoded = event_features(i, t, np.arange(len(channels)), len(channels))
        details = {'native_key': name+'/'+key, 'input_sha256': hashlib.sha256(np.stack([t,i],1).astype('<i8').tobytes()).hexdigest(),
                   'events_sha256': hashlib.sha256(np.stack([pop['spike_ticks'],pop['indices']],1).astype('<i8').tobytes()).hexdigest(),
                   'states_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in pop['states'].items()},
                   'seconds': result['wall_seconds'], 'external_spikes': len(i)}
        native_runs += 1
        del pop, result
        shutil.rmtree(runners[name].directory/key)
        (runners[name].directory/(key+'.spikes')).unlink()
        return neural, encoded, details

    blank = np.full((40,48,48), .5, dtype=np.float32)
    blanks = {n: simulate(blank, n, 'blank') for n in runners}
    baseline = blanks['intact'][0]
    np.savez_compressed(args.output/'blank-features.npz', **{n:v[0] for n,v in blanks.items()})
    save(args.output/'blank-events.json', {n:v[2] for n,v in blanks.items()})
    cache, reset_check = {}, None

    def collect(split, condition):
        nonlocal reset_check
        for seed in protocol[split+'_groups']:
            for metadata, frames in iter_orbit(seed, TRAVEL):
                kind = metadata['kind'];i,j = metadata['phase_index']
                if condition == 'static_first':
                    frames = np.repeat(frames[:1], 40, axis=0)
                elif condition == 'scrambled':
                    order = np.random.default_rng(np.random.SeedSequence([seed, i, j, 551])).permutation(np.arange(1,39))
                    frames = frames[np.r_[0,order,39]].copy()
                frame_hash = hashlib.sha256(frames.tobytes()).hexdigest()
                model_name = condition if condition in runners else 'intact'
                key = f'{split}-{condition}-{seed}-{i}-{j}-{kind}'
                reused = condition == 'static_first' and frame_hash in cache
                if reused:
                    neural, encoded, details = cache[frame_hash]
                    if reset_check is None:
                        rn, re, rd = simulate(frames, model_name, key+'-reset')
                        reset_check = (np.array_equal(rn,neural) and np.array_equal(re,encoded) and all(rd[k]==details[k] for k in ('events_sha256','states_sha256','input_sha256')))
                        if not reset_check: raise ValueError('static duplicate reset failed')
                else:
                    neural, encoded, details = simulate(frames, model_name, key)
                    if condition == 'static_first': cache[frame_hash] = neural, encoded, details
                values = {'neural_linear': neural, 'neural_motion': motion_features(neural, baseline, projection),
                          'input_motion': motion_features(encoded, np.zeros_like(encoded), input_projection)}
                for name, value in values.items(): features[name].append(value)
                labels.append(DIRECTIONS.index(kind))
                rows.append({**metadata, **details, 'split':split, 'condition':condition, 'movie_sha256':frame_hash, 'reused_identical_static':reused})
            save(args.output/'rows.json', rows)
            np.savez_compressed(args.output/'features.npz', **{n:np.asarray(v) for n,v in features.items()}, labels=labels)
            print(json.dumps({'split':split,'condition':condition,'logical_rows':len(rows),'native_runs':native_runs,'seconds':time.perf_counter()-started}),flush=True)

    collect('fit','intact');collect('validation','intact')
    y = np.asarray(labels);models, selected = {}, {}
    for name in features:
        x = np.asarray(features[name]);candidates=[]
        for count in ((32,128,512) if name=='neural_linear' else (0,)):
            for alpha in ALPHAS:
                m = fit_sparse(x[:128],y[:128],count,'raw',alpha) if name=='neural_linear' else fit(x[:128],y[:128],alpha)
                pred = (linear_scores(m,x[128:]) if name=='neural_linear' else scores(m,x[128:])).argmax(1)
                candidates.append((float(np.mean(pred==y[128:])), -count, alpha, m, pred))
        accuracy, negcount, alpha, m, pred = max(candidates,key=lambda q:q[:3])
        models[name]=m
        np.savez_compressed(args.output/(name+'-readout.npz'),**m)
        selected[name]={'readout_sha256':sha(args.output/(name+'-readout.npz')),'validation':evaluate(pred,y[128:],rows[128:]),'alpha':alpha,'cells':-negcount}
    shuffled=y[:128].reshape(8,16).copy();rng=np.random.default_rng(111926)
    for group in shuffled:rng.shuffle(group)
    models['labels_shuffled']=fit(np.asarray(features['neural_motion'])[:128],shuffled.ravel(),selected['neural_motion']['alpha'])
    np.savez_compressed(args.output/'labels_shuffled-readout.npz',**models['labels_shuffled'])
    selected['labels_shuffled']={'readout_sha256':sha(args.output/'labels_shuffled-readout.npz'),'alpha':selected['neural_motion']['alpha']}
    save(args.output/'selection.json',{'protocol_sha256':protocol_hash,'test_clips_executed':0,'readouts':selected})
    selection_hash=sha(args.output/'selection.json')
    print(json.dumps({'stage':'readouts_locked_before_test','validation':{n:v['validation']['accuracy'] for n,v in selected.items() if 'validation' in v}}),flush=True)
    for condition in CONDITIONS:collect('test',condition)
    # Restoring the target intervention means using the identical intact model.
    reference = next(r for r in rows if r['split']=='test' and r['condition']=='intact')
    frames = movie(reference['kind'],reference['group'],TRAVEL,tuple(reference['phase_index']))
    _, _, rescue = simulate(frames,'intact','post-intervention-rescue')
    rescue_check = all(rescue[k]==reference[k] for k in ('events_sha256','states_sha256','input_sha256'))
    if not rescue_check:raise ValueError('intact rescue differs from original')
    test_rows=rows[192:];test_y=np.asarray(labels[192:]);results={}
    for condition in CONDITIONS:
        mask=np.array([r['condition']==condition for r in test_rows]);selected_rows=[r for r in test_rows if r['condition']==condition]
        results[condition]={}
        for name,m in models.items():
            x=np.asarray(features['neural_motion' if name=='labels_shuffled' else name])[192:][mask]
            pred=(linear_scores(m,x) if name=='neural_linear' else scores(m,x)).argmax(1)
            results[condition][name]=evaluate(pred,test_y[mask],selected_rows)
    paired={}
    logical= [r for r in test_rows if r['condition']=='intact'];truth=np.array([DIRECTIONS.index(r['kind']) for r in logical])
    for name in models:
        correct={c:np.asarray(results[c][name]['predictions'])==truth for c in CONDITIONS}
        paired[name]={'dynamic_minus_static':paired_interval(correct['intact'],correct['static_first'],logical),
                      'intact_minus_target_cut':paired_interval(correct['intact'],correct['cut_input'],logical),
                      'intact_minus_matched_cut':paired_interval(correct['intact'],correct['matched_cut'],logical),
                      'matched_minus_target_cut':paired_interval(correct['matched_cut'],correct['cut_input'],logical),
                      'dynamic_minus_scrambled':paired_interval(correct['intact'],correct['scrambled'],logical)}
    if sha(args.output/'protocol.json')!=protocol_hash or sha(args.output/'selection.json')!=selection_hash:raise ValueError('locked artifacts changed')
    for name,v in selected.items():
        if sha(args.output/(name+'-readout.npz'))!=v['readout_sha256']:raise ValueError('readout changed after selection')
    final={'schema':protocol['schema'],'status':'complete','protocol_sha256':protocol_hash,'selection_sha256':selection_hash,
           'features_sha256':sha(args.output/'features.npz'),'test_groups':8,'test_clips_per_condition':128,
           'fit_clips':128,'validation_clips':64,'conditions':results,'paired':paired,'readouts':selected,
           'native_runs':native_runs,'logical_rows':len(rows),'seconds':time.perf_counter()-started,
           'static_reset_exact':reset_check,'intact_rescue_exact':rescue_check,'endpoint_balance_verified':True,
           'limits':['whole-orbit small pilot; fixed full-cycle speed and background','one topology-matched control mask only; intervention is all Mi1/Tm1 outputs, not a specific internal edge pathway','neural_motion computes motion externally using spatial correlations; no intrinsic FlyWire direction selectivity or topology benefit claim','all static/scrambled scores refer to source movie labels']}
    save(args.output/'report.json',final)
    print(json.dumps({'status':'complete','results':{c:{n:v['accuracy'] for n,v in m.items()} for c,m in results.items()}}),flush=True)


if __name__=='__main__':main()
