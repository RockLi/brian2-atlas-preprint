"""Analytic plasticity tests and scheduler differential checks for LK2014."""
from pathlib import Path
import os
import sys
import subprocess

import brian2 as b
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'examples'))
import brian2_rust  # noqa: F401,E402


def test_control_structure_detects_changed_arrays_with_unchanged_counts(tmp_path):
    import json
    from litwin_kumar_control_structure_review import review
    # Compact verifier fixture: this tests matching, not biological simulation.
    arrays = dict(ee_i=np.array([0, 1, 2]), ee_j=np.array([1, 2, 0]),
                  ee_initial=np.full(3, 2.76),
                  membership_e=np.array([[True, False, True]]),
                  membership_i=np.array([[True, False]]))
    conditions = {'full': None, 'no_stimulation': 'stimulation',
                  'no_istdp': 'inhibitory_plasticity', 'no_normalization': 'normalization'}
    jobs = []
    for seed in [20260906, 20260907, 20260908]:
        for condition, toggle in conditions.items():
            label = f'{condition}-seed-{seed}'
            jobs.append(dict(seed=seed, condition=condition, label=label))
            path = tmp_path/label
            path.mkdir()
            config = dict(seed=seed, scale=1, mode='learn', duration_s=2610,
                          ne=4000, ni=1000, stimulation=True,
                          inhibitory_plasticity=True, normalization=True)
            if toggle:
                config[toggle] = False
            (path/'result.json').write_text(json.dumps(dict(configuration=config, backend='rust',
                biological_seconds=2610, complete_training_protocol=True,
                topology_edges={'ee':3, 'ei':2, 'ie':2, 'ii':1})))
            np.savez(path/'activity.npz', **arrays)
    (tmp_path/'science_jobs.json').write_text(json.dumps(jobs))
    assert len(review(tmp_path, tmp_path/'passed.json')['checks']) == 12
    changed = tmp_path/'no_istdp-seed-20260907'/'activity.npz'
    for field in arrays:
        altered = {k: v.copy() for k, v in arrays.items()}
        altered[field].flat[0] = not altered[field].flat[0] if altered[field].dtype == bool else 9
        np.savez(changed, **altered)
        failed = tmp_path/f'failed-{field}.json'
        with pytest.raises(ValueError, match='initial structure differs'):
            review(tmp_path, failed)
        assert not failed.exists()
    np.savez(changed, **arrays)
    (tmp_path/'no_normalization-seed-20260908/result.json').unlink()
    with pytest.raises(FileNotFoundError):
        review(tmp_path, tmp_path/'incomplete.json')
    assert not (tmp_path/'incomplete.json').exists()


def test_phase_checkpoint_survives_latest_and_restores_fresh_process(tmp_path):
    import json
    command = [sys.executable, str(ROOT/'examples/litwin_kumar_device.py'),
               '--network-scale', '.01', '--threads', '2', '--warmup-s', '.02',
               '--train-repetitions', '1', '--stimulus-s', '.02', '--gap-s', '0',
               '--spontaneous-s', '.02', '--segment-seconds', '1',
               '--bounded-window-steps', '200', '--compact-artifacts']
    original, restored = tmp_path/'original', tmp_path/'restored'
    subprocess.run([*command, '--checkpoint', '--keep-phase-checkpoints',
                    '--output', str(original)], check=True, capture_output=True)
    report = json.loads((original/'result.json').read_text())
    assert set(report['retained_phase_checkpoints']) == {'warmup', 'training'}
    checkpoint = original/'checkpoint-training.pkl'
    assert checkpoint.stat().st_ino != (original/'checkpoint.pkl').stat().st_ino
    subprocess.run([*command, '--restore', str(checkpoint),
                    '--output', str(restored)], check=True, capture_output=True)
    with np.load(original/'state.npz') as expected, np.load(restored/'state.npz') as actual:
        assert set(expected.files) == set(actual.files)
        for field in expected.files:
            assert expected[field].dtype == actual[field].dtype
            assert expected[field].shape == actual[field].shape
            assert expected[field].tobytes() == actual[field].tobytes(), field
    continuation = json.loads((restored/'result.json').read_text())
    assert continuation['initial_time_seconds'] == pytest.approx(.42)
    assert continuation['restored_segment_count'] == 2


def test_timed_array_above_old_million_element_limit(tmp_path):
    # Reading the last row proves that both lowering and native validation
    # retain the full input, including entries beyond the former limit.
    select('reference', tmp_path/'large-input')
    values = np.zeros((1001, 1000))
    values[-1, 0] = 7
    stimulus = b.TimedArray(values, dt=.1*b.ms)
    group = b.NeuronGroup(1, 'v : 1', dt=.1*b.ms, namespace={'stimulus': stimulus})
    group.run_regularly('v = stimulus(t + 100*ms, i)')
    b.Network(group).run(.1*b.ms)
    assert group.v[0] == 7


def test_sensitivity_rejects_unplanned_configuration_changes():
    from litwin_kumar_model import LKConfig
    from litwin_kumar_sensitivity_analysis import CONDITIONS, matched_configuration
    baseline = LKConfig().to_dict()
    historical = {key: value for key, value in baseline.items()
                  if key not in {'delay_distribution', 'trace_integration'}}
    for condition, overrides in CONDITIONS.items():
        variant = {**baseline, **overrides}
        assert matched_configuration(historical, variant, condition) == baseline
        with pytest.raises(ValueError, match='unplanned configuration'):
            matched_configuration(historical, {**variant, 'inhibitory_plasticity': False}, condition)
    with pytest.raises(ValueError, match='unplanned configuration'):
        matched_configuration(historical, {**baseline, 'dt_ms': .025}, 'dt-005')
    with pytest.raises(ValueError, match='unplanned configuration'):
        matched_configuration(historical, baseline, 'joint')


def test_performance_figure_checks_raw_repeats_and_timing_scope(tmp_path):
    import json
    from litwin_kumar_performance_review import validated_report
    samples = [{'backend': backend, 'threads': threads, 'repeat': k,
                'simulation_seconds': value,
                'summary': {'neuron_count': 5000} if backend == 'rust' else {}}
               for backend, threads, value in [('rust', 8, 1.), ('cpp', 4, 2.)]
               for k in range(5)]
    gate = {backend: {'backend': backend, 'threads': threads, 'n': 5,
                     'median_seconds': value, 'min_seconds': value, 'max_seconds': value}
            for backend, threads, value in [('rust', 8, 1.), ('cpp', 4, 2.)]}
    report = {'suite': 'whole_run', 'profiled': False, 'all_outputs_byte_exact': True,
              'samples': samples, 'fixed_rust8_gate': {**gate, 'passed': True, 'speedup': 2.}}
    path = tmp_path/'report.json'
    path.write_text(json.dumps(report))
    validated_report(tmp_path, 'whole_run')
    with pytest.raises(ValueError, match='timing scope'):
        validated_report(tmp_path, 'performance')
    samples[0]['simulation_seconds'] = 3.  # Invalidates the claimed range separation.
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='raw samples'):
        validated_report(tmp_path, 'whole_run')
    samples[0]['simulation_seconds'] = 1.
    samples[1]['repeat'] = 0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='duplicate'):
        validated_report(tmp_path, 'whole_run')


def test_long_single_run_obeys_ten_million_step_budget(tmp_path):
    select('aot', tmp_path/'long-run')
    group = b.NeuronGroup(1, 'count : integer', dt=.1*b.ms)
    group.run_regularly('count += 1', when='end')
    b.Network(group).run(100.0001*b.second)
    assert group.count[0] == 1_000_001


def test_scientific_statistics_use_seed_pairs_and_remove_common_bursts():
    from litwin_kumar_analysis import sign_flip_test, holm_adjust, selective_activity
    assert sign_flip_test([1, 2, 3]) == .25
    assert sign_flip_test([1e-20, 2e-20, 3e-20]) == .25
    assert sign_flip_test([0, 0, 0]) == 1.
    assert sign_flip_test([1, -1]) == 1.
    np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])
    baseline = np.arange(40) < 20
    spontaneous = ~baseline
    population = np.where(baseline, 3., 20.)
    rates = np.tile(population, (3, 1))
    common_burst = selective_activity(rates, population, baseline, spontaneous)
    assert common_burst['selective_excess_change_hz'] == 0
    assert common_burst['selective_occupancy'] == 0
    rates[0, spontaneous] += 5
    selective = selective_activity(rates, population, baseline, spontaneous)
    assert selective['selective_excess_change_hz'] == 5
    assert selective['selective_occupancy'] == 1/3


def test_reactivation_episodes_remove_global_bursts_and_split_coverage_gaps():
    from litwin_kumar_analysis import reactivation_episodes
    time = (np.arange(40)+.5)*.05
    baseline = np.arange(40) < 20
    spontaneous = ~baseline
    coverage = np.ones(40, dtype=bool)
    population = np.where(baseline, 3., 30.)
    rates = np.tile(population, (3, 1))
    episodes, metrics = reactivation_episodes(rates, population, time,
                                            baseline, spontaneous, coverage)
    assert episodes == [] and metrics['reactivation_episode_count'] == 0
    rates[0, 22:29] += 5
    coverage[25] = False
    rates[:, 25] = np.nan
    rates[1, 29:32] += 7
    rates[2, 35] += 10  # A single 50 ms bin does not qualify.
    episodes, metrics = reactivation_episodes(rates, population, time,
                                            baseline, spontaneous, coverage)
    assert [row['assembly'] for row in episodes] == [1, 1, 2]
    assert [(row['start_bin'], row['stop_bin']) for row in episodes] == [(22, 25), (26, 29), (29, 32)]
    assert episodes[0]['right_censored'] and episodes[1]['left_censored']
    assert metrics['reactivated_assembly_count'] == 2
    assert metrics['adjacent_dominant_switches'] == 1
    assert metrics['censored_episode_count'] == 2
    assert metrics['observed_episode_duration_median_s'] == pytest.approx(.15)


def test_population_conditioning_removes_common_rate_and_variance_changes():
    from litwin_kumar_analysis import population_conditioned_activity, selective_activity
    rng = np.random.default_rng(91)
    baseline = np.arange(20000) < 10000
    # Different fixed assembly rates all undergo the same tenfold population
    # increase. This contains no new assembly preference or learned structure.
    means = np.linspace(10, 50, 20)[:, None]*np.where(baseline, 1, 10)
    counts = rng.poisson(means)
    rates = counts/(200*.05)
    population = counts.sum(axis=0)/(4000*.05)
    score, fractions = population_conditioned_activity(
        rates, population, np.full(20, 200), 4000, .05, baseline)
    raw = selective_activity(rates, population, baseline, ~baseline)
    adjusted = selective_activity(score, np.zeros(20000), baseline, ~baseline)
    assert raw['selective_excess_change_hz'] > 10
    assert abs(adjusted['selective_excess_change_hz']) < .2
    assert np.std(score[:, baseline]) == pytest.approx(1., abs=.02)
    assert np.std(score[:, ~baseline]) == pytest.approx(1., abs=.02)
    assert fractions.sum() == pytest.approx(1.)


@pytest.fixture(autouse=True)
def clean_device():
    previous = b.get_device()
    previous_target = b.prefs.codegen.target
    yield
    b.get_device().reinit()
    b.set_device(previous)
    b.prefs.codegen.target = previous_target
    b.start_scope()


def select(backend, directory, threads=1):
    from brian2.devices.device import all_devices
    all_devices["rust_standalone"].reinit()
    b.get_device().reinit()
    b.start_scope()
    if backend == 'numpy':
        b.set_device('runtime')
        b.prefs.codegen.target = 'numpy'
    else:
        b.set_device('rust_standalone', engine=backend,
                     runner=Path(os.environ.get('B2_RUNNER',
                         str(ROOT/'target/release/b2-runner'))),
                     directory=directory, threads=threads)


@pytest.mark.parametrize('slot', ['groups', 'end'])
@pytest.mark.parametrize('periodic_sum', [False, True])
def test_periodic_synapse_runner_matches_numpy(tmp_path, slot, periodic_sum):
    values = []
    for backend in ['numpy', 'reference', 'aot']:
        select(backend, tmp_path/backend)
        neurons = b.NeuronGroup(3, 'dv/dt = 0*Hz : 1\ntotal : 1', threshold='False', reset='', method='euler',
                               dt=1*b.ms, name='normalization_neurons')
        synapses = b.Synapses(neurons, neurons,
                             'w : 1\ntotal_post = w : 1 (summed)',
                             on_pre='w += 0', clock=neurons.clock, name='normalization_synapses')
        synapses.connect(i=[0, 1, 2], j=[1, 1, 2])
        synapses.w = [1, 3, 2]
        regular = synapses.run_regularly('w = w + dt/ms - total_post/4',
                              dt=2*b.ms, when=slot,
                              name='normalization_update')
        if periodic_sum:
            synapses.summed_updaters['total_post']._clock = regular.clock
        network = b.Network(neurons, synapses)
        network.run(6*b.ms)
        values.append((synapses.w[:].copy(), neurons.total[:].copy()))
    for value in values[1:]:
        for actual, expected in zip(value, values[0]):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('slot', ['groups', 'end'])
@pytest.mark.parametrize('random_inputs', [False, True])
def test_parallel_synapse_regular_preserves_edge_state_and_clock(tmp_path, slot, random_inputs):
    import json
    outputs = []
    arrays = []
    source = np.repeat(np.arange(256), 512)
    target = np.tile(np.arange(512), 256)
    for threads in [1, 4]:
        select('aot', tmp_path/str(threads), threads=threads)
        b.seed(43)
        group = b.NeuronGroup(512, 'v : 1', dt=.1*b.ms,
                             threshold='False', reset='',
                             name='edge_regular_neurons')
        group.v = np.arange(512)/512
        synapse = b.Synapses(group, group,
            'w : 1\ncount : integer\nflag : boolean\nscale : 1 (constant)',
            on_pre='w += 0',
            clock=group.clock, name='edge_regular_synapses')
        synapse.connect(i=source, j=target)
        synapse.w = .25
        synapse.scale = np.where(target % 2, .125, .25)
        update = '''
            w = clip(w + (v_pre - v_post)*scale + dt/ms, -1, 5)
            count += 1
            flag = not flag
            '''
        if random_inputs:
            update += '\nw += .01*rand() + .001*randn() + .001*sin(t/ms)'
        synapse.run_regularly(update, dt=.5*b.ms, when=slot, name='edge_regular_update')
        b.Network(group, synapse).run(2*b.ms)
        directory = b.get_device().last_run_directory/'rust'
        outputs.append((directory/'results.bin').read_bytes())
        arrays.append(np.asarray(synapse.w[:]).copy())
        summary = json.loads((directory/'summary.json').read_text())
        assert summary['parallel_synapse_regular'] == (threads > 1)
        assert summary['threads'] == threads  # This runner alone enables the pool.
        np.testing.assert_array_equal(synapse.count[:], 4)
        np.testing.assert_array_equal(synapse.flag[:], False)
    assert outputs[0] == outputs[1]
    expected = np.full(len(source), .25)
    scale = np.where(target % 2, .125, .25)
    for _ in range(4):
        expected = np.clip(expected + (source/512-target/512)*scale + .1, -1, 5)
    if not random_inputs:
        np.testing.assert_array_equal(arrays[0], expected)


def test_parallel_clock_driven_synapse_ode_preserves_delays_and_summed_state(tmp_path):
    """The generic edge updater must preserve an RK4/summed/event workload."""
    import json
    outputs = []
    states = []
    for threads in [1, 4]:
        select('aot', tmp_path / str(threads), threads=threads)
        group = b.NeuronGroup(
            256, 'dv/dt = -v/(10*ms) + total/(10*ms) : 1\ntotal : 1',
            threshold='t == 0*ms', reset='', method='rk4', dt=.1*b.ms,
            name='clock_edge_neurons')
        group.v = .1
        synapse = b.Synapses(
            group, group,
            '''total_post = s : 1 (summed)
               ds/dt = -s/(100*ms) + (0.5*kHz)*x*(1-s) : 1 (clock-driven)
               dx/dt = -x/(2*ms) : 1 (clock-driven)''',
            on_pre='x += 1', delay=.2*b.ms, method='rk4',
            clock=group.clock, name='clock_edge_synapses')
        synapse.connect(p=1)
        b.Network(group, synapse).run(1*b.ms)
        directory = b.get_device().last_run_directory / 'rust'
        outputs.append((directory / 'results.bin').read_bytes())
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['parallel_synapse_state'] == (threads > 1)
        assert summary['threads'] == threads
        states.append((np.asarray(synapse.s[:]).copy(),
                       np.asarray(synapse.x[:]).copy(),
                       np.asarray(group.total[:]).copy()))
    assert outputs[0] == outputs[1]
    for actual, expected in zip(states[1], states[0]):
        np.testing.assert_array_equal(actual, expected)
    assert np.any(states[0][0] > 0)
    assert np.any(states[0][1] > 0)


def triplet_oracle(pre_ms, post_ms, initial=.4):
    """Independent scalar event calculation, including left-limit slow traces."""
    import math
    traces = [0.0, 0.0, 0.0, 0.0]
    taus = [16.8, 101., 33.7, 125.]
    weight, previous = initial, 0.0
    for time, kind in sorted([(t, 0) for t in pre_ms]+[(t, 1) for t in post_ms]):
        traces = [x*math.exp(-(time-previous)/tau) for x, tau in zip(traces, taus)]
        r1, r2, o1, o2 = traces
        if kind == 0:
            weight -= 7e-3*o1+2.3e-4*o1*r2
            traces[0] += 1
            traces[1] += 1
        else:
            weight += 7.5e-10*r1+9.3e-3*r1*o2
            traces[2] += 1
            traces[3] += 1
        weight = min(1., max(0., weight))
        previous = time
    return weight


@pytest.mark.parametrize('pre,post', [([5], [15]), ([15], [5]),
                                       ([5, 25], [15]), ([15], [5, 25]),
                                       ([5, 15], [15, 25])])
def test_triplet_against_analytic_event_oracle(tmp_path, pre, post):
    from litwin_kumar_model import (TRIPLET_EQUATIONS, TRIPLET_PRE,
                                   TRIPLET_POST, TRIPLET_PARAMETERS)
    expected = triplet_oracle(pre, post)
    for backend in ['numpy', 'reference', 'aot']:
        select(backend, tmp_path/backend)
        clock = b.Clock(dt=1*b.ms)
        source = b.SpikeGeneratorGroup(1, np.zeros(len(pre), dtype=int), pre*b.ms,
                                       clock=clock, name='triplet_source')
        target = b.NeuronGroup(1, 'dv/dt=0*Hz : 1',
                              threshold=' or '.join(f't == {t}*ms' for t in post),
                              reset='v=0', method='euler', clock=clock, name='triplet_target')
        synapse = b.Synapses(source, target, TRIPLET_EQUATIONS,
                             on_pre=TRIPLET_PRE, on_post=TRIPLET_POST,
                             namespace={**TRIPLET_PARAMETERS, 'learning': 1.,
                                        'wmin': 0., 'wmax': 1.},
                             clock=clock, name='triplet')
        synapse.connect(i=[0], j=[0])
        synapse.w = .4
        b.Network(source, target, synapse).run(30*b.ms)
        np.testing.assert_allclose(synapse.w[:], [expected], rtol=1e-13, atol=1e-15)


def test_clustered_initialization_preserves_total_and_bounds():
    from litwin_kumar_model import project_incoming
    targets = np.array([0, 0, 0, 1, 1])
    expected = np.array([3*2.76, 2*2.76, 0])
    weights = project_incoming(np.array([21.4, 1.78, 1.78, 21.4, 21.4]),
                               targets, expected, 1.78, 21.4)
    np.testing.assert_allclose(np.bincount(targets, weights=weights, minlength=3), expected)
    assert np.all((weights >= 1.78) & (weights <= 21.4))


def test_adaptation_shared_expression_does_not_escape_refractory_branch(tmp_path):
    snapshots = []
    for backend in ['numpy', 'reference', 'aot']:
        select(backend, tmp_path/backend)
        group = b.NeuronGroup(2, '''
            dv/dt = (-adapt-v)/(10*ms) : 1 (unless refractory)
            dadapt/dt = -adapt/(10*ms) : 1
            ''', threshold='v>.5', reset='v=0; adapt+=.1', refractory=2*b.ms,
            method='euler', dt=b.ms, name='adaptation_probe')
        group.v = [1, .2]
        group.adapt = [.3, .4]
        state = b.StateMonitor(group, ['v', 'adapt'], record=True)
        b.Network(group, state).run(10*b.ms)
        snapshots.append((state.v[:].copy(), state.adapt[:].copy()))
    for snapshot in snapshots[1:]:
        for actual, expected in zip(snapshot, snapshots[0]):
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)


def test_checkpoint_atomic_replace_and_checksum(tmp_path, monkeypatch):
    import pickle
    from brian2_rust.device import _read_checkpoint, _write_checkpoint
    path = tmp_path/'checkpoint.pkl'
    _write_checkpoint(path, {'midpoint': {'value': 42}})
    original = path.read_bytes()
    with monkeypatch.context() as context:
        def fail_replace(*args):
            raise OSError('injected interruption before commit')
        context.setattr('brian2_rust.device.os.replace', fail_replace)
        with pytest.raises(OSError, match='injected interruption'):
            _write_checkpoint(path, {'midpoint': {'value': 99}})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    envelope = pickle.loads(original)
    envelope['payload'] = envelope['payload'][:-1]+bytes([envelope['payload'][-1] ^ 1])
    path.write_bytes(pickle.dumps(envelope))
    with pytest.raises(NotImplementedError, match='checksum mismatch'):
        _read_checkpoint(path)
    path.write_bytes(pickle.dumps({'legacy': {'value': 42}}))
    assert _read_checkpoint(path) == {'legacy': {'value': 42}}


@pytest.mark.parametrize('pre,post', [([5], []), ([5], [15]), ([15], [5]),
                                       ([5, 15], [15, 25])])
def test_inhibitory_plasticity_against_event_oracle(tmp_path, pre, post):
    import math
    from litwin_kumar_model import (INHIBITORY_EQUATIONS, INHIBITORY_PRE,
                                   INHIBITORY_POST)
    weight, previous, xpre, xpost = 100., 0., 0., 0.
    for event_time, kind in sorted([(t, 0) for t in pre]+[(t, 1) for t in post]):
        decay = math.exp(-(event_time-previous)/20.)
        xpre, xpost = xpre*decay, xpost*decay
        weight += xpost-.12 if kind == 0 else xpre
        weight = min(243., max(48.7, weight))
        xpre += kind == 0
        xpost += kind == 1
        previous = event_time
    for backend in ['numpy', 'reference', 'aot']:
        select(backend, tmp_path/backend)
        clock = b.Clock(dt=b.ms)
        source = b.SpikeGeneratorGroup(1, np.zeros(len(pre), dtype=int), pre*b.ms,
                                       clock=clock, name='inhibitory_source')
        target = b.NeuronGroup(1, 'v : 1',
            threshold=' or '.join(f't == {t}*ms' for t in post) or 'False',
            reset='v=0', clock=clock, name='inhibitory_target')
        synapse = b.Synapses(source, target, INHIBITORY_EQUATIONS,
            on_pre=INHIBITORY_PRE, on_post=INHIBITORY_POST,
            namespace={'tau_i': 20*b.ms, 'learning': 1., 'eta': 1.,
                       'alpha': .12, 'wmin': 48.7, 'wmax': 243.},
            clock=clock, name='inhibitory_probe')
        synapse.connect(i=[0], j=[0])
        synapse.w = 100.
        b.Network(source, target, synapse).run(30*b.ms)
        np.testing.assert_allclose(synapse.w[:], [weight], rtol=1e-14, atol=1e-14)


def test_assembly_summary_counts_existing_edges_and_overlapping_membership():
    from litwin_kumar_figures import assembly_weight_matrix
    members = np.array([[True, True, False], [False, True, True]])
    i, j, w = np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([2., 4., 8.])
    means, counts = assembly_weight_matrix(i, j, w, members)
    np.testing.assert_array_equal(counts, [[1, 1], [2, 1]])
    np.testing.assert_array_equal(means, [[2, 8], [3, 4]])


@pytest.mark.parametrize('slot', ['groups', 'end'])
def test_periodic_synapse_random_functions(tmp_path, slot):
    values = []
    for backend in ['reference', 'aot']:
        select(backend, tmp_path/backend)
        b.seed(43)
        group = b.NeuronGroup(9, 'v : 1', threshold='False', reset='',
                             dt=b.ms, name='regular_random_group')
        synapse = b.Synapses(group, group, 'w : 1', on_pre='w += 0',
                            clock=group.clock, name='regular_random_synapse')
        synapse.connect(i=[0, 1, 2, 3, 4], j=[4, 3, 2, 1, 0])
        synapse.w = .2
        synapse.run_regularly('w += sin(t/ms) + dt/ms + rand() + .01*i + .1*j',
                              dt=2*b.ms, when=slot, name='regular_random_update')
        b.Network(group, synapse).run(6*b.ms)
        values.append(np.asarray(synapse.w[:]).copy())
    np.testing.assert_array_equal(values[0], values[1])


def test_decimal_mixed_clocks_preserve_owner_time_and_refractory_state(tmp_path):
    """At 60 ms, 600*0.1 ms and 3*20 ms differ by one floating-point ULP."""
    snapshots = []
    for backend in ['numpy', 'reference', 'aot']:
        select(backend, tmp_path/backend)
        group = b.NeuronGroup(1, '''
            dv/dt=1*Hz : 1 (unless refractory)
            total : 1
            sampled_t : 1
            sampled_dt : 1
            ''', threshold='t >= 60*ms and t < 60.05*ms', reset='v=0',
            refractory=b.ms, dt=.1*b.ms, method='euler', name='decimal_clock_group')
        synapse = b.Synapses(group, group, 'w : 1\nx : 1\ntotal_post=w+t/second : 1 (summed)',
                            on_pre='w += 0', clock=group.clock, name='decimal_clock_synapse')
        synapse.connect(i=[0], j=[0])
        regular = synapse.run_regularly('w=t/second; x=dt/second', dt=20*b.ms,
                                       when='groups', name='decimal_clock_regular')
        synapse.summed_updaters['total_post']._clock = regular.clock
        group.run_regularly('sampled_t=t/second; sampled_dt=dt/second',
                            dt=20*b.ms, when='end', name='decimal_clock_population_regular')
        state = b.StateMonitor(group, ['v', 'total', 'sampled_t', 'sampled_dt'], record=True)
        spikes = b.SpikeMonitor(group)
        b.Network(group, synapse, state, spikes).run(100*b.ms)
        snapshots.append([np.asarray(value).copy() for value in
            [group.lastspike[:], group.not_refractory[:], synapse.w[:], synapse.x[:],
             state.v[:], state.total[:], state.sampled_t[:], state.sampled_dt[:], spikes.t[:]]])
    for snapshot in snapshots[1:]:
        for actual, expected in zip(snapshot, snapshots[0]):
            np.testing.assert_array_equal(actual, expected)


def test_backend_activity_comparison_excludes_unmatched_observation_bins():
    from litwin_kumar_backend_comparison import matched_activity_metrics
    time = (np.arange(80)+.5)*.05
    population = np.full(80, 10.)
    rates = np.array([10.+np.sin(np.arange(80)), 10.-np.sin(np.arange(80))])
    rust_coverage = np.ones(80, dtype=bool)
    rust_coverage[60::2] = False
    data = {backend: {'time_s': time.copy(), 'population_rate_hz': population.copy(),
                     'assembly_rates_hz': rates.copy(), 'coverage': np.ones(80, dtype=bool)}
            for backend in ['rust', 'cpp']}
    data['rust']['coverage'] = rust_coverage
    # A burst seen only by C++ must not change a matched-observation comparison.
    data['cpp']['assembly_rates_hz'][0, ~rust_coverage] = 80.
    data['rust']['assembly_rates_hz'][:, ~rust_coverage] = np.nan
    data['rust']['population_rate_hz'][~rust_coverage] = np.nan
    config = {'ne': 100, 'warmup_s': 2., 'training_s': 1.}
    rows, _ = matched_activity_metrics(data, np.array([20, 20]), config)
    assert rows['rust'] == rows['cpp']
    assert rows['rust']['common_baseline_seconds'] == pytest.approx(1.)
    assert rows['rust']['common_spontaneous_seconds'] == pytest.approx(.5)
    data['cpp']['time_s'] += .01
    with pytest.raises(ValueError, match='time grids differ'):
        matched_activity_metrics(data, np.array([20, 20]), config)


def test_delay_sensitivity_preserves_initial_network_and_uses_clock_grid():
    from litwin_kumar_model import LKConfig, make_network
    saved = []
    for distribution in ['fixed', 'uniform', 'uniform']:
        b.start_scope()
        b.set_device('runtime')
        b.prefs.codegen.target = 'numpy'
        model = make_network(LKConfig(scale=.01, delay_distribution=distribution))
        saved.append({
            'membership': {k: v.copy() for k, v in model.membership.items()},
            'edges': {k: (i.copy(), j.copy()) for k, (i, j) in model.edges.items()},
            'voltage': np.asarray(model.exc.v[:]).copy(),
            'delay': {k: np.asarray(s.pre.delay[:]/b.ms).copy() for k, s in model.synapses.items()}})
    for candidate in saved[1:]:
        np.testing.assert_array_equal(candidate['voltage'], saved[0]['voltage'])
        for name in saved[0]['membership']:
            np.testing.assert_array_equal(candidate['membership'][name], saved[0]['membership'][name])
        for name in saved[0]['edges']:
            np.testing.assert_array_equal(candidate['edges'][name], saved[0]['edges'][name])
            delays = candidate['delay'][name]
            np.testing.assert_allclose(delays/.1, np.rint(delays/.1), rtol=0, atol=1e-12)
            assert delays.min() >= 0 and delays.max() <= 1.5+1e-12
            np.testing.assert_array_equal(delays, saved[1]['delay'][name])
    assert set(np.rint(saved[1]['delay']['ee']/.1).astype(int)) == set(range(16))


def test_clock_driven_triplet_with_delayed_and_simultaneous_events(tmp_path):
    from litwin_kumar_model import TRIPLET_EQUATIONS, TRIPLET_PRE, TRIPLET_POST, TRIPLET_PARAMETERS
    traces = np.zeros(4)
    weight = 2.76
    decay = 1-1/np.array([16.8, 101., 33.7, 125.])
    for tick in range(10):
        traces *= decay
        if tick in [3, 7]:  # presynaptic emission at 1/5 ms, arrival after 2 ms
            weight = np.clip(weight-.007*traces[2]-.00023*traces[2]*traces[1], 1.78, 21.4)
            traces[:2] += 1
        if tick in [3, 6]:
            weight = np.clip(weight+7.5e-10*traces[0]+.0093*traces[0]*traces[3], 1.78, 21.4)
            traces[2:] += 1
    for backend in ['numpy', 'reference', 'aot']:
        select(backend, tmp_path/backend)
        clock = b.Clock(dt=b.ms)
        source = b.SpikeGeneratorGroup(1, [0, 0], [1, 5]*b.ms, clock=clock)
        target = b.NeuronGroup(1, 'v : 1', threshold='t == 3*ms or t == 6*ms',
                              reset='v = 0', clock=clock)
        synapse = b.Synapses(source, target,
            TRIPLET_EQUATIONS.replace('(event-driven)', '(clock-driven)'),
            on_pre=TRIPLET_PRE, on_post=TRIPLET_POST, clock=clock, method='euler',
            namespace={**TRIPLET_PARAMETERS, 'learning': 1., 'wmin': 1.78, 'wmax': 21.4})
        synapse.connect(i=[0], j=[0]); synapse.w = 2.76; synapse.pre.delay = 2*b.ms
        b.Network(source, target, synapse).run(10*b.ms)
        np.testing.assert_allclose(synapse.w[:], [weight], rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose([getattr(synapse, name)[0] for name in ['r1', 'r2', 'o1', 'o2']],
                                   traces, rtol=1e-13, atol=1e-14)


def test_memory_window_audit_checks_counters_and_boundary_spikes(tmp_path):
    from litwin_kumar_memory_analysis import verify_window
    full = {'exc_v': np.array([-.06, -.055]),
            'exc_spike_total': np.array([2., 1.]), 'inh_spike_total': np.array([1.]),
            'exc_spike_i': np.array([0, 1, 0], dtype=np.int32),
            'exc_spike_t': np.array([.1, .3, .4]),
            'inh_spike_i': np.array([0], dtype=np.int32), 'inh_spike_t': np.array([.2]),
            'voltage_t': np.arange(5)*.1, 'voltage_v': np.arange(10.).reshape(2, 5)}
    rolling = {k: v.copy() for k, v in full.items()}
    rolling.update(exc_spike_i=full['exc_spike_i'][1:], exc_spike_t=full['exc_spike_t'][1:],
                   inh_spike_i=full['inh_spike_i'][:0], inh_spike_t=full['inh_spike_t'][:0],
                   voltage_t=full['voltage_t'][3:], voltage_v=full['voltage_v'][:, 3:])
    a,b = tmp_path/'full.npz', tmp_path/'rolling.npz'
    np.savez(a, **full); np.savez(b, **rolling)
    result = verify_window(a,b,dt_s=.1,horizon_s=.5,window_steps=2)
    assert result['retained_from_tick'] == 3
    assert result['exc_retained_spikes'] == 2
    assert 'exc_spike_total' in result['dynamics_fields_byte_exact']
    rolling['exc_spike_total'][0] += 1
    np.savez(b, **rolling)
    with pytest.raises(AssertionError, match='exc_spike_total'):
        verify_window(a,b,dt_s=.1,horizon_s=.5,window_steps=2)
    rolling['exc_spike_total'] = full['exc_spike_total']
    rolling['exc_spike_i'] = full['exc_spike_i']
    rolling['exc_spike_t'] = full['exc_spike_t']
    np.savez(b, **rolling)
    with pytest.raises(AssertionError, match='exc_spike_t'):
        verify_window(a,b,dt_s=.1,horizon_s=.5,window_steps=2)


def test_activity_review_exposes_drift_hidden_by_whole_interval_mean():
    from litwin_kumar_activity_review import review_data
    time = (np.arange(4200)+.5)*.05
    population = np.full(len(time), 2.)
    population[(time >= 10) & (time < 110)] = 1.
    population[time >= 110] = 3.
    data = {'time_s': time, 'coverage': np.ones(len(time), dtype=bool),
            'population_rate_hz': population, 'assembly_rates_hz': np.tile(population, (2, 1))}
    config = {'ne': 100, 'warmup_s': 4, 'training_s': 6, 'duration_s': 210}
    arrays, metrics = review_data(data, np.array([20, 20]), config)
    assert arrays['population_rate_1s_hz'].mean() == metrics['baseline_population_hz'] == 2.
    assert metrics['first_100s_population_hz'] == 1.
    assert metrics['last_100s_population_hz'] == 3.
    assert metrics['late_over_early_population_rate'] == 3.
    np.testing.assert_allclose(arrays['conditioned_score'], 0., atol=1e-14)
    np.testing.assert_allclose(arrays['population_time_1s'], np.arange(200)+.5, atol=1e-12)
    # An unobserved interval cannot silently become a zero-rate/stable period.
    data['coverage'][2000] = False
    with pytest.raises(ValueError, match='complete baseline'):
        review_data(data, np.array([20, 20]), config)
    data['coverage'][2000] = True
    data['time_s'] = time.copy()
    data['time_s'][2000] += .01
    with pytest.raises(ValueError, match='uniform time grid'):
        review_data(data, np.array([20, 20]), config)


def test_cpp_placement_requires_consistent_observed_rust_cpu_order():
    from litwin_kumar_cpp_placement import matching_rust_places
    report = {'samples': [
        {'backend': 'rust', 'threads': 4,
         'summary': {'thread_affinity': True, 'thread_cpus': [0, 1, 36, 37]}}
        for _ in range(2)]}
    assert matching_rust_places(report, 4) == '{0},{1},{36},{37}'
    report['samples'][1]['summary']['thread_cpus'] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match='changes across'):
        matching_rust_places(report, 4)
    report['samples'][1]['summary']['thread_affinity'] = False
    with pytest.raises(ValueError, match='explicit unique'):
        matching_rust_places(report, 4)


def test_scaling_review_rejects_missing_pairs_and_unverified_worker_metadata(tmp_path):
    import copy
    import json
    from litwin_kumar_scaling_review import validated_report
    samples = [{'backend': backend, 'threads': n, 'repeat': k,
                'simulation_seconds': (2 if backend == 'cpp' else 1)/n + k*.01,
                'summary': {'neuron_count': 5000, 'threads': n,
                            'phase_profile': {'enabled': False}}}
               for backend in ['rust', 'cpp'] for n in [1, 2] for k in range(3)]
    report = {'suite': 'performance', 'profiled': False,
              'warmup_s': 10, 'timed_duration_s': 1,
              'recording': 'identical full-history SpikeMonitor and StateMonitor',
              'all_outputs_byte_exact': True, 'samples': samples,
              'new_cpp_best': {'backend': 'cpp', 'threads': 2, 'n': 3,
                               'median_seconds': 1.01, 'min_seconds': 1., 'max_seconds': 1.02}}
    def check(candidate):
        (tmp_path/'report.json').write_text(json.dumps(candidate))
        return validated_report(tmp_path, 'performance', [1, 2], 3)
    assert len(check(report)[1]) == 4
    for mutation in ['missing', 'duplicate', 'worker', 'profiled', 'winner', 'duration', 'nonfinite']:
        invalid = copy.deepcopy(report)
        if mutation == 'missing':
            invalid['samples'].pop()
        elif mutation == 'duplicate':
            invalid['samples'].append(invalid['samples'][0])
        elif mutation == 'worker':
            invalid['samples'][0]['summary']['threads'] = 2
        elif mutation == 'profiled':
            invalid['samples'][0]['summary']['phase_profile']['enabled'] = True
        elif mutation == 'winner':
            invalid['new_cpp_best']['threads'] = 1
        elif mutation == 'duration':
            invalid['timed_duration_s'] = 11
        else:
            invalid['samples'][0]['simulation_seconds'] = float('nan')
        with pytest.raises(ValueError):
            check(invalid)


def test_recovery_review_covers_new_segments_and_detects_earlier_spike_corruption(tmp_path):
    import copy
    import json
    from litwin_kumar_recovery_review import verify_trajectory
    original, restored = tmp_path/'original', tmp_path/'restored'
    report = {'configuration': {'dt_ms': 100}, 'initial_time_seconds': 0.,
              'biological_seconds': 3., 'recording_window_steps': 10,
              'segments': [{'start_s': float(i), 'stop_s': float(i+1)} for i in range(3)],
              'trajectory': [{'time_s': float(i+1), 'weights': {'mean': 2.76},
                              'spike_totals': {'exc': i*2, 'inh': 0}} for i in range(3)]}
    restored_report = copy.deepcopy(report)
    restored_report['initial_time_seconds'] = 1.
    restored_report['trajectory'][0]['weights']['mean'] = 999.  # Prefix is not replay evidence.
    for folder, metadata in [(original, report), (restored, restored_report)]:
        folder.mkdir()
        (folder/'result.json').write_text(json.dumps(metadata))
        for index in [2, 3]:
            segment = folder/'segments'/f'{index:04d}'
            segment.mkdir(parents=True)
            np.savez(segment/'spikes.npz', exc_i=np.array([0, 1]),
                     exc_t=np.array([index-.9, index-.1]),
                     inh_i=np.array([], dtype=int), inh_t=np.array([]))
    result = verify_trajectory(original, restored, 1.)
    assert result['verified_seconds'] == 2
    assert len(result['segments']) == 2
    for label in ['gap', 'short_window', 'diagnostics', 'incomplete']:
        changed = copy.deepcopy(restored_report)
        if label == 'gap':
            changed['segments'][1]['start_s'] = 1.1
        elif label == 'short_window':
            changed['recording_window_steps'] = 1
        elif label == 'diagnostics':
            changed['trajectory'][1]['weights']['mean'] = 9.
        else:
            changed['segments'].pop()
        (restored/'result.json').write_text(json.dumps(changed))
        with pytest.raises((ValueError, AssertionError)):
            verify_trajectory(original, restored, 1.)
    (restored/'result.json').write_text(json.dumps(restored_report))
    # Corrupt a replayed segment before the final one; endpoint-only comparison misses this.
    np.savez(restored/'segments/0002/spikes.npz', exc_i=np.array([0, 2]),
             exc_t=np.array([1.1, 1.9]), inh_i=np.array([], dtype=int), inh_t=np.array([]))
    with pytest.raises(AssertionError, match='spike trajectory differs'):
        verify_trajectory(original, restored, 1.)


def test_linux_confirmation_rejects_selection_and_raw_evidence_corruption(tmp_path):
    import copy
    import json
    from litwin_kumar_linux_performance_review import aggregate_samples, review

    def timing_report(suite, repeats, rust_workers):
        samples = []
        for backend, workers in [('rust', rust_workers), ('cpp', [4])]:
            for n in workers:
                for k in range(repeats):
                    samples.append(dict(backend=backend, threads=n, repeat=k,
                        simulation_seconds=(1 if backend == 'rust' and n == 4 else 2)+k*.01,
                        summary=dict(neuron_count=5000, threads=n, phase_profile={'enabled': False})))
        run = dict(configuration={'scale': 1}, recording_window_steps=None,
                   timed_interval={'synaptic_events': 100}, synaptic_events_from_full_history=1100)
        return dict(suite=suite, profiled=False, warmup_s=10,
                    timed_duration_s=1 if suite == 'performance' else 11,
                    recording='identical full-history SpikeMonitor and StateMonitor',
                    rust_worker_results_byte_exact=True, cpp_fixed_worker_repeats_byte_exact=True,
                    thread_counts={'rust': rust_workers, 'cpp': [4]}, samples=samples,
                    builds={'rust-1-build': {'run': run}, 'cpp-4-build': {'run': run}},
                    cpp_compile_profile='brian2-default', build_run_event_count_ratio=1.)

    contract = dict(cpp_profiles=['strict', 'native-strict', 'brian2-default'],
                    cpp_workers=[1, 2, 4, 8, 16], cpp_placements=['spread', 'close', 'matched-rust'],
                    cpp_selection_repeats=2, rust_selection_workers=[1, 4, 8])
    candidates, samples = [], []
    for p in contract['cpp_profiles']:
        for n in contract['cpp_workers']:
            for binding in contract['cpp_placements']:
                seconds = 2. if (p, n, binding) == ('brian2-default', 4, 'close') else 3.
                candidate = dict(profile=p, threads=n, binding=binding, n=2)
                for suite in ['performance', 'whole_run']:
                    candidate[suite] = dict(median_seconds=seconds, min_seconds=seconds, max_seconds=seconds)
                candidates.append(candidate)
                for k in range(2):
                    samples.append(dict(profile=p, threads=n, binding=binding, repeat=k,
                                        performance=seconds, whole_run=seconds))
    cpp = next(c for c in candidates if c['profile'] == 'brian2-default' and c['threads'] == 4 and c['binding'] == 'close')
    files = {'contract.json': contract, 'cpp-placement-samples.json': samples,
             'cpp-placement-candidates.json': {'candidates': candidates, 'all_outputs_byte_exact': True}}
    gates = {}
    for suite in ['performance', 'whole_run']:
        pilot = timing_report(suite, 2, [1, 4, 8])
        rust = next(r for r in aggregate_samples(pilot, suite, 2) if r['backend'] == 'rust' and r['threads'] == 4)
        confirmation = timing_report(suite, 5, [1, 4])
        rows = aggregate_samples(confirmation, suite, 5)
        gate = {b: next(r for r in rows if r['backend'] == b and r['threads'] == 4) for b in ['rust', 'cpp']}
        gate.update(speedup=gate['cpp']['median_seconds']/gate['rust']['median_seconds'], passed=True, cpp_selection=cpp)
        gates[suite] = gate
        files[suite+'-rust-selection/report.json'] = pilot
        files[suite+'-fixed-selection.json'] = {'rust': rust, 'cpp': cpp}
        files[suite+'-confirmation/report.json'] = confirmation
    files['linux-final-gates.json'] = gates
    files['completed.json'] = {'completed': True, 'both_scopes_passed': True}

    def write(name, data):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    for name, data in files.items():
        write(name, data)
    _, result = review(tmp_path)
    assert result['both_scopes_passed']
    assert result['gates']['performance']['rust']['threads'] == 4  # No fixed Rust8 assumption.
    bad = copy.deepcopy(files['performance-fixed-selection.json'])
    bad['cpp'] = candidates[0]  # Replacing the fastest default/close control is rejected.
    write('performance-fixed-selection.json', bad)
    with pytest.raises(ValueError, match='fastest measured'):
        review(tmp_path)
    write('performance-fixed-selection.json', files['performance-fixed-selection.json'])
    name = 'whole_run-confirmation/report.json'
    for mutation, error in [
            (lambda r: r['samples'].pop(), 'complete unique'),
            (lambda r: r['samples'][5].update(simulation_seconds=100.), 'recorded aggregate'),
            (lambda r: r['samples'][0]['summary']['phase_profile'].update(enabled=True), 'worker evidence'),
            (lambda r: r.update(timed_duration_s=1), 'timing and recording'),
            (lambda r: r.update(build_run_event_count_ratio=1.1), 'event workload')]:
        bad = copy.deepcopy(files[name])
        mutation(bad)
        write(name, bad)
        with pytest.raises(ValueError, match=error):
            review(tmp_path)
        write(name, files[name])
