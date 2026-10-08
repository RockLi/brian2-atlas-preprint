import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from mam_extend_duration import extend, audit_duration_delta, bits, output_bytes


def fixture():
    return dict(schema='test', instance=dict(rng_seed=1729, populations=[dict(refractory={'period_ticks': 21})], synapses=[dict(seed=42, weight='frozen')]),
        run=dict(duration=bits(2.5), clocks=[dict(start_tick=0, steps=25000)], start=bits(0.)),
        definition=dict(clocks=[dict(dt=bits(.0001))], populations=[dict(name='frozen', steps=25000, monitor=dict(window_steps=25000, variables=[], record=[]), count=3, states=[dict(name='v', dtype='f64')], parameters=[], events=['spike'], event_monitors=[])], synapses=[dict(states=[])]))


def test_only_declared_duration_changes():
    old = fixture();original = copy.deepcopy(old);new = extend(old)
    assert old == original and audit_duration_delta(old, new)['instance_exact']
    for mutate in [lambda m:m['instance'].update(rng_seed=1),
                   lambda m:m['definition']['clocks'][0].update(dt=bits(.0002)),
                   lambda m:m['run']['clocks'][0].update(start_tick=1),
                   lambda m:m['definition']['populations'][0]['monitor'].update(window_steps=25000),
                   lambda m:m['instance']['populations'][0]['refractory'].update(period_ticks=20)]:
        bad = copy.deepcopy(new);mutate(bad)
        with pytest.raises(ValueError):audit_duration_delta(old, bad)


def test_incomplete_baseline_rejected():
    old = fixture();old['definition']['populations'][0]['monitor']['window_steps'] = 100
    with pytest.raises(ValueError, match='prefix'):extend(old)


def test_output_budget_counts_event_bytes_and_rejects_new_state():
    model = fixture()
    # Header/footer 64 + pop header 64 + count/state/refractory 75 + synapse 24.
    assert output_bytes(model, 2, 1) == dict(results_bytes=267, events_bytes=72, combined_bytes=339)
    assert output_bytes(model, 2, 1, compact_spikes=True) == dict(results_bytes=251, events_bytes=56, combined_bytes=307)
    with pytest.raises(TypeError):output_bytes(model, 2, 1, compact_spikes=1)
    bad = copy.deepcopy(model);bad['run']['clocks'][0]['start_tick']=2**32
    with pytest.raises(ValueError, match='tick'):output_bytes(bad, 2, 1, compact_spikes=True)
    model['definition']['synapses'][0]['states'] = [dict(name='w', dtype='f64')]
    with pytest.raises(ValueError, match='stateless'):output_bytes(model, 2, 1)


@pytest.mark.parametrize('seconds,steps', [(10.5,105000),(50.5,505000),(100.5,1005000)])
def test_declared_long_targets_preserve_all_frozen_inputs(seconds, steps):
    old=fixture();saved=copy.deepcopy(old);new=extend(old,seconds)
    assert old==saved and new['instance']==old['instance']
    assert new['run']['duration']==bits(seconds) and new['run']['clocks'][0]['steps']==steps
    assert new['definition']['populations'][0]['steps']==steps
    assert new['definition']['populations'][0]['monitor']['window_steps']==steps
    assert audit_duration_delta(old,new,seconds)['target_steps']==steps
    wrong=copy.deepcopy(new);wrong['definition']['populations'][0]['monitor']['window_steps']=105000-1
    with pytest.raises(ValueError,match='prefix'):audit_duration_delta(old,wrong,seconds)
    wrong=copy.deepcopy(new);wrong['instance']['synapses'][0]['weight']='changed'
    with pytest.raises(ValueError,match='frozen instance'):audit_duration_delta(old,wrong,seconds)


@pytest.mark.parametrize('seconds', [2.5, 100, 100.5001, '100.5', True, float('inf'), float('nan')])
def test_undeclared_duration_is_rejected(seconds):
    with pytest.raises(ValueError,match='declared'):extend(fixture(),seconds)
