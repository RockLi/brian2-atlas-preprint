import copy
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'validation'), str(ROOT/'experiments'), str(ROOT/'python')]
import flywire_mnist_round3 as r
from flywire_mnist.config import Config


def test_regular_encoding_count_bounds_order_and_id_independence():
    image = np.arange(784).reshape(28,28).astype(np.uint8); config = Config(input_rate_hz=240.)
    indices, ticks = r.regular_encode(image, 1, config)
    expected = np.floor(image.reshape(-1)/255.*48+.5).astype(int)
    np.testing.assert_array_equal(np.bincount(indices, minlength=784), expected)
    assert np.all((ticks >= config.edges[0]) & (ticks < config.edges[-1]))
    pairs = np.c_[ticks, indices]
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    np.testing.assert_array_equal(np.lexsort((indices,ticks)), np.arange(len(ticks)))
    again = r.regular_encode(image, 999, config)
    np.testing.assert_array_equal(indices, again[0]); np.testing.assert_array_equal(ticks, again[1])
    other_i, other_t = r.regular_encode(image, 1, replace(config, seed=1783))
    np.testing.assert_array_equal(np.bincount(other_i, minlength=784), expected)
    assert not np.array_equal(ticks, other_t)
    assert len(r.regular_encode(np.zeros((28,28), np.uint8), 0, config)[0]) == 0


def test_candidate_charge_budget_and_no_internal_cut():
    cases = r.candidates()
    assert len(cases) == 5
    assert [c['rate']*c['gain'] for c in cases] == [1920.,1920.,1920.,1920.,3840.]
    assert all(not c['cut_alpn'] and c['mapping']=='excitatory_kc' for c in cases)


def test_fixed_internal_guard_accepts_only_external_pixel_synapses():
    base = dict(definition=dict(synapses=[dict(name='recurrent'), dict(name='pixel_connections')]),
                run=dict(ticks=3500), instance=dict(populations=[dict(v=[-52],ge=[.1])],
                                                   synapses=[dict(weight=[.3]), dict(weight=[1.])]))
    model = copy.deepcopy(base); model['instance']['synapses'][1]['weight'] = [32.]
    r.assert_fixed_internal(base, model)
    for change in ['weight','state','time','definition']:
        invalid = copy.deepcopy(model)
        if change == 'weight': invalid['instance']['synapses'][0]['weight'] = [.6]
        elif change == 'state': invalid['instance']['populations'][0]['ge'] = [.2]
        elif change == 'time': invalid['run']['ticks'] = 4000
        else: invalid['definition']['synapses'][0]['name'] = 'other'
        with pytest.raises(ValueError, match='changed'):
            r.assert_fixed_internal(base, invalid)
