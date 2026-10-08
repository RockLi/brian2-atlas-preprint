import copy
import pytest
import flywire_mnist_regular_cut as cut
from brian2_rust.protocol import attach_protocol


def models():
    intact = {'schema': 'b2ir-v1', 'definition': {'populations': [{'name': 'flywire_neurons'}]},
              'run': {'duration': 350}, 'instance': {'synapses': [{'weight': 1}],
              'populations': [{'initial_state': {'transmission': [cut.r.bits(1.)]*3, 'v': [-52]*3}}]}}
    attach_protocol(intact)
    changed = copy.deepcopy(intact)
    changed['instance']['populations'][0]['initial_state']['transmission'][1] = cut.r.bits(0.)
    attach_protocol(changed)
    return intact, changed


def test_permits_only_intended_transmission_cut():
    intact, changed = models()
    cut.assert_only_alpn_cut(intact, changed, [1])
    with pytest.raises(ValueError, match='more than'):
        cut.assert_only_alpn_cut(intact, intact, [1])


@pytest.mark.parametrize('mutation', ['weight', 'other_cell', 'initial_voltage', 'duration'])
def test_rejects_collateral_model_changes(mutation):
    intact, changed = models()
    if mutation == 'weight':
        changed['instance']['synapses'][0]['weight'] = 2
    elif mutation == 'other_cell':
        changed['instance']['populations'][0]['initial_state']['transmission'][0] = cut.r.bits(0.)
    elif mutation == 'initial_voltage':
        changed['instance']['populations'][0]['initial_state']['v'][0] = -51
    else:
        changed['run']['duration'] = 351
    attach_protocol(changed)
    with pytest.raises(ValueError, match='more than'):
        cut.assert_only_alpn_cut(intact, changed, [1])
