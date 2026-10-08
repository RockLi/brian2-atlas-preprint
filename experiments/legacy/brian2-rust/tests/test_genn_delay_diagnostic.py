"""Protect the diagnostic's attribution and original numerical evidence."""
import importlib
from pathlib import Path
import json
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'examples'))
    return (importlib.import_module('genn_postsynaptic_barrier'),
            importlib.import_module('modal_genn_delay_diagnostic'))


def kernel(count=16):
    groups=[]
    for i in range(count):
        condition=f'id < 4096' if i==0 else f'id >= {i*4096} && id < {(i+1)*4096}'
        groups.append('if('+condition+') {\n'
            'const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 64) + 1 : 64;\n'
            '    if (threadIdx.x < numSpikesInBlock) { shSpk[threadIdx.x] = spk; }\n'
            '    __syncthreads();\n}\n')
    return ('UNCHANGED_PRESYNAPTIC\nextern "C" __global__ void updatePostsynapticKernel(float t) {\n'
        'const unsigned int id = 64 * blockIdx.x + threadIdx.x;\n'+''.join(groups)+
        '}\nvoid updateSynapses(float t) { UNCHANGED_LAUNCH; }')


@pytest.mark.parametrize('groups',[1,4,8,16])
def test_patch_only_adds_group_uniform_barriers(modules,groups):
    patch,_=modules
    before=kernel(groups);after=patch.insert_barriers(before,groups=groups)
    inserted='    __syncthreads(); // diagnostic: finish previous shared-buffer readers\n'
    assert after.count(inserted)==groups
    assert after.replace(inserted,'')==before
    with pytest.raises(ValueError):patch.insert_barriers(after,groups=groups)
    with pytest.raises(ValueError):patch.insert_barriers(before,groups=8 if groups!=8 else 16)


@pytest.mark.parametrize('groups',[0,17,True,1.5,None])
def test_invalid_group_declaration_rejected(modules,groups):
    patch,_=modules
    with pytest.raises(ValueError):patch.insert_barriers(kernel(),groups=groups)


@pytest.mark.parametrize('change',[lambda s:s.replace('id < 4096','id < 4095'),
    lambda s:s.replace('id >= 4096 && id < 8192','id >= 8192 && id < 12288'),
    lambda s:s.replace('__syncthreads();',''),lambda s:s.replace('updatePostsynapticKernel','unknown')])
def test_unrecognized_or_nonuniform_source_is_rejected(modules,change):
    patch,_=modules
    with pytest.raises(ValueError):patch.insert_barriers(change(kernel()))


def test_failed_numeric_verdict_survives_successful_diagnostic(modules,tmp_path):
    _,runner=modules
    raw=json.dumps({'completed':True,'trials':{'stock':{'checks':{'passed':False}}}}).encode()
    runner.save_result(dict(passed=True,stdout='',payloads={'stock/report.json':raw}),tmp_path,{'function_call_id':'existing'})
    assert (tmp_path/'stock/report.json').read_bytes()==raw
    wrapper=json.loads((tmp_path/'report.json').read_text())
    assert wrapper['artifact_status']=='complete'
    assert wrapper['result_artifacts']['stock/report.json']['bytes']==len(raw)


@pytest.mark.parametrize('name',['report.json','call.json','source-hashes.json','../escape','/absolute'])
def test_payload_cannot_overwrite_provenance(modules,tmp_path,name):
    _,runner=modules
    with pytest.raises(ValueError):runner.save_result(dict(passed=True,stdout='',payloads={name:b'x'}),tmp_path,{})
    assert not list(tmp_path.iterdir())


def test_observation_phase_accounts_for_final_synapse_flush(modules):
    import numpy as np
    from genn_observed_reference import observed_reference
    result=observed_reference(16,1,1,np.array([0,15]),np.array([14,15]),
        drive=.125,delay_span=1,post_delay=0,topology_kind='ring',topology_seed=0)
    np.testing.assert_array_equal(result['previous_spike'],[[0,1],[0,0]])
    np.testing.assert_array_equal(result['v_after_reset'],[[.12109375,0],[.13671875,0]])
    np.testing.assert_array_equal(result['Apre'],[[0,0],[0,.0078125]])
    np.testing.assert_array_equal(result['Apost'],[[0,0],[-.00390625,0]])


@pytest.mark.parametrize('delays,post',[(1,0),(4,3),(16,16)])
def test_projected_final_state_matches_independent_full_oracle(modules,delays,post):
    import numpy as np
    from genn_observed_reference import observed_reference
    from gpu_stdp_compare import oracle
    options=dict(drive=.125,delay_span=delays,post_delay=post,topology_kind='ring',topology_seed=0)
    expected=oracle(32,4,48,**options)
    result=observed_reference(32,4,48,np.arange(32),np.arange(128),**options)
    for key in ['w','Apre','Apost','lastupdate']:
        np.testing.assert_array_equal(result[key][-1],expected[key])
    np.testing.assert_array_equal(result['v_after_reset'][-1],expected['v'])
    np.testing.assert_array_equal(result['v_start'][:-1,[0,31]],expected['trace'])
