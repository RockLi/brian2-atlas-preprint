import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('affinity', Path(__file__).resolve().parents[1]/'tools/mam_nest_rank_affinity.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def layout():
    return dict(schema='b2-mam-nest-affinity-v1', ranks_per_host=8, threads_per_rank=4,
                hosts=[dict(host='host'+str(i), rank_cpu_ids=[list(range(c, c+4))
                       for c in [0,12,24,36,48,60,72,84]]) for i in range(6)])


@pytest.mark.parametrize('rank', [0,7,8,15,16,23,24,31,32,39,40,47])
def test_rank_to_host_and_four_explicit_places(layout, rank):
    cpus, env = m.binding(layout, rank, 'host'+str(rank//8), range(192))
    assert cpus == layout['hosts'][rank//8]['rank_cpu_ids'][rank%8]
    assert env['OMP_PLACES'] == ','.join('{'+str(c)+'}' for c in cpus)
    assert env['OMP_NUM_THREADS'] == '4' and env['OMP_DYNAMIC'] == 'FALSE'
    assert env['OMP_PROC_BIND'] == 'close'


@pytest.mark.parametrize('fault', ['negative','too-large','missing-allowance','wrong-host',
    'duplicate-cpu','missing-worker','duplicate-host','wrong-thread-count','noninteger'])
def test_invalid_placement_is_rejected(layout, fault):
    rank, host, allowed = 0, 'host0', range(192)
    if fault == 'negative': rank = -1
    elif fault == 'too-large': rank = 48
    elif fault == 'missing-allowance': allowed = [0,1,2]
    elif fault == 'wrong-host': host = 'host1'
    elif fault == 'duplicate-cpu': layout['hosts'][0]['rank_cpu_ids'][1][0] = 0
    elif fault == 'missing-worker': layout['hosts'][0]['rank_cpu_ids'][0].pop()
    elif fault == 'duplicate-host': layout['hosts'][1]['host'] = 'host0'
    elif fault == 'wrong-thread-count': layout['threads_per_rank'] = 2
    elif fault == 'noninteger': layout['hosts'][0]['rank_cpu_ids'][0][0] = False
    with pytest.raises(ValueError):
        m.binding(layout, rank, host, allowed)
