import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from mam_nest_rank_affinity import binding as old_binding
from mam_benchmark_rank_affinity import binding


def original():
    return json.loads((ROOT/'mpi-evidence/primary-native-layout/layout.json').read_text())


def test_baseline_bindings_and_environments_are_unchanged():
    old=original();new=copy.deepcopy(old);new['schema']='b2-mam-nest-affinity-v2'
    for rank in range(48):
        host=old['hosts'][rank//8]
        allowed={c for group in host['rank_cpu_ids'] for c in group}
        assert binding(new,rank,host['host'],allowed)==old_binding(old,rank,host['host'],allowed)


def test_two_thread_candidate_covers_same_physical_cpu_set_without_overlap():
    old=original();new=copy.deepcopy(old)
    new.update(schema='b2-mam-nest-affinity-v2',ranks_per_host=16,threads_per_rank=2)
    for host in new['hosts']:
        cpus=[c for group in host['rank_cpu_ids'] for c in group]
        host['rank_cpu_ids']=[cpus[i:i+2] for i in range(0,32,2)]
    for index,host in enumerate(new['hosts']):
        allowed={c for group in old['hosts'][index]['rank_cpu_ids'] for c in group};seen=[]
        for local in range(16):
            cpus,env=binding(new,index*16+local,host['host'],allowed);seen+=cpus
            assert len(cpus)==2 and env['OMP_NUM_THREADS']=='2' and env['OMP_DYNAMIC']=='FALSE'
        assert len(seen)==len(set(seen))==32 and set(seen)==allowed
    assert 6*16*2==6*8*4==192


@pytest.mark.parametrize('mutate', [
    lambda d:d.update(ranks_per_host=32,threads_per_rank=1),
    lambda d:d['hosts'][0]['rank_cpu_ids'][1].__setitem__(0,d['hosts'][0]['rank_cpu_ids'][0][0]),
    lambda d:d['hosts'][0]['rank_cpu_ids'][0].append(99),
])
def test_unregistered_or_overlapping_layout_rejected(mutate):
    layout=original();layout['schema']='b2-mam-nest-affinity-v2';mutate(layout)
    with pytest.raises(ValueError):binding(layout,0,layout['hosts'][0]['host'],set(range(4096)))
