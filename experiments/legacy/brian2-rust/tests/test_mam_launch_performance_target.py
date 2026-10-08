import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import mam_launch_performance_target as target


def test_target_preserves_tested_model_and_resources(tmp_path):
    protocol=json.loads((ROOT/'mpi-evidence/performance-protocol-v1/protocol.json').read_text())
    assert target.selected_layout(ROOT/'mpi-evidence')=='nest48x4'
    _,pilot_label,pilot=target.tuning.options_for(protocol,'nest48x4-tuning',tmp_path)
    case,label,options=target.options_for(protocol,'nest48x4',tmp_path)
    changed=pilot['application'].copy()
    changed=[x.replace(pilot_label,label) for x in changed]
    changed[changed.index('--duration-ms')+1]='100500'
    changed[changed.index('--max-spikes-per-rank')+1]='268435456'
    assert options['application']==changed
    assert {k:v for k,v in options.items() if k not in ['application','timeout']}=={k:v for k,v in pilot.items() if k not in ['application','timeout']}
    assert options['timeout']==54000 and label.endswith('-seed1729-100500ms')
    layout=json.loads((ROOT/'mpi-evidence/performance-protocol-v1/nest48x4-layout.json').read_text())
    admission=target.admission_for(protocol,case,label,layout,options,{})
    assert admission['run_purpose']=='target'
    assert admission['identity']['duration_ms']==100500
    assert admission['limits']['wall_seconds']==54000
    assert admission['limits']['max_spikes_per_rank']==268435456
    assert len(admission['placements'])==48
    from mam_collect_benchmark_terminal import host_code
    for i in range(6):assert label in host_code(admission,i,'b2mpi-123456789abc')


@pytest.mark.parametrize('key,value',[('duration_ms',2500),('wall_seconds',54001),('max_spikes_per_rank',268435457)])
def test_changed_target_budget_rejected(key,value,tmp_path):
    protocol=json.loads((ROOT/'mpi-evidence/performance-protocol-v1/protocol.json').read_text())
    next(c for c in protocol['cases'] if c['id']==target.CASE_ID)[key]=value
    with pytest.raises(ValueError,match='budget changed'):target.options_for(protocol,'nest48x4',tmp_path)
