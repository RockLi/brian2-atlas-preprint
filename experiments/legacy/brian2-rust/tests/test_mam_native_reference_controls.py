"""Reference terminal-control admission. No remote jobs or recordings are read."""
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import mam_collect_native_primary_resources as collector
import mam_launch_native_full_reference as reference
from mam_launch_native_primary import launch_options


@pytest.fixture
def reference_inputs(tmp_path,monkeypatch):
    from test_mam_native_primary_resources import inputs
    admission,launch,*_=inputs.__wrapped__()
    # Protocol contents are tested against actual pinned sources elsewhere;
    # this fixture isolates run selection and terminal resource admission.
    monkeypatch.setattr(reference,'protocol_gate',lambda *a:{})
    monkeypatch.setattr(collector.subprocess,'run',lambda *a,**k:pytest.fail('network before terminal admission'))
    label=reference.label_for(1730)
    admission.update(label=label,seed=1730,campaign=reference.CAMPAIGN,
        protocol_sha256=reference.PROTOCOL_SHA,source_project=collector.PROJECT,
        source_project_is_retained_primary=True)
    options=launch_options(tmp_path/'logs',label=label,seed=1730)
    options['output']=str(options['output']);admission['launch_options']=options
    directory=tmp_path/reference.CAMPAIGN/'seed1730';directory.mkdir(parents=True)
    (directory/'admission.json').write_text(json.dumps(admission))
    return tmp_path,directory,admission,launch


def test_live_reference_never_uses_terminal_primary_or_creates_output(reference_inputs):
    root,directory,_,launch=reference_inputs
    old=root/'primary-native-layout';old.mkdir()
    (old/'launch.json').write_text(json.dumps(launch))
    (old/'admission.json').write_text('{}')
    output=root/'collected'
    result=collector.run(root,output,root/'parameters.json',reference_seed=1730)
    assert not result['ready'] and not result['collection_started']
    assert not output.exists() and not (directory/'launch.json').exists()


@pytest.mark.parametrize('fault',['failed-proxy','wrong-seed','wrong-protocol','wrong-source','wrong-label'])
def test_wrong_or_failed_reference_cannot_collect(reference_inputs,fault):
    root,directory,admission,launch=reference_inputs
    if fault=='failed-proxy': launch['returncodes']['proxy-5']=1
    elif fault=='wrong-seed': admission['seed']=1729
    elif fault=='wrong-protocol': admission['protocol_sha256']='0'*64
    elif fault=='wrong-source': admission['source_project']+='/different'
    else: admission['label']=collector.LABEL
    (directory/'admission.json').write_text(json.dumps(admission))
    (directory/'launch.json').write_text(json.dumps(launch))
    output=root/'collected'
    with pytest.raises(ValueError):
        collector.run(root,output,root/'parameters.json',reference_seed=1730)
    assert not output.exists()


def test_valid_reference_terminal_gate_reaches_parameter_read(reference_inputs,monkeypatch):
    root,directory,_,launch=reference_inputs
    (directory/'launch.json').write_text(json.dumps(launch))
    absent=root/'required-parameters.json'
    reads=[];original_read=collector.read
    def observed_read(path,*args,**kwargs):
        reads.append(path)
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(collector,'read',observed_read)
    with pytest.raises(ValueError,match='regular control file required'):
        collector.run(root,root/'collected',absent,reference_seed=1730)
    assert reads[-1]==absent
    assert not (root/'collected').exists()


def test_reference_cannot_reuse_primary_failure_recovery(reference_inputs):
    root,_,_,_=reference_inputs
    with pytest.raises(ValueError,match='cannot reuse'):
        collector.run(root,root/'collected',root/'parameters.json',
                      prior_attempt=root/'prior',prior_guard=root/'guard',reference_seed=1730)


@pytest.mark.parametrize('seed',[1729,1750,1754,True])
def test_unreserved_reference_identity_refused_before_network(tmp_path,monkeypatch,seed):
    monkeypatch.setattr(collector.subprocess,'run',lambda *a,**k:pytest.fail('unexpected network'))
    with pytest.raises(ValueError):
        collector.run(tmp_path,tmp_path/'output',tmp_path/'parameters',reference_seed=seed)
    assert list(tmp_path.iterdir())==[]
