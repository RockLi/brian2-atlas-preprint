"""No neural data: run binding, unchanged FC arithmetic and budget tests."""
import ast,copy,hashlib,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_confirmation_analysis as analysis
import analyze_mam_confirmation_fc as fc
import mam_launch_confirmation_raw as raw_launch
from test_mam_confirmation_terminal import controls
from test_mam_confirmation_raw import bound
from test_mam_launch_confirmation_raw import publication


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def accepted(bound,publication):
    case,binding,digest=bound;p,g,c,kw=publication
    raw=case/'raw';raw.mkdir();(raw/'binding.json').write_bytes(binding.read_bytes())
    source=sha(Path(analysis.__file__).with_name('mam_confirmation_raw.py'))
    p.update(binding_sha256=digest,implementation_sha256=source)
    g['command']=raw_launch.application(digest);c['command']=raw_launch.command(digest)
    kw.update(binding_sha=digest,raw_source_sha=source)
    def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
    rows={'pending.json':p,'guard.json':g,'controller.json':c,
        'admission.json':dict(source_catalog={'tools/mam_confirmation_raw.py':dict(sha256=source)}),
        'attempt.json':dict(binding_sha256=digest,automatic_retry=False),
        'intent.json':dict(binding_sha256=digest,automatic_retry=False,attempts=1,total_stage_seconds=7200)}
    for name,value in rows.items():write(raw/name,value)
    report=raw_launch.publish(p,g,c,**kw)
    report['evidence_sha256']={n:sha(raw/n) for n in [*rows,'binding.json']};write(raw/'report.json',report)
    names=['identity.json','protocol.json','admission.json','launch.json','terminal/report.json','raw/report.json','raw/intent.json']
    completion=dict(schema='b2-mam-confirmation-engineering-completion-v1',case_id=analysis.CASE,replicate=1750,
        identity_sha256=analysis.IDENTITY_SHA,protocol_sha256=kw['protocol_sha'],terminal_resource_audit_passed=True,
        raw_output_audit_passed=True,scientific_acceptance=False,performance_cost_acceptance=False,
        input_sha256={n:sha(case/n) for n in names})
    write(case/'completion.json',completion)
    return case,sha(case/'completion.json')


def test_engineering_gate_replays_guarded_raw_report(accepted):
    v,r,t=analysis.accepted_raw(*accepted)
    assert v['replicate']==1750 and r['raw_output_audit_passed'] and len(t['ranks'])==32
    assert not r['scientific_acceptance'] and not t['scientific_acceptance']


@pytest.mark.parametrize('name',['raw/guard.json','raw/report.json','raw/binding.json','identity.json'])
def test_altered_completed_inputs_rejected(accepted,name):
    case,digest=accepted
    with (case/name).open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError):analysis.accepted_raw(*accepted)


def test_missing_completion_is_no_partial_acceptance(tmp_path):
    with pytest.raises(ValueError):analysis.accepted_raw(tmp_path,'0'*64)


def test_six_stages_use_new_identity_and_do_not_repeat_full_raw_audit(controls):
    v=controls[0];source=Path('/data/brick2/test-source');rows=analysis.stages(source,v)
    assert [name for name,_,_ in rows]==list(analysis.REQUIRED) and sum(cap for _,cap,_ in rows)<10800
    for name,cap,command in rows:
        assert command[0]==sys.executable and command.count('--output')==1
        assert command[-1]==v['analysis']+'/'+name and 'output-audit' not in name
        if name in ['activity','cell','correlation','series']:
            assert command[command.index('--model')+1]==v['source_artifact']+'/model.json'
            assert command[command.index('--results')+1]==v['output'] and '--bounded-memory' in command
        if name=='activity':
            assert command[command.index('--spike-tick-offset')+1]=='1' and command[command.index('--end-tick')+1]=='1005000'
        if name=='fc':assert command[command.index('--identity')+1]==str(source/'case/identity.json')


def test_fc_accepts_actual_unsigned_key_not_replicate_label(controls):
    v=controls[0];identity=dict(simulator='Rust',model_sha256=v['model_sha256'],seed=v['random_keys']['runtime_input'])
    fc.admit_confirmation(identity,100000,v)
    for key in [1750,1729,float(v['random_keys']['runtime_input'])]:
        with pytest.raises(ValueError):fc.admit_confirmation({**identity,'seed':key},100000,v)
    with pytest.raises(ValueError):fc.admit_confirmation(identity,10000,v)
    with pytest.raises(ValueError):fc.admit_confirmation({**identity,'model_sha256':'0'*64},100000,v)


def test_fc_array_validation_normalization_and_numerical_tail_unchanged():
    root=Path(fc.__file__).parent
    def tail(name):
        tree=ast.parse((root/name).read_text());body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='analyze').body
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.With))
        return [ast.dump(n) for n in body[start:]]
    assert tail('analyze_mam_paper_fc.py')==tail('analyze_mam_confirmation_fc.py')
