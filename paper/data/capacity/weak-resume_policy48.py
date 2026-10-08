"""Final-layout admission continuation; never retry a started numerical run."""
from pathlib import Path
import json
import sys
import time
import traceback
import control_policy48 as c
import overlay_inputs
import resume_observer as prior

HERE=Path(__file__).resolve().parent
STATE={}


def read(name):return json.loads((HERE/name).read_text())


def save():
    (HERE/'sweep-state.json').write_text(json.dumps(STATE,indent=2)+'\n')


def action(name,case,call):
    STATE.update(action=name,case=case,hosts=30,last_update_epoch=time.time());save()
    print(json.dumps({k:STATE[k] for k in ['action','case','hosts']}),flush=True)
    call()


def wait(stage,seconds):
    deadline=time.monotonic()+seconds
    while True:
        action('status',stage,lambda:c.status(stage))
        data=read(stage+'-status.json');guard=data['guard']
        if guard and 'returncode' in guard:
            assert guard['returncode']==0,data['log'][-3000:]
            return
        assert time.monotonic()<deadline,stage+' observer deadline'
        time.sleep(35)


def main():
    c.configure(30)
    old=read('sweep-state.json')
    assert old['status']=='stopped_on_failure' and old['action']=='launch'
    assert old['case']=='weak1pct-h30-v2' and old['accepted_hosts']==[3,6,12,24]
    assert not (HERE/'weak1pct-h30-v2-launch').exists(), 'A started run requires separate diagnosis; this continuation is admission-only.'
    assert read('prepare-weak1pct-h30-v2-status.json')['guard']['returncode']==0
    assert read('weak1pct-h30-v2-deployment-audit.json')['passed']
    assert read('engine-source-readback.json')['passed']
    assert read('policy48-stage.json')['passed']
    (HERE/'sweep-state-policy48-admission-stop.json').write_text(json.dumps(old,indent=2)+'\n')
    STATE.update(old,status='running',resumed_after='Final-layout positive 48 GiB host-memory admission reserve; numerical inputs/binary/engine and eight-core physical placement reused unchanged.')
    STATE.pop('error',None)
    STATE['case_map']['30']='weak1pct-h30-v3';save()
    policy=read('protocol-revision-v3.json');policy.update(status='activated',activation_epoch=time.time())
    (HERE/'protocol-revision-v3.json').write_text(json.dumps(policy,indent=2)+'\n')
    pilot=c.pilot(30);case=c.major(30)
    action('inventory',case,lambda:c.inventory(case+'-policy48-preflight',node_memory_gib=dict(zip(c.NODES,c.resource_caps(case,30),strict=True))))
    action('overlay',pilot,lambda:overlay_inputs.overlay('pilot-weak1pct-h30-v1',pilot,pilot=True))
    action('launch',pilot,lambda:c.launch(pilot,30))
    action('validate',pilot,lambda:c.validate(pilot,30))
    action('collect',pilot,lambda:c.collect(pilot))
    validation=read(pilot+'-validation.json')
    assert validation['passed'] and validation['checks']==2403 and validation['max_state_absolute_difference']==0
    action('overlay',case,lambda:overlay_inputs.overlay('weak1pct-h30-v2',case))
    action('readback',case,c.readback)
    action('launch',case,lambda:c.launch(case,30))
    action('collect',case,lambda:c.collect(case))
    for row in read(case+'-resources.json')['rows']:
        assert all(g['reserved_host_memory_bytes']==48*2**30 for g in row['guards'].values())
    action('audit',case,lambda:c.audit(case))
    wait('audit-'+case,1850)
    action('collect-audit',case,lambda:c.collect_audit(case))
    action('inventory',case,lambda:c.inventory(case+'-post'))
    prior.STATE=STATE;prior.accept(case,30)
    accepted=read(case+'-accepted.json')
    accepted.update(protocol_revision='v3',host_memory_reserve_gib=48,
                    admission_guard_sha256=policy['guard_sha256'],numerical_pilot_case=pilot,
                    input_preparation_reused_from='weak1pct-h30-v2')
    (HERE/(case+'-accepted.json')).write_text(json.dumps(accepted,indent=2)+'\n')
    STATE.update(last_accepted=accepted,status='endpoint_complete',completed_epoch=time.time());save()
    print(json.dumps({'endpoint_complete':True,'neurons':accepted['neurons'],'case':case}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if STATE:
            STATE.update(status='stopped_on_failure',error=str(exc)[-4000:],last_update_epoch=time.time());save()
        traceback.print_exc();raise
