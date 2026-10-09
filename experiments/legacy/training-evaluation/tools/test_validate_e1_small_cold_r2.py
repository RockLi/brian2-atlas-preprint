#!/usr/bin/env python3
"""Tiny stdlib synthetic checks; never opens real evaluation/data evidence."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import tempfile


def run():
    source=Path(__file__).with_name('validate_e1_small_cold_r2.py')
    ast.parse(source.read_text())
    n=runpy.run_path(str(source),run_name='cold_synthetic_only')
    result=n['self_test']();tests=result['tests']
    with tempfile.TemporaryDirectory(prefix='cold-validator-tiny-') as tmp:
        root=Path(tmp);run=root/'evidence/cold';run.mkdir(parents=True)
        records=[dict(**v,status='not_launched_coordinator_failure',resource_qualification=False,strict_ranking_eligible=False) for v in n['order']()]
        term=dict(status='coordinator_failed',finite_slots=35,records=records,completed_slots=0,
                  supervisor_completed=False,spent_seconds={},error='synthetic preflight rejection')
        def dump(path,value):path.write_text(json.dumps(value))
        dump(run/'progress.json',records);dump(run/'terminal.json',term)
        value=n['validate'](root,run,'report-only')
        tests['preflight_missing_freeze_is_not_corruption']=not value['errors'] and value['status']=='consistent_non_execution_or_interrupted' and len(value['rows'])==35
        (run/'terminal.json').unlink();(run/'progress.json').unlink()
        phase=root/'phase.json'
        outer=dict(termination_reason='timeout',elapsed_s=13000.01,timeout_s=13000,
            command=['python',str(root/n['MEASURE']),'--allow-run','--output',str(run)],remaining_owned_processes=[])
        dump(phase,outer)
        value=n['validate'](root,run,'report-only',phase)
        tests['outer_kill_before_progress_retains35']=not value['errors'] and len(value['rows'])==35 and all(x['effective_status']=='not_launched_phase_timeout' for x in value['rows'])
        (run/'terminal.json').write_text('{')
        value=n['validate'](root,run,'report-only',phase)
        tests['outer_kill_partial_terminal_not_corruption']=not value['errors'] and len(value['rows'])==35
        (run/'terminal.json').unlink()
        dump(phase,{**outer,'termination_reason':'exited'})
        value=n['validate'](root,run,'report-only',phase)
        tests['outer_normal_exit_over_cap_still_timeout']=not value['errors'] and value['denominator']['phase_interrupted'] is True
        dump(run/'terminal.json',term);dump(run/'progress.json',records[:-1])
        value=n['validate'](root,run,'report-only')
        tests['final_progress_mismatch_rejected']=bool(value['errors'])
        tests['report_retains35']=n['markdown'](value).count('| not_launched_coordinator_failure |')==35
        swapped=[records[1],records[0],*records[2:]]
        dump(run/'terminal.json',{**term,'records':swapped});dump(run/'progress.json',swapped)
        value=n['validate'](root,run,'report-only')
        tests['rotated_slot_order_enforced']=bool(value['errors'])
    slot=dict(view='snn-layerwise',environment='cpu',engine='snntorch_fp64',compiled=False,seed=11)
    reference=dict(array_sha256='a'*64,raw_loss_trajectory=[.2]*60,resource_qualification=True)
    preflight=dict(environments=dict(cpu=dict(python='fake-python',historical_version_binding=dict(torch='fake-torch'))))
    child=dict(engine='snntorch_fp64',seed=11,compiled=False,array_sha256='a'*64,
        launcher_start_ns=1000000000,first_update_complete_ns=2000000000,process_to_first_update_s=1.,
        clock='same-host time.monotonic_ns across coordinator and fresh worker processes',
        reference_loss=.2,loss=.2,loss_finite=True,historical_runtime_versions=dict(torch='fake-torch'),
        current_python='fake-python',implementation=dict(torch='fake-torch'),runtime_version_match=True,
        resource_qualification=True,strict_ranking_eligible=True,status='completed')
    args={'--started-ns':'1000000000'}
    def worker(value):
        a=n['Audit']();matched=n['worker_check'](value,args,slot,reference,preflight,a);return matched,a.errors
    matched,errors=worker(child);tests['worker_clock_loss_versions_pass']=matched and not errors
    matched,errors=worker({**child,'loss':.4,'status':'first_loss_mismatch'})
    tests['scientific_loss_mismatch_is_not_corruption']=not matched and not errors
    matched,errors=worker({**child,'current_python':'changed','runtime_version_match':False,'status':'runtime_version_mismatch'})
    tests['scientific_runtime_mismatch_is_not_corruption']=not matched and not errors
    matched,errors=worker({**child,'loss':None,'loss_finite':False,'status':'numerical_divergence'})
    tests['scientific_nonfinite_is_not_corruption']=not matched and not errors
    _,errors=worker({**child,'process_to_first_update_s':.5})
    tests['worker_clock_arithmetic_tamper_rejected']=bool(errors)
    _,errors=worker({**child,'launcher_start_ns':True})
    tests['bool_clock_endpoint_rejected']=bool(errors)
    _,errors=worker({**child,'array_sha256':'b'*64})
    tests['worker_array_identity_mismatch_rejected']=bool(errors)
    refusal=dict(status='budget_rejected',failure=dict(status='budget_rejected',phase='admission'))
    a=n['Audit']();matched=n['worker_check'](refusal,args,{**slot,'engine':'atlas'},reference,preflight,a)
    tests['atlas_style_software_refusal_without_clock_retained']=not matched and not a.errors
    result.update(status='passed' if all(tests.values()) else 'failed',ast='passed',
        validator_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        synthetic_files_temporary_and_removed=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    result=run()
    if args.output:
        with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
