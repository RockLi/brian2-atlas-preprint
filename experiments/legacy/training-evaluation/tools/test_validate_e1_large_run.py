"""Synthetic stdlib validator tests only; never launch/import any model.

All fake evidence lives in TemporaryDirectory and is removed at test exit.
These tests verify bookkeeping and refusal behavior, not engine qualification.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import validate_e1_large_run as v


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def admission(byte_count=60_000_000):
    neurons=sum(v.SIZES[1:]);parameters=sum(a*b for a,b in zip(v.SIZES[:-1],v.SIZES[1:]))
    components=dict(tapes_and_returned_spikes=32*128*neurons*24,live_state=32*neurons*32,
                    logits=32*20*16,parameter_gradient_optimizer=parameters*48,topology=0,mpi_workspace=0,
                    inputs=32*128*(512*8+48)+32*8)
    rejected=byte_count>64*1024**2
    return dict(status='budget_rejected' if rejected else 'admitted',reasons=['request_json_exceeds_64_mib'] if rejected else [],
                request=dict(bytes=byte_count,sha256='0'*64),input_limit_bytes=64*1024**2,hard_max_tape_bytes=1024**3,
                plan_max_tape_bytes=1024**3,components=components,exact_native_tape_bytes=sum(components.values()),
                exact_initial_budget_bytes=32*neurons*64+parameters*48)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic-e1-large-validator-',dir='/private/tmp')
        self.root=Path(self.temp.name);self.run=self.root/'run';self.run.mkdir()
        self.folder=self.run/'worker';self.folder.mkdir()

    def tearDown(self):self.temp.cleanup()

    def test_exact_admission_guards(self):
        for size in [60_000_000,64*1024**2,64*1024**2+1]:
            issues=[];v.admission_check(admission(size),issues);self.assertEqual(issues,[])
        value=admission();value['exact_native_tape_bytes']-=1
        issues=[];v.admission_check(value,issues);self.assertTrue(issues)

    def test_initial_and_later_adam_rejection_distinct(self):
        for later in [False,True]:
            history=[]
            if later:history.append(dict(phase='qualification',index=1,optimizer_step_before=0,admission=admission()))
            history.append(dict(phase='qualification',index=2 if later else 1,optimizer_step_before=1 if later else 0,admission=admission(80_000_000)))
            jsonl(self.folder/'atlas-admissions.jsonl',history)
            row=dict(view='atlas',seed=11,effective_status='budget_rejected',errors=[])
            v.raw_and_resources(self.run,self.folder,{}, {},row)
            self.assertEqual(row['errors'],[])
            self.assertEqual(row['admission_outcome'],'later_optimizer_state_budget_rejected' if later else 'initial_request_budget_rejected')
            self.assertNotIn('seed_median_s',row)

    def test_complete_vs_partial_samples(self):
        for n,status in [(60,'completed'),(59,'completed'),(59,'timeout')]:
            samples=[dict(index=i,phase='warmup' if i<10 else 'measured',elapsed_s=.01+i/100000,loss=2.) for i in range(n)]
            jsonl(self.folder/'raw-steps.jsonl',samples)
            row=dict(view='snn-compile',seed=11,effective_status=status,errors=[])
            worker=dict(measured_seconds=[r['elapsed_s'] for r in samples[10:]])
            if n==60:worker['median_s']=v.statistics.median(worker['measured_seconds'])
            v.raw_and_resources(self.run,self.folder,{},worker,row)
            self.assertEqual(row['completed_measured_steps'],n-10)
            self.assertEqual(bool(row['errors']),n==59 and status=='completed')
            self.assertEqual('seed_median_s' in row,n==60)
            self.assertEqual('partial_measured_median_diagnostic_only_s' in row,n==59)

    def test_resource_peak_is_contemporaneous(self):
        path=self.run/'spyx-seed-11-resources.jsonl'
        rows=[dict(elapsed_s=.1,aggregate_rss=100,root_threads=20,rss_by_pid={'1':60,'2':40}),
              dict(elapsed_s=.2,aggregate_rss=120,root_threads=21,rss_by_pid={'1':100,'2':20})]
        jsonl(path,rows);row=dict(view='spyx',seed=11,effective_status='timeout',errors=[])
        external=dict(command=['synthetic'],peak_contemporaneous_job_rss=120,max_observed_root_threads=21)
        v.raw_and_resources(self.run,self.folder,external,{},row)
        self.assertEqual(row['errors'],[]);self.assertEqual(row['sampled_job_peak_rss_bytes'],120)
        external['peak_contemporaneous_job_rss']=140
        row=dict(view='spyx',seed=11,effective_status='timeout',errors=[])
        v.raw_and_resources(self.run,self.folder,external,{},row);self.assertTrue(row['errors'])

    def no_execution_fixture(self):
        rule=dict(sizes=v.SIZES,B=32,T=128,seeds=list(v.SEEDS))
        order=[dict(seed=s,view=[name,'synthetic',engine,[]]) for s in v.SEEDS for name,(engine,_,_) in v.VIEWS.items()]
        runtime=self.root/'runtime/b2-train';runtime.parent.mkdir();runtime.write_bytes(b'SYNTHETIC NONEXECUTABLE FIXTURE')
        snapshot=self.root/'sources/snapshot-manifest.json';write(snapshot,dict(source_hashes={}))
        hardware=self.root/'environment/hardware.json';write(hardware,dict(synthetic=True))
        freeze=dict(rule=rule,order=order,warmup_steps=10,measured_steps=50,per_seed_budget_s=360,per_view_budget_s=1800,
                    no_microbatch=True,no_batch_or_time_reduction=True,scripts={},gates={},environment_locks={},
                    runtime_sha256=v.sha(runtime),source_manifest_sha256=v.sha(snapshot),hardware_sha256=v.sha(hardware))
        write(self.run/'freeze.json',freeze)
        terminal=dict(supervisor_completed=True,preparation=dict(exit_code=2,termination_reason='exited'),
                      records=[dict(view=p['view'][0],seed=p['seed'],status='not_executed',reason='synthetic preparation failed') for p in order],
                      spent_seconds={name:0. for name in v.VIEWS})
        write(self.run/'terminal.json',terminal)
        return freeze,terminal

    def test_all_35_failure_slots_remain_and_jax_never_strict(self):
        self.no_execution_fixture()
        for mode in ['report-only','full']:
            result=v.validate(self.root,self.run,mode)
            self.assertEqual(result['errors'],[]);self.assertEqual(len(result['rows']),35)
            self.assertFalse(any(r['strict_ranking_eligible'] for r in result['rows']))
            for view in v.JAX:
                self.assertFalse(result['views'][view]['resource_qualification'])
                self.assertTrue(all(r['ratio_atlas_over_view'] is None for r in result['views'][view]['paired_seed_ratios']))

    def test_global_identity_failure_disables_all_ranking(self):
        self.no_execution_fixture();(self.root/'runtime/b2-train').write_bytes(b'CHANGED SYNTHETIC FIXTURE')
        result=v.validate(self.root,self.run,'full')
        self.assertEqual(result['status'],'evidence_invalid')
        self.assertTrue(any('runtime' in e for e in result['errors']))
        self.assertIn('拒绝生成已验证性能排名',v.report(result))
        self.assertFalse(any(r['strict_ranking_eligible'] for r in result['rows']))

    def test_aggregate_prelaunch_timeout_cannot_be_invented(self):
        _,terminal=self.no_execution_fixture();terminal['records'][0]['status']='timeout'
        write(self.run/'terminal.json',terminal)
        result=v.validate(self.root,self.run,'report-only')
        self.assertTrue(any('positive per-view budget' in e for e in result['errors']))

    def complete_fixture(self):
        """Tiny synthetic metadata for every slot; no real numerical artifacts.

        The validator checks artifact identity rather than rerunning numerical
        qualification. Opaque stand-ins explicitly exercise that bookkeeping
        boundary; they are never published as real engine evidence.
        """
        freeze,terminal=self.no_execution_fixture()
        generator=self.root/'tools/generate_dense_large.py';generator.parent.mkdir();generator.write_text('# SYNTHETIC, not a model generator\n')
        freeze['scripts']={'tools/generate_dense_large.py':v.sha(generator)}
        api=self.root/'snapshot/brian2-rust/python/brian2_rust/training.py';api.parent.mkdir(parents=True);api.write_text('# SYNTHETIC NONEXECUTABLE API\n')
        snapshot=self.root/'sources/snapshot-manifest.json';write(snapshot,{'source_hashes':{'brian2-rust/python/brian2_rust/training.py':v.sha(api)}})
        freeze['source_manifest_sha256']=v.sha(snapshot)
        manifest=dict(rule=freeze['rule'],generator_sha256=v.sha(generator),files={})
        for seed in v.SEEDS:
            case=self.root/f'fixtures/e1-large/seed-{seed}.json';write(case,{'synthetic_seed':seed})
            manifest['files'][case.name]=dict(sha256=v.sha(case),bytes=case.stat().st_size)
        mpath=self.root/'fixtures/e1-large/manifest.json';write(mpath,manifest)
        write(self.run/'fixture-identity.json',dict(manifest_sha256=v.sha(mpath)))
        qnames=['base_negative_count_input','batch_duplicate','initial_threshold_boundary','single_sample_no_carry']
        for view,(engine,compiled,_) in v.VIEWS.items():
            qpath=self.root/f'q0-{view}/report.json'
            q=dict(engine=engine,dense_qualification_status='passed',identities=freeze['scripts'],
                   cases={name:dict(passed=True,checks={'synthetic':dict(passed=True)}) for name in qnames})
            write(qpath,q)
            for name in qnames:
                write(qpath.parent/(name+'.json'),dict(raw=dict(engine=engine,compile=compiled,runner_sha256=freeze['runtime_sha256'],training_api_sha256=v.sha(api))))
            freeze['gates'][view]=dict(qualified=True,path=str(qpath),report_sha256=v.sha(qpath))
        write(self.run/'freeze.json',freeze)
        terminal.update(preparation=dict(exit_code=0,termination_reason='exited'),records=[],spent_seconds={name:0. for name in v.VIEWS})
        for entry in freeze['order']:
            view,seed=entry['view'][0],entry['seed'];engine,compiled,layerwise=v.VIEWS[view]
            folder=self.run/f'{view}-seed-{seed}';folder.mkdir();atlas=view=='atlas'
            def artifact(name):
                path=folder/name;path.write_bytes(b'SYNTHETIC OPAQUE ARTIFACT: identity test only')
                return dict(path=name,sha256=v.sha(path),bytes=path.stat().st_size,arrays={'synthetic':dict(shape=[],dtype='float64')})
            steps=[]
            for index in range(1,4):
                names=['loss']+[f'{key}_{i}' for key in ['weights','first','second'] for i in range(3)]
                if atlas:names+=['logits','spikes','initial_vjp','final_state','optimizer_counter','native_tape_count']+[f'gradient_{i}' for i in range(3)]
                meta=dict(qualified=True,checks={name:dict(passed=True) for name in names},artifact=artifact(f'qualification-step-{index}.npz'))
                if atlas:meta.update(phase='qualification',index=index,optimizer_step_before=index-1,admission=admission(),status='executed',public_api_ns=10_000_000,tape_count_matches_runtime=True)
                else:meta.update(step=index,actual_public_s=.01)
                write(folder/f'qualification-step-{index}.json',meta);steps.append(meta)
            latency=.01+list(v.VIEWS).index(view)/1000+seed/1_000_000
            raw=[]
            for index in range(60):
                meta=dict(index=index,phase='warmup' if index<10 else 'measured',loss=2.-index/100)
                if atlas:meta.update(status='executed',admission=admission(),public_api_ns=latency*1e9,optimizer_step_before=index,optimizer_step=index+1,tape_count_matches_runtime=True)
                else:meta['elapsed_s']=latency
                raw.append(meta)
            jsonl(folder/'raw-steps.jsonl',raw)
            if atlas:
                admissions=[dict(phase='qualification',index=i,optimizer_step_before=i-1,admission=admission()) for i in range(1,4)]
                admissions += [dict(phase=r['phase'],index=r['index'],optimizer_step_before=r['index'],admission=admission()) for r in raw]
                jsonl(folder/'atlas-admissions.jsonl',admissions)
            worker=dict(engine=engine,seed=seed,compile=compiled,layerwise=layerwise,status='completed',performance_run=True,
                        measured_seconds=[latency]*50,median_s=latency,warmup_steps=10,measured_steps=50,rule=freeze['rule'],
                        no_microbatch=True,identities=freeze['scripts'],identities_unchanged=True,wall_s=19.,
                        arrays_sha256=manifest['files'][f'seed-{seed}.json']['sha256'],fixture_manifest_sha256=v.sha(mpath),
                        qualification_steps=steps,qualification_status='passed_three_actual_public_Adam_updates' if atlas else 'passed_three_actual_public_Adam_updates_and_full_diagnostics')
            if atlas:worker.update(runtime_sha256=freeze['runtime_sha256'],constructor_ns=100000)
            else:
                checks={name:dict(passed=True) for name in ['loss','logits','states','spikes','initial_vjp']+[f'gradient_{i}' for i in range(3)]}
                write(folder/'diagnostics.json',dict(checks=checks,artifact=artifact('first-step-diagnostics.npz')))
                worker.update(diagnostic_checks=checks,cold_construction_and_first_update_s=.02)
            write(folder/'result.json',worker)
            command=['python','synthetic_worker','--engine',engine,'--seed',str(seed)]
            if compiled:command.append('--compile')
            if layerwise:command.append('--layerwise')
            external=dict(view=view,seed=seed,status='completed',command=command,elapsed_s=20.,timeout_s=360,
                          termination_reason='exited',exit_code=0,peak_contemporaneous_job_rss=100,max_observed_root_threads=20 if view in v.JAX else 1,
                          result_sha256=v.sha(folder/'result.json'))
            write(self.run/f'{view}-seed-{seed}-terminal.json',external)
            jsonl(self.run/f'{view}-seed-{seed}-resources.jsonl',[dict(elapsed_s=.1,aggregate_rss=100,rss_by_pid={'1':100},root_threads=external['max_observed_root_threads'])])
            terminal['records'].append(external);terminal['spent_seconds'][view]+=20.
        write(self.run/'terminal.json',terminal)
        return terminal

    def test_complete_35_slot_full_path_and_five_ratios(self):
        self.complete_fixture();result=v.validate(self.root,self.run,'full')
        self.assertEqual(result['errors'],[])
        self.assertTrue(result['all_declared_cases_completed'])
        self.assertEqual(sum(r['strict_ranking_eligible'] for r in result['rows']),25)
        for view,item in result['views'].items():
            self.assertEqual(item['completed_n'],5)
            self.assertEqual(len(item['paired_seed_ratios']),5)
            if view in v.JAX:
                self.assertFalse(item['strict_ranking_eligible']);self.assertTrue(all(p['ratio_atlas_over_view'] is None for p in item['paired_seed_ratios']))
            else:
                self.assertTrue(item['strict_ranking_eligible']);self.assertTrue(all(p['eligible'] for p in item['paired_seed_ratios']))
        preview=v.validate(self.root,self.run,'report-only')
        self.assertEqual(preview['errors'],[]);self.assertFalse(any(r['strict_ranking_eligible'] for r in preview['rows']))

    def test_supervisor_timeout_overrides_completed_worker(self):
        terminal=self.complete_fixture();first=terminal['records'][0]
        first.update(status='timeout',termination_reason='timeout',exit_code=-9,elapsed_s=360.05)
        terminal['spent_seconds'][first['view']]+=340.05
        write(self.run/'terminal.json',terminal);write(self.run/f'{first["view"]}-seed-{first["seed"]}-terminal.json',first)
        result=v.validate(self.root,self.run,'full')
        self.assertEqual(result['errors'],[])
        row=result['rows'][0];self.assertEqual(row['effective_status'],'timeout');self.assertFalse(row['strict_ranking_eligible'])
        self.assertEqual(row['completed_measured_steps'],50);self.assertNotIn('seed_median_s',row)
        self.assertFalse(result['views']['atlas']['complete_five_seed_summary'])
        self.assertNotIn('median_of_process_medians_s',result['views']['atlas'])


if __name__=='__main__':unittest.main()
