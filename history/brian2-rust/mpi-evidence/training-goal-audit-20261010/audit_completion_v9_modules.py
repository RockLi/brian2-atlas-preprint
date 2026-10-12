"""Latest-source closure: full original CPU/Metal scope plus actual current CUDA.

Historical source reviews retain their manually inspected unchanged excerpts.
Historical CUDA receipts are re-audited against their frozen sources and reused only after an explicit test-only equivalence check.
"""
import collections,hashlib,importlib.util,json,re
from pathlib import Path
import xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent;E=P.parent;ROOT=P.parents[2];V=E/'training-final-scope-v5-20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def historical_cuda_load(name,path):
 # These validators read ROOT only to check source bytes. Validate the genuine
 # v5 frozen tree, then apply the explicit production-equivalence proof below.
 # No digest is substituted and no current source is presented as old bytes.
 allowed={'audit_current_cuda_v5.py','audit_remaining_cuda.py','audit_named_cuda.py'}
 assert path.name in allowed
 module=load(name,path)
 if path.name in {'audit_current_cuda_v5.py','audit_remaining_cuda.py'}:
  module.ROOT=Path(read(V/'snapshot.json')['workspace']).resolve()
 if path.name=='audit_named_cuda.py':module.load=historical_cuda_load
 return module

def audit():
 snap=read(V/'snapshot.json');root=Path(snap['workspace']).resolve();scope=read(V/'goal-scope-collection.json')
 current_phase=E/'training-module-scope-v9-20261012'
 bridge=load('v6_test_only_bridge',current_phase/'run.py');cfg=bridge.configuration()
 current=cfg[0]
 assert len(snap['sources'])==946 and len(snap['assets'])==997
 assert all(sha(root/n)==h for n,h in snap['assets'].items())
 # configuration authenticates every current working/frozen byte, all prior
 # receipts, and permits only the documented test contract correction.
 assert current['runner_sha256']==snap['runner_sha256']
 collection=read(current_phase/'collection-v9/acceptance.json')
 terminal=read(current_phase/'collection-v9/terminal.json')
 assert all(collection[k]==v for k,v in terminal.items())
 assert terminal['actual_wait']==0 and terminal['wait']=='subprocess.wait' and terminal['pid']>0
 assert collection['snapshot_sha256']==sha(current_phase/'snapshot.json')
 assert collection['log_sha256']==sha(current_phase/'collection-v9/tests.log')
 assert collection['identities']==scope['identities'] and collection['count']==34754
 assert collection['command'][1:]==['-m','pytest','--collect-only','-q','-p','no:cacheprovider',*scope['modules']]
 assert [n for n in (current_phase/'collection-v9/tests.log').read_text().splitlines() if n.startswith('brian2-rust/') and '::' in n]==scope['identities']
 # Original frontend and harness preflight remains evidence for byte-identical
 # production code. The changed boundary test has fresh whole CPU/Metal runs.
 preflight=read(V/'preflight/receipt.json')
 assert preflight['actual_wait']==0 and preflight['passed']==39 and preflight['failed']==preflight['skipped']==0
 assert preflight['snapshot_sha256']==sha(V/'snapshot.json')
 assert all(sha(V/n)==h for n,h in preflight['controller_hashes'].items())
 assert [r['name'] for r in preflight['records']]==['harness-api','collection','tests']
 for record in preflight['records']:
  assert record==read(V/'preflight'/(record['name']+'-terminal.json'))
  assert record['actual_wait']==0 and record['pid']>0 and record['wait']=='subprocess.wait'
  assert record['log_sha256']==sha(V/'preflight'/(record['name']+'.log'))
 assert '5 harness invariants passed' in (V/'preflight/harness-api.log').read_text()
 assert preflight['junit_sha256']==sha(V/'preflight/tests.xml')
 cases=ET.parse(V/'preflight/tests.xml').findall('.//testcase')
 assert [c.get('classname').replace('.','/')+'.py::'+c.get('name') for c in cases]==preflight['identities']==[n for n in scope['identities'] if n.startswith('brian2-rust/tests/test_training_error_reports.py::')]
 assert all(not any(c.find(t) is not None for t in ('failure','error','skipped')) for c in cases)
 for mode in ('cpu','metal'):
  for index in (96,117,134,160,260):bridge.audit_one(index,list(cfg[2])[index],list(cfg[2].values())[index],mode,cfg)
 assert sha(Path(snap['runner']))==snap['runner_sha256']
 assert scope['actual_wait']==0 and scope['count']==34754 and scope['module_count']==272
 assert len(scope['identities'])==len(set(scope['identities']))==34754 and scope['original_34565_identities_preserved_in_order']
 previous_scope=read(E/'training-final-scope-v3-20261010/goal-scope-collection.json')
 assert scope['identities']==previous_scope['identities'] and scope['modules']==previous_scope['modules']
 collected_ids=[n for n in (V/'collection.log').read_text().splitlines() if n.startswith('brian2-rust/') and '::' in n]
 assert collected_ids==scope['identities']
 old=read(E/'training-final-scope-v3-20261010/snapshot.json');oldroot=Path(old['workspace'])
 assert all(sha(oldroot/n)==h for n,h in old['assets'].items())
 policy=load('latest_original_skip_policy',V/'skip_policy.py')
 assert sha(V/'skip_policy.py')==sha(E/'training-final-scope-v3-20261010/skip_policy.py')
 gates={'working_sources_match':True,'latest_frontend_and_harness_preflight':True};artifacts={};executions=[];all_passed=set()
 module_phase=current_phase
 module_audit=load('full_module_scope_audit',module_phase/'run.py')
 for mode in ('cpu','metal'):
  accepted=module_audit.audit_backend(mode)
  gates[mode+'_complete_original_scope']=accepted is not None
  if accepted is None:continue
  receipt=module_phase/(mode+'-accepted.json')
  assert accepted==read(receipt)
  assert accepted['identities']==scope['identities'] and accepted['modules']==scope['modules']
  assert accepted['snapshot_sha256']==sha(current_phase/'snapshot.json')
  assert accepted['all_original_identities_preserved'] and accepted['actual_wait']==0
  assert len(accepted['module_receipt_hashes'])==272
  passed=accepted['passed'];classified=accepted['classified_skips']
  all_passed.update(passed)
  executions.append(dict(backend=mode,identities=passed,receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt),module_receipt_hashes=accepted['module_receipt_hashes'],snapshot_sha256=sha(current_phase/'snapshot.json'),per_module_snapshot_hashes=accepted['module_snapshot_hashes']))
  artifacts[mode]=dict(passed=len(passed),skipped=len(classified),actual_wait=0,independent_module_processes=272)
 # Repaired syntax/delay modules are required by the complete original scope
 # above; older duplicate partial runs are not needed to inflate coverage.
 cuda=historical_cuda_load('latest_cuda',P/'audit_current_cuda_v5.py').audit_current();gates['CUDA_current_source_accepted']=cuda is not None
 if cuda is not None:
  assert cuda['actual_wait']==0 and cuda['distinct_passed']==83 and cuda['snapshot_sha256']==sha(V/'snapshot.json')
  assert cuda==read(P/'current-CUDA-acceptance-v5.json')
  artifacts['CUDA']=cuda
  for b in cuda['batches']:
   all_passed.update(b['identities']);executions.append(dict(backend='cuda',identities=b['identities'],receipt=str((P/'current-CUDA-acceptance-v5.json').relative_to(ROOT)),receipt_sha256=sha(P/'current-CUDA-acceptance-v5.json'),snapshot_sha256=sha(V/'snapshot.json')))
 named_cuda_phase=E/'training-requirement-cuda-v5-20261010'
 named_cuda_module=historical_cuda_load('original_named_cuda_parameters',named_cuda_phase/'audit_named_cuda.py')
 assert len(named_cuda_module.required())==175
 named_cuda=named_cuda_module.audit()
 gates['all_original_named_CUDA_parameters']=named_cuda is not None
 if named_cuda is not None:
  assert named_cuda['actual_wait']==0 and named_cuda['required_original_CUDA_count']==175
  assert named_cuda['snapshot_sha256']==sha(V/'snapshot.json')
  assert named_cuda==read(named_cuda_phase/'named-CUDA-accepted.json')
  all_passed.update(named_cuda['identities']);artifacts['all_original_named_CUDA_parameters']=named_cuda
  executions.append(dict(backend='cuda',identities=named_cuda['identities'],receipt=str((named_cuda_phase/'named-CUDA-accepted.json').relative_to(ROOT)),receipt_sha256=sha(named_cuda_phase/'named-CUDA-accepted.json'),snapshot_sha256=sha(V/'snapshot.json')))
 compiler=E/'training-cuda-capability-exports-20261010';compiled=read(compiler/'combined-receipt.json')
 assert compiled['actual_wait']==0 and compiled['passed']==21
 assert sha(ROOT/'brian2-rust/python/brian2_rust/training_cuda.py')==compiled['translator_sha256']
 assert sha(ROOT/'brian2-rust/python/brian2_rust/training_cuda.cu')==compiled['host_source_sha256']
 gates['compiler_checks_accepted']=cuda is not None
 readonly=read(E/'training-fresh-readonly-20261010/accepted-fix.json')
 assert readonly['actual_wait']==0 and readonly['distinct_passed']==48
 assert sha(ROOT/'brian2-rust/python/brian2_rust/training.py')==readonly['working_source_sha256']
 gates['fresh_readonly_current_atomicity_accepted']=set(readonly['identities'])<=all_passed
 # Reuse the actual Rust execution only for the identical Rust sources and native.
 rust=E/'training-current-rust-lib-20261010';r=read(rust/'receipt.json');terminal=read(rust/'terminal.json');base=read(E/'training-final-scope-v2-20261010/snapshot.json')
 assert r['actual_wait']==terminal['actual_wait']==0 and r['passed']==51 and r['wait']=='subprocess.wait' and r['pid']==terminal['pid']>0
 assert r['snapshot_sha256']==sha(E/'training-final-scope-v2-20261010/snapshot.json') and r['runner_sha256']==snap['runner_sha256']==base['runner_sha256']
 assert r['controller_sha256']==sha(rust/'run.py') and r['log_sha256']==sha(rust/'tests.log')
 launch=read(rust/'launch.json');assert launch['pid']==r['pid']
 assert launch['command']==['cargo','test','--release','--lib','--locked','--offline','-j','1','--target-dir',str(Path(base['base'])/'target'),'--','--test-threads=1']
 rust_names={n for n in snap['sources'] if n.startswith('brian2-rust/src/') or n in ('brian2-rust/Cargo.toml','brian2-rust/Cargo.lock')}
 assert all(base['sources'][n]==snap['sources'][n] for n in rust_names)
 text=(rust/'tests.log').read_text();assert re.findall(r'^test (.+) \.\.\. ok$',text,re.M)==r['identities'] and len(set(r['identities']))==51
 assert '51 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' in text
 gates['unchanged_Rust_library_actual_execution']=True
 artifacts['Rust']=dict(actual_wait=0,passed=51,unchanged_rust_files=len(rust_names),original_snapshot_sha256=r['snapshot_sha256'],latest_native_sha256=snap['runner_sha256'])
 original=read(E/'training-stp-full-vjp-20261009/requirements-audit.json')['requirements'];constraints=read(E/'training-stp-full-vjp-20261009/constraint-audit.json')['constraints'];rows={r['id']:r for r in original+constraints}
 manual=load('original_manual_source_review',P/'build_semantic_review.py').REVIEW
 required={f'S{i}' for i in range(1,5)}|{f'D{i}' for i in range(1,6)}|{f'C{i}' for i in range(1,9)}|{f'E{i}' for i in range(1,5)}
 assert set(rows)==set(manual)==required
 full_ready=gates['cpu_complete_original_scope'] and gates['metal_complete_original_scope'];review={};source_hashes={}
 for key,(reason,refs) in manual.items():
  references=[]
  for relative,start,end in refs:
   name='brian2-rust/'+relative;old_lines=(oldroot/name).read_text().splitlines();current_lines=(ROOT/name).read_text().splitlines()
   shift=13 if relative=='python/brian2_rust/training_brian_dynamic.py' and start>171 else 0
   assert current_lines[start+shift-1:end+shift]==old_lines[start-1:end],(key,name,start,end)
   excerpt='\n'.join(current_lines[start+shift-1:end+shift]);source_hashes[name]=sha(ROOT/name)
   references.append(dict(path=name,start=start+shift,end=end+shift,excerpt_sha256=hashlib.sha256(excerpt.encode()).hexdigest()))
  functions=rows[key].get('test_functions',rows[key].get('functions'));assert functions
  assert all(any(n.split('::')[1].split('[')[0]==fn for n in scope['identities']) for fn in functions)
  for module in rows[key]['modules']:source_hashes[module]=sha(ROOT/module)
  matches=sorted(n for n in all_passed if n.split('::')[1].split('[')[0] in functions)
  every_function=all(any(n.split('::')[1].split('[')[0]==fn for n in matches) for fn in functions)
  review[key]=dict(requirement=rows[key]['requirement'],source_reasoning=reason,source_references=references,source_reviewed=True,test_functions=functions,actual_passed_identities=matches,all_named_functions_actually_passed=every_function,final_scope_pending=not full_ready,proven=full_ready and cuda is not None and named_cuda is not None and every_function)
 gates['all_original_requirements_semantically_reviewed']=all(r['proven'] for r in review.values())
 # Original independently checked STP identities must all pass on this version.
 stp=E/'training-stp-full-vjp-20261009';stp_ids=[]
 for mode in ('cpu','metal'):stp_ids+=read(stp/(mode+'-v1-receipt.json'))['identities']
 assert len(stp_ids)==len(set(stp_ids))==80
 gates['STP_current_runner_accepted']=full_ready and set(stp_ids)<=all_passed
 result=dict(gates=gates,requirements=review,source_hashes=source_hashes,executions=executions,artifacts=artifacts,final_scope_modules=272,final_scope_identities=34754,original_modules=268,original_identities=34565,latest_snapshot_sha256=sha(current_phase/'snapshot.json'),prior_CUDA_snapshot_sha256=sha(V/'snapshot.json'),test_only_equivalence_lineage_sha256=sha(current_phase/'lineage.json'),cross_host_user_deferred=True,complete_goal_proven=all(gates.values()),missing_gates=[n for n,v in gates.items() if not v],cloud_calls=0)
 return result
if __name__=='__main__':
 r=audit();(P/'completion-gates-v9-modules.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({n:v for n,v in r.items() if n not in ('requirements','source_hashes','executions','artifacts')}))
 if not r['complete_goal_proven']:raise SystemExit(2)
