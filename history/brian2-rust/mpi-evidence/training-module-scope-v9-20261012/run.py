"""Full original scope with authenticated reuse across verified test-only ABI fixes."""
import argparse,ast,fcntl,importlib.util,json,os,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;E=P.parent;O=E/'training-module-scope-v5-20261011';R=E/'training-module-scope-v5-resume-20261011'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
n=load('current_modules',O/'run_full_modules.py');n.V=P
empty=load('current_empty',R/'run.py');empty.o=n;empty.P=P
read,sha,save=n.read,n.sha,n.save
original_verify=n.verify
def verify(snap,controls):
 original_verify(snap,controls)
 expected={'brian2-rust/target/release/b2-runner':0o755}
 assert snap['executable_modes']==expected
 for name,mode in expected.items():
  executable=Path(snap['workspace'])/name
  assert executable.stat().st_mode & 0o777==mode and os.access(executable,os.X_OK), 'Reference executable permissions changed'
 assert os.access(snap['runner'],os.X_OK), 'Training executable permissions changed'
n.verify=verify

def body_independent_tree(path,functions):
 tree=ast.parse(path.read_text());found=set()
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and node.name in functions:node.body=[];found.add(node.name)
 assert found==set(functions)
 return ast.dump(tree,include_attributes=False)


def configuration():
 snap,scope,groups,controls=n.configuration();lineage=read(P/'lineage.json')
 assert lineage['snapshot_sha256']==sha(P/'snapshot.json') and lineage['patch_sha256']==sha(P/'test-contract.patch')
 assert lineage['previous_controller_session']==89598 and not lineage['previous_cpu_complete']
 assert lineage['changed_module_indices']==[96,160] and lineage['production_changes']==0
 assert all(sha(Path(f))==h for f,h in lineage['prior_files'].items())
 previous=E/'training-module-scope-v8-20261012'
 terminal=read(previous/'controller-terminal.json')
 assert terminal['terminal'] is True and terminal['actual_exit']==lineage['previous_controller_actual_exit']
 assert terminal['actual_exit']==1 and terminal['failure_module']==260
 assert read(previous/'cpu/260/terminal.json')['actual_wait']==1
 reference='brian2-rust/target/release/b2-runner'
 assert lineage['reference_metadata_fix']==dict(path=reference,previous_mode=0o644,current_mode=0o755,sha256=snap['assets'][reference])
 allowed={'brian2-rust/tests/test_training_mutable_captures.py':['test_invalid_capture_storage_refused_without_execution'],
 'brian2-rust/tests/test_training_parameter_gather.py':['test_cuda_gather_translation_contains_typed_bounds_and_reverse_scatter','test_metal_gather_rejects_previous_abi'],
 'brian2-rust/tests/test_training_indirect_gpu.py':['test_metal_index_loader_rejects_previous_abi'],
 'brian2-rust/tests/test_training_precise_clock.py':['test_clock_rejects_previous_metal_abi']}
 assert lineage['allowed_test_bodies']==allowed
 rows=lineage['previous_receipts']
 assert [r['index'] for r in rows if r['mode']=='cpu']==[i for i in range(260) if i not in (96,160)]
 keys=[(r['mode'],r['index']) for r in rows]
 assert len(keys)==len(set(keys)) and all(mode in ('cpu','metal') and 0<=i<272 and i not in (96,160) for mode,i in keys)
 for directory in {r['snapshot_dir'] for r in rows}:
  d=Path(directory);old=read(d/'snapshot.json');oldroot=Path(old['workspace']);root=Path(snap['workspace'])
  assert read(d/'goal-scope-collection.json')==scope
  assert set(old['sources'])==set(snap['sources']) and set(old['assets'])==set(snap['assets'])
  assert old['runner']==snap['runner'] and old['runner_sha256']==snap['runner_sha256']
  assert all(sha(oldroot/k)==h for k,h in old['assets'].items())
  for key in ('assets','sources'):
   different={k for k,h in old[key].items() if snap[key][k]!=h};assert different<=set(allowed)
   for name in different:assert body_independent_tree(oldroot/name,allowed[name])==body_independent_tree(root/name,allowed[name])
 for row in rows:
  d=Path(row['snapshot_dir']);old=read(d/'snapshot.json');module=list(groups)[row['index']]
  assert row['snapshot_sha256']==sha(d/'snapshot.json')
  # A reused module's own test source must be exactly unchanged. Changed helper
  # imports may differ only in the two explicitly reviewed test bodies above.
  assert old['assets'][module]==snap['assets'][module]
  assert sha(Path(row['path'])/'acceptance.json')==row['acceptance_sha256']
  assert all(sha(Path(f))==h for f,h in row['controls'].items())
 controls.update({str(f):sha(f) for f in (P/'run.py',P/'lineage.json',P/'test-contract.patch',R/'run.py')})
 n.verify(snap,controls)
 return snap,scope,groups,controls


def prior_row(index,mode):
 return next((r for r in read(P/'lineage.json')['previous_receipts'] if r['mode']==mode and r['index']==index),None)

def location(index,mode):
 row=prior_row(index,mode);return Path(row['path']) if row else P/mode/f'{index:03d}'


def audit_one(index,module,expected,mode,cfg):
 snap,scope,groups,controls=cfg;row=prior_row(index,mode);out=location(index,mode)
 if row:
  directory=Path(row['snapshot_dir']);old=read(directory/'snapshot.json')
  validator=load('prior_validator',O/'run_full_modules.py');validator.V=directory
  helper=load('prior_empty',R/'run.py');helper.o=validator;helper.P=directory
  result=validator.audit_module(out,expected,mode,old,row['controls']) if expected else helper.audit_empty(out,module,mode,old,row['controls'])
 else:result=n.audit_module(out,expected,mode,snap,controls) if expected else empty.audit_empty(out,module,mode,snap,controls)
 assert result==read(out/'acceptance.json');return result


def audit_backend(mode):
 cfg=configuration();snap,scope,groups,controls=cfg;results=[];paths=[]
 for index,(module,expected) in enumerate(groups.items()):
  out=location(index,mode)
  if not (out/'acceptance.json').exists():return None
  results.append(audit_one(index,module,expected,mode,cfg));paths.append(out/'acceptance.json')
 assert [i for r in results for i in r['identities']]==scope['identities']
 assert len(results)==272 and sum(bool(r['identities']) for r in results)==266
 assert all(r['actual_wait']==0 for r in results if r['identities'])
 return dict(mode=mode,actual_wait=0,identities=scope['identities'],modules=scope['modules'],
 passed=[i for r in results for i in r['passed']],classified_skips=[s for r in results for s in r['classified_skips']],
 module_receipt_paths=list(map(str,paths)),module_receipt_hashes=[sha(p) for p in paths],module_snapshot_hashes=[r['snapshot_sha256'] for r in results],
 snapshot_sha256=sha(P/'snapshot.json'),empty_module_collections=[dict(module=r['module'],actual_wait=r['actual_wait']) for r in results if not r['identities']],
 actual_test_processes_all_exit_zero=True,all_original_identities_preserved=True,test_only_lineage_sha256=sha(P/'lineage.json'),unchanged_production_source=True)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 cfg=configuration();snap,scope,groups,controls=cfg
 for mode in ('cpu','metal'):
  for index,(module,expected) in enumerate(groups.items()):
   if args.preflight and index not in (96,160,260):continue
   save(P/'status.json',dict(status='running',mode=mode,module_index=index,module=module,module_count=272,controller_pid=os.getpid(),controls=controls,preflight=args.preflight,complete_goal_proven=False))
   out=location(index,mode)
   if prior_row(index,mode):result=audit_one(index,module,expected,mode,cfg)
   elif expected:result=n.execute(out,expected,mode,snap,controls)
   else:result=empty.execute_empty(out,module,mode,snap,controls)
   print(json.dumps(dict(mode=mode,module_index=index,module=module,counts=result['counts'],actual_wait=result['actual_wait'])),flush=True)
  if not args.preflight:
   result=audit_backend(mode);assert result is not None;save(P/(mode+'-accepted.json'),result)
 save(P/'status.json',dict(status='preflight-complete' if args.preflight else 'CPU-and-Metal-complete',controls=controls,cloud_calls=0,complete_goal_proven=False))

if __name__=='__main__':
 lock=(P/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:main()
 except BaseException as error:
  s=read(P/'status.json') if (P/'status.json').exists() else {}
  save(P/'status.json',dict(s,status='stopped-with-error',error=repr(error),stopped_at=time.time(),complete_goal_proven=False));traceback.print_exc();raise SystemExit(1)
