"""Isolated batch-size trial, with multiple-batch and full-host numeric gates."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('capacity_control', HERE / 'control.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

def prepare(case):
    if case == 'major-b128':
        assert all(json.loads((HERE/(x+'-validation.json')).read_text())['passed'] for x in ['batch-old','batch-new','pilot-b128'])
        assert all(json.loads((HERE/(x+'-resources.json')).read_text())['passed'] for x in ['batch-old','batch-new','pilot-b128'])
    small = case.startswith('batch-')
    n = 48000 if small else 24000 if case == 'pilot-b128' else 86000000
    ranks = 2 if small else 240
    source = c.BASE + ('/source' if case == 'batch-old' else '/source-b128')
    c.guarded('prepare-'+case,[c.c.PYTHON,source+'/experiment/prepare-p12.py','--source',source,'--output',c.BASE+'/'+case,'--neurons',str(n),'--ranks',str(ranks),'--compile'],8192 if small else 65536,2 if small else 4,1800)

def launch(case):
    if not case.startswith('batch-'):
        assert all(json.loads((HERE/(x+'-validation.json')).read_text())['passed'] for x in ['batch-old','batch-new'])
        assert all(json.loads((HERE/(x+'-resources.json')).read_text())['passed'] for x in ['batch-old','batch-new'])
        if case == 'major-b128':
            assert json.loads((HERE/'large-input-identity.json').read_text())['passed']
            assert all(json.loads((HERE/(x+'-validation.json')).read_text())['passed'] for x in ['batch-old','batch-new','pilot-b128'])
            assert json.loads((HERE/'pilot-b128-resources.json').read_text())['passed']
        return c.launch(case,memory=16384 if case=='pilot-b128' else 262144)
    # Two ranks exercise peer routing and multiple batches on both batch sizes.
    sys.path.insert(0,str(c.OLD/'frozen-tools'))
    from mpi_teleport_launch import launch as run
    admission=c.remote(c.NODES[0],"import subprocess,json\ns=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True);assert not s.strip();print(json.dumps({'no_other_mpi_job':True}))")
    c.record(case+'-admission.json',admission)
    command='exec env B2_MPI_MAX_LOCAL_EDGES=450000000 B2_THREAD_AFFINITY=required /usr/bin/time -v -o '+c.BASE+'/'+case+'/rank-${PMI_RANK}.time '+c.BASE+'/'+case+'/mpi/b2-mpi '+c.BASE+'/'+case+'/mpi/instance.bin '+c.BASE+'/'+case+'/result'
    try:
        r=run(nodes=c.NODES[:1],ips=c.selection['ips'][:1],ranks_per_node=2,remote_base=c.REAL,application=['sh','-c',command],output=HERE/(case+'-launch'),mpi_prefix=c.BASE+'/mpi',timeout=300,login='root',guard_script=c.REAL+'/source/tools/mpi_resource_guard.py',guard_volume='/data/brick2',guard_allow_root_volume=False,guard_memory_mib=8192,guard_cpu_percent=200,guard_cpu_count=2,guard_cpu_ids=[1,13],guard_file_mib=8192,guard_min_free_gib=128)
    finally:
        p=HERE/(case+'-launch/launch.json')
        if p.exists():(HERE/(case+'-launch-result.json')).write_bytes(p.read_bytes())

def validate(case):
    if not case.startswith('batch-'):return c.validate_pilot(case)
    code=f'''from pathlib import Path
import json,sys,numpy as np,hashlib
b=Path({c.BASE!r});p=b/{case!r};ref=b/'batch-old';sys.path.insert(0,str(b/'source/python'))
from brian2_rust.results import load_results
m=json.loads((p/'model.json').read_text());assert hashlib.sha256((p/'model.json').read_bytes()).digest()==hashlib.sha256((ref/'model.json').read_bytes()).digest()
a=load_results(m,ref/'reference');z=load_results(m,p/'result');checks=[];maximum=0
for x,y in zip(a['populations'],z['populations'],strict=True):
 for k in ['spike_ticks','indices','counts','last_spikes']:checks.append(bool(np.array_equal(x[k],y[k])))
 for kind in ['states','trace','refractory']:
  for k,v in x[kind].items():
   w=y[kind][k];assert np.shape(v)==np.shape(w);e=float(np.max(np.abs(np.asarray(v,dtype=float)-np.asarray(w,dtype=float)),initial=0));maximum=max(maximum,e);checks.append(e==0)
checks.extend([a['synaptic_events']==z['synaptic_events'],z['synaptic_events']>0,z['metadata']['spike_count']>0])
r=json.loads((p/'result/mpi-runtime.json').read_text());assert r['ranks']==2 and len(set(r['processor_names']))==1
assert len(r['procedural_topology'])==144 and all(t['construction']=='distributed-draw-ranges' for t in r['procedural_topology'])
print(json.dumps({{'passed':all(checks),'case':{case!r},'checks':len(checks),'max_state_absolute_difference':maximum,'spikes':z['metadata']['spike_count'],'synaptic_events':z['synaptic_events'],'largest_projection_connections':480000,'largest_projection_draws_per_rank':240000,'old_new_batches_per_rank_largest_projection':[15,2],'runtime':r}}))'''
    c.record(case+'-validation.json',c.remote(c.NODES[0],code,python=c.c.PYTHON))

def collect(case):
    if not case.startswith('batch-'):return c.collect(case)
    prefix=json.loads((HERE/(case+'-launch-result.json')).read_text())['resource_guard']['unit_prefix']
    code=f'''from pathlib import Path
import json,os
b=Path({c.BASE!r});g={{r:json.loads((b/'guards'/({prefix!r}+'-'+r+'.json')).read_text()) for r in ['proxy-0','controller']}}
assert all(x.get('returncode')==0 and 'oom 0' in x['after']['memory.events'] for x in g.values())
times={{p.name:p.read_text() for p in (b/{case!r}).glob('rank-*.time')}};assert len(times)==2 and all('Exit status: 0' in t for t in times.values())
print(json.dumps({{'passed':True,'rows':[{{'host':os.uname().nodename,'guards':g,'rank_times':times}}]}}))'''
    c.record(case+'-resources.json',c.remote(c.NODES[0],code))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','validate','collect']);p.add_argument('case');a=p.parse_args()
    {'prepare':prepare,'launch':launch,'validate':validate,'collect':collect}[a.action](a.case)
