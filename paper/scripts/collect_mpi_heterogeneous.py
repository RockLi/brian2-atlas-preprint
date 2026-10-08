"""Retain existing heterogeneous MPI contracts and qualification records; no execution."""
from pathlib import Path
import json, hashlib, xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/preprint/data/mpi_heterogeneous';OUT.mkdir(parents=True,exist_ok=True)
base=ROOT/'brian2-rust/mpi-evidence/heterogeneous-20260911'
inputs=[(base/'verification.json','simulation_verification.json'),(base/'example-runtime.json','simulation_example_runtime.json')]
inputs += [(base/n,n) for n in ['mixed-gpu.xml','gpu-inventory.xml','cpu-mpi-regression.xml','plan-device-regression.xml']]
inputs += [(ROOT/'brian2-rust/MPI_GPU.md','simulation_contract.md'),(ROOT/'brian2-rust/python/brian2_rust/mpi_gpu.py','inspected_mpi_gpu.py'),(ROOT/'brian2-rust/mpi-evidence/training-v4-gpu-20261002/cuda-l4-mpich5/verification.json','training_cuda_l4_verification.json'),(ROOT/'brian2-rust/mpi-evidence/training-v4-gpu-20261002/README.md','training_cuda_scope.md')]
sources=[]
for src,name in inputs:
 raw=src.read_bytes();(OUT/name).write_bytes(raw);sources.append({'source':str(src),'retained':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
v=json.loads((OUT/'simulation_verification.json').read_text());e=json.loads((OUT/'simulation_example_runtime.json').read_text());t=json.loads((OUT/'training_cuda_l4_verification.json').read_text())
passed=skipped=0
for n in ['mixed-gpu.xml','gpu-inventory.xml','cpu-mpi-regression.xml','plan-device-regression.xml']:
 x=ET.parse(OUT/n).getroot();ss=[x] if x.tag=='testsuite' else x.findall('.//testsuite')
 for s in ss:
  a={k:int(s.get(k,'0')) for k in ['tests','failures','errors','skipped']};assert a['failures']==a['errors']==0
  passed+=a['tests']-a['skipped'];skipped+=a['skipped']
assert (passed,skipped)==(v['passed'],v['skipped'])==(118,12)
assert e['rank_backends']==['cpu','metal'] and e['rank_gpu_dispatches']==[0,32]
assert t['accepted'] and t['cuda_mpi_passed']==31 and t['limits']['gpu_count']==1
result={'sources':sources,'inspection_date':'2026-10-07','simulation_delivery':{'passed':passed,'skipped':skipped,'mixed_and_inventory_passed':25,'hardware':v['hardware_verified'],'rank_backends':e['rank_backends'],'rank_gpu_dispatches':e['rank_gpu_dispatches'],'example_spikes':v['example_spikes'],'simulation_nvidia_hardware_qualified':False,'cross_host_qualified':False,'simultaneous_metal_cuda_qualified':False},'training_cuda':{'accepted':t['accepted'],'cuda_mpi_passed':t['cuda_mpi_passed'],'passed':t['passed'],'skipped':t['skipped'],'scope':t['scope'],'gpu':'L4','physical_gpus':1},'scope':'Retained contract documents, XML count consistency and recorded runtime/qualification fields; no new hardware or scientific validation.'}
(OUT/'evidence.json').write_text(json.dumps(result,indent=2)+'\n');print('Retained ten source records; checked 118/12 delivery counts, CPU/Metal dispatches and separate one-L4 training scope.')
