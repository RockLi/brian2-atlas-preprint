"""Paired Device activations isolating predecessor compilation reuse."""
import argparse
import json
from pathlib import Path
import time

import brian2 as b
import brian2_rust
import numpy as np


def benchmark(backend,output):
    output.mkdir(parents=True,exist_ok=False)
    device=b.get_device();records=[];saved={};expected={}
    from brian2.devices.device import all_devices
    device=all_devices['rust_standalone']
    previous=b.get_device()
    dt=b.second/1024;n=257
    try:
        for pair,order in enumerate((('fresh','reuse'),('reuse','fresh'))):
            for position,policy in enumerate(order):
                device.reinit();directory=output/f'{pair}-{policy}'
                b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
                    gpu_buffer_reuse=True,gpu_compile_reuse=policy=='reuse',directory=directory,
                    **({'cuda_dag_execution':'resident'} if backend=='cuda' else {}))
                p=b.NeuronGroup(n,'v:1',threshold='v>=1',reset='v-=1',dt=dt,name='population')
                p.v=(np.arange(n)%8)/8;p.run_regularly('v+=0.125',name='advance')
                s=b.Synapses(p,p,'w:1',on_pre='w+=0.015625;v_post+=w',clock=p.clock,name='projection')
                i=np.repeat(np.arange(n),4);s.connect(i=i,j=(i+np.tile(np.arange(1,5),n))%n)
                s.w=.015625;s.delay=dt
                m=b.StateMonitor(p,'v',record=[0,n-1],name='voltage');sp=b.SpikeMonitor(p,name='spikes')
                net=b.Network(p,s,m,sp)
                for activation in range(4):
                    start=time.perf_counter();net.run(8*dt);wall=time.perf_counter()-start
                    values=dict(v=np.asarray(p.v[:]).copy(),w=np.asarray(s.w[:]).copy(),
                        trace=np.asarray(m.v[:]).copy(),spike_t=np.asarray(sp.t[:]).copy(),spike_i=np.asarray(sp.i[:]).copy())
                    if activation not in expected:expected[activation]=values
                    for name,a in values.items():
                        np.testing.assert_array_equal(a,expected[activation][name])
                        saved[f'{pair}/{policy}/{activation}/{name}']=a
                    summary=json.loads((device.last_run_directory/'rust/summary.json').read_text())
                    runtime=summary[backend+'_runtime']
                    if activation==0:
                        assert not runtime['activation_buffer_reuse']['adopted']
                    else:
                        assert runtime['activation_buffer_reuse']['adopted']
                        counters=runtime if backend=='metal' else runtime['dag_execution']
                        assert counters['reused_buffer_count']>0
                        assert 0<counters['allocated_bytes']<counters['reused_buffer_bytes']
                    compilation=runtime['compilation']
                    if policy=='reuse' and activation>0:assert compilation['kernels_reused']>0
                    else:assert compilation['kernels_reused']==0
                    records.append(dict(pair=pair,order=position,policy=policy,activation=activation,
                        network_run_seconds=wall,device_seconds=device._last_run_time,runtime=runtime))
    finally:
        device.reinit();b.set_device(previous)
    np.savez_compressed(output/'activation-results.npz',**saved)
    report=dict(schema='gpu-compilation-reuse-v0',backend=backend,neurons=n,edges=n*4,steps_per_activation=8,
        records=records,passed=True,scope='Both policies transfer compatible allocations. Fresh disables compilation reuse; reuse enables exact-source predecessor reuse. Every activation independently validates and plans the model. Wall covers Network.run including model export, validation, compilation, GPU execution, full results and monitor append. Initial activations are warmups; two balanced groups are exploratory, not a statistical speedup claim.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('metal','cuda'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();benchmark(a.backend,a.output)
