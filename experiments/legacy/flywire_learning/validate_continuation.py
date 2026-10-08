"""Fresh-process checkpoint and delayed CSR events with local plasticity."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from spiking import NEURON_EQUATIONS, plastic, setup
from validate_spiking import SPIKE_IDS, SPIKE_TICKS, PRE, POST, CONTACTS, signals
from prepare import write_csr


def child(output, backend, mode, threads):
    import brian2 as b
    import brian2_rust as rust
    folder=output/f'{backend}-{threads}-{mode}'
    folder.mkdir(exist_ok=False)
    setup(backend,folder/'project',threads)
    g=b.NeuronGroup(2,NEURON_EQUATIONS,threshold='v>-45*mV',reset='v=-52*mV',
                    refractory=2.2*b.ms,dt=.1*b.ms,method='euler',name='neurons')
    g.v=[-44.5,-52]*b.mV
    source=b.SpikeGeneratorGroup(2,SPIKE_IDS,SPIKE_TICKS*.1*b.ms,clock=g.clock,sorted=True,name='source')
    reward,gate=signals()
    s=plastic(source,g,PRE,POST,CONTACTS,b.TimedArray(reward,dt=.1*b.ms,name='teaching'),
              b.TimedArray(gate,dt=.1*b.ms,name='learning'),delay_ms=.2,tau_ms=1,
              depression_hz=80,recovery_hz=40)
    static=b.Synapses(g,g,'multiplicity : 1 (constant)',on_pre='ge_post += .02*multiplicity',
                      delay=3*b.ms,clock=g.clock,name='static')
    if backend=='numpy':
        static.connect(i=[0,1],j=[1,0]);static.multiplicity=[2,3]
    else:
        rust.connect_binary_csr(static,output/('other.b2csr' if mode=='mismatch' else 'graph.b2csr'),
                                parameters={'multiplicity':0})
    m=b.StateMonitor(g,['v','ge','gi'],record=True,name='trace')
    spikes=b.SpikeMonitor(g,name='spikes')
    net=b.Network(g,source,s,static,m,spikes)
    checkpoint=output/f'{backend}-{threads}.checkpoint'
    if mode in ('restore','mismatch'):
        net.restore('boundary',filename=checkpoint,restore_random_state=True)
    if mode=='mismatch':
        try: net.run(.5*b.ms)
        except NotImplementedError as error:
            if 'binary CSR topology, delay or parameters changed' not in str(error): raise
            (folder/'rejected.json').write_text(json.dumps({'reason':str(error)}));return
        raise AssertionError('changed graph accepted on restore')
    if mode=='full': net.run(8*b.ms)
    elif mode in ('save','split'):
        net.run(1*b.ms)
        if mode=='save':
            pending=b.get_device()._pending_events
            assert any(e['delivery_tick']>=15 for events in pending.values() for e in events)
            net.store('boundary',filename=checkpoint)
            return
        net.run(.5*b.ms);net.run(6.5*b.ms)
    elif mode=='restore': net.run(.5*b.ms);net.run(6.5*b.ms)
    result=dict(v=np.array(g.v[:]),ge=np.array(g.ge[:]),gi=np.array(g.gi[:]),
                gain=np.array(s.gain[:]),eligibility=np.array(s.eligibility[:]),
                neuron_trace=np.array([m.v,m.ge,m.gi]),spike_i=np.array(spikes.i[:]),
                spike_t=np.array(spikes.t[:]),lastspike=np.array(g.lastspike[:]),
                not_refractory=np.array(g.not_refractory[:]),time=np.array(float(net.t/b.second)))
    np.savez(folder/'snapshot.npz',**result)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    write_csr(output/'graph.b2csr',2,[0,1,2],[1,0],[2.,3.])
    write_csr(output/'other.b2csr',2,[0,1,2],[1,0],[2.,4.])
    report={}
    for backend,threads in [('numpy',1),('reference',1),('aot',1),('aot',4)]:
        modes=['full'] if backend=='numpy' else ['full','split','save','restore','mismatch']
        for mode in modes:
            label=f'{backend}-{threads}-{mode}'
            with (output/f'{label}.log').open('w') as log:
                subprocess.run([sys.executable,__file__,'--output',str(output),'--backend',backend,
                                '--threads',str(threads),'--mode',mode],stdout=log,stderr=subprocess.STDOUT,check=True)
            if mode in ('save','mismatch'): continue
            with np.load(output/'numpy-1-full/snapshot.npz') as baseline,np.load(output/label/'snapshot.npz') as actual:
                for key in baseline.files:
                    np.testing.assert_allclose(actual[key],baseline[key],rtol=1e-12,atol=1e-14,err_msg=f'{label}/{key}')
            if mode in ('split','restore'):
                with np.load(output/f'{backend}-{threads}-full/snapshot.npz') as full,np.load(output/label/'snapshot.npz') as actual:
                    for key in full.files:
                        assert full[key].dtype==actual[key].dtype and full[key].tobytes()==actual[key].tobytes(),(label,key)
            report[label]='passed'
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['numpy','reference','aot'])
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--mode',choices=['full','split','save','restore','mismatch'])
    a=p.parse_args();child(a.output,a.backend,a.mode,a.threads) if a.mode else run(a.output)
