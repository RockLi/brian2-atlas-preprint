"""Input, API and independent small-network Euler/delay checks."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .simulation import SimulationConfig, build_model, cut_model, encode_movie
from .serve import validate_request


class InputChecks(unittest.TestCase):
    def setUp(self):
        self.channels=[dict(index=0,cell_type="Mi1",p=19,q=14),dict(index=1,cell_type="Tm1",p=19,q=20)]

    def test_blank_has_no_external_spikes(self):
        c=SimulationConfig()
        i,t,s=encode_movie(np.full((40,48,48),.5),self.channels,c)
        self.assertEqual(len(i),0);self.assertEqual(len(t),0);self.assertFalse(s.any())

    def test_contrast_polarity_timing_and_fresh_phase(self):
        c=SimulationConfig();frames=np.full((40,48,48),.9)
        i,t,s=encode_movie(frames,self.channels,c)
        self.assertTrue(len(i)>0);self.assertTrue(np.all(i==0))
        self.assertTrue(np.all(t>=1000)&np.all(t<5000))
        self.assertTrue(np.all(np.diff(t*2+i)>0))
        other=encode_movie(1-frames,self.channels,c)
        self.assertTrue(np.all(other[0]==1))
        again=encode_movie(frames,self.channels,c)
        np.testing.assert_array_equal(i,again[0]);np.testing.assert_array_equal(t,again[1])
        np.testing.assert_allclose(s[:,0],.8,atol=1e-6)

    def test_fresh_api_rejects_labels_and_unknown_fields(self):
        self.assertEqual(validate_request({"kind":"left","condition":"cut"}),("left","cut"))
        for value in ({"kind":"left","condition":"cut","label":1},{"kind":[],"condition":"cut"},{"kind":"secret","condition":"intact"},None):
            with self.assertRaises(ValueError):validate_request(value)


def numpy_reference(config, indices, ticks, channels, source, target, weights, cut):
    """Independent six-cell scalar Euler loop, SI converted here to mV/ms.

    Schedule: membrane/conductance update, threshold, synaptic delivery, reset.
    Positive and negative edges and an 18-tick recurrent delay are both exercised.
    """
    n=6;v=-52+np.random.default_rng(config.seed).uniform(-.8,.8,n)
    ge=np.zeros(n);gi=np.zeros(n);last=np.full(n,-10000000,dtype=int)
    pending={};events=[];cursor=0
    for tick in range(config.steps):
        available=(tick-last)>=22
        old_v=v.copy();old_ge=ge.copy();old_gi=gi.copy()
        for neuron in range(n):
            if available[neuron]:
                v[neuron]=old_v[neuron]+config.dt_ms*(-(old_v[neuron]+52)-old_ge[neuron]*old_v[neuron]-old_gi[neuron]*(old_v[neuron]+70))/20
            ge[neuron]=old_ge[neuron]-config.dt_ms*old_ge[neuron]/5
            gi[neuron]=old_gi[neuron]-config.dt_ms*old_gi[neuron]/5
        fired=np.flatnonzero((v>-45)&available)
        for neuron in fired:
            events.append((tick,int(neuron)))
            if cut and neuron in (0,1):continue
            for edge in np.flatnonzero(source==neuron):pending.setdefault(tick+18,[]).append(int(edge))
        for edge in pending.pop(tick,[]):
            if weights[edge]>0:ge[target[edge]]+=.275/52*weights[edge]
            else:gi[target[edge]]-=.275/52*4*weights[edge]
        while cursor<len(ticks) and ticks[cursor]==tick:
            ge[channels[indices[cursor]]["index"]]+=config.input_weight_mv/52
            cursor+=1
        v[fired]=-52;last[fired]=tick
    return v/1000,ge,gi,np.array(events,dtype=np.int64).reshape(-1,2)


class NumericChecks(unittest.TestCase):
    def test_native_matches_independent_euler_and_delay_loop(self):
        from flywire_mnist.graph import Graph
        from flywire_mnist.backends import CPU
        from brian2_rust.protocol import attach_protocol
        root=Path(__file__).resolve().parents[2]
        graph=Graph(np.arange(1,7,dtype=np.uint64),np.array([0,1]),np.array([2,3]),np.array([4,5]),"toy","diagnostic",
                    source=np.array([0,2,1,3]),target=np.array([2,3,4,5]),contacts=np.array([20.,20.,-8.,12.]),edge_count=4)
        channels=[dict(index=0,cell_type="Mi1",p=19,q=14),dict(index=1,cell_type="Tm1",p=19,q=20)]
        cfg=SimulationConfig(frames=8,warmup_ms=2,tail_ms=10,rate_hz=500,input_weight_mv=64,background_rate_hz=0)
        frames=np.full((8,48,48),.1);frames[:,:,:24]=.9
        indices,ticks,_=encode_movie(frames,channels,cfg)
        template=build_model(graph,channels,cfg,root/"target/release/b2-runner")
        pi=next(i for i,p in enumerate(template["definition"]["populations"]) if p["name"]=="visual_input")
        ni=next(i for i,p in enumerate(template["definition"]["populations"]) if p["name"]=="flywire_neurons")
        with tempfile.TemporaryDirectory(prefix="vision-numerics-") as temp:
            cpu=CPU(template,Path(temp)/"cpu")
            for cut in (False,True):
                model=cut_model(template,channels) if cut else copy.deepcopy(template)
                model["instance"]["populations"][pi]["spike_generator"]={"spike_indices":indices.tolist(),"spike_ticks":ticks.tolist()}
                attach_protocol(model)
                actual=cpu.run(model,"cut" if cut else "intact")["populations"][ni]
                v,ge,gi,events=numpy_reference(cfg,indices,ticks,channels,graph.source,graph.target,graph.contacts,cut)
                for name,expected in (("v",v),("ge",ge),("gi",gi)):
                    np.testing.assert_allclose(actual["states"][name],expected,atol=2e-12,rtol=2e-12)
                np.testing.assert_array_equal(actual["spike_ticks"],events[:,0])
                np.testing.assert_array_equal(actual["indices"],events[:,1])
                self.assertTrue(len(events)>0)
                if not cut:self.assertTrue(gi[4]>0)


if __name__=="__main__":unittest.main()
