"""Two direct input edges: rest-linear filter and blank-conductance local LIF."""
import numpy as np

GAIN=.275/52
N=6000

def schedule(shift,kind,weights):
    a,b=weights
    pairs=[(1538+shift+t,w) for start,w in ((0,a if kind=='AB' else b),(200,b if kind=='AB' else a)) for t in (start,start+100)]
    impulses=np.zeros(N)
    for t,w in pairs:impulses[t]+=GAIN*w
    return impulses

def direct_conductance(impulses):
    g=np.zeros(len(impulses))
    for t in range(len(g)-1):g[t+1]=.98*g[t]+impulses[t]
    return g

def linear(impulses):
    g=direct_conductance(impulses);x=np.zeros(len(g))
    for t in range(len(g)-1):x[t+1]=.995*x[t]+.005*.052*g[t]
    return x*1000

def local(ge,gi,initial,impulses):
    added=direct_conductance(impulses);v=np.zeros(len(ge));v[0]=initial;available=0;spikes=[]
    for t in range(len(v)):
        x=v[t]
        if t>=available:
            x+=.005*(-(x+.052)-(ge[t]+added[t])*x-gi[t]*(x+.070))
            if x>-.045:spikes.append(t);available=t+22;x=-.052
        if t+1<len(v):v[t+1]=x
    return v*1000,np.array(spikes,dtype=np.int64)

def weights_for(weights,variant):
    a,b=weights
    return (a,b) if variant=='original' else ((a+b)/2,(a+b)/2) if variant=='balanced' else (b,a)
