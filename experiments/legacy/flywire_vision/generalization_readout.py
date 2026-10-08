"""Broad temporal bands and harmonic pooling, selected by held-speed validation."""
import numpy as np
from .multispeed_readout import energy_pairs


def feature_pairs(grids):
    return energy_pairs(grids, np.arange(1,19,dtype=float))


def from_pairs(pairs, recipe):
    pos,neg=pairs
    if pos.shape!=neg.shape or pos.shape[1:]!=(24,18):raise ValueError('two-family energy tensor required')
    k=np.tile(np.repeat([1,2,3],2),4)[:,None];f=np.arange(1,19)[None]
    if recipe['band']=='temporal':mask=np.broadcast_to(f<=recipe['upper'],(24,18))
    elif recipe['band']=='speed':mask=(f>=.5*k)&(f<=recipe['upper']*k)
    elif recipe['band']=='p8':mask=(f==k)|(f==2*k)
    elif recipe['band']=='trained':mask=(f==k)|(f==2*k)|(f==4*k)
    else:raise ValueError('unknown frequency band')
    p,n=(pos*mask).sum(-1),(neg*mask).sum(-1)
    if recipe['pool']:
        p=p.reshape(-1,2,2,3,2).sum(3).reshape(-1,8)
        n=n.reshape(-1,2,2,3,2).sum(3).reshape(-1,8)
    return (p-n)/np.maximum(p+n,1e-12)


def transform(grids, recipe):
    return from_pairs(feature_pairs(grids),recipe)


def candidates():
    bands=([{'band':'p8'},{'band':'trained'}]+[{'band':'temporal','upper':u} for u in (3,6,12,18)]+
           [{'band':'speed','upper':u} for u in (4,6)])
    return [{**b,'pool':pool} for b in bands for pool in (False,True)]


def rank(result):
    # Explicitly prioritize the weakest leave-one-speed-out fold, then mean.
    return (min(result['held_speed_accuracy']),float(np.mean(result['held_speed_accuracy'])),-result['features'],result['alpha'])
