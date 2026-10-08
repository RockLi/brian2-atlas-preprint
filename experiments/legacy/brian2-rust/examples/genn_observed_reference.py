"""Independent recurrence projected onto GeNN's observed execution phases."""
import numpy as np
from gpu_stdp_compare import DT, topology


def observed_reference(neurons, degree, steps, watched, selected, *, drive, delay_span,
                       post_delay, topology_kind, topology_seed):
    source,target,delay=topology(neurons,degree,delay_span=delay_span,
        topology_kind=topology_kind,topology_seed=topology_seed)
    v=np.arange(neurons)%16/16;w=np.full(len(source),.25)
    ap=np.zeros(len(source));ao=ap.copy();last=ap.copy()
    history=np.zeros((steps,neurons),bool)
    order=np.lexsort((np.arange(len(source)),source,-delay))
    record={'v_start':[],'v_after_reset':[],'previous_spike':[],
            'w':[w[selected].copy()],'Apre':[ap[selected].copy()],
            'Apost':[ao[selected].copy()],'lastupdate':[last[selected].copy()]}
    for tick in range(steps):
        v+=drive;record['v_start'].append(v[watched].copy());v+=(-1/32)*v
        fired=v>1;history[tick]=fired
        record['v_after_reset'].append(np.where(fired,0,v)[watched])
        record['previous_spike'].append(fired[watched].astype(int))
        pre=order[(tick>=delay[order]) & history[np.maximum(tick-delay[order],0),source[order]]]
        post=np.flatnonzero(history[tick-post_delay,target]) if tick>=post_delay else np.empty(0,np.int64)
        for events,is_pre in ((pre,True),(post,False)):
            elapsed=tick*DT-last[events]
            ap[events]*=np.exp(-elapsed/(16*DT));ao[events]*=np.exp(-elapsed/(32*DT))
            if is_pre:
                np.add.at(v,target[events],w[events]/16);ap[events]+=.0078125
                w[events]=np.clip(w[events]+ao[events],0,.5)
            else:
                ao[events]-=.00390625;w[events]=np.clip(w[events]+ap[events],0,.5)
            last[events]=tick*DT
        v[fired]=0
        for key,values in [('w',w),('Apre',ap),('Apost',ao),('lastupdate',last)]:
            record[key].append(values[selected].copy())
    record['v_start'].append(record['v_start'][-1].copy())
    record['v_after_reset'].append(v[watched].copy())
    record['previous_spike'].append(np.zeros(len(watched),int))
    return {key:np.asarray(values) for key,values in record.items()}
