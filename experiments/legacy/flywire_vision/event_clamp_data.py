"""P12 actual-delivery checks and lossless, per-probe trace storage."""
from pathlib import Path
import numpy as np
from brian2_rust.binary_topology import inspect_csr,csr_arrays
from .motion_refinement import read,sha,load_npz
from .multispeed_data import digest
from .run_experiment import save


def outgoing_graph(artifact,cells):
    model=read(artifact/'model.json');top=model['instance']['synapses'][1]['topology']
    if sha(Path(top['path']))!=top['sha256']:raise ValueError('CSR identity failed')
    offsets,targets,weights=csr_arrays(inspect_csr(top['path']));rows=[]
    for cell in sorted(cells):
        for edge in range(int(offsets[cell]),int(offsets[cell+1])):rows.append((cell,edge,int(targets[edge]),float(weights[0,edge])))
    return {'source':np.array([r[0] for r in rows],dtype=np.int64),'edge':np.array([r[1] for r in rows],dtype=np.int64),
        'target':np.array([r[2] for r in rows],dtype=np.int64),'contacts':np.array([r[3] for r in rows]),'graph_sha256':np.array(top['sha256'])}


def verify_delivery(audit,events,selected,mode,graph):
    indices,ticks=events;expected=[];contacts=[]
    for tick,source in sorted(zip(map(int,ticks),map(int,indices))):
        if source not in selected or tick+18>=6000:continue
        mask=graph['source']==source
        for edge,target,w in zip(graph['edge'][mask],graph['target'][mask],graph['contacts'][mask]):
            expected.append((tick+18,source,int(edge),int(target)));contacts.append(w)
    expected=np.array(expected,dtype=np.uint64).reshape(-1,4)
    if not np.array_equal(audit[:,:4],expected):raise ValueError('actual per-edge deliveries differ from expected source events')
    w=np.asarray(contacts);transmission=0. if mode==2 else 1.
    ge=(.275/52)*(w*(w>0)*transmission);gi=-((.275/52)*4)*(w*(w<0)*transmission)
    values=audit[:,4:].copy().view(np.float64)
    if not (np.array_equal(values[:,0],ge) and np.array_equal(values[:,1],gi)):raise ValueError('contact gain/delay changed')
    if not (np.array_equal(values[:,2]+values[:,0],values[:,3]) and np.array_equal(values[:,4]+values[:,1],values[:,5])):raise ValueError('actual state write differs from logged increment')
    if len(audit)!=len({(int(a[0]),int(a[2])) for a in audit}):raise ValueError('duplicate edge at delivery tick')
    return {'all_passed':True,'delivered_edge_events':len(audit),'actual_write_sha256':digest(audit),'ge_budget':float(ge.sum()),'gi_budget':float(gi.sum())}


def extract(result,audit,cells,monitor_cells,visual,events):
    pop=result['populations'][1];columns=[monitor_cells.index(c) for c in cells];mask=np.isin(pop['indices'],cells)
    return {**{k:np.asarray(v[:,columns]).copy() for k,v in pop['trace'].items()},'cells':np.array(cells,dtype=np.int64),
        'indices':np.asarray(pop['indices'][mask]).copy(),'ticks':np.asarray(pop['spike_ticks'][mask]).copy(),
        'visual_indices':np.asarray(visual[0],dtype=np.int64),'visual_ticks':np.asarray(visual[1],dtype=np.int64),
        'outgoing_indices':np.asarray(events[0],dtype=np.int64),'outgoing_ticks':np.asarray(events[1],dtype=np.int64),'audit':audit.copy()}


def store_trial(directory,key,data,blank_key=None,blank=None):
    encoded={k:np.bitwise_xor(v.view(np.uint64),blank[k].view(np.uint64)) if blank is not None and k in ('v','ge','gi') else v for k,v in data.items()}
    path=directory/(key+'.npz');np.savez_compressed(path,**encoded)
    restored=load_npz(path)
    if blank is not None:
        for k in ('v','ge','gi'):restored[k]=np.bitwise_xor(restored[k],blank[k].view(np.uint64)).view(np.float64)
    hashes={k:digest(v) for k,v in data.items()}
    if hashes!={k:digest(v) for k,v in restored.items()}:raise ValueError('lossless archive mismatch')
    return {'key':key,'blank_key':blank_key,'archive_sha256':sha(path),'array_sha256':hashes,'array_shapes':{k:list(v.shape) for k,v in data.items()}}


def load_trial(directory,row):
    data=load_npz(directory/(row['key']+'.npz'))
    if row['blank_key']:
        blank=load_npz(directory/(row['blank_key']+'.npz'))
        for k in ('v','ge','gi'):data[k]=np.bitwise_xor(data[k],blank[k].view(np.uint64)).view(np.float64)
    return data
