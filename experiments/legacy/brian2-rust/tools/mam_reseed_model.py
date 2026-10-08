"""Change only random inputs of the full MAM, preserving definition and run.

The legacy initial-voltage realization must first reproduce bit for bit. New
replicate keys have explicit domain separation; no scientific outcomes are read.
"""
import hashlib
import json
import re
import struct
import numpy as np
from mam_replicate_seeds import allocation,registry


def need(ok,message):
    if not ok:raise ValueError(message)


def voltage_bits(random,parameters,count):
    p=parameters['params']['neuron_params']
    # Preserve the original adapter's normal draw order and mV-to-SI operation.
    values=random.normal(p['V0_mean'],p['V0_sd'],count)*0.001
    return [struct.pack('>d',float(v)).hex() for v in values]


def check_geometry(model,parameters):
    pops=model['definition']['populations'];syn=model['definition']['synapses']
    lookup={'mam_'+p['name'].replace('-','_'):p for p in parameters['populations']}
    need(len(lookup)==len(pops) and set(lookup)=={p['name'] for p in pops},'population identity mismatch')
    need(all(lookup[p['name']]['count']==p['count'] for p in pops),'population count mismatch')
    need(len(model['instance']['populations'])==len(pops),'population instance coverage')
    need(len(parameters['projections'])==len(syn)==len(model['instance']['synapses']),'projection coverage')
    ordinals=[]
    for d,s in zip(syn,model['instance']['synapses'],strict=True):
        m=re.fullmatch(r'mam_projection_(\d+)',d['name']);need(m is not None,'projection name')
        ordinal=int(m[1]);need(ordinal<len(syn),'projection ordinal')
        q=parameters['projections'][ordinal];ordinals.append(ordinal)
        for side in ['source','target']:
            wanted='mam_'+parameters['populations'][q[side]]['name'].replace('-','_')
            need(pops[d[side+'_population']]['name']==wanted,'projection endpoint identity')
        need(s['topology']['kind']=='fixed_total' and s['topology']['edge_count']==q['count'],'projection recipe/count')
    need(sorted(ordinals)==list(range(len(syn))),'projection ordinals not unique and complete')
    return ordinals


def audit_delta(old,new,parameters,keys):
    """Reject every field difference outside the three declared random inputs."""
    need(set(old)==set(new),'model key changes')
    for name in set(old)-{'instance','protocol'}:need(old[name]==new[name],'model definition/run changed')
    a,b=old['instance'],new['instance']
    need(set(a)==set(b),'instance key changes')
    need(len(a['populations'])==len(b['populations']) and len(a['synapses'])==len(b['synapses']),'instance coverage changed')
    for name in set(a)-{'rng_seed','populations','synapses'}:need(a[name]==b[name],'other instance field changed')
    need(b['rng_seed']==keys['runtime_input'],'runtime random key mismatch')
    lookup={d['name']:i for i,d in enumerate(old['definition']['populations'])}
    random=np.random.default_rng(keys['initial_voltage'])
    for p in parameters['populations']:
        i=lookup['mam_'+p['name'].replace('-','_')];x,y=a['populations'][i],b['populations'][i]
        need(set(x)==set(y),'population field changes')
        need({k:v for k,v in x.items() if k!='initial_state'}=={k:v for k,v in y.items() if k!='initial_state'},'non-voltage population change')
        sx,sy=x['initial_state'],y['initial_state']
        need(set(sx)==set(sy) and {k:v for k,v in sx.items() if k!='v'}=={k:v for k,v in sy.items() if k!='v'},'non-voltage state change')
        need(sy['v']==voltage_bits(random,parameters,p['count']),'new voltage realization differs')
    for x,y,ordinal in zip(a['synapses'],b['synapses'],check_geometry(old,parameters),strict=True):
        need(set(x)==set(y) and {k:v for k,v in x.items() if k!='topology'}=={k:v for k,v in y.items() if k!='topology'},'synaptic parameters or delays changed')
        tx,ty=x['topology'],y['topology']
        need(set(tx)==set(ty) and {k:v for k,v in tx.items() if k!='seed'}=={k:v for k,v in ty.items() if k!='seed'},'topology distribution changed')
        need(ty['seed']==keys['projections'][ordinal],'projection random key differs')
    return dict(only_declared_random_inputs_changed=True,definition_exact=True,run_exact=True,
                initial_voltage_neurons=sum(p['count'] for p in parameters['populations']),
                projection_keys=len(keys['projections']),runtime_key=keys['runtime_input'])


def reseed(old,parameters,replicate):
    from brian2_rust.protocol import attach_protocol,verify_protocol
    verify_protocol(old);ordinals=check_geometry(old,parameters)
    need(old['instance']['rng_seed']==1729,'frozen legacy seed required')
    keys=allocation(replicate,len(ordinals))
    random=np.random.default_rng(1729)
    lookup={d['name']:i for i,d in enumerate(old['definition']['populations'])}
    for p in parameters['populations']:
        i=lookup['mam_'+p['name'].replace('-','_')]
        need(old['instance']['populations'][i]['initial_state']['v']==voltage_bits(random,parameters,p['count']),
             'legacy voltage realization does not reproduce; no new artifact may be published')
    need(all(s['topology']['seed']==1729+i for s,i in zip(old['instance']['synapses'],ordinals,strict=True)),'legacy projection keys differ')
    new=dict(old,instance=dict(old['instance']))
    new['instance']['rng_seed']=keys['runtime_input']
    new['instance']['populations']=[dict(p,initial_state=dict(p['initial_state'])) for p in old['instance']['populations']]
    random=np.random.default_rng(keys['initial_voltage'])
    for p in parameters['populations']:
        i=lookup['mam_'+p['name'].replace('-','_')]
        new['instance']['populations'][i]['initial_state']['v']=voltage_bits(random,parameters,p['count'])
    new['instance']['synapses']=[dict(s,topology=dict(s['topology'],seed=keys['projections'][i]))
                               for s,i in zip(old['instance']['synapses'],ordinals,strict=True)]
    delta=audit_delta(old,new,parameters,keys)
    attach_protocol(new);verify_protocol(new)
    need(new['protocol']['layers']['definition']==old['protocol']['layers']['definition']
         and new['protocol']['layers']['run']==old['protocol']['layers']['run']
         and new['protocol']['layers']['instance']!=old['protocol']['layers']['instance'],'layer identity differs')
    return new,dict(keys=keys,delta=delta,legacy_voltage_reproduced_exactly=True,registry=registry(),
                    numpy_version=np.__version__,bit_generator='PCG64',neural_runs=0)
