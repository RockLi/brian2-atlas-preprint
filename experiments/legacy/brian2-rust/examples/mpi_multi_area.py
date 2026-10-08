"""Build a declared, scaled MAM area subset from pinned official parameters.

This is a Brian2 discretization adapter, not a validated NEST reproduction.
Unsimulated areas use homogeneous 10 Hz Poisson replacement, with the official
mean weights, indegrees and distance delays. External Poisson counts are exact
per-bin distributions; no Bernoulli or Gaussian approximation is used.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import brian2 as b
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import brian2_rust
from brian2_rust.export import lower_network
from brian2_rust.distributed import write_mpi_project,compile_mpi_project
from brian2_rust.resource_limits import initial_value_budget,ir_byte_budget,neuron_budget
from brian2_rust.multi_area_semantics import nest_grid_refractory_ms,nest_poisson_gate_tick


def make_model(parameters,areas,steps=100,seed=1729,*,max_neurons=100000,max_recurrent_edges=100000000,nest_grid=False,nest_poisson_start=False):
    if parameters['schema']!='b2-official-mam-parameters-v1':raise ValueError('unknown parameter schema')
    if not areas or len(areas)!=len(set(areas)) or set(areas)-{p['area'] for p in parameters['populations']}:
        raise ValueError('unknown or duplicate areas')
    if not 1<=steps<=1000:raise ValueError('pilot limited to 1..1000 ticks')
    if any(type(value) is not int or value<=0 for value in [max_neurons,max_recurrent_edges]):
        raise ValueError('preparation budgets must be positive integers')
    selected={i:p for i,p in enumerate(parameters['populations']) if p['area'] in areas}
    if sum(p['count'] for p in selected.values())>max_neurons:raise ValueError('pilot neuron preparation budget exceeded before allocation')
    # Two states and two refractory arrays are required for every MAM neuron.
    # Scalar parameters add a little more; the exporter checks the exact total.
    if 4*sum(p['count'] for p in selected.values())>initial_value_budget():raise ValueError('initial-value preparation budget exceeded before allocation')
    projections=[p for p in parameters['projections'] if p['source'] in selected and p['target'] in selected]
    if sum(p['count'] for p in projections)>max_recurrent_edges:raise ValueError('pilot recurrent-edge preparation budget exceeded before allocation')
    cell=parameters['params']['neuron_params']['single_neuron_dict']
    if cell['tau_syn_ex']!=cell['tau_syn_in']:raise ValueError('combined current requires equal synaptic time constants')
    clock=b.Clock(dt=0.1*b.ms)
    refractory_ms=nest_grid_refractory_ms(cell['t_ref'],0.1) if nest_grid else cell['t_ref']
    random=np.random.default_rng(seed)
    groups={};objects=[];names={}
    for index,p in selected.items():
        name='mam_'+p['name'].replace('-','_')
        namespace={'EL':cell['E_L']*b.mV,'tau_m':cell['tau_m']*b.ms,'tau_s':cell['tau_syn_ex']*b.ms,
                   'capacitance':cell['C_m']*b.pF,'dc':p['dc_pA']*b.pA,
                   'threshold_v':cell['V_th']*b.mV,'reset_v':cell['V_reset']*b.mV}
        group=b.NeuronGroup(p['count'],'dv/dt=(EL-v)/tau_m+(current+dc)/capacitance:volt (unless refractory)\ndcurrent/dt=-current/tau_s:amp',
                            threshold='v>=threshold_v',reset='v=reset_v',refractory=refractory_ms*b.ms,
                            namespace=namespace,clock=clock,name=name)
        init=parameters['params']['neuron_params']
        group.v=random.normal(init['V0_mean'],init['V0_sd'],p['count'])*b.mV
        # Superposition is exact only for identical weight and delay bin.
        # NEST static external connections default to 1 ms. The explicit
        # startup option accounts for the generator's strict start boundary.
        inputs=defaultdict(float)
        inputs[(p['external_weight_pA'],10)]+=p['external_indegree']*10.0
        for projection in parameters['projections']:
            if projection['target']==index and projection['source'] not in selected:
                delay=int(np.floor(projection['delay_mean_ms']/0.1+0.5))
                inputs[(projection['weight_mean_pA'],delay)]+=projection['indegree']*10.0
        terms=[f'int(timestep(t, dt)>={nest_poisson_gate_tick(delay) if nest_poisson_start else delay})*({weight!r}*pA)*poisson({rate*0.0001!r})'
               for (weight,delay),rate in sorted(inputs.items()) if rate>0]
        group.run_regularly('current += '+' + '.join(terms),when='synapses')
        groups[index]=group;names[name]=p['area'];objects += [group,b.SpikeMonitor(group),b.StateMonitor(group,'v',record=[0])]
    for ordinal,p in enumerate(projections):
        syn=b.Synapses(groups[p['source']],groups[p['target']],'w:amp (constant)',on_pre='current_post+=w',clock=clock,name=f'mam_projection_{ordinal}')
        mean,sd=p['weight_mean_pA']*b.pA,p['weight_sd_pA']*b.pA
        weight=brian2_rust.ClippedNormal(mean,sd,minimum=0*b.pA if p['excitatory'] else None,maximum=None if p['excitatory'] else 0*b.pA)
        delay=brian2_rust.ClippedNormal(p['delay_mean_ms']*b.ms,p['delay_sd_ms']*b.ms,minimum=clock.dt)
        brian2_rust.connect_fixed_total(syn,p['count'],seed=seed+ordinal,initializers={'w':weight},delay_initializer=delay)
        objects.append(syn)
    model=lower_network(b.Network(*objects),steps*clock.dt,rng_seed=seed)
    # Export order is canonical and may differ from source dictionary order.
    owners=[areas.index(names[p['name']]) for p in model['definition']['populations']]
    return model,owners


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parameters',type=Path,required=True);p.add_argument('--areas',nargs='+',default=['V1','V2'])
    p.add_argument('--steps',type=int,default=100);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nest-grid',action='store_true',help='Opt-in NEST refractory grid translation validated on fixed-input cells; full stochastic MAM equivalence remains open')
    p.add_argument('--nest-poisson-start',action='store_true',help='Opt-in NEST Poisson generator start=0 arrival-bin translation; independent of refractory mapping')
    p.add_argument('--max-neurons',type=int,default=100000,help='Explicit preparation budget; runtime admission is separate')
    p.add_argument('--max-recurrent-edges',type=int,default=100000000,help='Explicit preparation budget; runtime admission is separate')
    p.add_argument('--max-initial-values',type=int,help='Explicit B2IR initialization-value budget; default 4 million or B2_MAX_INITIAL_VALUES')
    p.add_argument('--max-ir-bytes',type=int,help='Explicit validator JSON-byte budget; default 128 MiB or B2_MAX_IR_BYTES')
    p.add_argument('--opt-level',type=int,choices=range(4),default=3)
    p.add_argument('--compact-projections',action='store_true',help='Opt-in table and ordered-loop emission for homogeneous fixed-total projections')
    p.add_argument('--compact-populations',action='store_true',help='Aggregate population storage; also enables projection compaction')
    p.add_argument('--panic-strategy',choices=['unwind','abort'],help='Explicit Rust panic behavior; abort exits immediately without unwinding')
    p.add_argument('--mpicc',default='mpicc');p.add_argument('--rustc',default='rustc');p.add_argument('--compile',action='store_true')
    a=p.parse_args()
    os.environ['B2_MAX_NEURONS']=str(a.max_neurons)
    if a.max_initial_values is not None:os.environ['B2_MAX_INITIAL_VALUES']=str(a.max_initial_values)
    if a.max_ir_bytes is not None:os.environ['B2_MAX_IR_BYTES']=str(a.max_ir_bytes)
    initial_value_budget();ir_byte_budget();neuron_budget()
    a.output.mkdir(parents=True,exist_ok=False)
    b.set_device('rust_standalone',runner=ROOT/'target/release/b2-runner')
    parameters=json.loads(a.parameters.read_text());started=time.monotonic()
    model,owners=make_model(parameters,a.areas,a.steps,max_neurons=a.max_neurons,max_recurrent_edges=a.max_recurrent_edges,nest_grid=a.nest_grid,nest_poisson_start=a.nest_poisson_start)
    (a.output/'model.json').write_text(json.dumps(model)+'\n')
    write_mpi_project(model,a.output/'mpi',ranks=len(a.areas),population_owners=owners,
                      compact_projections=a.compact_projections,compact_populations=a.compact_populations)
    report={'schema':'b2-mam-adapter-pilot-v1','official_parameters_sha256':hashlib.sha256(a.parameters.read_bytes()).hexdigest(),
            'areas':a.areas,'N_scaling':parameters['N_scaling'],'K_scaling':parameters['K_scaling'],
            'replacement':'hom_poisson_stat at 10 Hz; mean weights and distance delays',
            'integration':'Brian2 exact linear state update, dt=0.1 ms; cross-NEST timing/statistical validation pending',
            'nest_grid_requested':a.nest_grid,
            'nest_poisson_start_requested':a.nest_poisson_start,
            'biological_refractory_ms':parameters['params']['neuron_params']['single_neuron_dict']['t_ref'],
            'brian_refractory_ms':nest_grid_refractory_ms(parameters['params']['neuron_params']['single_neuron_dict']['t_ref'],0.1) if a.nest_grid else parameters['params']['neuron_params']['single_neuron_dict']['t_ref'],
            'neurons':sum(p['count'] for p in model['definition']['populations']),
            'recurrent_edges':sum(s['topology']['edge_count'] for s in model['instance']['synapses']),
            'projections':len(model['definition']['synapses']),'population_owners':owners,
            'compact_projections_requested':a.compact_projections,
            'compact_populations_requested':a.compact_populations,
            'preparation_budgets':{'max_neurons':a.max_neurons,'max_recurrent_edges':a.max_recurrent_edges,
                                   'max_initial_values':initial_value_budget(),'max_ir_bytes':ir_byte_budget()},
            'prepare_seconds':time.monotonic()-started,'compiled':False}
    (a.output/'adapter-report.json').write_text(json.dumps(report,indent=2)+'\n')
    if a.compile:
        started=time.monotonic();compile_mpi_project(a.output/'mpi',mpicc=a.mpicc,rustc=a.rustc,opt_level=a.opt_level,panic_strategy=a.panic_strategy)
        report.update(compiled=True,opt_level=a.opt_level,compile_seconds=time.monotonic()-started)
        (a.output/'adapter-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
