"""Observe actual MPI MAM builders, with bounded, explicitly instrumented dumps.

The observer is inserted after initialization and before original-edge IDs are
dropped. Original source/manifest and the exact observer diff remain archived.
No production sampler or runtime source is modified by this tool.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def decode(bits):return struct.unpack('>d',bytes.fromhex(bits))[0]
def rust_float(bits):return f'f64::from_bits(0x{bits}u64)'
def rust_bound(bits):return 'None' if bits is None else f'Some({rust_float(bits)})'


def prepare(parameters,out,runner):
    import brian2 as b
    import brian2_rust
    spec=importlib.util.spec_from_file_location('mam_audit_adapter',ROOT/'examples/mpi_multi_area.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    p=json.loads(parameters.read_text());assert p['N_scaling']==p['K_scaling']==.01
    b.set_device('rust_standalone',runner=runner)
    model,owners=adapter.make_model(p,['V1','V2'],steps=1,seed=20260908,
        max_neurons=5000,max_recurrent_edges=200000,nest_grid=True,nest_poisson_start=True)
    (out/'model.json').write_text(json.dumps(model)+'\n');(out/'owners.json').write_text(json.dumps(owners)+'\n')
    (out/'parameters.json').write_bytes(parameters.read_bytes())
    # Call the actual reference library functions used by executor.rs, rather
    # than implementing an independent Python version of the same RNG.
    library=ROOT/'src/large_topology.rs'
    lines=[f'#[path={json.dumps(str(library))}] mod topology;',
        'use std::io::Write; fn main()->Result<(),Box<dyn std::error::Error>>{',
        'let out=std::path::PathBuf::from(std::env::args().nth(1).ok_or("output required")?);']
    for q,(d,v) in enumerate(zip(model['definition']['synapses'],model['instance']['synapses'],strict=True)):
        t=v['topology'];assert t['kind']=='fixed_total' and len(d['parameters'])==1 and d['parameters'][0]['name']=='w'
        w=t['initializers']['w'];delay=v['pathways'][0]['delay_initializer'];dt=model['definition']['populations'][d['source_population']]['dt']
        assert w['kind']==delay['kind']=='clipped_normal'
        def draw(x):return f'topology::deterministic_clipped_normal({t["seed"]}u64,{x["stream"]}u64,edge,{rust_float(x["mean"])},{rust_float(x["std"])},{rust_bound(x["minimum"])},{rust_bound(x["maximum"])})?'
        lines += ['{',f'let (s,t)=topology::build_fixed_total_indices({d["source_count"]},{d["target_count"]},{t["edge_count"]},{t["seed"]}u64)?;',
            f'let mut f=std::io::BufWriter::new(std::fs::File::create_new(out.join("q{q}.bin"))?);',
            f'for edge in 0..s.len(){{ let w={draw(w)};let delay=({draw(delay)}/{rust_float(dt)}+0.5).floor()as u64;',
            'for value in [edge as u64,s[edge]as u64,t[edge]as u64,w.to_bits(),delay]{f.write_all(&value.to_le_bytes())?;}}f.flush()?;}']
    lines.append('Ok(())}')
    (out/'reference.rs').write_text('\n'.join(lines)+'\n');(out/'reference').mkdir()
    subprocess.run(['rustc','--edition=2021','-Awarnings','-C','opt-level=1',str(out/'reference.rs'),'-o',str(out/'reference-builder')],check=True,timeout=90)
    subprocess.run([str(out/'reference-builder'),str(out/'reference')],check=True,timeout=30)
    receipt=dict(parameters_sha256=sha(parameters),model_sha256=sha(out/'model.json'),reference_library_sha256=sha(library),
        reference_source_sha256=sha(out/'reference.rs'),reference_executable_sha256=sha(out/'reference-builder'))
    (out/'prepare.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


def run_project(fixture,out,runner,layout):
    from brian2_rust.distributed import write_mpi_project,compile_mpi_project,run_mpi_project
    model=json.loads((fixture/'model.json').read_text());ranks=2 if layout in ['shard2','owner2'] else 4
    owners=json.loads((fixture/'owners.json').read_text()) if layout=='owner2' else None
    project=out/'mpi';dump=out/'dump';dump.mkdir()
    write_mpi_project(model,project,ranks=ranks,runner=runner,population_owners=owners,
        compact_populations=layout=='compact4',prebuild_shared_topology=layout=='compact4')
    source=(project/'main.rs').read_text();original=source
    for q,v in enumerate(model['instance']['synapses'] if layout!='compact4' else []):
        assert v['topology']['edge_count']<=200000
        anchor=f'    drop(s{q}_original_edges);';assert source.count(anchor)==1,(q,'observer anchor missing')
        observer=f'''    {{
        check(s{q}_local_edge_count<=200000,"topology observer edge limit")?;
        let path=std::path::PathBuf::from(std::env::var("B2_MAM_TOPOLOGY_DUMP")?).join(format!("q{q}.rank{{}}.bin",mpi.rank));
        let mut f=std::io::BufWriter::new(std::fs::File::create_new(path)?);
        for source in 0..s{q}_offsets.len().saturating_sub(1) {{
            for edge in s{q}_offsets[source]..s{q}_offsets[source+1] {{
                for value in [s{q}_original_edges[edge] as u64,source as u64,s{q}_target_index[edge] as u64,s{q}_parameter_0[edge].to_bits(),s{q}_delay_ticks[edge] as u64] {{
                    std::io::Write::write_all(&mut f,&value.to_le_bytes())?;
                }}
            }}
        }}
        std::io::Write::flush(&mut f)?;
    }}
'''
        source=source.replace(anchor,observer+anchor,1)
    if layout=='compact4':
        # Compaction funnels the same arrays through one generic loader.
        # Unique recipe seeds recover the original projection ordinal.
        seeds=[v['topology']['seed'] for v in model['instance']['synapses']]
        assert len(seeds)==len(set(seeds))
        cases=','.join(f'{seed}u64=>{q}usize' for q,seed in enumerate(seeds))
        anchor='    drop(original_edges);';assert source.count(anchor)==1
        observer=f'''    {{
        check(local_edge_count<=200000,"topology observer edge limit")?;
        let q=match seed{{{cases},_=>return Err("unknown topology observer seed".into())}};
        let path=std::path::PathBuf::from(std::env::var("B2_MAM_TOPOLOGY_DUMP")?).join(format!("q{{}}.rank{{}}.bin",q,mpi.rank));
        let mut f=std::io::BufWriter::new(std::fs::File::create_new(path)?);
        for source in 0..offsets.len().saturating_sub(1) {{
            for edge in offsets[source]..offsets[source+1] {{
                for value in [original_edges[edge] as u64,source as u64,target_index[edge] as u64,parameter_0[edge].to_bits(),delay_0[edge] as u64] {{
                    std::io::Write::write_all(&mut f,&value.to_le_bytes())?;
                }}
            }}
        }}
        std::io::Write::flush(&mut f)?;
    }}
'''
        source=source.replace(anchor,observer+anchor,1)
    (out/'original-main.rs').write_text(original)
    (out/'observer.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='original-main.rs',tofile='mpi/main.rs')))
    (project/'main.rs').write_text(source)
    manifest_path=project/'manifest.json';(out/'original-manifest.json').write_bytes(manifest_path.read_bytes())
    manifest=json.loads(manifest_path.read_text());manifest['files']['main.rs']=sha(project/'main.rs')
    manifest['observer_instrumentation']=dict(original_source_sha256=sha(out/'original-main.rs'),diff_sha256=sha(out/'observer.diff'),scope='Post-initialization read-only array dump; production sampler functions unchanged')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    compile_mpi_project(project,opt_level=1,panic_strategy='abort')
    import os
    os.environ['B2_MAM_TOPOLOGY_DUMP']=str(dump)
    report=run_mpi_project(project,out/'result',timeout=120)
    (out/'probe.json').write_text(json.dumps(dict(layout=layout,ranks=ranks,plan_sha256=report['plan_sha256'],observer=manifest['observer_instrumentation']),indent=2)+'\n')


def read(path):
    assert path.stat().st_size%40==0 and path.stat().st_size<=8_000_000
    return np.fromfile(path,dtype='<u8').reshape(-1,5)


def distribution(model,arrays):
    mean_delta=0.;degree_variance=0.;degree_df=0;correlation=0.;total=0;autapses=0;auto_mean=0.;auto_var=0.;duplicates=0
    thresholds=[-2.,-1.,0.,1.,2.];cdf={key:dict(observed=0,expected=0.,variance=0.) for kind in ['weight','delay'] for key in [kind+':'+str(x) for x in thresholds]}
    def normal_cdf(z):return .5*(1+math.erf(z/math.sqrt(2)))
    def truncated(x,descriptor):
        mu,sd=decode(descriptor['mean']),decode(descriptor['std'])
        low=-math.inf if descriptor['minimum'] is None else decode(descriptor['minimum'])
        high=math.inf if descriptor['maximum'] is None else decode(descriptor['maximum'])
        if x<low:return 0.
        if x>=high:return 1.
        if sd==0:return float(x>=mu)
        a=normal_cdf((low-mu)/sd);b=normal_cdf((high-mu)/sd)
        return (normal_cdf((x-mu)/sd)-a)/(b-a)
    for d,v,a in zip(model['definition']['synapses'],model['instance']['synapses'],arrays,strict=True):
        e=len(a);s=a[:,1].astype(np.int64);t=a[:,2].astype(np.int64);w=a[:,3].view('<f8');delay=a[:,4].astype(np.int64)
        ns,nt=d['source_count'],d['target_count'];assert e==v['topology']['edge_count'] and np.all(s<ns) and np.all(t<nt)
        total+=e;duplicates+=e-len(np.unique(s*nt+t))
        for indices,count in [(s,ns),(t,nt)]:
            degree=np.bincount(indices,minlength=count);chi=float(np.sum((degree-e/count)**2/(e/count)))
            degree_df+=count-1;mean_delta+=chi-(count-1);degree_variance+=2*(count-1)*(e-1)/e
        if ns>1 and nt>1:correlation+=float(np.sum((s-(ns-1)/2)*(t-(nt-1)/2)))/math.sqrt((ns*ns-1)*(nt*nt-1)/144)
        if d['source_population']==d['target_population']:
            assert d['source_start']==d['target_start']==0 and ns==nt
            autapses+=int(np.sum(s==t));auto_mean+=e/ns;auto_var+=e/ns*(1-1/ns)
        dt=decode(model['definition']['populations'][d['source_population']]['dt'])
        for kind,descriptor,values in [('weight',v['topology']['initializers']['w'],w),('delay',v['pathways'][0]['delay_initializer'],delay)]:
            mu,sd=decode(descriptor['mean']),decode(descriptor['std']);assert sd>0
            assert np.isfinite(values).all()
            if kind=='weight':
                if descriptor['minimum'] is not None:assert np.all(values>=decode(descriptor['minimum']))
                if descriptor['maximum'] is not None:assert np.all(values<=decode(descriptor['maximum']))
            else:assert np.all(values>=1)
            for z in thresholds:
                x=mu+z*sd
                if kind=='weight':prob=truncated(x,descriptor);observed=int(np.sum(values<=x))
                else:
                    tick=math.floor(x/dt);prob=truncated((tick+.5)*dt,descriptor);observed=int(np.sum(values<=tick))
                row=cdf[kind+':'+str(z)];row['observed']+=observed;row['expected']+=e*prob;row['variance']+=e*prob*(1-prob)
    for row in cdf.values():
        # A threshold below the minimum permitted delay has probability zero.
        # Check such deterministic boundaries exactly, not with a 0/0 score.
        row['z']=None if row['variance']==0 else (row['observed']-row['expected'])/math.sqrt(row['variance'])
        row['passed']=row['observed']==row['expected'] if row['variance']==0 else abs(row['z'])<6
    result=dict(edges=total,degree_pearson_z=mean_delta/math.sqrt(degree_variance),degree_df=degree_df,
        endpoint_covariance_z=correlation/math.sqrt(total),autapses=autapses,expected_autapses=auto_mean,
        autapse_z=(autapses-auto_mean)/math.sqrt(auto_var),duplicate_pairs=duplicates,cdf_checks=cdf)
    result['passed']=all(abs(z)<6 for z in [result['degree_pearson_z'],result['endpoint_covariance_z'],result['autapse_z']]) and all(r['passed'] for r in cdf.values())
    return result


def analyze(fixture,out,nest_fixture):
    model=json.loads((fixture/'model.json').read_text());gold=[read(fixture/'reference'/f'q{q}.bin') for q in range(len(model['definition']['synapses']))]
    parity=[]
    for layout in ['shard2','shard4','owner2','compact4']:
        info=json.loads((out/layout/'probe.json').read_text());ranks=info['ranks']
        for q,(d,v,expected) in enumerate(zip(model['definition']['synapses'],model['instance']['synapses'],gold,strict=True)):
            shards=[read(out/layout/'dump'/f'q{q}.rank{rank}.bin') for rank in range(ranks)]
            size=model['definition']['populations'][d['target_population']]['count']
            owners=json.loads((fixture/'owners.json').read_text()) if layout=='owner2' else None
            for rank,a in enumerate(shards):
                target=a[:,2]+d['target_start']
                assert np.all((target>=size*rank//ranks)&(target<size*(rank+1)//ranks)) if owners is None else (len(a)==0 or owners[d['target_population']]==rank)
            merged=np.concatenate(shards);merged=merged[np.argsort(merged[:,0],kind='stable')]
            assert np.array_equal(merged[:,0],np.arange(len(expected)))
            assert np.array_equal(merged,expected),(layout,q,'snapshot mismatch')
        parity.append(dict(layout=layout,projections=len(gold),all_edge_fields_exact=True))
    meta=json.loads((nest_fixture/'fixture.json').read_text());assert sha(nest_fixture/'fixture.npz')==meta['fixture_sha256']
    n=dict(np.load(nest_fixture/'fixture.npz'));mapping={'mam_'+p['name'].replace('-','_'):p for p in meta['populations']};nest_arrays=[]
    for d in model['definition']['synapses']:
        source=mapping[model['definition']['populations'][d['source_population']]['name']];target=mapping[model['definition']['populations'][d['target_population']]['name']]
        mask=(n['source']>=source['start'])&(n['source']<source['end'])&(n['target']>=target['start'])&(n['target']<target['end']);e=int(mask.sum());a=np.empty((e,5),dtype='<u8')
        a[:,0]=np.arange(e);a[:,1]=n['source'][mask]-source['start'];a[:,2]=n['target'][mask]-target['start'];a[:,3]=(n['weight_pa'][mask]*1e-12).view('<u8');a[:,4]=n['delay_ticks'][mask];nest_arrays.append(a)
    stats={'rust':distribution(model,gold),'nest':distribution(model,nest_arrays)}
    result=dict(scope='Full-array parity of bounded actual MPI builders and limited distribution diagnostics; not exhaustive RNG or full-model equivalence.',
        model_sha256=sha(fixture/'model.json'),nest_fixture_sha256=meta['fixture_sha256'],parity=parity,statistics=stats,
        limits='Predeclared |z|<6: aggregated uniform endpoint degree Pearson statistic, endpoint covariance, autapses, and 5 CDF thresholds each for truncated-normal weights and quantized delays. Duplicate pairs reported without an acceptance gate.',
        passed=all(s['passed'] for s in stats.values()))
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit('Topology distribution diagnostic failed; raw dumps retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','shard2','shard4','owner2','compact4','analyze'])
    parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parameters',type=Path);parser.add_argument('--runner',type=Path);parser.add_argument('--nest-fixture',type=Path)
    args=parser.parse_args()
    if args.mode=='analyze':analyze(args.fixture,args.output,args.nest_fixture)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        if args.mode=='prepare':prepare(args.parameters,args.output,args.runner)
        else:run_project(args.fixture,args.output,args.runner,args.mode)
