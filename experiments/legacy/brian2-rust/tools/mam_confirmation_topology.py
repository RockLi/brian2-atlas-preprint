"""Recount only shared-population incoming edges for the new immutable seed.

The old tested integer counter is reused unchanged; its helpers must exactly
match the newly compiled model source. This is not a neural simulation.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def run(root):
    started=time.monotonic();identity=json.loads((root/'identity.json').read_text());source=Path(identity['source_artifact'])
    need(type(identity['replicate']) is int and identity['replicate'] in (1750,1751)
         and identity['neural_launch_admitted'] is False,'candidate identity')
    need(sha(source/'model.json')==identity['model_sha256'],'new model changed')
    need(sha(source/'mpi/execution-plan.json')==identity['files']['mpi/execution-plan.json']['sha256'],'new plan changed')
    need(sha(source/'mpi/main.rs')==identity['files']['mpi/main.rs']['sha256'],'new generated source changed')
    counter=(root/'count.rs').read_text();program=(source/'mpi/main.rs').read_text()
    names=['fn topology_mix64(','fn topology_draw(','fn topology_bounded(']
    helpers=[]
    for name in names:
        a=[line for line in counter.splitlines() if line.startswith(name)]
        b=[line for line in program.splitlines() if line.startswith(name)]
        need(len(a)==len(b)==1 and a==b,'counter differs from current generated topology helper')
        helpers.extend(a)
    model=json.loads((source/'model.json').read_text());plan=json.loads((source/'mpi/execution-plan.json').read_text())
    need(model['protocol']['layers']['instance']==identity['instance_sha256'] and plan['instance_sha256']==identity['instance_sha256'],'instance layers differ')
    owners=plan['population_owners'];need(len(owners)==254 and owners.count(None)==1 and owners[88] is None,'shared population placement changed')
    recipes=[]
    for q,(d,s) in enumerate(zip(model['definition']['synapses'],model['instance']['synapses'],strict=True)):
        if owners[d['target_population']] is not None:continue
        t=s['topology'];need(t['kind']=='fixed_total','unsupported topology')
        ordinal=int(d['name'].removeprefix('mam_projection_'))
        need(t['seed']==identity['random_keys']['projections'][ordinal],'projection random key mismatch')
        recipes.append(dict(q=q,population=d['target_population'],seed=t['seed'],edges=t['edge_count'],
            sources=d['source_count'],targets=d['target_count'],target_start=d['target_start'],
            population_size=model['definition']['populations'][d['target_population']]['count']))
    need(0<len(recipes)<=71 and sum(r['edges'] for r in recipes)<=1100000000,'counting work exceeds declared budget')
    layers=model['protocol']['layers'];del model,program
    write(root/'recipes.json',recipes)
    names=['q','population','seed','edges','sources','targets','target_start','population_size']
    (root/'recipes.tsv').open('x').write('\n'.join(' '.join(str(r[n]) for n in names) for r in recipes)+'\n')
    rustc='/atlas-home/0003/workspace/brian2-lk-20260906/.cargo/bin/rustc'
    subprocess.run([rustc,'-C','opt-level=3','--edition=2021',str(root/'count.rs'),'-o',str(root/'count')],check=True,timeout=30)
    begin=time.monotonic()
    with (root/'exact-counts.jsonl').open('x') as f:
        subprocess.run([str(root/'count'),str(root/'recipes.tsv')],stdout=f,check=True,timeout=120)
        f.flush();os.fsync(f.fileno())
    seconds=time.monotonic()-begin
    rows=[json.loads(line) for line in (root/'exact-counts.jsonl').read_text().splitlines()]
    need(len(rows)==len(recipes),'counter output coverage')
    for spec,row in zip(recipes,rows,strict=True):
        need(row['q']==spec['q'] and row['population']==spec['population'],'counter projection identity')
        need(len(row['counts'])==32 and all(type(n) is int and n>=0 for n in row['counts'])
             and sum(row['counts'])==spec['edges'],'counter totals/ownership')
    report=dict(schema='b2-mam-confirmation-topology-v1',replicate=identity['replicate'],model_sha256=identity['model_sha256'],
        instance_sha256=identity['instance_sha256'],plan_sha256=identity['plan_sha256'],model_layers=layers,
        exact_target_histograms=True,full_shared_projection_coverage=True,shared_populations=[88],
        projections=len(recipes),edges=sum(r['edges'] for r in recipes),ranks=32,
        counter_source_sha256=sha(root/'count.rs'),counter_executable_sha256=sha(root/'count'),
        helpers_match_new_model_source=True,helper_sha256=hashlib.sha256('\n'.join(helpers).encode()).hexdigest(),
        recipe_sha256=sha(root/'recipes.json'),counts_sha256=sha(root/'exact-counts.jsonl'),
        identity_sha256=sha(root/'identity.json'),count_seconds=seconds,wall_seconds=time.monotonic()-started,
        legacy_counts_reused=False,neural_runs=0,scientific_acceptance=False)
    write(root/'pending.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':run(Path(sys.argv[1]))
