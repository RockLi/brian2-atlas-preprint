"""Read pinned experiment dictionaries without importing or running their code."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


def extract_conditions(source):
    tree=ast.parse(source);env={};lines={}
    if len(list(ast.walk(tree)))>10000:raise ValueError('excessive experiment source')
    def value(node):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.Name):return env[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -value(node.operand)
        if isinstance(node,ast.Dict):return {value(k):value(v) for k,v in zip(node.keys,node.values,strict=True)}
        if isinstance(node,(ast.List,ast.Tuple)):
            items=[value(x) for x in node.elts];return tuple(items) if isinstance(node,ast.Tuple) else items
        if isinstance(node,ast.Subscript):return value(node.value)[value(node.slice)]
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and isinstance(node.func.value,ast.Name) and node.func.value.id=='copy'
                and node.func.attr=='deepcopy' and len(node.args)==1 and not node.keywords):
            return copy.deepcopy(value(node.args[0]))
        raise ValueError('unsupported experiment expression: '+type(node).__name__)
    def statements(nodes):
        for node in nodes:
            if isinstance(node,(ast.Import,ast.ImportFrom)):continue
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant):continue
            if isinstance(node,ast.Assign):
                v=value(node.value)
                for target in node.targets:
                    if isinstance(target,ast.Name):env[target.id]=v
                    elif isinstance(target,ast.Subscript):
                        value(target.value)[value(target.slice)]=v
                        if isinstance(target.value,ast.Name) and target.value.id=='params_stab':lines[str(value(target.slice))]=node.lineno
                    else:raise ValueError('unsupported assignment')
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
                call=node.value
                if not (isinstance(call.func,ast.Attribute) and call.func.attr=='update' and len(call.args)==1 and not call.keywords):raise ValueError('unsupported experiment call')
                target=value(call.func.value)
                if not isinstance(target,dict):raise ValueError('update target must be data')
                target.update(value(call.args[0]))
            elif isinstance(node,ast.For) and isinstance(node.target,ast.Name) and not node.orelse:
                items=value(node.iter)
                if not isinstance(items,list) or len(items)>20:raise ValueError('excessive experiment loop')
                for item in items:env[node.target.id]=item;statements(node.body)
            else:raise ValueError('unsupported experiment statement: '+type(node).__name__)
    prefix=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NEW_SIM_PARAMS' for t in node.targets):break
        prefix.append(node)
    statements(prefix)
    return env,lines


def audit(source_dir, parameters, output):
    catalog=json.loads((source_dir/'catalog.json').read_text())
    for name,row in catalog['files'].items():
        p=source_dir/'source'/name
        if p.stat().st_size!=row['bytes'] or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=row['sha256']:raise ValueError('source changed')
    path=source_dir/'source/figures/Schmidt2018_dyn/network_simulations.py'
    env,lines=extract_conditions(path.read_text());p=json.loads(parameters.read_text())
    assert p['source_commit']==catalog['commit']=='0a658be40bef3249cbe452f38809edf7d2f524ba'
    quantitative=env['params_stab'][1.9];raster=env['params_stab']['1.9_spikes']
    assert quantitative[1]['t_sim']==100500 and raster[1]['t_sim']==10500
    checks=[]
    def check(exp,actual,prefix=''):
        for key,v in exp.items():
            name=prefix+key
            if isinstance(v,dict):check(v,actual[key],name+'.')
            elif key=='K_stable':
                q=Path(actual[key]);expected='cc970f6089dd1c997d7978c9fe6db44017cb478b7ad48e6963343d87fa9c3c58'
                h=hashlib.file_digest(q.open('rb'),'sha256').hexdigest();assert h==expected
                checks.append(dict(field=name,exact_content=True,sha256=h,script_path=v,export_path=str(q)))
            else:checks.append(dict(field=name,script=v,export=actual[key],exact=v==actual[key]))
    check(quantitative[0],p['params'])
    check(env['input_params'],p['params']['input_params'],'input_params.')
    assert all(x.get('exact',x.get('exact_content',False)) for x in checks)
    conn=p['params']['connection_params'];ground=env['params_stab'][1.0][0]['connection_params']
    inherited={k:conn[k] for k in ['cc_weights_factor','cc_weights_I_factor'] if k not in ground}
    notebook=json.loads((source_dir/'source/M2E_statistical_test.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code');settings={}
    for node in ast.parse(code).body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Dict):
            for key,v in zip(node.value.keys,node.value.values,strict=True):
                if isinstance(key,ast.Constant) and key.value in ['N_scaling','K_scaling','t_sim']:settings[key.value]=ast.literal_eval(v)
    assert settings==dict(N_scaling=.006,K_scaling=.006,t_sim=2000.)
    r=dict(schema='b2-mam-paper-condition-audit-v1',source_commit=catalog['commit'],parameters_sha256=hashlib.file_digest(parameters.open('rb'),'sha256').hexdigest(),explicit_metastable_fields_match=True,checks=checks,
        paper_script_conditions=dict(quantitative_ms=100500,raster_ms=10500,quantitative_line=lines['1.9'],raster_line=lines['1.9_spikes']),
        ground_control_default_hazard=dict(label=1.0,assignment_line=lines['1.0'],missing_explicit_overrides=list(inherited),effective_values_if_combined_with_current_export_defaults=inherited,required_action='Define explicit condition manifests; do not infer chi=1 from the dictionary label or replay this legacy fragment with changed defaults.'),
        m2e_fixture=dict(settings=settings,usable_as_full_model_reference=False,reason='Notebook runs N=K=0.006 for 2000 ms; its small fixture does not validate full-model activity.'),
        scope='Static data-only audit of explicit experiment fields and stabilized matrix content. No official simulation executed; implicit historical defaults, NEST-version semantics, sampling and full figure reproduction remain open.')
    output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source-dir','parameters','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.source_dir,a.parameters,a.output)
