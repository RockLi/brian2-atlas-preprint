"""Hardware skips must identify disabled hardware and an authenticated guard."""
import ast
from pathlib import Path

GPU='B2_TEST_GPU';CUDA={'B2_TEST_CUDA','B2_TEST_CUDA_TRAIN'}

def classify(mode,identity,message,evidence,root,assets):
    assert mode in ('cpu','metal')
    assert evidence['nodeid']==identity and evidence['mode']==mode
    lower=message.lower()
    assert any(word in lower for word in ('hardware','gpu','metal','cuda','toolchain','runtime')), (identity,message,'non-hardware skip')
    assert not any(word in lower for word in ('mutable capture','not implemented','unsupported','compiler required','build ')),(identity,message,'logical/toolchain skip')
    source,line,reason=evidence.get('guard_location') or evidence['location'];source=Path(source)
    source=source if source.is_absolute() else root/source
    name=str(source.relative_to(root));assert name in assets,(identity,name,'unauthenticated skip location')
    tree=ast.parse(source.read_text());line=int(line)
    parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('skip','skipif') and node.lineno<=line<=node.end_lineno]
    # Runtime skips require their own enclosing environment check. A flag
    # elsewhere in the same function cannot authorize a logical skip.
    if calls:
        call=min(calls,key=lambda n:n.end_lineno-n.lineno)
        if call.func.attr=='skipif':guard=call.args[0] if call.args else None
        else:
            ancestor=parents.get(call)
            while ancestor is not None and not isinstance(ancestor,ast.If):ancestor=parents.get(ancestor)
            guard=ancestor.test if ancestor is not None else None
        assert guard is not None,(identity,message,'skip lacks a conditional guard')
        text=ast.unparse(guard)
        assert 'os.environ.get' in text or 'os.getenv' in text,(identity,message,text,'skip is not guarded by a hardware environment check')
    else:
        # skipif reports the decorator/test definition, not a pytest.skip call.
        decorators=[d for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and min([n.lineno,*[d.lineno for d in n.decorator_list]])<=line<=n.end_lineno for d in n.decorator_list if isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr=='skipif']
        assert any('os.environ.get' in ast.unparse(d) or 'os.getenv' in ast.unparse(d) for d in decorators),(identity,message,'skipif has no authenticated environment guard')
    functions=[node for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and min([node.lineno,*[d.lineno for d in node.decorator_list]])<=line<=node.end_lineno]
    region=min(functions,key=lambda n:n.end_lineno-n.lineno) if functions else tree
    flags={node.value for node in ast.walk(region) if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in {GPU,*CUDA}}
    assert flags,(identity,message,'no hardware environment guard at skip site')
    devices=set(evidence['devices'].values())
    if devices:
        candidates=devices & {'metal','cuda'}
    else:
        candidates=({'metal'} if GPU in flags and not flags&CUDA else {'cuda'} if flags&CUDA and GPU not in flags else set())
    assert len(candidates)==1,(identity,message,evidence['devices'],flags,'ambiguous hardware')
    hardware=next(iter(candidates))
    runtime=evidence.get('runtime_flags');assert runtime is not None,(identity,message,'missing real environment flags')
    assert runtime['B2_TEST_MPI']=='1' and runtime['B2_TEST_GPU']==('1' if mode=='metal' else '0') and runtime['B2_TEST_CUDA']==runtime['B2_TEST_CUDA_TRAIN']=='0',(identity,message,runtime,'runtime flags differ from declared backend')
    assert hardware=='cuda' or mode=='cpu',(identity,message,'enabled Metal skipped')
    assert (GPU in flags if hardware=='metal' else bool(flags&CUDA)),(identity,message,'guard/hardware mismatch')
    if 'cuda' in lower:assert hardware=='cuda',(identity,message,'wrong CUDA reason')
    if 'metal' in lower:assert hardware=='metal',(identity,message,'wrong Metal reason')
    return dict(hardware=hardware,source=name,line=line,flags=sorted(flags),devices=evidence['devices'],pending_hardware_verification=True)
