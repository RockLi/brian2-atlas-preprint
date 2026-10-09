"""Tiny stdlib-only controls: fake Mach-O bytes, fake trainer, and source ASTs.
Never execute native artifacts, import numerical frameworks, or compile anything.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import arm64_artifact_gate_r1 as gate


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def thin(cpu=gate.ARM64,kind=2,subtype=0):
    return struct.pack('<8I',0xFEEDFACF,cpu,subtype,kind,0,0,0,0)


def fat(cpus,wide=False):
    offset=8+len(cpus)*(32 if wide else 20)
    body=b'';table=b''
    for cpu in cpus:
        data=thin(cpu)
        table+=struct.pack('>IIQQII' if wide else '>5I',cpu,0,offset,len(data),0,*([0] if wide else []))
        body+=data;offset+=len(data)
    return struct.pack('>2I',0xCAFEBABF if wide else 0xCAFEBABE,len(cpus))+table+body


def dump(node):
    return ast.dump(node,include_attributes=False)


def funcs(filename):
    return {n.name:n for n in ast.parse((ROOT/'tools'/filename).read_text()).body if isinstance(n,ast.FunctionDef)}


def checks():
    results=[]
    def check(name,function):
        function();results.append(dict(name=name,passed=True))
    def truth(value):
        assert value
    def rejects(function,contains=None):
        try:function()
        except (gate.ArchitectureGateError,ValueError) as e:
            if contains:assert contains in str(e),str(e)
        else:raise AssertionError('expected rejection')
    with tempfile.TemporaryDirectory(prefix='arm64-control-only-') as temp:
        temp=Path(temp)
        def artifact(name,data,executable=True):
            p=temp/name;p.write_bytes(data);p.chmod(0o755 if executable else 0o644);return p
        arm=artifact('arm',thin())
        x86=artifact('x86',thin(0x01000007))
        lib=artifact('lib.dylib',thin(kind=6))
        badlib=artifact('x86.dylib',thin(0x01000007,6))
        armid=gate.gate_artifact(arm,'runner')
        check('thin_arm64_execute_accept',lambda:truth(armid['gate_status']=='passed'))
        check('thin_arm64_dylib_accept',lambda:truth(gate.gate_artifact(lib,'dylib')['gate_status']=='passed'))
        check('x86_runner_reject',lambda:rejects(lambda:gate.gate_artifact(x86,'runner'),'ARM64'))
        check('x86_dylib_reject',lambda:rejects(lambda:gate.gate_artifact(badlib,'dylib'),'ARM64'))
        check('filetype_runner_reject',lambda:rejects(lambda:gate.gate_artifact(lib,'runner'),'filetype'))
        check('filetype_dylib_reject',lambda:rejects(lambda:gate.gate_artifact(arm,'dylib'),'filetype'))
        check('missing_execute_bit_reject',lambda:rejects(lambda:gate.gate_artifact(artifact('nonexec',thin(),False),'runner'),'permission'))
        check('sha_mismatch_reject',lambda:rejects(lambda:gate.gate_artifact(arm,'runner','0'*64),'SHA'))
        check('sha_match_accept',lambda:truth(gate.gate_artifact(arm,'runner',armid['sha256'])==armid))
        for wide in (False,True):
            check(f'fat{64 if wide else 32}_single_arm64_accept',lambda wide=wide:truth(gate.gate_artifact(artifact('fatarm'+str(wide),fat([gate.ARM64],wide)),'runner')['gate_status']=='passed'))
            check(f'fat{64 if wide else 32}_mixed_reject',lambda wide=wide:rejects(lambda:gate.gate_artifact(artifact('fatmixed'+str(wide),fat([gate.ARM64,0x01000007],wide)),'runner'),'ARM64'))
        check('unknown_binary_reject',lambda:rejects(lambda:gate.gate_artifact(artifact('unknown',b'not a Mach-O'),'runner'),'invalid'))
        check('truncated_thin_reject',lambda:rejects(lambda:gate.parse_macho(thin()[:20]),'truncated'))
        check('fat_empty_reject',lambda:rejects(lambda:gate.parse_macho(struct.pack('>2I',0xCAFEBABE,0)),'table'))
        check('fat_truncated_table_reject',lambda:rejects(lambda:gate.parse_macho(struct.pack('>2I',0xCAFEBABE,2)),'table'))
        bad=bytearray(fat([gate.ARM64]));struct.pack_into('>I',bad,8,0x01000007)
        check('fat_cpu_mismatch_reject',lambda:rejects(lambda:gate.parse_macho(bad),'differs'))
        overlap=bytearray(fat([gate.ARM64,gate.ARM64]));struct.pack_into('>I',overlap,36,48)
        check('fat_overlap_reject',lambda:rejects(lambda:gate.parse_macho(overlap),'overlapping'))
        badcommands=bytearray(thin());struct.pack_into('<I',badcommands,16,1)
        check('load_command_bounds_reject',lambda:rejects(lambda:gate.parse_macho(badcommands),'commands'))
        goodcommands=bytearray(thin()+struct.pack('<2I',1,8));struct.pack_into('<2I',goodcommands,16,1,8)
        check('bounded_load_command_accept',lambda:truth(gate.parse_macho(goodcommands)['slices'][0]['command_count']==1))
        retention=gate.capture_and_gate_library(lib,temp/'retain',hashlib.sha256(lib.read_bytes()).hexdigest())
        check('actual_dylib_copy_exact',lambda:truth(Path(retention['retained']['path']).read_bytes()==lib.read_bytes()))
        check('actual_dylib_identity_durable',lambda:truth(json.loads((temp/'retain/library-identity.json').read_text())==retention))
        check('rejected_dylib_retained',lambda:rejects(lambda:gate.capture_and_gate_library(badlib,temp/'retain-bad',hashlib.sha256(badlib.read_bytes()).hexdigest()),'ARM64'))
        check('rejected_dylib_metadata_durable',lambda:truth(json.loads((temp/'retain-bad/library-identity.json').read_text())['architecture_gate']['gate_status']=='rejected'))
        dense=module('qualify_dense_arm64_r2');metal=module('qualify_metal_arm64_r2')
        check('drivers_import_without_frameworks',lambda:truth(not any(n in sys.modules for n in ('numpy','torch','jax','brian2_rust','oracle'))))
        oldargv=sys.argv
        try:
            sys.argv=['qualify_dense_arm64_r2','--engine','atlas','--runner',str(x86),'--output',str(temp/'dense-reject')]
            with contextlib.redirect_stdout(io.StringIO()):code=dense.main()
        finally:sys.argv=oldargv
        rejected=json.loads((temp/'dense-reject/report.json').read_text())
        check('dense_reject_before_numeric_import',lambda:truth(code==1 and 'numpy' not in sys.modules and rejected['stage']=='runner_architecture_gate'))
        check('dense_reject_keeps_four_case_denominator',lambda:truth(len(rejected['cases'])==4 and all(v['status']=='not_launched' for v in rejected['cases'].values())))
        def refuses_existing_dense_directory(directory):
            saved=sys.argv
            try:
                sys.argv=['qualify_dense_arm64_r2','--engine','atlas','--runner',str(arm),'--output',str(directory)]
                try:dense.main()
                except FileExistsError:return
                raise AssertionError('existing evidence directory accepted')
            finally:sys.argv=saved
        previous=(temp/'dense-reject/report.json').read_bytes()
        check('dense_terminal_directory_reuse_rejected',lambda:refuses_existing_dense_directory(temp/'dense-reject'))
        check('dense_terminal_evidence_not_overwritten',lambda:truth((temp/'dense-reject/report.json').read_bytes()==previous))
        partial=temp/'dense-partial';partial.mkdir();(partial/'runner-identity.json').write_text('synthetic incomplete evidence')
        check('dense_partial_directory_reuse_rejected',lambda:refuses_existing_dense_directory(partial))
        check('dense_partial_evidence_not_overwritten',lambda:truth((partial/'runner-identity.json').read_text()=='synthetic incomplete evidence'))
        check('metal_separate_destination',lambda:truth(metal.DEST==ROOT/'evidence/h0r-arm64-r2'))
        check('metal_architecture_failure_classification',lambda:truth(metal.failure(gate.ArchitectureGateError('x',{}))=='architecture_rejected'))
        executions=[];options={}
        class FakeTrainer:
            def __init__(self,plan,runner,weights):
                options['runner']=runner;self._metal_library=options['library'];self._metal_hash=hashlib.sha256(self._metal_library.read_bytes()).hexdigest();self.state={}
            def execute(self,*a,**k):
                executions.append(k);raise RuntimeError('synthetic dlopen diagnostic: no native code executed')
        fake_api=types.SimpleNamespace(lif_training_plan=lambda sizes,**kw:dict(sizes=sizes,**kw),NativeLIFTrainer=FakeTrainer)
        prior=sys.modules.get('atlas_adapter')
        sys.modules['atlas_adapter']=types.SimpleNamespace(admission=lambda *a,**k:dict(status='admitted',exact_native_tape_bytes=0),load_api=lambda source:fake_api)
        fakecase=dict(sizes=[2,4,2],weights=[[0]*8,[0]*8],inputs=[[[0,0]]],labels=[0])
        try:
            options['library']=badlib;out={}
            check('metal_wrong_library_blocks_execute',lambda:rejects(lambda:metal.atlas_case(fakecase,one_adam=False,state_trace=False,output=out,runner=arm,runner_identity=armid,artifact_directory=temp/'driver-reject'),'ARM64'))
            check('metal_wrong_library_no_execute_and_captured',lambda:truth(not executions and out['native_library']['retained']['sha256']==hashlib.sha256(badlib.read_bytes()).hexdigest()))
            options['library']=lib;out={}
            try:metal.atlas_case(fakecase,one_adam=False,state_trace=False,output=out,runner=arm,runner_identity=armid,artifact_directory=temp/'driver-loader')
            except RuntimeError as e:assert 'synthetic dlopen diagnostic' in str(e)
            else:raise AssertionError('fake loader should fail')
            check('metal_actual_runner_parameter_used',lambda:truth(options['runner']==arm))
            check('metal_dylib_saved_before_loader_failure',lambda:truth(len(executions)==1 and (temp/'driver-loader/library-identity.json').is_file() and out['native_library']['architecture_gate']['gate_status']=='passed'))
        finally:
            if prior is None:sys.modules.pop('atlas_adapter',None)
            else:sys.modules['atlas_adapter']=prior
        check('test_never_imported_frameworks',lambda:truth(not any(n in sys.modules for n in ('numpy','torch','jax','brian2_rust','oracle'))))
    olddense,newdense=funcs('qualify_dense.py'),funcs('qualify_dense_arm64_r2.py')
    oldmetal,newmetal=funcs('qualify_metal_q0.py'),funcs('qualify_metal_arm64_r2.py')
    check('dense_compare_AST_unchanged',lambda:truth(dump(olddense['compare'])==dump(newdense['compare'])))
    for name in ('plain','write','cases','command_inventory','dependency_inventory','compare','reference','compare_result','optimizer_checks','torch_case'):
        check('metal_'+name+'_AST_unchanged',lambda name=name:truth(dump(oldmetal[name])==dump(newmetal[name])))
    class StripGate(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='gate_artifact':return None
            return self.generic_visit(node)
    dense_eval=StripGate().visit(copy.deepcopy(newdense['evaluate']))
    dense_eval.args=copy.deepcopy(olddense['evaluate'].args)
    for node in ast.walk(dense_eval):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='run_case' and len(node.args)==3 and isinstance(node.args[2],ast.Name) and node.args[2].id=='runner':
            node.args[2]=ast.parse("ROOT/'runtime/b2-train'",mode='eval').body
    check('dense_evaluate_AST_unchanged_after_runner_gate_normalization',lambda:truth(dump(dense_eval)==dump(olddense['evaluate'])))
    def numerical_fixture_statements(node):
        start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base' for t in n.targets))
        end=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='report' for t in n.targets))
        return [dump(n) for n in node.body[start:end]]
    check('dense_all_four_fixture_AST_unchanged',lambda:truth(numerical_fixture_statements(olddense['main'])==numerical_fixture_statements(newdense['main'])))
    oldcall=next(n for n in oldmetal['atlas_case'].body if isinstance(n,ast.FunctionDef) and n.name=='call')
    newcall=next(n for n in newmetal['atlas_case'].body if isinstance(n,ast.FunctionDef) and n.name=='call')
    check('metal_native_call_AST_unchanged_after_gate_normalization',lambda:truth(dump(oldcall)==dump(StripGate().visit(copy.deepcopy(newcall)))))
    def field_value(node,key):
        values=[k.value for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' for k in n.keywords if k.arg==key]
        assert len(values)==1,key
        return dump(values[0])
    for name in ('atlas_profile','torch_profile','strict_same_numeric_profile','tolerances','optimizer_tolerance_policy','boundary','positive_margin','default_state_scope','prefix_replay','excluded'):
        check('metal_contract_'+name+'_AST_unchanged',lambda name=name:truth(field_value(oldmetal['contract'],name)==field_value(newmetal['contract'],name)))
    def plan_call(node):
        return next(n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='lif_training_plan')
    check('metal_native_plan_AST_unchanged',lambda:truth(dump(plan_call(oldmetal['atlas_case']))==dump(plan_call(newmetal['atlas_case']))))
    for source in ('arm64_artifact_gate_r1.py','qualify_dense_arm64_r2.py','qualify_metal_arm64_r2.py'):
        ast.parse((ROOT/'tools'/source).read_text())
    return dict(schema='ARM64-qualification-preparation-controls-r1',status='passed',
                checks=results,check_count=len(results),model_executed=False,compiler_executed=False,
                actual_runner_executed=False,actual_metal_library_built=False,framework_imported=False,
                numerical_qualification_executed=False,source_identities={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in (
                    'tools/qualify_dense.py','tools/qualify_metal_q0.py','tools/arm64_artifact_gate_r1.py',
                    'tools/qualify_dense_arm64_r2.py','tools/qualify_metal_arm64_r2.py',str(Path(__file__).relative_to(ROOT)))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');args=p.parse_args()
    result=checks()
    if args.output:
        with Path(args.output).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],check_count=result['check_count'],model_executed=False,framework_imported=False)))
