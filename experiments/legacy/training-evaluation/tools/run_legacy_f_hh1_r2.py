#!/usr/bin/env python3
"""Prepare-only legacy F-HH1 driver; --allow-run is required on remote host.

Future serial stages resolve a new hash lock, install into a new private venv,
then call the unchanged published fitting_probe against the existing teacher.
No existing environment, r1 result, source or dependency lock is overwritten.
"""
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from pathlib import Path
import re
import shutil
import socket
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evidence/frontier-compat-r2'
INPUT=BASE/'dependencies/brian2modelfitting-legacy-py311.in'
TEACHER=ROOT/'evidence/frontier/fixtures/F-HH1-r1.json'
ORIGINAL=ROOT/'tools/frontier_capability.py'
LOCK=BASE/'.legacy-run.lock'
FROZEN={
 'tools/frontier_capability.py':'7918d5b1d6a7fef920f72c7831d491b723f97cd6312eb20c1d89c336a62d522c',
 'evidence/frontier/fixtures/F-HH1-r1.json':'af22580d06f74210a80752ae6254230293423ba6454b15dc46abbba39e624034',
 'evidence/frontier/F-HH1-contract.json':'e86cd0e35db501ff2fd5eaef7e80deb498b4784a74e4c21b12d688bb66a560e5',
 'evidence/frontier-compat-r2/dependencies/brian2modelfitting-legacy-py311.in':'46f09a96b238e0f1c0067428913fd4b88f79930678eab66c0f93cc721f92a088',
 'tools/measure_e1_small_cold_r1.py':'bdc2e0420f1df2556fdbdde1c0b5e13a82f2ab25fad280b0330cbcd4bcc9c94b',
 'tools/run_a1_queue_v4.py':'cdf0f6ba85745bd425099336ed115cd706adcaff897eac944ac76e745ca164b5',
 'tools/run_recurrent_queue.py':'2f29be1bdc9548866587a46eb7255ddf597c29a04330d585ea89ae8409106f3f',
}
STAGES=[('uv-version',30),('python-bootstrap',300),('interpreter',60),('resolve',900),('venv',60),
        ('install',1200),('metadata',60),('probe',600)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value,update=False):
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n'
    path=Path(path)
    if update:
        temporary=path.with_suffix(path.suffix+'.partial')
        temporary.write_text(text);temporary.replace(path)
    else:
        with path.open('x') as stream:stream.write(text)


def verify():
    for name,expected in FROZEN.items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Frozen input changed: '+name)


def norm(name):
    return re.sub(r'[-_.]+','-',name).lower()


def lock_pins(path,require_hashes=True):
    """Accept only exact PyPI version pins and sha256 artifact hashes."""
    logical=Path(path).read_text().replace('\\\n',' ')
    result={}
    for raw in logical.splitlines():
        line=raw.split('#',1)[0].strip()
        if not line:continue
        match=re.fullmatch(r'([A-Za-z0-9_.-]+)==([^ ;\s]+)(.*)',line)
        if not match:raise ValueError('Unsupported non-exact lock entry: '+line)
        name,version,tail=match.groups();name=norm(name)
        hashes=re.findall(r'--hash=sha256:([a-fA-F0-9]{64})',tail)
        residual=re.sub(r'--hash=sha256:[a-fA-F0-9]{64}','',tail).strip()
        if residual:raise ValueError('Unexpected lock marker/option: '+residual)
        if require_hashes and not hashes:raise ValueError('Missing hash for '+name)
        if name in result:raise ValueError('Duplicate lock package '+name)
        result[name]=version
    if not result:raise ValueError('Empty lock')
    return result


@contextmanager
def phase_lock(token,directory):
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a+') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another legacy F coordinator is active') from None
        stream.seek(0);stream.truncate()
        json.dump(dict(pid=os.getpid(),token=token,directory=str(directory)),stream)
        stream.flush();os.fsync(stream.fileno())
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def worker_lease(args):
    record=json.loads(LOCK.read_text())
    if record['pid']!=os.getppid() or record['token']!=args.token or record['directory']!=str(args.directory):
        raise RuntimeError('Private worker requires active direct-parent phase lease')
    with LOCK.open('r+') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        raise RuntimeError('Private worker lease is no longer active')


def clean_env(directory):
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('UV_','PIP_')) or key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','VIRTUAL_ENV','CONDA_PREFIX'):
            env.pop(key,None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        UV_NO_CONFIG='true',UV_PYTHON_DOWNLOADS='never',UV_LINK_MODE='copy',
        UV_CACHE_DIR=str(directory/'uv-cache'),UV_DEFAULT_INDEX='https://pypi.org/simple',
        TMPDIR=str(directory/'tmp'),MPLCONFIGDIR=str(directory/'matplotlib'),
        MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    env.pop('PYTHONUSERBASE',None)
    Path(env['TMPDIR']).mkdir()
    return env


def installed(directory):
    if sys.version_info[:2]!=(3,11) or platform.machine()!='arm64' or platform.python_implementation()!='CPython':
        raise RuntimeError('Private probe requires native arm64 Python 3.11')
    prefix=(directory/'venv').resolve()
    if Path(sys.prefix).resolve()!=prefix:
        raise RuntimeError('Probe is outside the new private venv')
    if os.environ.get('PYTHONPATH'):
        raise RuntimeError('Outer snapshot PYTHONPATH must be absent')
    expected=lock_pins(directory/'resolved.lock')
    actual={};duplicates=[]
    for dist in importlib.metadata.distributions():
        name=norm(dist.metadata['Name'])
        if name in actual:duplicates.append(name)
        actual[name]=dist.version
    if duplicates or actual!=expected:
        raise ImportError('Installed distributions differ from new hash lock: '+json.dumps(dict(duplicates=duplicates,expected=expected,actual=actual)))
    for name,version in lock_pins(INPUT,require_hashes=False).items():
        if actual.get(name)!=version:raise ImportError('Declared top-level pin changed: '+name)
    return dict(python=sys.version,machine=platform.machine(),implementation=platform.python_implementation(),executable=sys.executable,prefix=sys.prefix,
                installed_distributions=actual,lock_sha256=sha(directory/'resolved.lock'))


def worker(args):
    worker_lease(args);verify()
    directory=args.directory
    report=dict(status='started',performance_run=False,profile='legacy-private-py311-arm64',
                teacher_sha256=sha(TEACHER),original_probe_sha256=sha(ORIGINAL))
    target=directory/('metadata.json' if args.worker=='metadata' else 'probe/result.json')
    target.parent.mkdir(parents=True,exist_ok=args.worker=='metadata')
    def publish():save(target,report,update=True)
    publish()
    try:
        report.update(installed(directory))
        if args.worker=='metadata':
            report['status']='installed_lock_verified';publish();return 0
        work=directory/'probe/work';work.mkdir()
        os.chdir(work)
        # Load the frozen module as source only; its old worker/py312 lock gate
        # and fixture-generating function are deliberately never invoked.
        spec=importlib.util.spec_from_file_location('frozen_frontier_probe',ORIGINAL)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        import brian2
        import brian2modelfitting
        if not Path(brian2.__file__).resolve().is_relative_to((directory/'venv').resolve()):
            raise ImportError('Brian2 resolved outside the private venv')
        if brian2.__version__!='2.9.0':
            raise ImportError('Wrong Brian2 release profile')
        report.update(brian2_source=str(Path(brian2.__file__).resolve()),
            brian2modelfitting_source=str(Path(brian2modelfitting.__file__).resolve()),
            same_physical_model_contract_sha256=sha(ROOT/'evidence/frontier/F-HH1-contract.json'),
            numerical_profile_change='Python3.11/Brian2.9.0/NumPy1.23.5; original unmodified HH equations, NumPy RK4, same teacher')
        publish()
        data=json.loads(TEACHER.read_text())
        module.fitting_probe(data,report,publish,directory/'probe')
        report['status']='finite_fitting_probe_completed'
        report['qualification_limit']='Finite forward/one fit round only; inspect refine status separately, no convergence/independent sensitivity qualification or full F claim.'
        publish();return 0
    except BaseException as error:
        report.update(status='dependency_failure' if isinstance(error,(ImportError,ModuleNotFoundError)) else 'probe_failed',
                      error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        publish();return 1


def run(args):
    sys.path.insert(0,str(ROOT/'tools'))
    from run_recurrent_queue import launch
    from measure_e1_small_cold_r1 import require_idle
    directory=BASE/'runs'/args.run_id
    directory.mkdir(parents=True,exist_ok=False)
    token=uuid.uuid4().hex
    records=[dict(name=name,cap_s=cap,status='not_launched') for name,cap in STAGES]
    report=dict(status='not_completed',performance_run=False,stages=records,
        r1_preserved=True,denominator='one separately versioned HH fitting probe; never the 12 migration models',
        total_internal_caps_s=sum(cap for _,cap in STAGES),outer_recommended_cap_s=3600)
    active=None
    try:
        with phase_lock(token,directory):
            require_idle();verify()
            if shutil.disk_usage(ROOT).free<50*1024**3:
                raise RuntimeError('50GiB free disk floor; no installation attempted')
            python=None
            uv=shutil.which('uv')
            if uv is None:raise FileNotFoundError('uv is not available; no bootstrap attempted')
            env=clean_env(directory)
            save(directory/'preexecution.json',dict(status='declared_before_resolution',source_sha256=sha(__file__),
                frozen_inputs=FROZEN,requirements_sha256=sha(INPUT),teacher_sha256=sha(TEACHER),
                python_request='3.11',required_machine='arm64',managed_python_directory=str(directory/'managed-python'),
                uv_path=str(Path(uv).resolve()),uv_sha256=sha(Path(uv).resolve()),
                source_changes=[],monkeypatches=[],allow_python_downloads='only explicit private bootstrap stage; --no-bin',
                dependency_resolution_status='not_yet_executed',caps=STAGES,
                binding_note='Resolved transitive versions will be materialized in a new hash lock before any install or probe; no result-driven pin changes.'))
            vpython=directory/'venv/bin/python'
            identities=dict(FROZEN)
            identities[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
            identities['tools/measure_e1_small_cold_r1.py']=sha(ROOT/'tools/measure_e1_small_cold_r1.py')
            resolved_sha=None
            for index,(name,cap) in enumerate(STAGES):
                active=index
                require_idle()
                if shutil.disk_usage(ROOT).free<50*1024**3:
                    raise RuntimeError('50GiB free disk floor before '+name+'; stage not launched')
                for path,expected in identities.items():
                    if sha(ROOT/path)!=expected:raise RuntimeError('Preexecution identity changed: '+path)
                if resolved_sha is not None and sha(directory/'resolved.lock')!=resolved_sha:
                    raise RuntimeError('Resolved hash lock changed after resolution')
                stage_env=env.copy()
                if name=='python-bootstrap':
                    stage_env.pop('UV_PYTHON_DOWNLOADS',None)
                    stage_env['UV_PYTHON_INSTALL_DIR']=str(directory/'managed-python')
                    stage_env['UV_PYTHON_BIN_DIR']=str(directory/'unused-private-bin')
                    command=[uv,'python','install','3.11','--install-dir',str(directory/'managed-python'),'--no-bin']
                elif name=='uv-version':
                    command=[uv,'--version']
                elif name=='interpreter':
                    command=[str(python),'-I','-c','import json,platform,sys; print(json.dumps(dict(version=sys.version,major_minor=list(sys.version_info[:2]),machine=platform.machine(),implementation=platform.python_implementation(),executable=sys.executable))); assert sys.version_info[:2]==(3,11) and platform.machine()=="arm64" and platform.python_implementation()=="CPython"']
                elif name=='resolve':
                    command=[uv,'pip','compile',str(INPUT),'--python',str(python),'--generate-hashes','--output-file',str(directory/'resolved.lock')]
                elif name=='venv':
                    command=[uv,'venv','--python',str(python),str(directory/'venv')]
                elif name=='install':
                    command=[uv,'pip','sync','--python',str(vpython),'--require-hashes',str(directory/'resolved.lock')]
                else:
                    command=[str(vpython),'-I',str(Path(__file__).resolve()),'--worker',name,'--directory',str(directory),'--token',token]
                records[index].update(status='launching',command=command)
                save(directory/'progress.json',report,update=True)
                output=directory/'probe' if name=='probe' else directory
                result=launch(name,command,cap,64*1024**3,directory,output,stage_env,'legacy_compatibility')
                records[index].update(status='exited',supervisor=result)
                if result['exit_code']!=0 or result['termination_reason']!='exited':
                    raise RuntimeError(f'{name} failed; subsequent stages remain unexecuted')
                if name=='python-bootstrap':
                    candidates=[entry/'bin/python3.11' for entry in (directory/'managed-python').iterdir()
                                if entry.is_dir() and entry.name.startswith('cpython-3.11') and (entry/'bin/python3.11').is_file()]
                    if len(candidates)!=1:raise RuntimeError('Private bootstrap did not produce exactly one Python3.11 candidate')
                    python=candidates[0].resolve()
                    if not python.is_relative_to((directory/'managed-python').resolve()):
                        raise RuntimeError('Downloaded interpreter escaped private directory')
                if name=='interpreter':
                    metadata=json.loads((directory/'interpreter.log').read_text())
                    if metadata['major_minor']!=[3,11] or metadata['machine']!='arm64' or metadata['implementation']!='CPython':
                        raise RuntimeError('Private interpreter is not arm64 Python3.11')
                    save(directory/'interpreter-receipt.json',dict(**metadata,sha256=sha(python),
                        binding='Exact downloaded interpreter version and binary recorded before dependency resolution'))
                if name=='resolve':
                    pins=lock_pins(directory/'resolved.lock')
                    for key,version in lock_pins(INPUT,require_hashes=False).items():
                        if pins.get(key)!=version:raise RuntimeError('Resolver changed declared pin '+key)
                    resolved_sha=sha(directory/'resolved.lock')
                    save(directory/'resolved-lock-receipt.json',dict(status='resolved_not_installed_or_qualified',
                        lock_sha256=resolved_sha,requirements_sha256=sha(INPUT),pins=pins,
                        python=str(python.resolve()),performance_run=False))
                records[index]['status']='completed'
                active=None
                save(directory/'progress.json',report,update=True)
            report['status']='finite_legacy_probe_pipeline_completed'
            report['claim']='Inspect probe/result.json including refine outcome; no full F, convergence or performance claim.'
    except BaseException as error:
        report.update(status='dependency_or_pipeline_failure',error_type=type(error).__name__,
                      error=str(error),traceback=traceback.format_exc())
        if active is not None:
            records[active]['status']='stage_failed'
            terminal=directory/(records[active]['name']+'-terminal.json')
            if terminal.exists():records[active]['partial_supervisor']=json.loads(terminal.read_text())
        for row in records:
            if row['status']=='not_launched':row['reason']='prior stage or prerequisite failed'
    finally:
        report['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(directory/'terminal.json',report)
    return 0 if report['status']=='finite_legacy_probe_pipeline_completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-run',action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--worker',choices=('metadata','probe'))
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--token')
    args=parser.parse_args()
    if socket.gethostname()!='rock-mac-studio-1.local':
        raise RuntimeError('This driver is reserved for 100.90.28.27')
    if args.worker:
        args.directory=args.directory.resolve();return worker(args)
    if not args.allow_run or not args.run_id:
        parser.error('Prepared only: future coordinated --allow-run --run-id NEW_ID required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):
        parser.error('Invalid fresh run id')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
