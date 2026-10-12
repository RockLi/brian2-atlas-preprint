"""Explicit one-L4 acceptance of a selected Atlas checkout; no deployment/retries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def run_gate(guard):
    if not guard.put('attempt', {'started_unix':time.time()}, skip_if_exists=True):
        raise RuntimeError('Refusing replay of already started GPU acceptance')
    import os
    import sys
    import platform
    output=Path('/tmp/acceptance');output.mkdir()
    command=[sys.executable,'/gate/run_gate.py','--atlas','/workspace','--output',str(output),'--mode','cuda','--reuse-cuda-build']
    env=dict(os.environ,B2_RUNNER='/workspace/brian2-rust/target/release/b2-runner',B2_TRAIN_RUNNER='/workspace/brian2-rust/target/release/b2-train')
    process=subprocess.run(command,env=env,capture_output=True,text=True,timeout=3300)
    report={'exit_code':process.returncode,'stdout':process.stdout,'stderr':process.stderr,
            'gpu':subprocess.check_output(['nvidia-smi'],text=True),
            'nvcc':subprocess.check_output(['nvcc','--version'],text=True),
            'mpi':subprocess.check_output(['mpiexec','--version'],text=True),
            'host':platform.platform(), 'python':sys.version,
            'scope':'One physical L4, same-host MPI ranks sharing device 0; no cross-host or multi-GPU claim.',
            'files':{name:(output/name).read_text() for name in ('results.xml','result.json','selected-tests.json') if (output/name).exists()}}
    guard.put('report',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.atlas.resolve();args.output.mkdir(parents=True,exist_ok=False)
    import modal
    with tempfile.TemporaryDirectory(prefix='atlas-cuda-upload-') as temp:
        snapshot=Path(temp)
        names=subprocess.check_output(['git','-C',str(root),'ls-files','-c','-o','--exclude-standard'],text=True).splitlines()
        headers={'setup.py','pyproject.toml','_atlas_build.py','README.md','LICENSE','LICENSE_SCOPE.md','NOTICE','AUTHORS','CONTRIBUTORS'}
        prefixes=('brian2/','brian2-rust/src/','brian2-rust/python/','brian2-rust/tests/','brian2-rust/examples/','LICENSES/')
        names=sorted({n for n in names if n in headers or n.startswith(prefixes) or n in ('brian2-rust/Cargo.toml','brian2-rust/Cargo.lock','brian2-rust/rust-toolchain.toml')})
        hashes={}
        for name in names:
            p=root/name
            if not p.is_file():continue
            q=snapshot/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        (args.output/'source-hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
        image=(modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu24.04',add_python='3.12')
               .entrypoint([]).apt_install('curl','clang','build-essential')
               .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1")
               .env({'PATH':'/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin','PYTHONPATH':'/workspace/brian2-rust/python:/workspace','MPLCONFIGDIR':'/tmp/matplotlib','B2_TRAIN_CUDA_DEVICE':'0'})
               .pip_install('numpy==2.2.6','pytest==8.4.1','pytest-timeout==2.4.0','cython==3.0.12','sympy==1.14.0','pyparsing==3.2.3','jinja2==3.1.6','setuptools==80.9.0','packaging==25.0','mpich==5.0.1')
               .add_local_dir(snapshot,'/workspace',copy=True)
               .run_commands('cargo build --release --locked --manifest-path /workspace/brian2-rust/Cargo.toml','cd /workspace && python setup.py build_ext --inplace')
               .pip_install('scipy==1.15.3')
               .add_local_file(Path(__file__),'/root/modal_gate.py',copy=True)
               .add_local_file(Path(__file__).with_name('run_gate.py'),'/gate/run_gate.py',copy=True)
               .add_local_file(Path(__file__).with_name('test-modules.json'),'/gate/test-modules.json',copy=True)
               .run_commands('B2_RUNNER=/workspace/brian2-rust/target/release/b2-runner B2_TRAIN_RUNNER=/workspace/brian2-rust/target/release/b2-train python /gate/run_gate.py --atlas /workspace --output /tmp/collection --mode cuda --collect-only'))
        app=modal.App('atlas-training-f73-acceptance',include_source=False)
        function=app.function(image=image,gpu='L4',cpu=4,memory=8192,max_containers=1,timeout=3600,retries=0)(run_gate)
        started=time.time()
        with modal.enable_output(),modal.Dict.ephemeral() as guard:
            try:
                with app.run():
                    print('APP_ID='+app.app_id,flush=True)
                    report=function.remote(guard)
            except Exception as exc:
                report=guard.get('report') or {'exit_code':None,'error':str(exc),'error_type':type(exc).__name__,'attempt':guard.get('attempt')}
        report.update(started_unix=started,ended_unix=time.time(),source_manifest_sha256=hashlib.sha256((args.output/'source-hashes.json').read_bytes()).hexdigest())
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        for name,data in report.get('files',{}).items():(args.output/name).write_text(data)
        print(json.dumps({k:report.get(k) for k in ('exit_code','error','started_unix','ended_unix')}))
        return 0 if report.get('exit_code')==0 else 1

if __name__=='__main__':raise SystemExit(main())
