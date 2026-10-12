"""Validate the training increment against an explicitly selected Atlas checkout.

CPU runs all non-GPU parameterizations; GPU gates select the first case per
function/backend/rank count (at most two ranks), retaining exact selected IDs.
These bounded device gates are not full Cartesian GPU qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

class Selection:
    def __init__(self, mode, output):
        self.mode, self.output = mode, output
        self.fixed = set(json.loads(Path(__file__).with_name('fixed-mpi-functions.json').read_text())) if mode in ('mpi', 'mpi-fixed') else set()
    def pytest_collection_modifyitems(self, config, items):
        kept, dropped, seen = [], [], set()
        for item in items:
            params = getattr(getattr(item, 'callspec', None), 'params', {})
            backend = next((params[k] for k in ('engine', 'backend', 'gpu_backend') if k in params), None)
            ranks = params.get('ranks', params.get('mpi_ranks'))
            if self.mode == 'cpu':
                accept = backend not in ('metal', 'cuda')
            else:
                accept = backend == self.mode or (self.mode == 'cuda' and item.path.name.startswith('test_training_cuda'))
                if self.mode in ('mpi', 'mpi-fixed'):
                    fixed = item.nodeid.split('[')[0] in self.fixed
                    accept = backend not in ('metal', 'cuda') and (fixed or (self.mode == 'mpi' and (ranks == 2 or '_mpi' in item.path.stem)))
                if isinstance(ranks, int) and ranks > 2:
                    accept = False
                key = (str(item.path), item.originalname, str(backend), str(ranks))
                if key in seen:
                    accept = False
                if accept:
                    seen.add(key)
            (kept if accept else dropped).append(item)
        items[:] = kept
        config.hook.pytest_deselected(items=dropped)
        (self.output/'selected-tests.json').write_text(json.dumps([x.nodeid for x in kept], indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['cpu','metal','mpi','mpi-fixed','cuda'],required=True)
    parser.add_argument('--collect-only',action='store_true')
    parser.add_argument('--reuse-cuda-build',action='store_true',help='Compile the unchanged CUDA library once, then copy it between test lifetimes')
    parser.add_argument('--test-module',action='append',help='Explicit module override for follow-up validation')
    args=parser.parse_args(); root=args.atlas.resolve(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(root),str(root/'brian2-rust/python')]
    os.environ['PYTHONPATH']=os.pathsep.join(sys.path[:2])
    os.environ['B2_TEST_GPU']='1' if args.mode=='metal' else '0'
    os.environ['B2_TEST_CUDA_TRAIN']='1' if args.mode=='cuda' else '0'
    os.environ['B2_TEST_MPI']='0' if args.mode=='cpu' else '1'
    os.chdir(root)
    import brian2, brian2_rust, pytest
    from brian2.tests import PreferencePlugin
    assert Path(brian2.__file__).resolve().is_relative_to(root)
    assert Path(brian2_rust.__file__).resolve().is_relative_to(root)
    brian2.prefs.codegen.target='numpy'
    brian2.prefs.codegen.runtime.cython.cache_dir=str(output/'cython')
    paths=args.test_module or json.loads(Path(__file__).with_name('test-modules.json').read_text())
    metadata={'schema':'atlas-training-increment-gate-v1','gate_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'mode':args.mode,'started_unix':time.time(),'python':sys.version,'modules':paths}
    for key in ('B2_RUNNER','B2_TRAIN_RUNNER'):
        value=Path(os.environ[key]); assert value.is_file()
        metadata[key.lower()+'_sha256']=hashlib.sha256(value.read_bytes()).hexdigest()
    if args.reuse_cuda_build and args.mode == 'cuda' and not args.collect_only:
        import shutil
        from brian2_rust import training_cuda
        directory=output/'cuda-library';directory.mkdir()
        library=training_cuda.build(directory)
        library_hash=hashlib.sha256(library.read_bytes()).hexdigest()
        metadata['cuda_build']={'compiled_once':True,'library_sha256':library_hash,
            'kernel_sha256':hashlib.sha256(training_cuda.kernel_source().encode()).hexdigest(),
            'host_source_sha256':hashlib.sha256(Path(training_cuda.__file__).with_suffix('.cu').read_bytes()).hexdigest()}
        def reuse_library(target):
            assert hashlib.sha256(library.read_bytes()).hexdigest()==library_hash
            return Path(shutil.copy2(library,Path(target)/library.name))
        training_cuda.build=reuse_library
    cmd=['-q','--maxfail=3','--timeout=240','--junitxml='+str(output/'results.xml'),'--basetemp='+str(output/'tmp'),'-o','cache_dir='+str(output/'pytest-cache'),*paths]
    if args.collect_only:cmd+=['--collect-only']
    code=pytest.main(cmd,plugins=[Selection(args.mode,output),PreferencePlugin(dict(brian2.prefs))])
    metadata.update(exit_code=int(code),ended_unix=time.time(),collect_only=args.collect_only)
    (output/'result.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return int(code)

if __name__=='__main__':raise SystemExit(main())
