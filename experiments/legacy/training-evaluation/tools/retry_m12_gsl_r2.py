"""M12 original-source retry with a private Brian GSL preference file.
The r1 compile failure and original corpus source remain unchanged.
Executed only by the later serial queue; no package installation.
"""
import argparse, datetime, hashlib, json, os, pathlib, socket, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-run',action='store_true')
    p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args()
    if not a.allow_run or socket.gethostname()!='rock-mac-studio-1.local':
        p.error('Explicit remote serial execution required')
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    work=out/'work';work.mkdir()
    gsl=pathlib.Path('/opt/homebrew/opt/gsl')
    required=[gsl/'include/gsl'/x for x in ['gsl_odeiv2.h','gsl_errno.h','gsl_matrix.h']]
    required += [gsl/'lib/libgsl.dylib',gsl/'lib/libgslcblas.dylib']
    if not all(path.is_file() for path in required):
        raise RuntimeError('Declared installed GSL headers/libraries absent; no fallback or install')
    corpus=json.loads((ROOT/'protocol/brian-corpus-manifest.json').read_text())
    model=next(m for m in corpus['models'] if m['id']=='M12')
    original=ROOT/'snapshot'/model['repository_path']
    if sha(original)!=model['source_sha256']:raise RuntimeError('Original source changed')
    prefs="[GSL]\ndirectory = "+repr(str(gsl/'include'))+"\n[codegen.cpp]\nruntime_library_dirs = "+repr([str(gsl/'lib')])+"\n"
    (work/'brian_preferences').write_text(prefs)
    record=dict(status='retry_declared_before_source_execution',source_transformations=[],
        configuration_change='Private cwd Brian preferences point to existing verified GSL include/lib directories',
        original_failure_preserved='M12 original r1 Cython compile failure: gsl/gsl_odeiv2.h not found',
        source_path=str(original),source_sha256=sha(original),driver_sha256=sha(pathlib.Path(__file__)),
        frozen_original_wrapper_sha256=sha(ROOT/'tools/capability_migration.py'),
        gsl_files={str(q):dict(resolved=str(q.resolve()),sha256=sha(q),bytes=q.stat().st_size) for q in required},
        preferences_sha256=sha(work/'brian_preferences'),preferences_text=prefs,
        executed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        limit_s=600,scope='original source smoke only; not Atlas migration or performance')
    (out/'configuration.json').write_text(json.dumps(record,indent=2)+'\n')
    env={**os.environ,'MPLBACKEND':'Agg','PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONPATH':str(ROOT/'snapshot')+':'+str(ROOT/'snapshot/brian2-rust/python'),
         'MPLCONFIGDIR':str(out/'mpl'),'TMPDIR':str(out/'tmp')}
    pathlib.Path(env['TMPDIR']).mkdir()
    cpu=ROOT/'environment/cpu/bin/python'
    command=[str(cpu),str(ROOT/'tools/capability_migration.py'),'--child','smoke','--model','M12',
             '--child-output',str(out/'result.json')]
    os.chdir(work)
    os.execve(str(cpu),command,env)
if __name__=='__main__':main()
