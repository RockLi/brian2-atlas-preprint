"""Corrupt semantic evidence in isolated copies, never rerun the native simulator."""
import argparse
import contextlib
import io
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from .verify_onset_diagnostic import verify,read,save,sha,digest


def run(source):
    assert read(source/'verification.json')['all_passed'];before={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()};results=[]
    for name,flag in [('forecast_time','forecast_precedes_all_pairs'),('stimulus_tick','only_stimulus_onset_changed'),('missing_restore','two_prespecified_restorations')]:
        with tempfile.TemporaryDirectory(prefix='p13c-mutation-') as tmp:
            root=Path(tmp)/'view'
            def copy(src,dst):
                if Path(src).suffix in ('.bin','.npz') or Path(src).name=='b2-native':os.link(src,dst)
                else:shutil.copy2(src,dst)
                return dst
            shutil.copytree(source,root,copy_function=copy)
            if name=='forecast_time':
                preds=read(root/'predictions.json');preds[0]['time_unix']=1e12;save(root/'predictions.json',preds)
            elif name=='missing_restore':save(root/'restorations.json',read(root/'restorations.json')[:-1])
            else:
                rows=read(root/'rows.json');row=next(r for r in rows if r['kind']=='AB' and r['delay']==200);path=root/'trials'/(row['key']+'.npz')
                with np.load(path) as z:d={k:z[k] for k in z.files}
                d['stimulus_ticks']=d['stimulus_ticks']+1;temp=path.with_name('changed.npz');np.savez_compressed(temp,**d);os.replace(temp,path);row['archive_sha256']=sha(path);row['array_sha256']['stimulus_ticks']=digest(d['stimulus_ticks']);save(root/'rows.json',rows)
            rejected=False
            with contextlib.redirect_stdout(io.StringIO()):
                try:verify(root)
                except ValueError as e:
                    if str(e)!='onset diagnostic verification failed':raise
                    rejected=True
            result=read(root/'verification.json');assert rejected and not result['checks'][flag];results.append({'mutation':name,'rejected':True,'expected_check':flag});print(results[-1],flush=True)
    assert before=={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()}
    save(source/'mutation-tests.json',{'all_passed':True,'original_evidence_unchanged':True,'tests':results,'verifier_sha256':sha(Path(__file__).with_name('verify_onset_diagnostic.py')),'test_source_sha256':sha(__file__)})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);run(a.parse_args().root)
