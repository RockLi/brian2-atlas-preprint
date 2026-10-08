"""Semantic corruption checks for the simplified-model archive."""
import argparse
import contextlib
import io
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from .verify_minimal_response import read,save,sha,verify


def run(source):
    assert read(source/'verification.json')['all_passed'];before={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()};out=[]
    for name,flag in [('metric','all_240_metrics_recomputed'),('weights','anatomical_strengths_and_blank_identity'),('prediction','linear_analytic_solution')]:
        with tempfile.TemporaryDirectory(prefix='minimal-model-mutation-') as tmp:
            root=Path(tmp)/'view'
            def copy(src,dst):
                if Path(src).suffix=='.npz':os.link(src,dst)
                else:shutil.copy2(src,dst)
                return dst
            shutil.copytree(source,root,copy_function=copy)
            if name=='metric':m=read(root/'metrics.json');m[0]['models']['linear']['mean_mv']+=1.;save(root/'metrics.json',m)
            else:
                rows=read(root/'prediction-rows.json')
                if name=='weights':rows[0]['contacts'][0]+=1
                else:
                    path=root/'predictions'/(rows[0]['key']+'.npz')
                    with np.load(path) as z:d={k:z[k] for k in z.files}
                    d['linear_original_AB'][1800]+=.1;temp=path.with_name('changed.npz');np.savez_compressed(temp,**d);os.replace(temp,path);rows[0]['archive_sha256']=sha(path)
                save(root/'prediction-rows.json',rows);seal=read(root/'prediction-seal.json');seal['prediction_rows_sha256']=sha(root/'prediction-rows.json');save(root/'prediction-seal.json',seal)
            rejected=False
            with contextlib.redirect_stdout(io.StringIO()):
                try:verify(root)
                except ValueError as e:
                    if str(e)!='minimal-model verification failed':raise
                    rejected=True
            result=read(root/'verification.json');assert rejected and not result['checks'][flag];out.append({'mutation':name,'rejected':True,'expected_check':flag});print(out[-1],flush=True)
    assert before=={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()}
    save(source/'mutation-tests.json',{'all_passed':True,'original_evidence_unchanged':True,'tests':out,'verifier_sha256':sha(Path(__file__).with_name('verify_minimal_response.py')),'test_source_sha256':sha(__file__)})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);run(a.parse_args().root)
