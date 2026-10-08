"""Semantic mutation checks on a complete real P13 archive; no native launches."""
import argparse
import contextlib
import io
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from .verify_repeat_study import verify
from .motion_refinement import read,sha,load_npz
from .multispeed_data import digest
from .run_experiment import save


def run(source):
    if not read(source/'verification.json')['all_passed']:raise ValueError('verified study required')
    before={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()};out=[]
    for name,flag in [('metric','all_160_metrics_independently_recomputed'),('background','literal_stimuli_and_fixed_background'),('duplicate','no_duplicate_injection'),('restoration','four_prespecified_restorations'),('reference','fixed_reference_from_first_background_only')]:
        with tempfile.TemporaryDirectory(prefix='p13-verify-') as td:
            root=Path(td)/'view'
            def copy(src,dst):
                if Path(src).suffix in ('.npz','.bin') or Path(src).name in ('b2-native','unit-tests'):os.link(src,dst)
                else:shutil.copy2(src,dst)
                return dst
            shutil.copytree(source,root,copy_function=copy);report=read(root/'report.json')
            if name=='metric':report['per_site'][0]['voltage_interaction_mean_mv']+=1
            elif name=='reference':
                ref=read(root/'reference-signs.json');ref['reference'][0]['J_reference_sign']=2;save(root/'reference-signs.json',ref)
            elif name=='restoration':save(root/'postchecks.json',read(root/'postchecks.json')[:-1]);report['postchecks_sha256']=sha(root/'postchecks.json')
            else:
                rows=read(root/'rows.json');row=next(r for r in rows if r['background']==785 and r['site']==0 and r['kind']=='AB' and r['delay']==200);path=root/'trials'/(row['key']+'.npz');data=load_npz(path)
                if name=='background':
                    # Empty background also must remain exactly empty.
                    if len(data['background_ticks']):data['background_ticks']=data['background_ticks']+1
                    else:data['background_ticks']=np.array([100],dtype=np.int64);data['background_indices']=np.array([row['selected'][0]],dtype=np.int64)
                else:data['audit']=np.concatenate([data['audit'],data['audit'][:1]])
                temp=path.with_name('mutated.npz');np.savez_compressed(temp,**data);os.replace(temp,path);blank=load_npz(root/'trials'/(row['blank_key']+'.npz'));decoded={k:v.copy() for k,v in data.items()}
                for k in ('v','ge','gi'):decoded[k]=np.bitwise_xor(decoded[k],blank[k].view(np.uint64)).view(np.float64)
                row.update(archive_sha256=sha(path),array_sha256={k:digest(v) for k,v in decoded.items()},array_shapes={k:list(v.shape) for k,v in decoded.items()});save(root/'rows.json',rows);report['rows_sha256']=sha(root/'rows.json')
            save(root/'report.json',report);rejected=False
            with contextlib.redirect_stdout(io.StringIO()):
                try:verify(root)
                except ValueError as e:
                    if str(e)!='P13 independent verification failed':raise
                    rejected=True
            result=read(root/'verification.json');assert rejected and not result['all_passed'] and not result['checks'][flag];out.append({'mutation':name,'rejected':True,'expected_check':flag,'failed_checks':[k for k,v in result['checks'].items() if not v]});print(out[-1],flush=True)
    assert before=={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()}
    save(source/'verifier-mutation-tests.json',{'all_passed':True,'original_files_unchanged':True,'mutations':out,'verifier_sha256':sha(Path(__file__).with_name('verify_repeat_study.py')),'test_source_sha256':sha(Path(__file__))})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);a=parser.parse_args();run(a.root)
