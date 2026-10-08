"""Post-hoc total-input-count diagnostic; no P9 model or selection changes."""
import argparse
from pathlib import Path
import numpy as np
from .motion_refinement import read, sha, load_npz
from .refinement import fit_linear, linear_scores
from .pilot import DIRECTIONS
from .direction_study import evaluate
from .generalization_data import digest
from .run_experiment import save


def audit(root, verify=False):
    protocol=read(root/'protocol.json'); report=read(root/'report.json')
    assert report['status']=='complete' and read(root/'verification.json')['all_passed']
    rows=[]; totals=[]
    for speed,record in zip((1,2,4),protocol['development']):
        directory=Path(record['directory'])
        with np.load(directory/'features.npz',allow_pickle=False) as data:totals.append(data['encoded'].sum(1))
        rows.extend({**r,'speed':speed} for r in read(directory/'rows.json'))
    x=np.concatenate(totals); y=np.array([DIRECTIONS.index(r['kind']) for r in rows]); options=[]
    for alpha in (.01,.1,1.,10.,100.):
        accuracy=[]
        for held in (1,2,4):
            train=np.array([r['split']=='fit' and r['speed']!=held for r in rows]); valid=np.array([r['split']=='validation' and r['speed']==held for r in rows])
            model=fit_linear(x[train],y[train],'raw_standard',alpha,None)
            accuracy.append(float((linear_scores(model,x[valid]).argmax(1)==y[valid]).mean()))
        options.append({'alpha':alpha,'held_speed_accuracy':accuracy})
    chosen=max(options,key=lambda v:(min(v['held_speed_accuracy']),np.mean(v['held_speed_accuracy']),v['alpha']))
    train=np.array([r['split']=='fit' for r in rows]); model=fit_linear(x[train],y[train],'raw_standard',chosen['alpha'],None)
    path=root/'totals-audit-model.npz'
    if verify:
        stored=load_npz(path)
        assert set(model)==set(stored) and all(np.array_equal(v,stored[k]) for k,v in model.items())
    else:
        if path.exists():raise ValueError('audit model already exists')
        np.savez_compressed(path,**model)
    testrows=read(root/'rows.json')
    with np.load(root/'features.npz',allow_pickle=False) as data:test=data['encoded'].sum(1)
    predictions=linear_scores(model,test).argmax(1); truth=np.array([DIRECTIONS.index(r['kind']) for r in testrows])
    result={'scope':'post-hoc after P9 and input-shuffle results; total counts per actual input channel, no time or spatial pooling; standard ridge, no reversal augmentation; alpha selected on original fit/validation groups at speeds 1/2/4 only; no native run or P9 model changes',
        'report_sha256':sha(root/'report.json'),'protocol_sha256':sha(root/'protocol.json'),'features_sha256':sha(root/'features.npz'),
        'source_sha256':sha(Path(__file__)),'model_sha256':sha(path),'development_totals_sha256':digest(x),'test_totals_sha256':digest(test),
        'features':x.shape[1],'candidates':options,'chosen':chosen,'conditions':{}}
    for condition in ('speed_2','speed_3','speed_5','static_first'):
        mask=np.array([r['condition']==condition for r in testrows]);rr=[r for r,k in zip(testrows,mask) if k]
        result['conditions'][condition]=evaluate(predictions[mask],truth[mask],rr)
    output=root/'totals-audit.json'
    if verify:
        assert read(output)==result
        save(root/'totals-audit-verification.json',{'all_passed':True,'audit_sha256':sha(output)})
        print({'all_passed':True},flush=True)
    else:
        if output.exists():raise ValueError('audit already exists')
        save(output,result)
        print({'chosen':chosen,'accuracy':{k:v['accuracy'] for k,v in result['conditions'].items()}},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args();audit(a.root,a.verify)
