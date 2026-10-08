"""Development-only selection of external motion feature recipes."""
import argparse
import json,numpy as np
from pathlib import Path
from flywire_vision.motion_readout import cell_weights,pooled_grids,grid_features,fit_readout,predict
from flywire_vision.run_experiment import save
from flywire_vision.pilot import DIRECTIONS
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('--feature',choices=['correlation','spectral','hybrid'],default='correlation')
parser.add_argument('--mapping',type=Path,required=True)
parser.add_argument('--artifact',type=Path,required=True)
args=parser.parse_args();p=args.directory
assert json.loads((p/'report.json').read_text())['status']=='complete'
with np.load(p/'features.npz') as f:raw=f['neural']
with np.load(p/'blank.npz') as f:blank=f['neural']
with np.load(args.mapping) as f:m=dict(f)
groups=json.loads((args.artifact/'groups.json').read_text())
m['subtype']=np.concatenate([np.full(len(groups[f'T{n}{a}']),idx) for idx,(n,a) in enumerate((n,a) for n in [4,5] for a in 'abcd')]+[np.full(len(groups['LC4']),8)])
rows=json.loads((p/'rows.json').read_text());y=np.array([DIRECTIONS.index(r['kind']) for r in rows]);results=[]
for width in [1,2,5]:
 for stop in [50,60]:
  for weight in ['none','std']:
   for separation in ['family','subtype']:
    recipe=dict(start=10,stop=stop,width=width,weight=weight,separation=separation,feature=args.feature)
    w=cell_weights(raw[:128],blank,recipe);grids=pooled_grids(raw,blank,m,w,recipe)
    for mode in ['both','signed']:
     recipe={**recipe,'transform':mode};x=grid_features(grids,recipe)
     for reverse in [False,True]:
      best=(-1,None)
      for alpha in [.01,.1,1,10]:
       model=fit_readout(x[:128],y[:128],alpha,reverse);pred=predict(model,x[128:]).argmax(1);acc=float(np.mean(pred==y[128:]))
       if acc>best[0]:best=acc,alpha
      r={'recipe':recipe.copy(),'reverse':reverse,'accuracy':best[0],'alpha':best[1]};results.append(r)
      print(json.dumps(r),flush=True)
 save(p/('development-search-'+args.feature+'.json'),results)
print('BEST',json.dumps(sorted(results,key=lambda r:r['accuracy'],reverse=True)[:12]),flush=True)
