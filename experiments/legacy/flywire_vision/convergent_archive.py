"""Lossless float-bit XOR archive against each condition's blank trace."""
import argparse
import io
import hashlib
from pathlib import Path
import numpy as np
from .motion_refinement import read,sha,load_npz
from .multispeed_data import digest
from .run_experiment import save


def encode(data,blank):
    return {k:np.bitwise_xor(v.view(np.uint64),blank[k].view(np.uint64)) if k in ('v','ge','gi') else v for k,v in data.items()}


def decode(data,blank):
    return {k:np.bitwise_xor(v,blank[k].view(np.uint64)).view(np.float64) if k in ('v','ge','gi') else v for k,v in data.items()}


def archive(root,destination,prune_source=False):
    if not read(root/'verification.json')['all_passed']:raise ValueError('verified study required')
    destination.mkdir(parents=True,exist_ok=False);rows=read(root/'rows.json');blanks={}
    for c in ('intact','cut_input'):
        row=next(r for r in rows if r['condition']==c and r['kind']=='blank');blanks[c]=load_npz(root/'trials'/(row['key']+'.npz'))
        np.savez_compressed(destination/(c+'-blank.npz'),**blanks[c])
    files=[(r['key'],r['condition'],r['data_sha256']) for r in rows]
    files += [(key,'intact',read(root/name)['data_sha256']) for key,name in [('monitor_control','monitor-control.json'),('restoration','restoration.json')]]
    manifest=[]
    for key,c,rawhash in files:
        raw=root/'trials'/(key+'.npz');assert sha(raw)==rawhash
        data=load_npz(raw);path=destination/(key+'.npz');np.savez_compressed(path,**encode(data,blanks[c]))
        restored=decode(load_npz(path),blanks[c]);assert all(np.array_equal(v.view(np.uint8),restored[k].view(np.uint8)) for k,v in data.items())
        rebuilt=io.BytesIO();np.savez_compressed(rebuilt,**restored)
        assert hashlib.sha256(rebuilt.getvalue()).hexdigest()==rawhash
        if prune_source:raw.unlink()  # Only after exact original-container reconstruction succeeds.
        manifest.append({'key':key,'blank':c+'-blank.npz','encoded_sha256':sha(path),'original_npz_sha256':rawhash,
            'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':digest(v)} for k,v in data.items()}})
    save(destination/'manifest.json',{'encoding':'lossless IEEE-754 uint64 XOR for v/ge/gi against condition blank; other arrays copied; compressed NPZ; original zip-container AND array bytes reconstruct exactly with the recorded NumPy environment',
        'source_sha256':sha(Path(__file__)),'numpy_version':np.__version__,'source_trials_pruned':prune_source,'blank_sha256':{c:sha(destination/(c+'-blank.npz')) for c in blanks},'rows':manifest})
    verify(destination)


def verify(root):
    m=read(root/'manifest.json');assert sha(Path(__file__))==m['source_sha256'];blanks={c+'.npz':load_npz(root/(c+'.npz')) for c in ('intact-blank','cut_input-blank')}
    for c,h in m['blank_sha256'].items():assert sha(root/(c+'-blank.npz'))==h
    for row in m['rows']:
        path=root/(row['key']+'.npz');assert sha(path)==row['encoded_sha256'];data=decode(load_npz(path),blanks[row['blank']])
        rebuilt=io.BytesIO();np.savez_compressed(rebuilt,**data);assert hashlib.sha256(rebuilt.getvalue()).hexdigest()==row['original_npz_sha256']
        assert {k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':digest(v)} for k,v in data.items()}==row['arrays']
    save(root/'verification.json',{'all_passed':True,'arrays_exact':True,'original_npz_exact':True,'records':len(m['rows']),'manifest_sha256':sha(root/'manifest.json')});print({'all_passed':True,'archived_records':len(m['rows'])},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--destination',type=Path);ap.add_argument('--prune-source',action='store_true');a=ap.parse_args()
    if a.destination:archive(a.root,a.destination,a.prune_source)
    else:verify(a.root)
