"""Fetch a size-bounded, commit-pinned official MAM parameter source bundle."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT='0a658be40bef3249cbe452f38809edf7d2f524ba'


def fetch(output):
    with urlopen(f'https://api.github.com/repos/INM-6/multi-area-model/git/trees/{COMMIT}?recursive=1',timeout=30) as response:
        tree=json.loads(response.read(4_000_001))
    if tree['sha']!=COMMIT or tree.get('truncated'):raise ValueError('incomplete/wrong official tree')
    files=[x for x in tree['tree'] if x['type']=='blob' and
           (x['path'].startswith('multiarea_model/') or x['path'].startswith('tests/') or
            x['path']=='figures/SchueckerSchmidt2017/K_prime_original.npy' or
            x['path'] in ['LICENSE','config_template.py','requirements.txt'])]
    if not files or sum(x['size'] for x in files)>8_000_000:raise ValueError('source exceeds 8 MB budget')
    output.mkdir(parents=True,exist_ok=False)
    def download(item):
        path=output/item['path']
        if not path.resolve().is_relative_to(output.resolve()):raise ValueError('invalid source path')
        with urlopen(f'https://raw.githubusercontent.com/INM-6/multi-area-model/{COMMIT}/'+item['path'],timeout=30) as response:
            data=response.read(item['size']+1)
        if len(data)!=item['size'] or hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=item['sha']:
            raise ValueError('Git object identity mismatch: '+item['path'])
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return {'path':item['path'],'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    with ThreadPoolExecutor(max_workers=4) as pool:manifest=list(pool.map(download,files))
    (output/'source-manifest.json').write_text(json.dumps({'commit':COMMIT,'files':manifest},indent=2)+'\n')
    print(json.dumps({'commit':COMMIT,'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    fetch(p.parse_args().output.resolve())
