"""Archive one complete P13c study without launching a simulator."""
import argparse
import os
import shutil
from pathlib import Path
from .verify_onset_diagnostic import read,save,sha


def archive(source,destination):
    assert not destination.exists(),'refuse overwrite'
    p=read(source/'protocol.json');v=read(source/'verification.json');b=read(source/'budget.json');m=read(source/'mutation-tests.json')
    assert v['all_passed'] and m['all_passed'] and v['verifier_sha256']==m['verifier_sha256']
    assert len(b['attempts'])==116 and b['limit']==128 and all(x['status']=='completed' for x in b['attempts'])
    assert not list((source/'executor/work').glob('*/result'))
    for n in ('README.md','DECISION.md','summary.json','new-response-accounting.json','tests.log'):assert (source/n).is_file()
    def copy(src,dst):
        if Path(src).suffix in ('.bin','.npz') or Path(src).name=='b2-native':os.link(src,dst)
        else:shutil.copy2(src,dst)
        return dst
    shutil.copytree(source,destination,copy_function=copy);snap=destination/'sources';snap.mkdir();names=set(p['sources'])|{'onset_report.py','verify_onset_diagnostic.py','test_onset_verifier.py','archive_onset_diagnostic.py','verify_background_clamp.py'}
    for n in names:shutil.copy2(Path(__file__).with_name(n),snap/n)
    save(destination/'source-hashes.json',{n:sha(snap/n) for n in sorted(names)});save(destination/'archive-manifest.json',{'source':str(source),'source_file_sha256':{str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()},'policy':'immutable arrays and binaries hardlinked; metadata copied; native/model/graph and old conditions reused by identity; report regenerated for archive paths'})
    print({'archive':str(destination),'files':sum(f.is_file() for f in destination.rglob('*'))},flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('source',type=Path);a.add_argument('destination',type=Path);p=a.parse_args();archive(p.source,p.destination)
