"""Archive the verified zero-native-run P13d analysis."""
import argparse
import os
import shutil
from pathlib import Path
from .verify_minimal_response import read,save,sha


def archive(source,destination):
    assert not destination.exists(),'refuse overwrite';p=read(source/'protocol.json');v=read(source/'verification.json');m=read(source/'mutation-tests.json')
    assert v['all_passed'] and m['all_passed'] and v['verifier_sha256']==m['verifier_sha256'] and p['new_native_runs']==0
    for n in ('README.md','DECISION.md','summary.json','tests.log'):assert (source/n).exists()
    def copy(src,dst):
        if Path(src).suffix=='.npz':os.link(src,dst)
        else:shutil.copy2(src,dst)
        return dst
    shutil.copytree(source,destination,copy_function=copy);snap=destination/'sources';snap.mkdir();names=set(p['sources'])|{'verify_minimal_response.py','minimal_response_report.py','test_minimal_verifier.py','archive_minimal_response.py'}
    for n in names:shutil.copy2(Path(__file__).with_name(n),snap/n)
    save(destination/'source-hashes.json',{n:sha(snap/n) for n in sorted(names)});save(destination/'archive-manifest.json',{'source':str(source),'source_file_sha256':{str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()},'policy':'minimal predictions hardlinked, metadata copied, original full-network data referenced by frozen identity; README regenerated for archive paths'})
    print({'archive':str(destination),'files':sum(f.is_file() for f in destination.rglob('*'))},flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('source',type=Path);a.add_argument('destination',type=Path);p=a.parse_args();archive(p.source,p.destination)
