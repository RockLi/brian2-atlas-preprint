"""Archive a verified P12b run without copying large immutable arrays twice."""
import argparse
import os
import shutil
from pathlib import Path
from .motion_refinement import read,sha
from .run_experiment import save


def archive(source,destination):
    if destination.exists():raise ValueError('archive already exists; never overwrite evidence')
    v=read(source/'verification.json');p=read(source/'protocol.json');budget=read(source/'budget.json')
    if not v['all_passed'] or v['report_sha256']!=sha(source/'report.json'):raise ValueError('verified report required')
    if any(a['status']!='completed' for a in budget['attempts']) or len(budget['attempts'])>200:raise ValueError('ledger not complete within budget')
    if any((source/'executor/work').glob('*/result')):raise ValueError('unexpected unarchived transient native results')
    for name in ('README.md','METHODS.md','DECISION.md','summary.json','independent-metrics.json','postchecks.json','tests.log','verifier-mutation-tests.json'):
        if not (source/name).is_file():raise ValueError('missing '+name)
    def copy(src,dst):
        path=Path(src)
        # Immutable arrays, native binary and sidecar .bin files are never written again.
        if path.suffix in ('.npz','.bin') or path.name in ('b2-native','unit-tests'):os.link(src,dst)
        else:shutil.copy2(src,dst)
        return dst
    shutil.copytree(source,destination,copy_function=copy)
    sources=destination/'sources';sources.mkdir()
    names=set(p['sources'])|{'test_background_clamp.py','verify_background_clamp.py','background_clamp_report.py','archive_background_clamp.py','test_background_clamp_verifier.py'}
    for name in sorted(names):shutil.copy2(Path(__file__).with_name(name),sources/name)
    save(destination/'source-hashes.json',{name:sha(sources/name) for name in sorted(names)})
    save(destination/'archive-manifest.json',{'source':str(source),'source_file_sha256':{str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()},'copy_policy':'hardlinks for immutable npz/bin/native binaries; separate copies for metadata; parent graph/model reused by frozen hashes'})
    print({'archive':str(destination),'files':sum(f.is_file() for f in destination.rglob('*'))},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path);args=parser.parse_args();archive(args.source,args.destination)
