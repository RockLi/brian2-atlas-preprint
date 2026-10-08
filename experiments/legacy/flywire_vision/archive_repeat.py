"""Finalize an independently verified P13 repetition archive; no simulator runs."""
import argparse
import os
import shutil
from pathlib import Path
from .motion_refinement import read,sha
from .run_experiment import save


def archive(source,destination):
    if destination.exists():raise ValueError('refuse overwriting a stage archive')
    p=read(source/'protocol.json');v=read(source/'verification.json');b=read(source/'budget.json');mut=read(source/'verifier-mutation-tests.json')
    if not v['all_passed'] or v['report_sha256']!=sha(source/'report.json') or not mut['all_passed']:raise ValueError('all independent checks must pass')
    if b['limit']!=300 or len(b['attempts'])!=265 or any(r['status']!='completed' for r in b['attempts']):raise ValueError('study/budget incomplete')
    if list((source/'executor/work').glob('*/result')):raise ValueError('unexpected transient result directory')
    for name in ('README.md','DECISION.md','summary.json','tests.log','independent-metrics.json','reference-signs.json'):
        if not (source/name).exists():raise ValueError('missing '+name)
    def copy(src,dst):
        if Path(src).suffix in ('.npz','.bin') or Path(src).name in ('b2-native','unit-tests'):os.link(src,dst)
        else:shutil.copy2(src,dst)
        return dst
    shutil.copytree(source,destination,copy_function=copy);snap=destination/'sources';snap.mkdir();names=set(p['sources'])|{'verify_repeat_study.py','repeat_report.py','test_repeat_verifier.py','archive_repeat.py','verify_background_clamp.py'}
    for name in sorted(names):shutil.copy2(Path(__file__).with_name(name),snap/name)
    save(destination/'source-hashes.json',{name:sha(snap/name) for name in sorted(names)})
    save(destination/'archive-manifest.json',{'source':str(source),'source_file_sha256':{str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()},'copy_policy':'immutable arrays and native/sidecar binaries hardlinked; metadata copied; frozen parent graph/model reused by hash'})
    print({'archive':str(destination),'files':sum(f.is_file() for f in destination.rglob('*'))},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path);a=parser.parse_args();archive(a.source,a.destination)
