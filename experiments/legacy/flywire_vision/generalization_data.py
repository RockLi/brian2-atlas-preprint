"""Additional speed-4 development data; reuse verified immutable native bases."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from .multispeed_data import NativeRecorder, digest, speed_movie
from .motion_challenge import centers, render
from .motion_refinement import read, sha, load_npz
from .run_experiment import save


def generalization_movie(kind, group, phase, speed):
    if type(speed) is not int or not 1 <= speed <= 6:raise ValueError('integer speed 1 through 6 required')
    if speed <= 4:return speed_movie(kind, group, phase, speed)
    if kind in ('left', 'down'):
        return generalization_movie({'left':'right','down':'up'}[kind], group, phase, speed)[::-1].copy()
    start = centers(kind, group, 2., tuple(phase))[0]
    position = np.repeat(start[None], 40, axis=0)
    axis, sign = (0,1) if kind == 'right' else (1,-1)
    position[:,axis] += sign*2*speed*np.linspace(0,1,40)
    position = (position+1)%2-1;position[0] = position[-1] = start
    return render(position, group)


class SharedRecorder(NativeRecorder):
    def __init__(self, artifact, directory, parent, template=None, identity='intact'):
        super().__init__(artifact, directory, parent, template, identity)
        old = parent/identity/'base.bin'
        if sha(old) != self.runner.base_hash:raise ValueError('shared native base changed')
        # Only replace this newly created duplicate. Both runners check the
        # snapshot hash before every read; neither writes to the base file.
        self.runner.base.unlink()
        os.link(old, self.runner.base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact','parent','development','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=read(args.development/'report.json');rows=read(args.development/'rows.json')
    if report['status']!='complete' or sha(args.development/'features.npz')!=report['features_sha256']:raise ValueError('development changed')
    if len(rows)!=192 or any(r['split']!=('fit' if i<128 else 'validation') for i,r in enumerate(rows)):raise ValueError('development rows only')
    save(args.output/'protocol.json',{'schema':'flywire-generalization-development-v1','parent':str(args.parent),
        'parent_protocol_sha256':sha(args.parent/'protocol.json'),'development':str(args.development),
        'development_report_sha256':sha(args.development/'report.json'),'source_sha256':sha(Path(__file__)),
        'speed':4,'test_used':False})
    recorder=SharedRecorder(args.artifact,args.output/'execution',args.parent)
    blank,encoded,record=recorder.run(np.full((40,48,48),.5,dtype=np.float32),'blank')
    np.testing.assert_array_equal(blank,load_npz(args.development/'blank.npz')['neural'])
    np.savez_compressed(args.output/'blank.npz',neural=blank,encoded=encoded);save(args.output/'blank-events.json',record)
    neural=[];inputs=[];records=[];start=time.perf_counter()
    for idx,row in enumerate(rows):
        nn,ee,rr=recorder.run(generalization_movie(row['kind'],row['group'],row['phase_index'],4),f'development-{idx}')
        neural.append(nn);inputs.append(ee);records.append({**row,**rr,'speed':4})
        if (idx+1)%16==0:
            np.savez_compressed(args.output/'features.npz',neural=np.array(neural),encoded=np.array(inputs));save(args.output/'rows.json',records)
            print({'completed':idx+1,'native_runs':recorder.runs,'seconds':time.perf_counter()-start},flush=True)
    save(args.output/'report.json',{'status':'complete','native_runs':recorder.runs,'test_used':False,
        'protocol_sha256':sha(args.output/'protocol.json'),'rows_sha256':sha(args.output/'rows.json'),
        'features_sha256':sha(args.output/'features.npz'),'seconds':time.perf_counter()-start})


if __name__=='__main__':main()
