"""Descriptive, post-hoc voltage diagnostics for the two already frozen pilots.

These diagnostics never select a model parameter or change the formal protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def diagnose(folders, output):
    rows=[]
    for folder in folders:
        config=json.loads((folder/'configuration.json').read_text())
        result=json.loads((folder/'report.json').read_text())
        with np.load(folder/'snapshot.npz') as snapshot:
            indices=[config['recorded_indices'].index(i) for i in config['mbon_indices']]
            for trial in result['trials']:
                start=(trial['start_ms']+50)/1000
                selected=(snapshot['trace_t']>=start)&(snapshot['trace_t']<start+.25)
                v,ge,gi=[snapshot['neuron_trace'][k,indices][:,selected] for k in range(3)]
                rows.append({'amplitude_mv':config['amplitude_mv'],'cue':trial['cue'],
                             'ORN_hz':trial['input_hz'],'KC_hz':trial['kc_hz'],'MBON01_hz':trial['mbon_hz'],
                             'MBON01_mean_v_mv':float(v.mean()*1000),
                             'MBON01_max_v_mv':float(v.max()*1000),
                             'mean_ge':float(ge.mean()),'mean_gi':float(gi.mean()),
                             'threshold_mv':-45,
                             'source':str(folder.resolve()),
                             'snapshot_sha256':hashlib.sha256((folder/'snapshot.npz').read_bytes()).hexdigest()})
    output.write_text(json.dumps({'scope':'post-hoc description of the two calibration runs; no parameter selection',
                                 'rows':rows},indent=2)+'\n')
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folders',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();diagnose(a.folders,a.output)
