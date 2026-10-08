"""Describe two matched observations without asserting scientific equivalence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(baseline, corrected, output):
    old = json.loads((baseline/'activity.json').read_text())
    new = json.loads((corrected/'activity.json').read_text())
    assert old['window'] == new['window'] and old['sampling'] == new['sampling']
    assert old['window']['spike_tick_offset'] == 1
    assert old['neurons'] == new['neurons'] == 4129924
    assert [p['name'] for p in old['populations']] == [p['name'] for p in new['populations']]
    assert [a['area'] for a in old['areas']] == [a['area'] for a in new['areas']]
    output.mkdir(exist_ok=False)
    populations = []
    for before, after in zip(old['populations'], new['populations'], strict=True):
        assert before['neurons'] == after['neurons']
        populations.append(dict(name=before['name'], neurons=before['neurons'],
            old_rate_hz=before['mean_rate_hz'], new_rate_hz=after['mean_rate_hz'],
            delta_hz=after['mean_rate_hz']-before['mean_rate_hz'],
            old_lvr=before['lvr_eligible_mean'], new_lvr=after['lvr_eligible_mean'],
            old_lvr_eligible=before['lvr_eligible_cells'], new_lvr_eligible=after['lvr_eligible_cells'],
            old_correlation=before['pairwise_corr_mean'], new_correlation=after['pairwise_corr_mean']))
    report = dict(schema='b2-mam-timing-comparison-v1', scientific_equivalence=False,
        scope='One seed; two timing implementations; descriptive comparison only. No reference tolerance fitted.',
        window=new['window'], old_model_sha256=old['model_sha256'], new_model_sha256=new['model_sha256'],
        old_observed_spikes=old['observed_spikes'], new_observed_spikes=new['observed_spikes'],
        old_global_rate_hz=old['mean_rate_hz'], new_global_rate_hz=new['mean_rate_hz'],
        old_silent_populations=[p['name'] for p in old['populations'] if p['observed_spikes']==0],
        new_silent_populations=[p['name'] for p in new['populations'] if p['observed_spikes']==0],
        old_below_001_hz=[p['name'] for p in old['populations'] if p['mean_rate_hz']<.01],
        new_below_001_hz=[p['name'] for p in new['populations'] if p['mean_rate_hz']<.01],
        areas=[dict(area=a['area'], old=a, new=b) for a,b in zip(old['areas'],new['areas'],strict=True)],
        populations=populations,
        sampling_limit='LvR eligibility changes with activity; correlations sample active cells independently in each observation. Differences are not a paired statistical equivalence test.',
        input_sha256={str(p):sha(p) for folder in [baseline,corrected] for p in [folder/'activity.json',folder/'activity-arrays.npz']})
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(baseline/'activity-arrays.npz') as archive:
        old_rates=archive['area_rates_hz']
    with np.load(corrected/'activity-arrays.npz') as archive:
        new_rates=archive['area_rates_hz']
    assert old_rates.shape==new_rates.shape==(32,2000)
    areas=[a['area'] for a in new['areas']]
    fig,axes=plt.subplots(5,1,figsize=(12,12),sharex=True,layout='constrained')
    for axis,name in zip(axes,['V1','V2','FEF','MT','MIP'],strict=True):
        index=areas.index(name)
        for values,color,label in [(old_rates,'#777777','Legacy timing'),(new_rates,'#1267b3','Audited NEST timing')]:
            smooth=np.convolve(values[index],np.ones(20)/20,mode='valid')
            axis.plot(.5+(np.arange(len(smooth))+10)*.001,smooth,color=color,label=label,lw=1)
        before,after=old['areas'][index],new['areas'][index]
        axis.set(title=f"{name}   mean {before['mean_rate_hz']:.3f} → {after['mean_rate_hz']:.3f} Hz",
                 ylabel='Hz / neuron',xlim=(.5,2.5))
        axis.grid(alpha=.2)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel('Physical time (s); 20 ms smoothing for display')
    fig.suptitle('Full MAM: matched seed and topology, corrected timing\nOne 2 s observation; paper reproduction remains unproven')
    fig.savefig(output/'selected-area-comparison.png',dpi=150);plt.close(fig)
    catalog={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file()}
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps(dict(old_rate=old['mean_rate_hz'],new_rate=new['mean_rate_hz'],
        old_silent=len(report['old_silent_populations']),new_silent=len(report['new_silent_populations']),
        old_below_001=len(report['old_below_001_hz']),new_below_001=len(report['new_below_001_hz']),
        largest_new_rates=sorted(populations,key=lambda p:p['new_rate_hz'],reverse=True)[:5],
        selected_areas=[a for a in report['areas'] if a['area'] in ['V1','V2','FEF','MT','MIP']]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','corrected','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    compare(args.baseline,args.corrected,args.output)
