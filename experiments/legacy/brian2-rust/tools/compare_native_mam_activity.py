"""Descriptive native NEST/Rust comparison; independent stochastic realizations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import zipfile


def compare(rust_dir,nest_dir,output):
    rust=json.loads((rust_dir/'activity.json').read_text());nest=json.loads((nest_dir/'activity.json').read_text())
    for folder in [rust_dir,nest_dir]:
        catalog=json.loads((folder/'catalog.json').read_text())
        for name in ['activity.json','activity-arrays.npz']:
            path=folder/name
            assert path.stat().st_size==catalog[name]['bytes']
            assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==catalog[name]['sha256']
        assert (folder/'activity-arrays.npz').stat().st_size <= 512*2**20
        with zipfile.ZipFile(folder/'activity-arrays.npz') as archive:
            entries=archive.infolist()
            assert len(entries)<=2000 and sum(row.file_size for row in entries)<=512*2**20
    for key in ['start_tick','end_tick','endpoint','dt_seconds','bin_ticks','seconds']:
        assert rust['window'][key]==nest['window'][key],key
    assert (rust['window']['spike_tick_offset'],nest['window']['spike_tick_offset'])==(1,0)
    for report in [rust,nest]:
        w=report['window'];assert w['raw_start_tick']+w['spike_tick_offset']==w['start_tick'] and w['raw_end_tick']+w['spike_tick_offset']==w['end_tick']
    w=rust['window']
    assert w['start_tick']==5000 and w['end_tick'] in (25000,105000,1005000) and w['dt_seconds']==.0001 and w['bin_ticks']==10 and w['endpoint']=='[start,end)'
    start,end=w['start_tick']*w['dt_seconds'],w['end_tick']*w['dt_seconds']
    duration=end-start;bins=(w['end_tick']-w['start_tick'])//w['bin_ticks']
    assert w['seconds']==duration
    assert rust['sampling']==nest['sampling']
    assert rust['neurons']==nest['neurons']==4129924
    assert [r['name'] for r in rust['populations']]==[r['name'] for r in nest['populations']]
    assert [r['area'] for r in rust['areas']]==[r['area'] for r in nest['areas']]
    rows=[]
    for a,b in zip(rust['populations'],nest['populations'],strict=True):
        assert a['neurons']==b['neurons']
        rows.append(dict(name=a['name'],neurons=a['neurons'],rust_rate_hz=a['mean_rate_hz'],nest_rate_hz=b['mean_rate_hz'],delta_hz=b['mean_rate_hz']-a['mean_rate_hz'],rust_lvr=a['lvr_eligible_mean'],nest_lvr=b['lvr_eligible_mean'],rust_lvr_eligible=a['lvr_eligible_cells'],nest_lvr_eligible=b['lvr_eligible_cells'],rust_correlation=a['pairwise_corr_mean'],nest_correlation=b['pairwise_corr_mean']))
    with np.load(rust_dir/'activity-arrays.npz',allow_pickle=False) as f:rust_rates=f['area_rates_hz']
    with np.load(nest_dir/'activity-arrays.npz',allow_pickle=False) as f:nest_rates=f['area_rates_hz']
    assert rust_rates.shape==nest_rates.shape==(32,bins)
    output.mkdir(exist_ok=False)
    result=dict(schema='b2-native-rust-mam-description-v1',scientific_equivalence=False,
        scope=f'One realization per simulator over physical [{start:g},{end:g}) s. Intended full-model condition; this tool checks physical windows, population sizes and sampling, not every input semantic. Random graph/input realizations and rank layouts differ. Descriptive differences are not an equivalence test or evidence of a speed/cost advantage.',
        physical_window={k:nest['window'][k] for k in ['start_tick','end_tick','endpoint','dt_seconds','bin_ticks','seconds']},sampling=nest['sampling'],rust_model_sha256=rust['model_sha256'],nest_parameters_sha256=nest['parameters_sha256'],rust_spikes=rust['observed_spikes'],nest_spikes=nest['observed_spikes'],rust_rate_hz=rust['mean_rate_hz'],nest_rate_hz=nest['mean_rate_hz'],populations=rows,
        areas=[dict(area=a['area'],rust=a,nest=b) for a,b in zip(rust['areas'],nest['areas'],strict=True)],
        silent_populations={name:[p['name'] for p in r['populations'] if p['observed_spikes']==0] for name,r in [('rust',rust),('nest',nest)]},
        below_001_hz={name:[p['name'] for p in r['populations'] if p['mean_rate_hz']<.01] for name,r in [('rust',rust),('nest',nest)]},
        sampling_limit='Active-cell correlation samples and LvR eligibility can differ. No paired-cell claim, tolerance fitting, or outcome-based calibration.',
        input_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for folder in [rust_dir,nest_dir] for p in [folder/'activity.json',folder/'activity-arrays.npz']})
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    areas=[r['area'] for r in nest['areas']];fig,axes=plt.subplots(5,1,figsize=(12,12),sharex=True,layout='constrained')
    for ax,name in zip(axes,['V1','V2','FEF','MT','MIP'],strict=True):
        i=areas.index(name)
        for rates,color,label in [(rust_rates,'#777777','Rust'),(nest_rates,'#1267b3','Native NEST')]:
            values=np.convolve(rates[i],np.ones(20)/20,mode='valid');ax.plot(start+(np.arange(len(values))+10)*.001,values,label=label,color=color,lw=1)
        ax.set(title=f"{name}: Rust {rust['areas'][i]['mean_rate_hz']:.3f} / NEST {nest['areas'][i]['mean_rate_hz']:.3f} Hz",ylabel='Hz / neuron',xlim=(start,end));ax.grid(alpha=.2)
    axes[0].legend(ncol=2);axes[-1].set_xlabel('Physical time (s); 20 ms display smoothing')
    fig.suptitle(f'Full MAM: independent native NEST and Rust realizations\nOne {duration:g} s observation; scientific equivalence unproven')
    fig.savefig(output/'selected-area-comparison.png',dpi=150);plt.close(fig)
    catalog={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in output.iterdir() if p.is_file()};(output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps(dict(rust_rate=result['rust_rate_hz'],nest_rate=result['nest_rate_hz'],largest_nest_rates=sorted(rows,key=lambda p:p['nest_rate_hz'],reverse=True)[:5],selected_areas=[r for r in result['areas'] if r['area'] in ['V1','V2','FEF','MT','MIP']]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['rust-dir','nest-dir','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();compare(a.rust_dir,a.nest_dir,a.output)
