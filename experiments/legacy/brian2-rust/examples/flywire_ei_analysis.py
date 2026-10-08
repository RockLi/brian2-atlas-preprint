"""Explicit numerical/activity checks, not a clinical or behavioural diagnosis."""
import numpy as np


def metrics(snapshot,groups,duration=1.):
    ids,times=snapshot['spike_i'],snapshot['spike_t']; n=len(snapshot['v'])
    start=min(.1,duration/10)
    valid=(times>=start)&(times<duration)
    bins=np.floor(times[valid]/.005).astype(np.int64)
    pairs=np.unique(bins*n+ids[valid])
    participation=np.bincount(pairs//n,minlength=int(np.ceil(duration/.005)))/n
    rate=np.bincount(ids[valid].astype(np.int64),minlength=n)/(duration-start)
    return {'mean_rate_hz':float(rate.mean()),'full_run_mean_rate_hz':len(ids)/(n*duration),
            'active_fraction':float(np.mean(rate>0)),
            'neuron_rate_quantiles_hz':np.quantile(rate,[0,.5,.9,.99,1]).tolist(),
            'peak_5ms_participating_fraction':float(participation.max()),
            'voltage_range_mv':[float(snapshot['v'].min()*1000),float(snapshot['v'].max()*1000)],
            'conductance_nonnegative':bool(np.all(snapshot['ge']>=0) and np.all(snapshot['gi']>=0)),
            'voltage_within_reversals':bool(np.all(snapshot['v']>=-.070-1e-12) and np.all(snapshot['v']<=1e-12)
                                            and np.all(snapshot['trace_v']>=-.070-1e-12) and np.all(snapshot['trace_v']<=1e-12)),
            'cell_sets':{k:{'count':len(v),'mean_rate_hz':float(rate[v].mean())} for k,v in groups.items()}}


def analyze(snapshots,groups,config):
    duration=config['duration_ms']/1000; begin=config['stimulus_start_ms']/1000;end=config['stimulus_end_ms']/1000
    results={k:metrics(v,groups,duration) for k,v in snapshots.items()}
    n=len(snapshots['rest']['v'])
    def counts(snapshot):
        t=snapshot['spike_t'];mask=(t>=begin)&(t<end)
        return np.bincount(snapshot['spike_i'][mask].astype(np.int64),minlength=n)
    window={k:counts(v) for k,v in snapshots.items()}
    response={}
    for name,indices in groups.items():
        rates={k:float(v[indices].mean()/(end-begin)) for k,v in window.items()}
        delta=rates['odor']-rates['rest'];cut_delta=rates['cut']-rates['cut_rest']
        def histogram(snapshot):
            mask=np.isin(snapshot['spike_i'],indices)
            return np.histogram(snapshot['spike_t'][mask],bins=np.arange(begin,end+.0005,.001))[0]
        difference=histogram(snapshots['odor'])-histogram(snapshots['rest'])
        positive=np.flatnonzero(difference>0)
        response[name]={**rates,'stimulus_delta_hz':delta,'cut_stimulus_delta_hz':cut_delta,
                        'difference_in_differences_hz':delta-cut_delta,
                        'first_positive_1ms_bin_after_onset_ms':int(positive[0]) if len(positive) else None}
    sensory=groups['sensory']
    def outside_spikes(snapshot):
        keep=~np.isin(snapshot['spike_i'],sensory)
        return snapshot['spike_i'][keep],snapshot['spike_t'][keep]
    cut_i,cut_t=outside_spikes(snapshots['cut']);rest_i,rest_t=outside_spikes(snapshots['cut_rest'])
    cut_blocks=bool(np.array_equal(cut_i,rest_i) and np.array_equal(cut_t,rest_t))
    gates={'rest_rate_1_to_10_hz':1<=results['rest']['mean_rate_hz']<=10,
           'preferred_rest_rate_1_to_5_hz':1<=results['rest']['mean_rate_hz']<=5,
           'bounded_voltage_and_conductance':all(v['voltage_within_reversals'] and v['conductance_nonnegative'] for v in results.values()),
           'sparse_5ms_participation_below_10_percent':all(v['peak_5ms_participating_fraction']<.1 for v in results.values()),
           'pn_and_kc_stimulus_response':all(response[k]['stimulus_delta_hz']>0 for k in ('pn','kc')),
           'cut_blocks_non_sensory_stimulus_response':cut_blocks}
    return {'conditions':results,'response_300_to_700ms':response,'gates':gates,
            'interpretation':'Engineering acceptance thresholds; rate and synchrony checks do not establish physiological or seizure diagnoses. Frozen shared background is a modelling assumption.'}
