"""Export descriptive EI response and cross-host timing figures (not confidence intervals)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True,help='Aggregate JSON with hosts -> report')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    r=json.loads(a.report.read_text())
    with np.load(a.graph/'annotations.npz') as z:groups={k:z[k] for k in ('sensory','pn','kc','mbon')}
    snapshots={}
    for condition in ('rest','odor','cut_rest','cut'):
        with np.load(a.snapshots/f'aot-{condition}-t1/snapshot.npz') as z:
            snapshots[condition]={k:z[k] for k in ('spike_i','spike_t')}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    bins=np.linspace(0,1,51);mid=(bins[:-1]+bins[1:])*500
    titles={'sensory':'DM1 ORNs (68)','pn':'DM1 projection neurons (2)',
            'kc':'Kenyon cells (5,177)','mbon':'Mushroom body output neurons (96)'}
    for ax,(group,ids) in zip(axes.flat,groups.items()):
        rates={c:np.histogram(s['spike_t'][np.isin(s['spike_i'],ids)],bins=bins)[0]/len(ids)/.02
               for c,s in snapshots.items()}
        ax.axvspan(300,700,color='#eedbae',alpha=.5,label='Input on')
        ax.plot(mid,rates['rest'],color='#717b86',label='Background only')
        ax.plot(mid,rates['odor'],color='#197f87',label='DM1 input')
        ax.plot(mid,rates['cut']-rates['cut_rest'],color='#ba6047',ls='--',label='Input delta with ORN output cut')
        ax.set(title=titles[group],xlabel='Simulation time (ms)',ylabel='Population mean rate (Hz)',xlim=(0,1000))
    axes.flat[0].set_ylim(-4,120)
    axes.flat[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('One frozen-input trial: downstream response depends on ORN output')
    fig.savefig(a.output/'sensory-response.png',dpi=180);fig.savefig(a.output/'sensory-response.svg');plt.close(fig)
    fig,axes=plt.subplots(1,len(r['hosts']),figsize=(12,4.2),sharey=True,constrained_layout=True)
    host_titles={'local':'Local M3\n4P + 4E, 16 GiB; noisy',
                 'mac27':'27 Mac Studio M1 Ultra\n20 cores, 128 GiB',
                 'linux23':'23 Linux EPYC 9454\n16-core NUMA allocation'}
    for ax,(host,item) in zip(np.atleast_1d(axes),r['hosts'].items()):
        report=item['report'];ts=sorted(map(int,report['performance']))
        for backend,color,label in [('aot','#197f87','Rust AOT'),('cpp','#ba6047','Brian2 C++')]:
            med=np.array([report['performance'][str(t)][backend]['median_seconds'] for t in ts])
            vals=[[v['loop_seconds'] for v in report['samples'][str(t)][backend]] for t in ts]
            error=np.array([med-np.array([min(v) for v in vals]),np.array([max(v) for v in vals])-med])
            ax.errorbar(ts,med,yerr=error,label=label,color=color,marker='o',capsize=3)
        ax.set(title=host_titles.get(host,item['label']),xlabel='Requested threads',ylabel='Simulation + recording (s)',xticks=ts,yscale='log')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('1,000 ms full-graph EI workload; median and full observed range')
    fig.savefig(a.output/'cross-host-timings.png',dpi=180);fig.savefig(a.output/'cross-host-timings.svg');plt.close(fig)


if __name__=='__main__':main()
