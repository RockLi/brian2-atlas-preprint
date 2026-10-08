"""Render source-checked initial/final assembly weights and recorded learning history."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from litwin_kumar_figures import COLORS, csv_rows


def render(primary_sources, output):
    import matplotlib.pyplot as plt
    manifest = json.loads((primary_sources/'complete-source-manifest.json').read_text())['files']
    hashes, runs, matrices, trajectories = {}, {}, {}, {}

    def verified(relative):
        path = primary_sources/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative not in manifest or actual != manifest[relative]['sha256']:
            raise ValueError(f'source differs from scientific manifest: {relative}')
        hashes[relative] = actual
        return path

    for backend, suite, analysis in [('rust', 'science', 'rust-primary-analysis'),
                                     ('cpp', 'cpp-science', 'cpp-science-analysis')]:
        for seed in [20260906, 20260907, 20260908]:
            label = f'full-seed-{seed}'
            run = json.loads(verified(f'results/{suite}/{label}/result.json').read_text())
            config = run['configuration']
            if (run['backend'] != backend or config['scale'] != 1 or config['mode'] != 'learn'
                    or config['seed'] != seed or run['biological_seconds'] != 2610
                    or not run['complete_training_protocol']):
                raise ValueError('complete matched full-scale learned protocols required')
            with np.load(verified(f'results/{analysis}/{label}/connectivity_source.npz')) as source:
                matrix = {name:source[name].copy() for name in ['initial','final','edge_counts','membership']}
            if (matrix['initial'].shape != (20,20) or matrix['final'].shape != (20,20)
                    or not np.isfinite(matrix['final']).all() or np.any(matrix['edge_counts'] <= 0)):
                raise ValueError('complete finite assembly matrices required')
            np.testing.assert_allclose(matrix['initial'], 2.76, rtol=0, atol=1e-10)
            with verified(f'results/{analysis}/{label}/weight_trajectory.csv').open() as stream:
                trajectory = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
            times = np.array([r['time_s'] for r in trajectory])
            if times[0] != 0 or times[-1] != 2610 or np.any(np.diff(times) <= 0):
                raise ValueError('unique ordered complete weight observations required')
            recorded = run.get('trajectory', [])
            if len(trajectory) != (1+len(recorded) if recorded else 2):
                raise ValueError('weight observations differ from run report')
            expected = recorded or [{'time_s':2610, 'weights':run['weights']}]
            for row, reference in zip(trajectory[1:], expected):
                if row['time_s'] != reference['time_s']:
                    raise ValueError('weight observation time differs from run report')
                for key, field in [('within_pf','within_mean_pf'),('between_pf','between_mean_pf')]:
                    if row[key] != reference['weights'][field]:
                        raise ValueError('weight observation differs from run report')
            runs[backend,seed], matrices[backend,seed], trajectories[backend,seed] = run,matrix,trajectory
    for seed in [20260906,20260907,20260908]:
        if runs['rust',seed]['configuration'] != runs['cpp',seed]['configuration']:
            raise ValueError('backend model configurations differ')
        for key in ['initial','edge_counts','membership']:
            a,b=matrices['rust',seed][key],matrices['cpp',seed][key]
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                raise ValueError('paired initial assembly structure differs')
    output.mkdir(parents=True,exist_ok=False)
    rows=[dict(backend=backend,seed=seed,**row) for (backend,seed),values in trajectories.items() for row in values]
    csv_rows(output/'weight_observations.csv',rows)
    matrix_rows=[]
    for (backend,seed),m in matrices.items():
        for post in range(20):
            for pre in range(20):
                matrix_rows.append(dict(backend=backend,seed=seed,post_assembly=post+1,pre_assembly=pre+1,
                    existing_edges=int(m['edge_counts'][post,pre]),initial_mean_pf=m['initial'][post,pre],final_mean_pf=m['final'][post,pre]))
    csv_rows(output/'assembly_matrix_source.csv',matrix_rows)
    with plt.rc_context({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,3,figsize=(9,5.2),layout='constrained')
        for row,backend in enumerate(['rust','cpp']):
            label='Rust' if backend=='rust' else 'Brian2 C++'
            m=matrices[backend,20260906]
            for col,(key,time) in enumerate([('initial',0),('final',2610)]):
                ax=axes[row,col]
                image=ax.imshow(m[key],origin='lower',extent=(.5,20.5,.5,20.5),vmin=1.78,vmax=21.4,cmap='viridis')
                ax.set(xlabel='Presynaptic membership',ylabel='Postsynaptic membership',
                       title=f'{label} · {time:,} s',xticks=[1,10,20],yticks=[1,10,20])
            ax=axes[row,2]
            for index,seed in enumerate([20260906,20260907,20260908]):
                values=trajectories[backend,seed]
                for key,color,name in [('within_pf',COLORS['within'],'Within'),('between_pf',COLORS['between'],'Between')]:
                    ax.plot([x['time_s'] for x in values],[x[key] for x in values],
                            color=color,linestyle=['-','--',':'][index] if backend=='rust' else 'None',
                            marker='o',markersize=2.5,label=f'{name} · {str(seed)[-2:]}')
            if backend=='rust':
                ax.axvspan(10,1610,color='#777777',alpha=.09,zorder=0)
                ax.axvline(1610,color='#777777',linewidth=.6)
            ax.set(xlabel='Biological time (s)',ylabel='Mean EE weight (pF)',ylim=(0,15),
                   title=f'{label} · '+('recorded learning history' if backend=='rust' else 'recorded endpoints only'))
            ax.legend(frameon=False,fontsize=6,ncol=1 if backend=='rust' else 2,
                      loc='center right' if backend=='rust' else 'upper left')
        fig.colorbar(image,ax=axes[:,:2],location='bottom',shrink=.7,pad=.04,label='Mean weight over existing edges (pF)',ticks=[1.78,10,21.4])
        for letter,ax in zip('abcdef',axes.flat):
            ax.text(-.13,1.07,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)
        fig.suptitle('Formation from uniform initial weights · N = 5,000',fontsize=11)
        fig.savefig(output/'assembly_formation.pdf')
        fig.savefig(output/'assembly_formation.png',dpi=220)
        plt.close(fig)
    summary=dict(all_sources_verified=True,paired_initial_structure_byte_exact=True,representative_matrix_seed=20260906,
                 source_sha256=hashes,observations={f'{b}/{s}':len(v) for (b,s),v in trajectories.items()},
                 cpp_intermediate_history_inferred=False)
    (output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'caption.md').write_text(
        'Initial and final mean EE weights between the 20 overlapping stimulus memberships are shown for '
        'the first declared network seed (20260906), using one fixed 1.78–21.4 pF scale for both backends '
        'and times. Means include existing edges only; absent edges are excluded. A neuron can belong '
        'to multiple memberships, so this is a membership-level summary rather than disjoint clusters. '
        'All six runs start from uniform 2.76 pF existing EE weights. Right panels include all three seeds '
        '(legend suffixes 06/07/08). Rust has recorded intermediate observations; the shaded interval '
        'is patterned training from 10 to 1610 s. C++ recorded only initial and final weights, which '
        'are shown as points without an inferred learning trajectory. Initial matrices, memberships and '
        'edge-count matrices match byte-for-byte across backends. The displayed seed and intervals '
        'are conventions selected during review, not preregistered. Separate controls and complete '
        'activity analysis are required for biological interpretation. Source matrices and weight '
        'observations are exported in the accompanying CSV files.\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    render(**vars(parser.parse_args()))
