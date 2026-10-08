"""Descriptive, consistently ordered Rust/NEST/published FC matrices."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_mam_paper_fc import checked,sha,compare_fc


def compare(rust,native,output):
    matrices=[];reports=[]
    for folder,simulator in [(rust,'Rust'),(native,'NEST')]:
        catalog=json.loads((folder/'catalog.json').read_text())
        report=json.loads(checked(folder,catalog,'fc.json',2**20).read_text())
        with np.load(checked(folder,catalog,'fc.npz',64*2**20),allow_pickle=False) as data:
            matrix=data['functional_connectivity'];reference=data['reference_fc']
        if report['schema']!='b2-mam-paper-fc-v1' or report['identity']['simulator']!=simulator or not report['validated_rate_input']:
            raise ValueError('unexpected analyzed FC artifact')
        if matrices:np.testing.assert_array_equal(reference,original)
        else:original=reference
        matrices.append(matrix);reports.append(report)
    first,second=reports
    for key in ['parameters_sha256','area_names','observation_seconds','settings','reference_observation_seconds']:
        if first[key]!=second[key]:raise ValueError('incompatible comparison: '+key)
    if first['source_sha256']['reference_report']!=second['source_sha256']['reference_report']:
        raise ValueError('reference reconstruction differs')
    a,b=matrices;areas=first['area_names'];seconds=first['observation_seconds']
    metrics={'rust_vs_published':compare_fc(a,original),'nest_vs_published':compare_fc(b,original),'rust_vs_nest':compare_fc(a,b)}
    result=dict(schema='b2-mam-fc-comparison-v1',scientific_equivalence=False,
        model_observation_seconds=seconds,published_observation_seconds=first['reference_observation_seconds'],
        equal_observation_duration=first['equal_observation_duration'] and second['equal_observation_duration'],
        metrics=metrics,area_names=areas,parameters_sha256=first['parameters_sha256'],
        source_sha256=dict(rust_report=sha(rust/'fc.json'),rust_arrays=sha(rust/'fc.npz'),nest_report=sha(native/'fc.json'),nest_arrays=sha(native/'fc.npz')),
        implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('analyze_mam_paper_fc.py')]},
        scope='Descriptive synaptic-input FC matrices in the same official area order, no fitted clustering or acceptance cutoff. The 496 dependent pair entries are not independent trials; seeds and observation duration need replication. This does not override unresolved rate/LvR/correlation discrepancies or demonstrate simulator, BOLD, propagation, causal or cost equivalence.')
    output.mkdir(exist_ok=False)
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    top=[original,a,b];titles=[f'Published model | {first["reference_observation_seconds"]:g} s',f'Rust | {seconds:g} s',f'NEST | {seconds:g} s']
    for ax,matrix,title in zip(axes[0],top,titles,strict=True):
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r',interpolation='nearest');ax.set_title(title,fontsize=12)
    fig.colorbar(im,ax=axes[0].tolist(),label='Synaptic-input correlation',shrink=.8)
    diffs=[a-original,b-original,a-b];labels=['Rust - published','NEST - published','Rust - NEST']
    vmax=max(.05,np.ceil(max(float(np.max(np.abs(d))) for d in diffs)*20)/20)
    for ax,matrix,title,key in zip(axes[1],diffs,labels,metrics,strict=True):
        im=ax.imshow(matrix,vmin=-vmax,vmax=vmax,cmap='RdBu_r',interpolation='nearest')
        metric=metrics[key];ax.set_title(f'{title}\nMAE {metric["mean_absolute_difference"]:.4f}; entry r {metric["entry_correlation"]:.4f}',fontsize=11)
    fig.colorbar(im,ax=axes[1].tolist(),label='Correlation difference',shrink=.8)
    for ax in axes.flat:
        ax.set_xticks(range(32),areas,rotation=90,fontsize=6)
        ax.set_yticks(range(32),areas,fontsize=6)
    duration='Unequal observation durations' if not result['equal_observation_duration'] else 'Matched observation durations'
    fig.suptitle('Multi-Area Cortical Model | Figure 8 synaptic-input FC\n'+duration+'; descriptive comparison, scientific equivalence unproven',fontsize=14)
    fig.savefig(output/'fc-comparison.png',dpi=150);plt.close(fig)
    (output/'catalog.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir()},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['rust','native','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();compare(a.rust,a.native,a.output)
