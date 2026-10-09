"""Build portable HTML and vector/PNG figures from frozen manuscript data."""
from pathlib import Path
import base64
import hashlib
import html
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/b2-preprint-matplotlib')
# Use the bundled DejaVu fonts for portable, headless figure generation.
os.environ.setdefault('MPL_IGNORE_SYSTEM_FONTS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
try:
    import mistune
except ImportError:
    mistune = None
    from markdown_it import MarkdownIt
from PIL import Image, ImageOps, ImageDraw

D = json.loads((ROOT/'data/figure_data.json').read_text())
P = json.loads((ROOT/'data/plan_selection.json').read_text())
V3 = json.loads((ROOT/'data/v3/evidence.json').read_text())
MODELS = json.loads((ROOT/'data/published_models/evidence.json').read_text())
HET = json.loads((ROOT/'data/mpi_heterogeneous/evidence.json').read_text())
CAP = json.loads((ROOT/'data/capacity/evidence.json').read_text())
WEAK = json.loads((ROOT/'data/capacity/weak-evidence.json').read_text()) if (ROOT/'data/capacity/weak-evidence.json').exists() else None
table = ['# GPU comparator accounting', '',
         'Generated from retained cohort reports. All times are complete replay medians in milliseconds; failed qualifications have no ranked time. Modified adapters remain explicitly labeled.', '',
         '| Host | Ring neurons | Worker | Median (ms) | Qualification |',
         '|---|---:|---|---:|---|']
worker_names = {'rust-f64':'Rust CPU f64 (serial slot)', 'cpu-f32':'Compiled f32 expression control',
                'brian2genn-corrected':'Brian2GeNN schedule-corrected variant',
                'brian2genn-f32-factor':'Brian2GeNN f32-factor variant',
                'genn-barrier':'GeNN barrier variant', 'genn-barrier-gather':'GeNN barrier + gather variant',
                'genn':'Direct GeNN original ring adapter'}
for host,cases in D['gpu_ring'].items():
    for case,result in cases.items():
        for r in result['summary']:
            passed = r.get('complete') and r.get('matched_gate_passed')
            median = f"{r['median_seconds']*1000:.2f}" if passed else '—'
            table.append(f"| {host.upper()} | {case.split('-')[-1]} | {worker_names.get(r['backend'], r['backend'])} | {median} | {'passed matched gate' if passed else 'excluded: failed/incomplete qualification'} |")
table += ['', '## Recurrent CUBA', '',
          'Direct GeNN here uses the disclosed Brian-Euler variant. Matched f32 qualification does not imply f64 equivalence.', '',
          '| Host | Worker | Median (ms) | Min–max (ms) |', '|---|---|---:|---|']
for host,result in D['gpu_cuba'].items():
    for r in result['summary']:
        name = 'Direct GeNN Brian-Euler variant' if r['backend']=='genn' else r['backend']
        table.append(f"| {host.upper()} | {name} | {r['median_seconds']*1000:.2f} | {r['min_seconds']*1000:.2f}–{r['max_seconds']*1000:.2f} |")
(ROOT/'data/gpu_comparisons.md').write_text('\n'.join(table)+'\n')
selection_table=['# GPU policy-selection and exact-input cache records', '',
                 'Derived from six hash-verified archived reports. Profiling ranges are observed minima and maxima, not confidence intervals.', '',
                 '## All calibration candidates', '',
                 '| Host / case | Candidate | Status | Median (ms) | Min–max (ms) | Selected |',
                 '|---|---|---|---:|---:|---|']
for r in P['autotune']['rows']:
    for name,c in r['candidates'].items():
        ys=np.array(c['seconds'])*1000
        summary=(f'{np.median(ys):.3f}',f'{min(ys):.3f}–{max(ys):.3f}') if len(ys) else ('—','—')
        selection_table.append(f"| {r['host']} {r['case']} | {name} | {c['status']} | {summary[0]} | {summary[1]} | {'yes' if name==r['selected'] else 'no'} |")
selection_table += ['', '## Later exact-input cache cohort', '',
                    'Activation costs include transport and enabled allocation/compilation reuse, excluding Brian frontend lowering. These are a separate source cohort.', '',
                    '| Host / case | Selected | Calibration + transport (s) | Three hit activations (s) | Hit median (s) | f64 diagnostic |',
                    '|---|---|---:|---|---:|---|']
for r in P['tuning-cache']['rows']:
    hits=', '.join(f'{x:.3f}' for x in r['hit_seconds'])
    selection_table.append(f"| {r['host']} {r['case']} | {r['selected']} | {r['calibration_transport_seconds']:.3f} | {hits} | {r['hit_median_seconds']:.3f} | {'passed' if r['f64_passed'] else 'failed'} |")
(ROOT/'data/plan_selection_tables.md').write_text('\n'.join(selection_table)+'\n')
OUT = ROOT/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                     'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
                     'svg.hashsalt':'brian2-preprint-v1','figure.dpi':140,'savefig.dpi':170})
BLUE, ORANGE, GREEN, GREY = '#245c85', '#bc6731', '#327b64', '#70777e'
FILES=[]

def save(fig, name):
    fig.savefig(OUT/f'{name}.svg', bbox_inches='tight', metadata={'Date':None})
    fig.savefig(OUT/f'{name}.png', bbox_inches='tight', facecolor='white')
    FILES.append(name)
    plt.close(fig)

def panel(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc='left', fontweight='bold', pad=12)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis='y',alpha=.16,zorder=0)

def box(ax,x,y,w,h,label,color=BLUE,size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.015',
                 linewidth=1.1,edgecolor=color,facecolor=color+'0d'))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color='#172a39',linespacing=1.5)

def arrow(ax,a,b,color=GREY):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.3})

# 1: architecture. Explicitly separate retained frontend and target implementations.
fig,ax=plt.subplots(figsize=(11,7.6)); ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.20,.87,.60,.095,'Retained Brian2 frontend\nObjects • equations • units • abstract update statements')
box(ax,.22,.70,.56,.105,'AtlasIR: Definition / Instance / Run\nTypes • clocks • events • effects • identities',GREEN)
arrow(ax,(.5,.87),(.5,.817))
box(ax,.12,.52,.76,.115,'Independent validation → LogicalPlan\nCompleted dependencies • clocks • transformation eligibility',GREEN)
arrow(ax,(.5,.70),(.5,.647))
box(ax,.12,.355,.76,.105,'Physical planning within the selected backend\nRules + work estimates • optional verified GPU calibration',BLUE)
arrow(ax,(.5,.52),(.5,.472))
labels=['CPU plan\nRust AOT','GPU plans\nMetal / CUDA','DistributedPlan\nMPI ranks','WasmPlan\nShared reference']
for x,l in zip([.02,.265,.51,.755],labels):
    box(ax,x,.17,.215,.10,l,BLUE,10)
    arrow(ax,(.5,.355),(x+.1075,.282))
    arrow(ax,(x+.1075,.17),(x+.1075,.117))
box(ax,.02,.015,.95,.09,'Validated execution → results + observed runtime binding\nExplain: selected paths, reasons and available execution context',GREEN)
save(fig,'fig1_architecture')

# 2: deterministic semantic example; no measured/error-bar representation.
fig,axes=plt.subplots(1,2,figsize=(11,3.8),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
a=axes[0];a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
for y,t in [(.72,'Clock-driven synaptic weight w'),(.42,'Summed write: target.total ← w'),(.12,'Linked reader: x ← x + target.total')]:box(a,.06,y,.86,.14,t)
arrow(a,(.49,.72),(.49,.57));arrow(a,(.49,.42),(.49,.27))
a.set_title('a  A consumer makes intermediate state observable',loc='left',fontweight='bold',fontsize=10)
a=axes[1];d=D['semantics_example'];a.plot(d['ticks'],d['canonical_x'],'o-',color=BLUE,label='Canonical schedule')
a.plot(d['ticks'],d['invalid_final_only_x'],'s--',color=ORANGE,label='Invalid final-only evaluation')
panel(a,'b  Four-step illustrative trace','Start-of-tick observation','Reader state x');a.set_xticks(d['ticks']);a.legend(fontsize=8,loc='upper left')
save(fig,'fig2_semantics')

# 3: machine-readable MPI samples and independent primary resource peaks.
fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
x=np.arange(3)
for v,c,dx in [('before',ORANGE,-.17),('after',BLUE,.17)]:
    rows=[r for r in D['mpi_ranklocal'] if r['version']==v]
    axes[0].bar(x+dx,[r['max_rank_rss_mib'] for r in rows],.31,color=c,label=v.capitalize(),zorder=3)
    for i,r in enumerate(rows):
        axes[1].scatter(np.full(3,i+dx)+np.linspace(-.035,.035,3),r['samples_seconds'],color=c,s=20,zorder=4)
        axes[1].plot([i+dx-.09,i+dx+.09],[r['median_seconds']]*2,color=c,lw=2.5)
for a in axes[:2]:a.set_xticks(x,['1','2','4'])
panel(axes[0],'a  Rank-local storage','MPI ranks','Maximum rank RSS (MiB)');axes[0].legend(fontsize=8)
panel(axes[1],'b  Time incl. recording and gather','MPI ranks','Seconds');axes[1].set_ylim(0,15)
peaks=np.array(D['mpi_primary']['proxy_peak_bytes'])/2**30
axes[2].bar(np.arange(4),peaks,color=GREEN,width=.6,zorder=3)
axes[2].axhline(256,color=GREY,ls='--',lw=1);axes[2].set_ylim(0,300);axes[2].set_xticks(range(4),['A','B','C','D'])
panel(axes[2],'c  Separate 24.1B-synapse run','Node / 8 ranks per node','Independent proxy peak (GiB)')
axes[2].text(.04,.91,'256 GiB ceiling',transform=axes[2].transAxes,fontsize=8,color=GREY)
save(fig,'fig3_distributed')

# 4: CPU report medians. Do not invent samples or confidence intervals.
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for col,host in enumerate(['M1 Ultra','EPYC 9454 × 2']):
    rows=[r for r in D['cpu_flywire'] if r['host']==host];xs=[r['threads'] for r in rows]
    for backend,color,label in [('rust',BLUE,'Rust'),('cpp',ORANGE,'Brian C++')]:
        axes[0,col].plot(xs,[r[f'{backend}_seconds'] for r in rows],'o-',color=color,label=label)
        axes[1,col].plot(xs,[r[f'{backend}_mib'] for r in rows],'o-',color=color,label=label)
    for r in rows:
        if r['cpp_unstable']:
            axes[0,col].scatter([r['threads']],[r['cpp_seconds']],marker='x',s=110,lw=2.5,color='black',zorder=5)
    for a in axes[:,col]:a.set_xticks(xs);a.legend(fontsize=8)
    panel(axes[0,col],f'{chr(97+col)}  {host}', 'Threads','Simulation + recording (s)')
    panel(axes[1,col],f'{chr(99+col)}  Native process memory','Threads','Peak RSS (MiB)')
    axes[0,col].set_ylim(bottom=0);axes[1,col].set_ylim(bottom=0)
axes[0,0].text(.03,.94,'×  Variability threshold exceeded',transform=axes[0,0].transAxes,fontsize=8,va='top')
save(fig,'fig4_cpu')

# 5: ring raw samples, recurrent report min–max. Every panel is one host.
fig=plt.figure(figsize=(12,8.6),layout='constrained');gs=fig.add_gridspec(2,6)
fig.set_constrained_layout_pads(h_pad=.18, hspace=.15, wspace=.12)
labels={'rust-f64':'Rust CPU f64','cpu-f32':'CPU f32 control','metal':'Metal f32','cuda':'CUDA f32','genn':'GeNN variant f32'}
colors={'rust-f64':BLUE,'cpu-f32':GREY,'metal':GREEN,'cuda':GREEN,'genn':ORANGE}
for col,host in enumerate(['m3','l4','a100']):
    a=fig.add_subplot(gs[0,col*2:col*2+2]);gpu='metal' if host=='m3' else 'cuda'
    rows={r['backend']:r for r in D['gpu_ring'][host]['ring-16384']['summary']}
    keys=['rust-f64','cpu-f32',gpu]
    for i,k in enumerate(keys):
        r=rows[k];ys=np.array(r['samples_seconds'])*1000
        assert len(ys)==5 and r['matched_gate_passed']
        a.scatter(i+np.linspace(-.07,.07,5),ys,color=colors[k],s=19)
        a.plot([i-.16,i+.16],[np.median(ys)]*2,color=colors[k],lw=2)
    a.set_xticks(range(3),[labels[k].replace(' ','\n',1) for k in keys],fontsize=8)
    panel(a,f'{chr(97+col)}  Ring STDP · {host.upper()}',ylabel='Replay (ms)');a.set_ylim(bottom=0)
for col,host in enumerate(['l4','a100']):
    a=fig.add_subplot(gs[1,col*3:col*3+3]);rows={r['backend']:r for r in D['gpu_cuba'][host]['summary']}
    keys=['rust-f64','cuda','genn']
    for i,k in enumerate(keys):
        r=rows[k];m=r['median_seconds']*1000
        a.bar(i,m,width=.5,color=colors[k],alpha=.85)
        a.errorbar(i,m,yerr=[[m-r['min_seconds']*1000],[r['max_seconds']*1000-m]],color='#222222',capsize=4,fmt='none')
    a.set_xticks(range(3),[labels[k] for k in keys],fontsize=9)
    panel(a,f'{chr(100+col)}  Recurrent CUBA · {host.upper()}',ylabel='Replay (ms)');a.set_ylim(bottom=0)
save(fig,'fig5_gpu')

# 6: different memory scopes, single observations, separate recovery result.
fig,axes=plt.subplots(1,3,figsize=(11.4,3.5),layout='constrained');life=D['lifecycle']
for a,key,title in zip(axes[:2],['native_gb','summed_tree_gb'],['a  Native-child RSS','b  Summed process-tree peaks']):
    a.bar(range(3),life[key],color=[BLUE,GREEN,ORANGE],width=.6,zorder=3)
    a.set_xticks(range(3),['Rust\nfull','Rust\nrolling','Brian\nC++']);a.set_ylim(0,5.6)
    panel(a,title,ylabel='Memory (decimal GB)')
a=axes[2];a.axis('off');a.set_title('c  Process-restart conformance',loc='left',fontweight='bold')
a.text(.06,.80,'3 fresh processes\n\n1,000 s replay each\n\n41 final fields exact\n\n11 trajectory segments exact',va='top',fontsize=12,color=BLUE)
save(fig,'fig6_lifecycle')

# 7: real, unmodified Neural Lab workspace plus explicitly separate browser profiles.
capture=json.loads((ROOT/'data/neural_lab_capture.json').read_text())
screenshot=OUT/'source/neural-lab-workspace.png'
expected=next(r['sha256'] for r in capture['sources'] if r['path'].endswith('figures/source/neural-lab-workspace.png'))
assert hashlib.sha256(screenshot.read_bytes()).hexdigest()==expected
fig=plt.figure(figsize=(11.4,13.1))
ax=fig.add_axes([.025,.785,.95,.19]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
ax.text(0,1.04,'a  Browser execution profiles',fontweight='bold',fontsize=12)
for x,title,body in [(.012,'Generic WASM','AtlasIR + WasmPlan validation\nShared f64 reference / Worker\nNeural Lab interface below'),
                     (.347,'Experimental WebGPU','WASM validation → WGSL\nRestricted independent-cell f32\nNot used in the screenshot'),
                     (.682,'Model-specific WASM AOT','Separate frozen generated model\n139,255 neurons / 15.1M edges\n13 fixed-input CPU/WASM checks')]:
    box(ax,x,.32,.302,.58,title+'\n\n'+body,GREEN if x>.6 else BLUE,9)
ax.text(.5,.13,'Separate AOT study: matching activity hashes and predictions',ha='center',fontsize=10,color=BLUE)
ax.text(.5,.025,'Maximum score difference 7.42 × 10⁻¹⁴ • 533.8 MiB WASM linear memory',ha='center',fontsize=9,color=GREY)
ax=fig.add_axes([.025,.015,.95,.735]);ax.axis('off')
ax.imshow(Image.open(screenshot),interpolation='none')
ax.set_title('b  Neural Lab: a completed WASM/f64 experiment',loc='left',fontweight='bold',fontsize=12,pad=10)
save(fig,'fig7_browser')

# 8: independent frozen NMDA timing cohorts; no reconstructed error intervals.
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), layout='constrained')
rows = MODELS['nmda_cpu_rows']
x = np.arange(len(rows))
neuron_counts = np.array([r['neurons'] for r in rows])
labels = [f"{r['neurons']:,}" for r in rows]
axes[0].plot(neuron_counts, [r['cpp_seconds'] for r in rows], 'o-', color=ORANGE, label='Brian2 C++')
axes[0].plot(neuron_counts, [r['rust_seconds'] for r in rows], 'o-', color=BLUE, label='Atlas CPU')
axes[0].set_xscale('log', base=2)
axes[0].set_yscale('log')
axes[0].set_xticks(neuron_counts, labels, rotation=25, fontsize=8)
axes[0].legend(fontsize=8)
panel(axes[0], 'a  Explicit NMDA · matched 8 cores', xlabel='Neurons', ylabel='Compiled-region median (s)')
axes[1].bar(x, [r['ratio'] for r in rows], color=BLUE, width=.65, zorder=3)
axes[1].axhline(1, color=GREY, linewidth=1, linestyle='--')
axes[1].set_xticks(x, labels, rotation=25, fontsize=8)
axes[1].set_ylim(0, 2.6)
for i, r in enumerate(rows):
    axes[1].text(i, r['ratio']+.08, f"{r['ratio']:.2f}×", ha='center', fontsize=9)
panel(axes[1], 'b  Within-cohort time ratios', xlabel='Neurons', ylabel='Brian2 / Atlas')
same40 = MODELS['nmda_mpi_same40']
axes[2].bar([0, 1], [same40['cpu_seconds'], same40['mpi_seconds']], color=[BLUE, GREEN], width=.6, zorder=3)
axes[2].set_xticks([0, 1], ['40 threads', '40 MPI ranks'], fontsize=9)
axes[2].set_ylim(0, 320)
axes[2].text(.5, 292, f"MPI ratio: {same40['ratio']:.2f}×", ha='center', fontsize=10)
panel(axes[2], 'c  Separate same-40-core control', xlabel='10,240 neurons · one host', ylabel='Simulation/recording median (s)')
save(fig, 'fig8_nmda')

# 9: rank-specific simulation execution; qualification records remain distinct.
fig=plt.figure(figsize=(11.4,6.1));ax=fig.add_axes([.02,.02,.96,.96]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(.01,.98,'a  Explicit simulation rank assignment',fontsize=16,fontweight='bold',va='top')
box(ax,.19,.865,.62,.065,'AtlasIR → DistributedPlan\nOwnership + ordered rank backends + numerical profile',BLUE,14)
box(ax,.04,.725,.41,.085,'CPU rank\nf64 neuron-state update',BLUE,14)
box(ax,.55,.725,.41,.085,'GPU rank · alternative adapter\nMetal or CUDA · f32 neuron-state update',GREEN,14)
box(ax,.04,.58,.41,.095,'Rank-local CPU host work\nThreshold/reset · synapses · queues · recording',BLUE,14)
box(ax,.55,.58,.41,.095,'GPU state writeback → rank-local CPU\nThreshold/reset · synapses · queues · recording',BLUE,14)
for x in [.245,.755]:
 ax.annotate('',xy=(x,.815),xytext=(.5,.855),arrowprops={'arrowstyle':'->','color':GREY})
 ax.annotate('',xy=(x,.685),xytext=(x,.715),arrowprops={'arrowstyle':'->','color':GREY})
box(ax,.20,.445,.60,.055,'Canonical spike exchange within the MPI communicator',BLUE,14)
for x in [.245,.755]:ax.annotate('',xy=(x,.515),xytext=(x,.56),arrowprops={'arrowstyle':'<->','color':GREY})
ax.text(.01,.42,'b  Retained qualification · separate paths and snapshots',fontsize=16,fontweight='bold',va='top')
box(ax,.015,.09,.30,.255,'CPU + Metal simulation\nActual local MPI processes\n25 mixed/inventory checks\nExample: 82 spikes\nGPU dispatches [0, 32]',GREEN,14)
box(ax,.35,.09,.30,.255,'CUDA simulation offload\nSource + runtime adapter\nNVIDIA hardware qualification\nnot established\nNo cross-host result',ORANGE,14)
box(ax,.685,.09,.30,.255,'CUDA MPI native training\nSeparate training-plan family\n31 CUDA MPI tests passed\nOne L4 · shared device 0\nNot multi-GPU validation',BLUE,14)
ax.text(.5,.025,'Explicit assignment; no automatic placement, mixed-vendor cluster proof or speedup claim',ha='center',fontsize=12,color=GREY)
save(fig,'fig9_heterogeneous_mpi')

# 10: complete accepted five-point weak scaling, when the study is available.
if WEAK is not None:
    rows=WEAK['rows'];assert [r['hosts'] for r in rows]==[3,6,12,24,30]
    assert rows[-1]['neurons']==860_000_000
    fig,axes=plt.subplots(2,2,figsize=(10.8,6.6),layout='constrained');axes=axes.ravel()
    large=rows[-1];e_pools=large['excitatory_populations']
    for excitatory,label,color in [(True,'Excitatory (80%)',BLUE),(False,'Inhibitory (20%)',ORANGE)]:
        pops=[p for p in large['populations'] if (int(p['name'].rsplit('_',1)[1])<e_pools)==excitatory]
        count=sum(p['neurons'] for p in pops)
        rates=np.sum([p['spikes_per_1ms_bin'] for p in pops],axis=0)/(count*.001)
        axes[0].plot(np.arange(20)*5+2.5,rates.reshape(20,5).mean(1),lw=1.8,color=color,linestyle='-' if excitatory else '--',label=label)
    axes[0].legend(fontsize=8,frameon=False)
    panel(axes[0],'a  860M-neuron activity',xlabel='Biological time (ms)',ylabel='Population firing rate (Hz)')
    x=[r['hosts'] for r in rows]
    for key,label,color,marker in [('launch_wall_seconds','Launch to exit',BLUE,'o'),('initialization_max_seconds','Initialization max',ORANGE,'s'),('simulation_max_seconds','Simulation max',GREEN,'^')]:
        axes[1].plot(x,[r[key] for r in rows],marker+'-',lw=1.5,color=color,label=label)
    axes[1].set_xticks(x);axes[1].legend(fontsize=8,frameon=False)
    panel(axes[1],'b  Execution intervals',xlabel='Hosts (8 worker cores per host)',ylabel='Wall time (s)')
    axes[2].plot(x,[max(r['worker_cgroup_peak_gib'][1:]) for r in rows],'o-',color=GREEN,label='Other hosts max')
    axes[2].plot(x,[r['worker_cgroup_peak_gib'][0] for r in rows],'s-',color=ORANGE,label='Coordinator host')
    axes[2].fill_between(x,[min(r.get('worker_memory_caps_gib',[768]+[665]*(r['hosts']-1))[1:]) for r in rows],[max(r.get('worker_memory_caps_gib',[768]+[665]*(r['hosts']-1))[1:]) for r in rows],color=GREEN,alpha=.12,label='Worker cap range')
    axes[2].plot(x,[max(r.get('worker_memory_caps_gib',[768]+[665]*(r['hosts']-1))[1:]) for r in rows],color=GREEN,ls='--',lw=1)
    axes[2].axhline(768,color=ORANGE,ls=':',lw=1,label='Coordinator cap')
    axes[2].set_xticks(x);axes[2].set_ylim(500,795);axes[2].legend(fontsize=8,frameon=False,ncol=2)
    panel(axes[2],'c  Per-host memory peaks',xlabel='Hosts (8 worker cores per host)',ylabel='Worker cgroup peak (GiB)')
    for key,label,color,marker in [('simulation_max_seconds','Simulation',GREEN,'o'),('launch_wall_seconds','Launch',BLUE,'s')]:
        axes[3].plot(x,[rows[0][key]/r[key] for r in rows],marker+'-',color=color,label=label)
    axes[3].axhline(1,color='#777777',ls='--',lw=1);axes[3].set_xticks(x);axes[3].set_ylim(0,1.08);axes[3].legend(fontsize=8,frameon=False)
    panel(axes[3],'d  Descriptive weak efficiency',xlabel='Hosts (8 worker cores per host)',ylabel='Baseline time / measured time')
    # Keep labels legible when the four-panel figure is scaled to A4 width.
    axes[3].set_ylabel('Baseline time /\nmeasured time')
    for ax in axes:
        ax.title.set_fontsize(14)
        ax.xaxis.label.set_fontsize(12.5)
        ax.yaxis.label.set_fontsize(12.5)
        ax.tick_params(labelsize=12)
        for label in ax.get_legend().get_texts():
            label.set_fontsize(12)
    save(fig,'fig10_connected_capacity')
else:
    # 10: accepted single-observation connected capacity configurations.
    capacity_rows=CAP['rows'];assert [r['neurons'] for r in capacity_rows]==[86_000_000,128_000_000]
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.7),layout='constrained')
    large=capacity_rows[-1]
    for excitatory,label,color in [(True,'Excitatory (80%)',BLUE),(False,'Inhibitory (20%)',ORANGE)]:
        pops=[p for p in large['populations'] if (int(p['name'].rsplit('_',1)[1])<24)==excitatory]
        count=sum(p['neurons'] for p in pops)
        rates=np.sum([p['spikes_per_1ms_bin'] for p in pops],axis=0)/(count*.001)
        axes[0].plot(np.arange(20)*5+2.5,rates.reshape(20,5).mean(1),lw=1.8,color=color,
                     linestyle='-' if excitatory else '--',label=label,zorder=3)
    axes[0].legend(fontsize=8,frameon=False)
    panel(axes[0],'a  128M-neuron activity',xlabel='Biological time (ms)',ylabel='Population firing rate (Hz)')
    keys=['initialization_max_seconds','simulation_max_seconds','launch_wall_seconds']
    x=np.arange(3)
    for i,(row,color) in enumerate(zip(capacity_rows,[BLUE,GREEN])):
        axes[1].bar(x+(i-.5)*.36,[row[k] for k in keys],width=.34,color=color,
                    label=f"{row['neurons']//1_000_000}M neurons",zorder=3)
    axes[1].set_xticks(x,['Init.','Sim.','Launch'])
    axes[1].legend(fontsize=8,frameon=False)
    panel(axes[1],'b  Measured time',ylabel='Wall time (s)')
    peaks=[r['maximum_worker_cgroup_peak_gib'] for r in capacity_rows]
    axes[2].bar([0,1],peaks,width=.6,color=[BLUE,GREEN],zorder=3)
    axes[2].set_xticks([0,1],['86M neurons','128M neurons'],fontsize=8)
    axes[2].axhline(256,color=ORANGE,ls='--',lw=1,label='Per-host cap')
    axes[2].set_ylim(0,280);axes[2].legend(fontsize=8,frameon=False)
    for i,value in enumerate(peaks):axes[2].text(i,value+5,f'{value:.1f}',ha='center',fontsize=9)
    panel(axes[2],'c  Maximum node peak',ylabel='Worker cgroup peak (GiB)')
    save(fig,'fig10_connected_capacity')

# Portable HTML contains the full figures as data URIs, with no network assets.
md = (mistune.create_markdown(escape=False, plugins=['table']) if mistune is not None
      else MarkdownIt('commonmark', {'html': True}).enable('table').render)
source=(ROOT/'MANUSCRIPT.md').read_text()
table3=['| Host / case | Selected policy | Baseline profiling (ms) | Selected profiling (ms) | Later winner replay (ms) | Total calibration (s) |',
        '|---|---|---:|---:|---:|---:|']
policy_names={'baseline':'Baseline','prefix':'Prefix','bitset':'Bitset','prefix-bitset':'Prefix + bitset'}
for r in P['autotune']['rows']:
    table3.append(f"| {r['host']} {r['case']} | {policy_names[r['selected']]} | {r['baseline_profiling_ms']:,.2f} | {r['selected_profiling_ms']:,.2f} | {r['later_replay_median_ms']:,.2f} | {r['calibration_seconds']:.2f} |")
source,n=re.subn(r'\| Host / case \| Selected policy \|[^\n]*\n\|---[^\n]*\n(?:\|[^\n]*\n)+','\n'.join(table3)+'\n',source)
assert n==1
table4 = ['| Workload | Model time | Samples per backend | Brian2 Cython (s) | Rust AOT (s) | Cython / Rust |',
          '|---|---|---:|---:|---:|---:|']
for r in V3['dendritic_rows']:
    ratio = f"{r['ratio']:.2f}" if r['ratio'] >= 1 else f"{r['ratio']:.3f}"
    table4.append(f"| {r['workload']} | {r['model_time']} | {r['samples_per_backend']} | {r['cython_seconds']:.6f} | {r['rust_seconds']:.6f} | {ratio} |")
source, n = re.subn(r'\| Workload \| Model time \|[^\n]*\n\|---[^\n]*\n(?:\|[^\n]*\n)+', '\n'.join(table4)+'\n', source)
assert n == 1
nmda_table = ['| Neurons | Total synapses | Samples per backend | Brian2 C++ (s) | Atlas CPU (s) | Brian2 / Atlas | Reported RSS increase |',
              '|---:|---:|---:|---:|---:|---:|---:|']
for r in MODELS['nmda_cpu_rows']:
    nmda_table.append(f"| {r['neurons']:,} | {r['synapses']:,} | {r['samples_per_backend']} | {r['cpp_seconds']:,.3f} | {r['rust_seconds']:,.3f} | {r['ratio']:.2f} | {r['rss_increase_percent']:.1f}% |")
source, n = re.subn(r'\| Neurons \| Total synapses \|[^\n]*\n\|---[^\n]*\n(?:\|[^\n]*\n)+', '\n'.join(nmda_table)+'\n', source)
assert n == 1
(ROOT/'MANUSCRIPT.md').write_text(source)
# Separate inline anchors from numbered entries for consistent CommonMark lists.
render_source = re.sub(r'(<a id="ref\d+"></a>)\n(?=\d+\. )', r'\1\n\n', source)
body=md(render_source)
def embed(m):
    p=ROOT/m.group(1)
    return 'src="data:image/svg+xml;base64,'+base64.b64encode(p.read_bytes()).decode()+'"'
body=re.sub(r'src="(figures/[^\"]+\.svg)"',embed,body)
style='''body{margin:0;background:#f4f5f6;color:#202830;font:17px/1.7 Georgia,serif}main{max-width:1040px;margin:32px auto;padding:60px 72px;background:white;box-shadow:0 2px 20px #0000000a}h1{font-size:34px;line-height:1.25;margin:0 0 25px}h2{font-size:25px;margin-top:50px;border-top:1px solid #dde3e7;padding-top:22px}h3{font-size:20px;margin-top:30px}h1,h2,h3{font-family:Arial,sans-serif;color:#1d405b}a{color:#245c85}p{margin:16px 0}table{border-collapse:collapse;width:100%;font:12px/1.5 Arial,sans-serif;margin:25px 0}th,td{padding:9px 8px;border-bottom:1px solid #dbe1e5;text-align:left;vertical-align:top}th{background:#edf3f6}img{max-width:100%;height:auto;display:block;margin:30px auto 10px}p:has(>strong:first-child){font-size:15px}@media(max-width:700px){main{padding:26px 20px;margin:0}h1{font-size:27px}table{display:block;overflow:auto}}@media print{body{background:white;font-size:11pt}main{margin:0;padding:0;box-shadow:none}h2,h3{break-after:avoid}img,table{break-inside:avoid}a{color:inherit}}'''
style += '.author-block{font:18px/1.5 Arial,sans-serif;margin:0 0 20px}.author-block span{font-size:15px;color:#64707a}'
style += 'h4{font:700 18px/1.4 Arial,sans-serif;color:#1d405b;margin:24px 0 12px;break-after:avoid}'
(ROOT/'MANUSCRIPT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="author" content="Xinjun Li"><meta name="viewport" content="width=device-width, initial-scale=1"><title>A Unified Intermediate Representation and Execution Architecture for Heterogeneous and Distributed Neural Simulation</title><style>'+style+'</style><main>'+body+'</main></html>')
supplement = (ROOT/'SUPPLEMENTARY.md').read_text()
dendritic_section = '## S13.' + supplement.split('## S13.', 1)[1].split('## S14.', 1)[0]
(ROOT/'data/common_source/section.html').write_text(md(dendritic_section))

# Contact sheet for human visual inspection.
tiles=[]
for name in FILES:
    im=Image.open(OUT/f'{name}.png').convert('RGB');im.thumbnail((770,500))
    tile=Image.new('RGB',(800,550),'white');tile.paste(im,((800-im.width)//2,35+(500-im.height)//2))
    ImageDraw.Draw(tile).text((20,12),name,fill='black');tiles.append(tile)
sheet=Image.new('RGB',(1600,550*5),'#dbe1e5')
for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*800,(i//2)*550))
sheet.save(ROOT/'validation/figure_contact_sheet.png')

# Structural and numerical checks; no simulator performance tests are run here.
assert len(re.findall(r'!\[Figure \d',source))==10
assert source.count('```')%2==0
assert len(re.findall(r'^\*\*Table \d+ \|',source,re.M))==(7 if WEAK is not None else 6)
assert not re.search(r'Writing target:|Editorial continuation:|TODO|TBD',source)
for n in range(1,17):assert f'id="ref{n}"' in source
for r in D['mpi_ranklocal']:assert abs(np.median(r['samples_seconds'])-r['median_seconds'])<1e-12
for host,cases in D['gpu_ring'].items():
    for case,result in cases.items():
        for r in result['summary']:
            if r.get('complete') and r.get('matched_gate_passed'):
                assert len(r['samples_seconds'])==5
                assert abs(np.median(r['samples_seconds'])-r['median_seconds'])<1e-12
assert D['mpi_primary']['raw_audit_passed_later']
assert all(r['activity_exact'] and r['image_exact'] and r['prediction']==r['expected_prediction'] for r in D['browser']['records'])
linux=[r for r in D['cpu_flywire'] if r['host'].startswith('EPYC')]
ratio=min(r['cpp_seconds'] for r in linux)/min(r['rust_seconds'] for r in linux)
assert round(ratio,2)==5.43
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.svg'))}
for r in P['autotune']['rows']:
    baseline=r['candidates']['baseline']['seconds']
    eligible=[k for k,v in r['candidates'].items() if v['status']=='eligible' and len(v['seconds'])==3
              and max(v['seconds'])<min(baseline) and np.median(v['seconds'])<=.95*np.median(baseline)]
    expected=min(eligible,key=lambda k:np.median(r['candidates'][k]['seconds'])) if eligible else 'baseline'
    assert expected==r['selected']
    assert np.isclose(r['selected_profiling_ms'],1000*np.median(r['candidates'][expected]['seconds']))
for r in P['tuning-cache']['rows']:
    assert np.isclose(np.median(r['hit_seconds']),r['hit_median_seconds'])
for r in V3['sources']:
    retained = ROOT/'data/v3'/r['retained']
    raw = retained.read_bytes()
    assert len(raw) == r['bytes'] and hashlib.sha256(raw).hexdigest() == r['sha256']
for r in V3['dendritic_rows']:
    assert np.isclose(r['cython_seconds']/r['rust_seconds'], r['ratio'], rtol=1e-12, atol=0)
for r in V3['training_cohorts']:
    assert sum(r['counts'].values()) == r['identities']
assert V3['mam']['scientific_equivalence_accepted'] is False
assert V3['mam']['matched_performance_accepted'] is False
for r in MODELS['sources']:
    raw = (ROOT/'data/published_models'/r['retained']).read_bytes()
    assert len(raw) == r['bytes'] and hashlib.sha256(raw).hexdigest() == r['sha256']
for r in MODELS['nmda_cpu_rows']:
    assert np.isclose(np.median(r['cpp_samples']), r['cpp_seconds'])
    assert np.isclose(np.median(r['rust_samples']), r['rust_seconds'])
    assert np.isclose(r['cpp_seconds']/r['rust_seconds'], r['ratio'])
assert MODELS['complete_onasch_paper_reproduction_accepted'] is False
assert MODELS['source_environment_reproduction_is_atlas_execution'] is False
main_table_count=len(re.findall(r'^\*\*Table \d+ \|',source,re.M))
assert main_table_count==(7 if WEAK is not None else 6)
assert 'We present brian2-atlas,' in source
assert 'Full working draft' not in source and 'Not yet submitted or posted' not in source
for item in HET['sources']:
    raw=(ROOT/'data/mpi_heterogeneous'/item['retained']).read_bytes()
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
assert HET['simulation_delivery']['rank_gpu_dispatches']==[0,32]
assert HET['simulation_delivery']['simulation_nvidia_hardware_qualified'] is False
assert HET['training_cuda']['cuda_mpi_passed']==31 and HET['training_cuda']['physical_gpus']==1
for item in CAP['sources']:
    raw=(ROOT/'data/capacity'/item['retained']).read_bytes()
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
assert CAP['pilot']['passed'] and CAP['pilot']['max_state_absolute_difference']==0
assert all(r['duration_ms']==100 and r['ranks']==30 and r['hosts']==30 for r in CAP['rows'])
if WEAK is not None:
    for item in WEAK['sources']:
        raw=(ROOT/'data/capacity'/item['retained']).read_bytes()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    assert all(r['duration_ms']==100 and r['ranks']==r['hosts']*8 and r['precision']=='reference-f64' for r in WEAK['rows'])
    assert all(p['passed'] and p['max_state_absolute_difference']==0 for p in WEAK['pilots'])
report={'status':'passed','manuscript_version':'final-layout','figures':10,'tables':main_table_count,'bibliography_entries':16,'manuscript_words_whitespace':len(source.split()),
        'capacity_evidence_sha256':hashlib.sha256((ROOT/'data/capacity/evidence.json').read_bytes()).hexdigest(),
        'linux_cpu_best_measured_ratio':ratio,'mpi_raw_audit_verified_from_later_report':True,
        'heterogeneous_mpi_evidence_sha256':hashlib.sha256((ROOT/'data/mpi_heterogeneous/evidence.json').read_bytes()).hexdigest(),'browser_fixed_inputs':len(D['browser']['records']),'svg_sha256':manifest,
        'figure_data_sha256':hashlib.sha256((ROOT/'data/figure_data.json').read_bytes()).hexdigest(),
        'plan_selection_sha256':hashlib.sha256((ROOT/'data/plan_selection.json').read_bytes()).hexdigest(),
        'plan_selection_cases':len(P['autotune']['rows']),'decision_cache_cases':len(P['tuning-cache']['rows']),
        'v3_evidence_sha256':hashlib.sha256((ROOT/'data/v3/evidence.json').read_bytes()).hexdigest(),
        'v3_retained_sources':len(V3['sources']),'v3_training_cohorts':len(V3['training_cohorts']),
        'v3_dendritic_ratios':[r['ratio'] for r in V3['dendritic_rows']],
        'review_snapshot_sha256':hashlib.sha256((ROOT/'data/v3/review_snapshot.json').read_bytes()).hexdigest(),
        'published_models_sha256':hashlib.sha256((ROOT/'data/published_models/evidence.json').read_bytes()).hexdigest(),
        'nmda_cpu_ratios':[r['ratio'] for r in MODELS['nmda_cpu_rows']],
        'nmda_same40_ratio':same40['ratio'],
        'scope':'Artifact structure, retained-source hashes and reported-number consistency; no new simulation, hardware benchmark or full scientific raw-array audit.'}
if WEAK is not None:
    report['weak_scaling_evidence_sha256']=hashlib.sha256((ROOT/'data/capacity/weak-evidence.json').read_bytes()).hexdigest()
    report['weak_scaling_hosts']=[r['hosts'] for r in WEAK['rows']]
    report['weak_scaling_endpoint_neurons']=WEAK['rows'][-1]['neurons']
resource_analysis=ROOT/'data/full_scale/analysis.json'
if resource_analysis.exists():
    conditional=json.loads(resource_analysis.read_text())
    assert conditional['status']=='passed'
    assert conditional['evidence_sha256']==report['weak_scaling_evidence_sha256']
    for item in conditional['sources']:
        assert hashlib.sha256((ROOT/'data/full_scale'/item['retained']).read_bytes()).hexdigest()==item['sha256']
    report['conditional_resource_analysis_sha256']=hashlib.sha256(resource_analysis.read_bytes()).hexdigest()
    report['conditional_resource_scope']=conditional['scope']
(ROOT/'validation/build_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
