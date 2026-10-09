#!/usr/bin/env python3
"""Materialize bounded evaluation scope. No benchmark execution or downloads."""
import copy
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ORIGIN = Path('/atlas-home/0004/workspace/next-brain.ai/next-brain.ai/model-platform/docs')
HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

def sha(b):
    return hashlib.sha256(b).hexdigest()

def write_json(name, data):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    return sha(p.read_bytes())

sources = []
for name in ['training-competitive-analysis-2026-10-04.md', 'training-engine-evaluation-v1.md', 'validation/training-engine-evaluation-v1.plan.json']:
    src = ORIGIN / name
    content = src.read_bytes()
    assert content == src.read_bytes(), f'concurrent change: {src}'
    dest = OUT / 'sources' / src.name
    dest.parent.mkdir(exist_ok=True)
    dest.write_bytes(content)
    sources.append({'origin': str(src), 'archived_path': str(dest.relative_to(OUT)), 'sha256': sha(content), 'bytes': len(content)})
write_json('source-manifest.json', {'research_date': '2026-10-04', 'purpose': 'Verbatim research sources, not new benchmark evidence', 'sources': sources})
source = json.loads((OUT / 'sources/training-engine-evaluation-v1.plan.json').read_text())
contenders = {x['id']: x for x in source['contenders']}
workloads = {x['id']: x for x in source['workloads']}
assert len(contenders) == 23 and len(workloads) == 17

# Explicit finite scope. Each remaining cell is still materialized with its own reason.
selection = {
 'atlas': {'mandatory': 'Q0 E1 E3 E4 E6 E7 L1 L2 A1 A2 M1', 'specialized': 'E2 E5 A3 F1 F2 F3'},
 'brian2': {'mandatory': 'L1 L2 M1', 'specialized': 'Q0 F1 F2'},
 'brian2cuda': {'specialized': 'Q0 L1 L2 M1'},
 'brian2genn': {'specialized': 'Q0 L1 L2 M1'},
 'genn': {'specialized': 'L1 L2'},
 'snntorch': {'mandatory': 'Q0 E1 E3 E4 E6 E7 A1 A2', 'specialized': 'E2 E5 A3 F3 M1'},
 'spikingjelly_stable': {'specialized': 'Q0 E1 E4 L1 A1 F3'},
 'spikingjelly_frontier': {'mandatory': 'Q0 E1 E3 E4 E6 E7 A1 A2 F3', 'specialized': 'E2 E5 A3 M1'},
 'norse': {'specialized': 'Q0 E1 E3 E4 L1 A1'},
 'spyx': {'mandatory': 'Q0 E1 E3 E4 E6 E7 A1 A2', 'specialized': 'E2 E5 A3'},
 'sinabs_exodus': {'specialized': 'Q0 E1 E2 A1'},
 'lava_dl_slayer': {'specialized': 'Q0 E4 E5 A2 A3'},
 'rockpool': {'specialized': 'Q0 E4 E5 A2 A3'},
 'bindsnet': {'specialized': 'L1 L2'},
 'brainpy_classic': {'specialized': 'Q0 E1 E4 M1'},
 'brainx_state': {'mandatory': 'Q0 E1 E3 E4 E6 E7 A1 A2', 'specialized': 'E2 E5 A3 M1'},
 'braintrace': {'mandatory': 'A3', 'specialized': 'Q0'},
 'mlgenn': {'mandatory': 'A3', 'specialized': 'Q0 A1'},
 'nest': {'mandatory': 'L1 L2 A3', 'specialized': 'Q0'},
 'nest_gpu': {'specialized': 'L1 L2'},
 'brian2modelfitting': {'mandatory': 'F1', 'specialized': 'Q0 F2'},
 'jaxley': {'mandatory': 'F1 F2', 'specialized': 'Q0'},
 'braincell': {'mandatory': 'F1 F2', 'specialized': 'Q0'},
}
focus = {
 'atlas': 'Atlas 原生接口与 Brian 转换路径', 'brian2': '原版 Cython/C++ 单线程/OpenMP 的原方程和局部学习',
 'brian2cuda': 'Brian 方程及局部学习 CUDA 扩展', 'brian2genn': 'Brian 到 GeNN 转换兼容性', 'genn': '原生 GeNN 局部可塑性',
 'snntorch': '核心 SG-BPTT 及 Torch 合法优化', 'spikingjelly_stable': '稳定版的 E1/E4 与流程交叉参考；完整曲线由独立前沿版本承担',
 'spikingjelly_frontier': '核心 SG-BPTT 的 Triton/FlexSN/memopt 强优化资格与深度边界', 'norse': 'E1/E3/E4 数值交叉参考，有限范围不扩展全容量曲线',
 'spyx': 'NNX/JAX 普通 reset LIF、循环和长序列', 'sinabs_exodus': 'EXODUS CUDA 前馈 LIF/卷积限定子集',
 'lava_dl_slayer': '可学习动力学/延迟的 E4/E5 资格及 SHD 原生算法', 'rockpool': 'E4/E5 参数约束与 SHD；纯 LIF 重复内核由 EXODUS 承担',
 'bindsnet': '局部 STDP 规则对照', 'brainpy_classic': '旧栈 E1/E4 交叉参考；完整曲线由 brainx_state 承担',
 'brainx_state': 'brainpy.state/brainstate 的 JIT/scan/remat SG-BPTT', 'braintrace': 'SHD 在线算法与自身资格，非同梯度 BPTT',
 'mlgenn': 'SHD EventProp/e-prop、CUDA 数据并行', 'nest': 'CPU 局部网络分区与 SHD e-prop',
 'nest_gpu': 'GPU 模型分区及匹配局部可塑性，不能继承 NEST CPU e-prop',
 'brian2modelfitting': 'HH 参数拟合，原仿真与外层优化分别核验', 'jaxley': '单室/多室生物物理求导与拟合', 'braincell': '单室/多室生物物理求导与拟合',
}
matrix = []
for cid in contenders:
    for wid, w in workloads.items():
        role = next((role for role, ids in selection[cid].items() if wid in ids.split()), 'not_applicable')
        if role != 'not_applicable':
            reason = f"{w['name']}：纳入{focus[cid]}；{'完整指定案例与失败记录必须交付' if role == 'mandatory' else '先做指定语义/能力资格，合格且资源到位后执行指定子集'}。"
        elif w['track'] == 'E' and cid in ['brian2', 'brian2cuda', 'brian2genn', 'genn', 'bindsnet', 'nest', 'nest_gpu', 'brian2modelfitting', 'jaxley', 'braincell', 'braintrace', 'mlgenn']:
            reason = f"{wid} 是指定 surrogate 全 BPTT 引擎赛道；本轮 {cid} 评估 {focus[cid]}，不把其他梯度/局部学习/仿真算法硬接入同梯度速度榜。"
        elif w['track'] == 'L':
            reason = f"{wid} 是原 Brian 事件/局部更新规则；本轮 {cid} 的有限任务是{focus[cid]}，未安排另写局部学习引擎；并非框架不支持的结论。"
        elif w['track'] == 'F':
            reason = f"{wid} 的{'HH/多室' if wid != 'F3' else '深层卷积'}边界不在 {cid} 本轮专项；其选定范围为{focus[cid]}；不外推能力缺失。"
        elif wid == 'M1':
            reason = f"迁移完整分母本轮重点核验 Atlas/Brian 及明确选定适配路径；{cid} 的有限工程预算用于{focus[cid]}，不宣称它不能迁移。"
        else:
            reason = f"{wid} 未纳入 {cid} 的本轮有限专项，预算聚焦{focus[cid]}；其他框架承担该覆盖，保留此格且不判不支持。"
        matrix.append({'contender_id': cid, 'workload_id': wid, 'coverage': role, 'reason': reason,
                       'capability_status': 'not_applicable' if role == 'not_applicable' else 'unqualified',
                       'qualification_status': 'not_applicable' if role == 'not_applicable' else 'not_run',
                       'execution_status': 'not_run', 'performance_run': False,
                       'pending': [] if role == 'not_applicable' else ['environment_pin', 'adapter', 'semantic_qualification', 'hardware_resource_manifest']})
write_json('coverage-matrix.json', {'dimensions': [23, 17], 'cell_count': len(matrix), 'counts': dict(Counter(c['coverage'] for c in matrix)), 'scope_note': 'coverage is an evaluation obligation, not a verified framework capability; pending cells never count as completed', 'cells': matrix})
with (OUT / 'coverage-matrix.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['contender_id', 'workload_id', 'coverage', 'reason'])
    writer.writeheader()
    writer.writerows({k: c[k] for k in writer.fieldnames} for c in matrix)
legend = {'mandatory': '必', 'specialized': '专', 'not_applicable': '—'}
rows = ['# 23 × 17 有限覆盖矩阵', '', '必：指定完整案例；专：指定专项资格/边界，合格后执行子集；—：本轮不适用，逐格具体原因见 CSV/JSON。资源未到位保持 pending，不能转成不适用或 unsupported。', '', '| 候选 | ' + ' | '.join(workloads) + ' |', '|---|' + '---|' * len(workloads)]
for cid in contenders:
    rows.append('| ' + cid + ' | ' + ' | '.join(legend[c['coverage']] for c in matrix if c['contender_id'] == cid) + ' |')
(OUT / 'coverage-matrix.md').write_text('\n'.join(rows) + '\n')

corpus_specs = [
 ('IF_curve_LIF.py', ['point_LIF', 'refractory'], '基本点神经元及 refractory exact flow'),
 ('CUBA.py', ['sparse_recurrent', 'three_state', 'exponential_current'], '稀疏循环及精确积分'),
 ('adaptive_threshold.py', ['adaptive_two_state', 'poisson'], '适应阈值与随机输入'),
 ('frompapers/Brette_Gerstner_2005.py', ['AdEx', 'two_state'], '非线性适应动力学'),
 ('frompapers/Izhikevich_2003.py', ['Izhikevich', 'delayed_pathways', 'run_regularly', 'noise'], '双路径突触、噪声及参数异质性'),
 ('COBAHH.py', ['conductance', 'HH', 'six_state'], 'HH 电导网络与多状态'),
 ('synapses/STDP.py', ['local_plasticity', 'event_driven_traces'], '事件驱动 STDP trace'),
 ('synapses/jeffress.py', ['heterogeneous_delay', 'timed_array', 'noise'], '异质延迟、声音刺激和噪声'),
 ('advanced/stochastic_odes.py', ['multiplicative_noise', 'Milstein', 'Heun'], '随机积分器保留，不可改为确定性 Euler'),
 ('advanced/custom_events.py', ['custom_event', 'boolean_state', 'multiple_pathways'], '自定义事件及门控'),
 ('compartmental/hodgkin_huxley_1952.py', ['spatial', 'multi_compartment', 'HH'], '真实形态、电缆方程和空间通道差异'),
 ('advanced/compare_GSL_to_conventional.py', ['multiple_clocks', 'run_regularly', 'GSL_adaptive_solver'], '多个时钟和可变步长积分器')]
corpus = []
for i, (rel, tags, rationale) in enumerate(corpus_specs, 1):
    path = 'examples/' + rel
    data = subprocess.check_output(['git', 'show', HEAD + ':' + path], cwd=ROOT)
    dest = OUT / 'corpus/originals' / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    corpus.append({'id': f'M{i:02}', 'upstream_repository': 'https://github.com/brian-team/brian2', 'source_commit': HEAD,
                   'repository_path': path, 'archived_path': str(dest.relative_to(OUT)), 'source_sha256': sha(data),
                   'source_bytes': len(data), 'strata': tags, 'inclusion_reason': rationale, 'selected_before_adapter': True,
                   'selection_scope': 'purposeful mechanism-stratified corpus; not a random statistical sample of all Brian models',
                   'stages': {k: 'pending' for k in ['original_run', 'support', 'rewrite', 'forward', 'gradient', 'learning', 'writeback', 'independent_replay']}})
corpus_hash = write_json('brian-corpus-manifest.json', {'denominator': len(corpus), 'frozen_membership': True, 'membership_revision': 'M1-corpus-r1', 'freeze_date': '2026-10-04', 'models': corpus,
 'constraints': ['Original source bytes preserved before adaptation.', 'May remove plotting/output only with recorded patch; no mechanism removal counted as exact migration.', 'A small qualification derivative is a separately named variant, never a success of the original full model.', 'Unsupported, source dependency failure and cap reached remain in the denominator.', 'Random draws must be materialized and reused across engines; same seed is insufficient.', 'No actual model runs or migration outcomes are implied by this manifest.']})

contracts = {
 'status': 'specified_pending_oracle_and_adapter_qualification', 'revision': 'equations-r1',
 'shared': {'loss': source['semantic_profiles']['canonical_lif_v1']['loss'], 'optimizer': source['semantic_profiles']['canonical_lif_v1']['optimizer'],
            'reset_gate': 'stop_gradient on reset condition; continuous reset expressions remain differentiable', 'batch_state': 'zero or explicitly declared initial state for each independent sample',
            'input_rule': 'count-bin arrays; all populations threshold from pre-synaptic-update states; feedforward and recurrent events added only afterwards',
            'output': 'canonical LIF readout; outputs use the same global scheduling as hidden states', 'finite_difference': 'smooth subgraphs only; hand-derived surrogate VJP is the spike gradient oracle'},
 'E4': {'name': 'adaptive_threshold_euler_r1', 'units': 'dimensionless v,a; time in ms', 'dt_ms': 1,
   'equations': ['uv = (1-dt/tau_v)*v', 'ua = (1-dt/tau_a)*a', 'margin = uv-(1+ua)', 's = 1[margin>0]',
                 'v_next = uv + W*feed + R*s - (1+ua)*stop_gradient(s)', 'a_next = ua+k*s'],
   'surrogate': 'ds/dmargin = 1/(1+5*abs(margin))^2; adaptation increment k*s keeps the surrogate path',
   'initial': {'v': 0, 'a': 0, 'tau_v_ms': 20, 'tau_a_ms': 200, 'k': 0.1},
   'trainable_transforms': {'tau_v_ms': '5+95*sigmoid(z_v)', 'tau_a_ms': '20+980*sigmoid(z_a)', 'k': '2*sigmoid(z_k)'},
   'parameter_modes': ['fixed_dynamics', 'weights_plus_shared_hidden_tau_v_tau_a_k'],
   'raw_initialization': 'z=log((physical-low)/(high-physical)) evaluated in FP64, stored common scalar arrays',
   'topology': 'feedforward [128,256,20]; R absent in this fixture', 'qualification_shape': [2,4,2], 'qualification_T':32, 'qualification_B':2,
   'performance_T': [512,2048], 'performance_B':16, 'refractory':False, 'delays':'zero', 'noise':False,
   'identity_note':'Explicit adaptive threshold model, not interchangeable with AdEx or a library default ALIF.'},
 'E5': {'common': {'shape':[128,128,20], 'B':8, 'T':512, 'dt_ms':0.1, 'qualification_shape':[2,4,2], 'qualification_B':2, 'qualification_T':32,
                   'solver':'simultaneous forward Euler from old state', 'trainable':'synaptic weights only', 'hidden_to_output':'canonical LIF, beta=1-0.1/20; theta=1',
                   'input_weights':'hidden voltage impulses in the declared voltage units, output weights dimensionless', 'refractory':False, 'noise':False,
                   'schedule':'Euler drift all groups; threshold all groups; deliver input/internal spikes; apply reset last. Hard reset overrides same-tick voltage input.'},
   'AdEx': {'source_model':'corpus/originals/frompapers/Brette_Gerstner_2005.py', 'units':'mV, ms, pA, pF, nS; pA/pF=mV/ms',
     'equations':['uv=v+dt*(gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT)+I-a)/C', 'ua=a+dt*(a_adapt*(v-EL)-a)/tau_a',
                  's=1[uv>Vcut]', 'v_next=(1-stop(s))*(uv+voltage_feed)+stop(s)*Vr', 'a_next=ua+b*stop(s)'],
     'parameters':{'C_pF':281,'gL_nS':30,'EL_mV':-70.6,'VT_mV':-50.4,'DeltaT_mV':2,'Vcut_mV':-40.4,'tau_a_ms':144,'a_adapt_nS':4,'b_pA':80.5,'Vr_mV':-70.6,'I_pA':500},
     'initial':{'v_mV':-70.6,'a_pA':0}, 'surrogate':'margin=(uv-Vcut)/2mV; ds/du=g(margin)/(2mV)',
     'boundary':'No exp clipping; non-finite evaluation is numerical_divergence, not silently stabilized.'},
   'Izhikevich': {'source_model':'corpus/originals/frompapers/Izhikevich_2003.py', 'units':'standard dimensionless Izh numeric variables; time in ms',
     'equations':['uv=v+dt*(0.04*v*v+5*v+140-u+I)', 'uu=u+dt*a*(b*v-u)', 's=1[uv>=30]',
                  'v_next=(1-stop(s))*(uv+voltage_feed)+stop(s)*c', 'u_next=uu+d*stop(s)'],
     'parameters':{'a':0.02,'b':0.2,'c':-65,'d':8,'I':5}, 'initial':{'v':-65,'u':-13},
     'surrogate':'margin=(uv-30)/10; ds/duv=g(margin)/10',
     'difference_from_original':'Controlled homogeneous/noise-free network at dt=.1ms; explicitly new E fixture, not an unchanged migration success.'},
   'synthetic_four_state': {'identity':'Controlled four-state linear-filter adaptive spiking fixture; not HH', 'units':'dimensionless states, ms',
     'equations':['uv=v+dt*(-v-.2*a+.4*p)/2', 'ua=a+dt*(-a)/10', 'up=p+dt*(q-p)', 'uq=q+dt*(-q)/.5',
                  's=1[uv>1]', 'v_next=uv-stop(s)', 'a_next=ua+.1*s', 'p_next=up', 'q_next=uq+W*feed'],
     'initial':{'v':0,'a':0,'p':0,'q':0}, 'surrogate':'g(uv-1)', 'reset_note':'v reset gate detached; a increment keeps surrogate path'}},
 'L2': {'name':'CUBA_exact_fixed_indegree_with_optional_pair_STDP_r1', 'source_model':'corpus/originals/CUBA.py',
   'equations':['dv/dt=(ge+gi-(v-El))/tau_m, except refractory holds v', 'dge/dt=-ge/tau_e', 'dgi/dt=-gi/tau_i'],
   'parameters':{'tau_m_ms':20,'tau_e_ms':5,'tau_i_ms':10,'El_mV':-49,'threshold_mV':-50,'reset_mV':-60,'refractory_ms':5,'we_initial_mV':1.62,'wi_mV':-9},
   'flow':['em=exp(-dt/tau_m); ee=exp(-dt/tau_e); ei=exp(-dt/tau_i)',
           'uv=El+(v-El)*em+ge*tau_e/(tau_m-tau_e)*(em-ee)+gi*tau_i/(tau_m-tau_i)*(em-ei)', 'uge=ge*ee; ugi=gi*ei'],
   'schedule':'Exact drift from old states; eligible threshold uv>-50; all presynaptic arrivals in (target,source,edge_id) order; post events after pre; reset last. Initial next_allowed_tick=0; a spike at tick t permits threshold again at t+50 for dt=.1ms.',
   'initial':'v=-60+10*materialized_uniform01 mV; ge=gi=0; common arrays', 'drive':'El=-49mV tonic drive, no external random runtime drive',
   'edges':'Every target draws exactly 80 distinct E sources and 20 distinct I sources without replacement, excluding itself; no multiedges. E count=.8*N.',
   'delay':'edge_id sorted lexicographically (target,source); delay_ms=1+(edge_id mod 3), exact integer bins 10/20/30',
   'plasticity':'Only E-to-E edges plastic, all other weights static. L1 exact pair rule with dimensionless wbar multiplied by 3.24mV; initial wbar=.5, clip [0,1]; tau_pre=10ms,tau_post=12ms,pre increment=.08,post increment=-.09. Same-tick pre before post. The input current uses weight before its STDP update.',
   'dimensions':{'N':[1000,4000,16000],'duration_seconds':[1,10],'dt_ms':0.1,'modes':['static','stdp']},
   'qualification':{'N':125,'duration_ms':100,'reason':'100 distinct incoming sources fit smallest graph while retaining 80:20 sources'},
   'output':'spike/event counts all populations; voltage traces selected fixed IDs; per-edge final weight and trace hashes; no task quality inference from firing rate',
   'pending':['L1 executed artifact hash', 'independent exact-flow/refractory qualification', 'materialized edge lists and input arrays']}}
contracts['L1']={'source':'../adapters/brian_validation.py','status':'adapter_defined_results_pending_at_protocol_authoring','Nsyn':8,'dt_ms':1,'T':16,
 'rule':'all-to-all additive event-driven pair STDP, exact exponential lazy decay of both traces to current event; pre then post at same tick; weight clip[0,1]',
 'tau_pre_ms':10,'tau_post_ms':12,'pre_increment':.08,'post_increment':-.09,'w0':[.5,.5,.5,.5,.5,.5,.99,.01],
 'pre_delay_ticks':[0,0,0,0,1,3,0,0],'post_delay_ticks':[0]*8,
 'pre_times':[[2],[5],[4],[1,3,5,7,9],[2,6],[1,5],[1,2,3,4],[4,5,6,7]],
 'post_times':[[5],[2],[4],[2,3,6,7,10],[3,7],[4,8],[1,2,3,4],[1,2,3,4]],
 'pre_event':'decay both traces; apre+=.08; w=clip(w+apost,0,1)', 'post_event':'decay both traces; apost-=.09; w=clip(w+apre,0,1)',
 'proof':'per tick weights/lazy traces/lastupdate, pre/post arrival counts and per-event independent oracle log; exact spike schedule'}
contract_hash = write_json('semantic-contracts.json', contracts)

# Nineteen fixed engine shapes; intersect each contender's matrix before creating jobs.
cases = []
def case(cid, workloads_, **kw):
    cases.append({'case_id':cid,'workload_ids':workloads_,**kw})
case('E1-small',['E1'],layers=[128,128,10],T=128,B=16)
case('E1-large',['E1'],layers=[512,1024,1024,20],T=128,B=32)
case('E2-conv',['E2'],input=[1,28,28],conv_channels=[16,32],kernel=3,stride=2,padding=1,readout=[128,10],T=100,B=16,weight_tying=True)
for N,T,p,tags in [(128,512,.1,['E3','E7']),(512,512,.1,['E3','E6','E7']),(2048,512,.1,['E3','E7']),(512,512,.01,['E3']),(512,512,1.,['E3']),(512,128,.1,['E6']),(512,2048,.1,['E6']),(512,8192,.1,['E6']),(8192,512,.1,['E7'])]:
    case(f'R-N{N}-T{T}-p{p:g}',tags,input=128,hidden=N,output=20,T=T,B=16,density=p,recurrence='same-tick spikes, next-threshold influence')
for T in [512,2048]:
    for mode in ['fixed_dynamics','trainable_dynamics']:
        case(f'E4-T{T}-{mode}',['E4'],layers=[128,256,20],T=T,B=16,model='E4',parameter_mode=mode)
for model in ['AdEx','Izhikevich','synthetic_four_state']:
    case('E5-'+model,['E5'],layers=[128,128,20],T=512,B=8,dt_ms=.1,model=model)
assert len(cases)==19
write_json('finite-engine-cases.json', {'case_count':len(cases),'cases':cases,'deduplication':'A result may carry several workload IDs only if arrays, model, state/precision/adapter, hardware and run scope are identical. E3 is the five-point one-axis scan; no unregistered 3x3 grid.',
 'capacity_refinement': {'max_new_points_per_boundary':3,'maximum_total_additional_engine_points_per_implementation_hardware':6,'rule':'One T boundary and one N boundary; floor(sqrt(lower*upper)) strictly inside neighboring success/failure sizes; retain software/physical/timeout boundary type.'}})

plan=copy.deepcopy(source)
plan.update({'schema_version':'1.1-execution-scope','plan_id':'atlas-training-v1-20261004-exec-r1','status':'scope_frozen_execution_pending','executable':False,'benchmark_results':[],
 'documents':{'protocol':'execution-protocol.zh.md','source_manifest':'source-manifest.json','matrix':'coverage-matrix.json','equations':'semantic-contracts.json','cases':'finite-engine-cases.json','corpus':'brian-corpus-manifest.json'},
 'source_archive_manifest_sha256':sha((OUT/'source-manifest.json').read_bytes()),'semantic_contracts_sha256':contract_hash,'corpus_manifest_sha256':corpus_hash,
 'meaning':'Finite study obligations and proposed missing equations fixed before performance data; each runnable slice needs its own immutable qualified execution manifest. No pending cell is completed by this document.'})
plan['coverage_matrix_counts']=dict(Counter(c['coverage'] for c in matrix))
plan['resource_plan']={
 'known_local_constraint':{'platform':'Mac arm64','physical_CPU_count':8,'RAM_GiB':16,'initial_available_disk_approx_MiB':992,'latest_parent_observation_available_GiB':2.4,'sandbox_CUDA_available':False,'sandbox_MPS_available':False,'MPS_note':'Sandbox unavailability does not mean the machine has no physical Metal GPU','source':'../environment/hardware.json and parent P0 observations; refresh before every write/build','policy':'Isolated, bounded local environments and small artifacts only; no deleting user data, shared cache, shared venv or active outputs.'},
 'local_P0':{'authorized':'existing local resources only','wall_hours_cap':4,'max_new_artifact_MiB':128,'free_disk_floor_MiB':768,'new_large_dependency_install':False,'benchmark_concurrency':1,'threads':1,'status':'can_continue_small_CPU_fixtures'},
 'resource_configurations':[
  {'id':'H0-expansion','configuration':'same local Apple arm64 CPU; reserve 50 GiB free storage for isolated CPU environments and artifacts; Metal needs a separately qualified accessible runtime','approval_or_provisioning':'pending_storage_available; no paid purchase authorized','phase_cap_wall_hours':60},
  {'id':'H1','configuration':'Linux x86_64 host, 32 physical CPU cores, 128 GiB RAM, 200 GiB SSD; record SMT/NUMA and bind 1/8/32 physical cores','approval_or_provisioning':'pending_resource_manifest_and_permission','phase_cap_wall_hours':12},
  {'id':'H2','configuration':'Linux, one NVIDIA L4 24 GiB, >=8 CPU cores, 64 GiB RAM, 200 GiB SSD; compatible pinned driver/CUDA; exclusive GPU','approval_or_provisioning':'pending_resource_manifest_and_permission','phase_cap_wall_hours':400,'budget_device_hours_cap':400},
  {'id':'H3','configuration':'Linux, 4 physical same-model NVIDIA L4 24 GiB, >=32 CPU cores, 256 GiB RAM, 300 GiB SSD; 1/2/4 UUIDs; PCIe topology recorded','approval_or_provisioning':'pending_resource_manifest_and_permission','phase_cap_wall_hours':24,'budget_device_hours_cap':96},
  {'id':'H4-CUDA','configuration':'2 physical Linux nodes, each 2 physical L4 24 GiB, >=16 cores,128 GiB RAM,200 GiB SSD; >=25Gb/s declared link; same total resources H3 comparator','approval_or_provisioning':'pending_resource_manifest_and_permission','phase_cap_wall_hours':24,'budget_device_hours_cap':96},
  {'id':'H4-CPU','configuration':'2 physical Linux nodes each16 physical cores/64GiB RAM, >=10Gb/s link; compare to H1 32-core single node','approval_or_provisioning':'pending_resource_manifest_and_permission','phase_cap_wall_hours':12}],
 'price_policy':{'currency_budget':None,'no_paid_jobs_authorized':True,'quote_required_fields':['provider','region','SKU','price_timestamp','hourly_rate','storage_rate','egress_rate','tax_policy'],'not_to_exceed_formula':'sum(configuration hourly quote * phase wall-hour cap) + locked storage/egress allowance','gpu_hours_upper_cap':592},
 'estimate':{'adapter_qualification_engineering_hours':[40,80],'CPU_H0_H1_reserved_wall_hours':72,'CPU_reserved_free_disk_GiB':50,'single_device_execution_elapsed_serial_hours_upper_cap':472,'distributed_elapsed_serial_hours_upper_cap':60,'M_F_engineering_hours':[16,32], 'microcase_theoretical_full_timeout_bound_hours_per_hardware':112.5,'microcase_bound_formula':'Core Atlas1+four competitors2 implementation views=9; (19 fixed + at most6 refined shapes)*9*.5hour=112.5h, before specialists. The 72h CPU reservation is a stage ceiling, not proof all cells fit; stop preserves pending cells and requires a revised resource reservation, never silently drops cases.', 'interpretation':'Planning ceilings, not measured runtimes or promises. Software admission rejection and early failures may reduce usage; resources provisioned before execution.'},
 'stop_rules':['Stop new disk-producing work below 768 MiB available and record resource_pending; existing lightweight read-only checks may continue.', 'No shared venv/source/build writes. All caches, compiled artifacts and TMPDIR under isolated evaluation paths.', 'Any missing resource remains pending, not unsupported or completed.', 'Do not expand model/seed/search/implementation grids after results; publish new protocol revision for changes.', 'At stage budget ceiling preserve evidence and list remaining pending cells; complete means all required cells have supported terminal evidence, not merely budget exhaustion.']}
plan['finite_execution']={
 'H0':'Core five Q0 + 19 engine cases intersecting matrix; default-correct and strongest qualified legal implementation, at most 2 views per competitor, Atlas baseline 1. L1 and L2 Brian Cython/standalone1/OpenMP and Atlas. A1 five core only when data and disk gates satisfied. No full SHD training claimed from CPU fixtures.',
 'H1':'CPU scaling E1-large and R-N512-T512-p.1; 1/8/32 physical-core budgets; Atlas MPI identified separately from threaded competitors. L2 N1k/4k/16k 1s static/stdp, plus two-node placement later.',
 'H2':'Core five 19 engine cases, EXODUS E1/E2, SLAYER/Rockpool E4/E5 qualification and native SHD; Brian2CUDA/GeNN L1/L2. A1 pipeline and A2/A3 formal training. Jaxley/braincell F boundary when environment qualified.',
 'specialist_E_limits':'Norse E1 small/large,E3 anchor,E4 T512 both modes. Stable SpikingJelly E1 small/large,E4 T512 both modes. BrainPy classic same E1/E4. EXODUS E1 small/large,E2. SLAYER/Rockpool E4 T512 both modes,E5 three models; mismatch goes to native A3, never strict ratio.',
 'H3':'E1-large and R-N512-T512-p.1, each 1/2/4 GPUs, fixed global B32 and B16 respectively; weak throughput separately local B16 on E1-large. A3 mlGeNN qualifying fixed global B32 uses exact SHD task shape; no full distributed HPO. Model partition: F3 frontier minimum and L2 NEST GPU, named as separate models.',
 'H4':'Same E1-large/R anchor global batch and total devices, 1+1 and2+2 versus H3 2/4; CPU L2 NEST/Atlas same32physicalcores over16+16 versus H1 32. Record topology/resources and checkpoint round-trip.',
 'qualification_budget':'Per contender/backend semantic group at most 30min execution and two adapter repair iterations within declared engineering budget; preserve first failure. Engine fix only isolated version and requalification; scope extension is a new revision.',
 'microbenchmark':{'seeds':[11,23,37,51,71],'independent_processes':5,'warmup_steps_per_process':10,'measured_steps_per_process':50,'wall_cap_total_case_seconds':1800,'wall_cap_single_process_seconds':360,'clarification':'Resolved ambiguity in source: cap applies to all five processes total, each process gets equal 360s. Timeout results retain all five scheduled seed rows. No reduction of T/B/steps after timing.'},
 'L2':{'points':'3 sizes x 2 durations x2 plasticity modes=12; H1 distributed scaling limits duration1s only; all other original cells remain in H0/H2 scope','repetitions':5,'warmup':'One separate 100ms graph-equivalent untimed run then restore exact arrays; full cold build reported separately','wall_cap_total_case_seconds':1800},
 'F':{'performance_run':False,'scope':'F1 original single-compartment HH and F2 original multicompartment HH: support, solver/model comparison and smallest explicit forward/gradient/update fixture only. F3 fixed four-conv network build/forward/backward/one update. Full fitting/large deep training is a new preregistered extension, not implied by capability completion.',
      'qualification_execution_cap_seconds_per_cell':1800,'F3_architecture':'input1x28x28, Conv16/32/64/64 all k3p1; avgpool2 after conv2/4; flatten->10; no bias/BN/dropout; T16 B2; canonical LIF between conv layers; no invented Spikformer equivalence'},
 'M1':{'model_count':12,'per_model_active_engineering_minutes_cap':120,'per_original_run_wall_seconds_cap':600,'scope':'All12 source/support decisions mandatory. Attempt rewrite/semantic/learning/writeback/replay within supported subsets and bounded engineering cap, preserve first failure and stage reach. Cap means incomplete migration, not unsupported.'}}
plan['task_freeze']={
 'formal_seeds':[11,23,37,51,71], 'tuning_seeds':[101,103],
 'A1':{'epochs':10,'run_wall_seconds':1800,'tuning':'no HPO; fixed canonical Adam lr1e-3, pipeline sanity only','encoding':'x[t,pixel]=pixel_uint8/255 constant across100ticks; deterministic analogue-current encoding, not Bernoulli spikes','validation':'500 examples/class sampled once from official train with generator PCG64 seed20261004; archive exact indices before fitting','quality_targets':[0.9,0.95],'test_checkpoint':'best validation accuracy, earliest epoch tie; test once per formal seed'},
 'A2_A3':{'run_epochs':50,'run_wall_seconds':7200,'targets':[0.5,0.7,0.8],'validation_frequency':'every complete epoch','selection':'best validation accuracy, earliest epoch tie; score test once after selection',
  'HPO_configs':8,'runs_per_config':2,'HPO_run_epochs':10,'HPO_run_wall_seconds':1800,'SG_BPTT_search':{'learning_rate':[0.0003,0.001,0.003,0.01],'global_initial_weight_multiplier':[0.5,1.]},
  'online_search_status':'pending algorithm-native exact 8-config search artifacts; cannot be finalized without pinned APIs/algorithm parameter meaning',
  'SHD_split_rule':'Enumerate nonempty speaker subsets of official train only; require every class in both parts; choose validation subset minimizing abs(size-.2*train_size), tie by lexicographic speaker ID tuple. Archive exact speaker/sample lists and counts before HPO. If no subset meets coverage, stop pending protocol revision, never access test labels.',
  'T_rule':workloads['A2']['steps_rule'],'encoding':'uint16 count bins only after proving max count fits uint16; otherwise uint32; no clipping. JSON exact request bytes include weights/optimizer/metadata; original monolithic B32 cell retained.',
  'microbatch':{'enabled':False,'candidate_global_batch':32,'candidate_microbatch':4,'qualification':'sum gradients in declared FP64 sample order, same loss normalization, exactly one Adam per global batch; separate version and Q0 required; no automatic bypass'}},
 'statistics':{'latency':'5 process medians; report median, min/max and all paired ratios; no significance claim from 250 dependent steps', 'quality':'paired seed accuracy difference; 95% t interval on five differences, df4, tcrit2.776445105; noninferior only lower bound>-1 percentage point; all five outcomes shown; failed training never dropped','time_to_quality':'report five seed attainment/time values, censored-at-cap runs and success count; ratio only paired attained cases accompanied by full success counts, no global ratio when success<4/5'}}
plan['completion_checklist']=[
 {'id':'P0-source','done_when':'Exact Atlas snapshot + dirty patch + Python/native source/runtime hashes and provenance consistent; no active-tree mutations','status':'pending_parent_manifest'},
 {'id':'P0-qualification','done_when':'Q0 full states/spikes/loss/specified VJP/all gradients/SGD/Adam and corresponding Brian forward/L1 artifacts; every runnable slice oracle qualified','status':'pending_runs'},
 {'id':'P0-freeze','done_when':'Specific adapter/dependency/hardware/arrays/optimizer profile hashes complete; per-slice immutable manifest issued; equations and L1/L2 reconciled','status':'pending'},
 {'id':'P1-CPU','done_when':'All scoped H0/H1 cells qualified-and-completed or terminal failures with logs; 5seed raw timings and admission/memory evidence; strongest legal baselines audited','status':'pending'},
 {'id':'P1-GPU','done_when':'All scoped H2 cells as CPU with physicalGPUUUID and sync; exact build qualification, no inherited historicalCUDA pass','status':'pending_resources'},
 {'id':'P1-task','done_when':'A1 pipeline and all A2/A3 scoped algorithms have frozen data/search, 5formal seed records, all failure/censoring, learningcurves,testselectedcheckpoints','status':'pending_data_resources'},
 {'id':'P2','done_when':'True1/2/4physicalGPU, rankmapping, strong/weak throughput separated, perrank state allocation, all scoped cases and failures documented','status':'pending_resources'},
 {'id':'P3','done_when':'True2host placement versus sametotalresource1host, transport/restart verified, CPU/CUDA panels separate','status':'pending_resources'},
 {'id':'F','done_when':'All selected F cells model/solver/source-hashed capability stage terminal with forward/gradient/update evidence where supported; performance_run=false explicit','status':'pending'},
 {'id':'M1','done_when':'12model denominator public; each stage status and effort/deletion ledger complete; supported training paths include writeback and unseen-stimulus originalBrian replay or explicit failed/pending evidence','status':'membership_frozen_execution_pending'},
 {'id':'report','done_when':'RawJSON/logs, standalone validator, versions/devices/data inventory, plots, Chinese evidence-based conclusion; no unavailable resource cell falsely called complete','status':'pending'}]
plan['pending_freeze_items']=['parent isolated source/runtime snapshot reference','each contender exact version/complete lock/build flags','independent fixture hashes and full Q0 report','E4/E5/L2 oracle and adapter pass','L1 exact executed rule reconciliation','synthetic generator materialized arrays per five seeds','MNIST/SHD archives, actual T, split indices and count-bin checksums','online algorithm exact HPO search sets','GPU profile batch/rank reduction implementation audit','hardware availability, disk expansion, exclusivity and quotes if paid','memory measurement tool/source and sample interval','new protocol hashes recorded after final per-slice freezes']
write_json('execution-plan.json',plan)

def manifest():
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name != 'SHA256SUMS.json':
            files.append({'path':str(p.relative_to(OUT)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size})
    write_json('SHA256SUMS.json',{'purpose':'Protocol and source inventory; regenerate after intentional protocol revision, not a claim of executed performance','files':files})

if __name__ == '__main__':
    manifest()
    print(json.dumps({'matrix':dict(Counter(c['coverage'] for c in matrix)),'cells':len(matrix),'engine_shapes':len(cases),'corpus_denominator':len(corpus),'plan':str(OUT/'execution-plan.json')},ensure_ascii=False))
