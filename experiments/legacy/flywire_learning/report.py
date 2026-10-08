"""Audit complete experiment artifacts and generate the CPU v1 research report."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS=[11,23,47,83,131]
CONDITIONS=['paired','frozen','teaching_off','shuffled_reward','reversal']


def audit_rewards(config):
    """Check actual trial declarations against each frozen control definition."""
    condition=config['condition'];counts={}
    for block in ('pre','acquisition','post','second','final'):
        training=block in ('acquisition','second')
        trials=[t for t in config['trials'] if t['block']==block]
        assert len(trials)==(12 if training else 4)
        assert all(t['training']==training and t['reward'] in (0,1) for t in trials)
        for cue in (0,1):assert sum(t['cue']==cue for t in trials)==len(trials)//2
        if not training or condition=='teaching_off':
            assert all(t['reward']==0 for t in trials)
        elif condition=='shuffled_reward':
            assert sum(t['reward'] for t in trials)==6
        else:
            rewarded_cue=int(condition=='reversal' and block=='second')
            assert all(t['reward']==int(t['cue']==rewarded_cue) for t in trials)
        counts[block]={str(cue):sum(t['reward'] for t in trials if t['cue']==cue) for cue in (0,1)}
    return counts


def generate(study,prepared,output):
    final=json.loads((study/'final_report.json').read_text())
    identity=json.loads((study/'study_identity.json').read_text())
    correctness=json.loads((study/'correctness.json').read_text())
    assert final['completed_cases']==25 and final['seeds']==SEEDS and final['conditions']==CONDITIONS
    assert final['all_correctness_gates_passed'] and correctness['native_replay_exact']
    for name in ('full_aot_1_4','full_cpp','full_partition_frozen','full_continuous_segmented','full_fresh_process_restore'):
        assert name in correctness and isinstance(correctness[name],dict),name
    source_manifest=json.loads((prepared/'manifest.json').read_text())
    assert source_manifest['neurons']==139255 and source_manifest['plastic_edges']==2560
    assert source_manifest['partition_reconstructs_original_exactly']
    counts=np.load(prepared/'plastic.npz')['signed_contacts']
    audits=[]; verified_rows=[]; seed_inputs={}
    for seed in SEEDS:
        for condition in CONDITIONS:
            folder=study/f'seed-{seed}-{condition}'
            config=json.loads((folder/'configuration.json').read_text())
            observed=json.loads((folder/'report.json').read_text())
            assert (config['seed'],config['condition'],config['scope'],config['protocol'])==(seed,condition,'full','formal')
            assert config['neurons']==139255 and config['plastic_edges']==2560 and config['static_edges']==15089423
            assert config['duration_ms']==19100 and not config['unsplit']
            assert config['amplitude_mv']==identity['amplitude_mv']==40
            assert config['backend']=='aot' and config['threads']==final['threads']
            reward_counts=audit_rewards(config)
            for field,key in [('experiment_sha256','experiment.py'),('rule_sha256','spiking.py'),
                              ('protocol_sha256','PROTOCOL.md'),('prepared_manifest_sha256','manifest.json')]:
                assert config[field]==identity[key],field
            with np.load(folder/'input.npz') as inputs:
                digest=hashlib.sha256()
                for name in sorted(inputs.files):digest.update(name.encode()+inputs[name].tobytes())
                assert digest.hexdigest()==config['input_sha256']
                signature={name:hashlib.sha256(inputs[name].tobytes()).hexdigest()
                           for name in inputs.files if name not in ('learning','teaching')}
                if seed in seed_inputs:assert signature==seed_inputs[seed],'within-seed input mismatch'
                else:seed_inputs[seed]=signature
                assert len(inputs['learning'])==len(inputs['teaching'])==191000
                allowed=np.zeros(len(inputs['learning']),dtype=bool)
                expected_learning=np.zeros(191000);expected_teaching=np.zeros(191000)
                for trial in config['trials']:
                    if trial['training']:
                        start=int(round(trial['start_ms']/.1));allowed[start+500:start+3000]=True
                        if condition!='frozen':expected_learning[start+500:start+3000]=1
                        if trial['reward']:expected_teaching[start+2000:start+3000]=1
                np.testing.assert_array_equal(inputs['learning'],expected_learning)
                np.testing.assert_array_equal(inputs['teaching'],expected_teaching)
                assert not np.any(inputs['teaching'][~allowed]) and not np.any(inputs['learning'][~allowed])
                assert np.all(np.isin(inputs['learning'],[0,1])) and np.all(np.isin(inputs['teaching'],[0,1]))
            saved={}
            for block in ['pre','acquisition','post','second','final']:
                with np.load(folder/f'{block}-plastic.npz') as state:
                    saved[block]=state['gain'].copy()
                    assert state['gain'].shape==(2560,) and np.all((state['gain']>=0)&(state['gain']<=1))
                    assert np.all((state['eligibility']>=0)&(state['eligibility']<=1))
                    if condition in ('frozen','teaching_off'):np.testing.assert_array_equal(state['gain'],1)
            np.testing.assert_array_equal(saved['pre'],1)
            np.testing.assert_array_equal(saved['post'],saved['acquisition'])
            np.testing.assert_array_equal(saved['final'],saved['second'])
            with np.load(folder/'snapshot.npz') as state:
                assert state['v'].shape==(139255,) and state['counts'].shape==(139255,)
                assert np.all((state['spike_i']>=0)&(state['spike_i']<139255))
                np.testing.assert_array_equal(np.bincount(state['spike_i'],minlength=139255),state['counts'])
                np.testing.assert_array_equal(state['gain'],saved['final'])
                for key in ('v','ge','gi','gain','eligibility','neuron_trace'):assert np.isfinite(state[key]).all()
                assert np.all(state['ge']>=0) and np.all(state['gi']>=0)
                assert observed['spikes']==len(state['spike_i'])
                assert observed['final_gain_mean']==float(state['gain'].mean())
                assert observed['final_gain_min']==float(state['gain'].min())
                assert observed['changed_edges']==int(np.count_nonzero(state['gain']!=1))
                # Independently recompute each reported trial using sorted
                # integer-tick slices, rather than trusting summary metrics.
                spike_ids,spike_times=state['spike_i'],state['spike_t']
                ticks=np.rint(spike_times/.0001).astype(np.int64)
                assert np.all(np.diff(ticks)>=0) and np.all((ticks>=0)&(ticks<191000))
                np.testing.assert_allclose(spike_times,ticks*.0001,rtol=0,atol=1e-14)
                masks={key:np.isin(np.arange(139255),group) for key,group in
                       [('mbon',config['mbon_indices']),('kc',config['kc_indices']),
                        ('input0',config['stimulus_groups'][0]),('input1',config['stimulus_groups'][1])]}
                for declared,trial in zip(config['trials'],observed['trials'],strict=True):
                    assert all(trial[key]==value for key,value in declared.items())
                    start=int(round((declared['start_ms']+50)/.1));end=start+2500
                    left,right=np.searchsorted(ticks,[start,end],side='left')
                    local_ids=spike_ids[left:right]
                    for key,mask_key in [('mbon','mbon'),('kc','kc'),('input',f"input{declared['cue']}")]:
                        spike_count=int(masks[mask_key][local_ids].sum())
                        assert spike_count==trial[key+'_spikes']
                        assert spike_count/(int(masks[mask_key].sum())*.25)==trial[key+'_hz']
            for block in ('pre','acquisition','post','second','final'):
                trials=[t for t in observed['trials'] if t['block']==block]
                expected_count=6 if block in ('acquisition','second') else 2
                rates=[]
                for cue in (0,1):
                    samples=[t for t in trials if t['cue']==cue]
                    assert len(samples)==expected_count
                    rates.append(float(np.mean([t['mbon_hz'] for t in samples])))
                assert observed['contrasts'][block]==dict(A_hz=rates[0],B_hz=rates[1],B_minus_A_hz=rates[1]-rates[0])
            for stage in observed['stages']:
                artifact=Path(stage['artifact_directory'])
                native=json.loads((artifact/'native_metrics.json').read_text())
                raw=json.loads((artifact/'rust/summary.json').read_text())
                assert native==stage['native'] and raw['timings']==stage['timings']
                assert native['exit_code']==0 and native['native_peak_rss_bytes']>0
                assert raw['neuron_count']==139889 and raw['population_count']==3
                assert abs(raw['final_time_seconds']-stage['end_ms']/1000)<1e-12
            baseline=json.loads((study/f'seed-{seed}-frozen/report.json').read_text())
            verified_rows.append(dict(seed=seed,condition=condition,
                post_effect_hz=observed['contrasts']['post']['B_minus_A_hz']-baseline['contrasts']['post']['B_minus_A_hz'],
                final_effect_hz=observed['contrasts']['final']['B_minus_A_hz']-baseline['contrasts']['final']['B_minus_A_hz'],
                contrasts=observed['contrasts'],changed_edges=observed['changed_edges'],
                final_gain_mean=observed['final_gain_mean'],spikes=observed['spikes'],
                native_seconds=sum(s['timings']['simulation_and_recording_seconds'] for s in observed['stages']),
                native_peak_rss_bytes=max(s['native']['native_peak_rss_bytes'] for s in observed['stages']),
                frontend_peak_rss_bytes=observed['frontend_peak_rss_bytes'],end_to_end_seconds=observed['end_to_end_seconds']))
            audits.append({'seed':seed,'condition':condition,'test_learning_and_teaching_off':True,
                           'reward_counts_by_block_and_cue':reward_counts,
                           'all_test_gain_arrays_exact':True,'contact_weighted_final_gain':float(counts@saved['final']/counts.sum()),
                           'native_stages_measured':all(s['native']['exit_code']==0 for s in observed['stages'])})
    for name in ('background_ids','background_ticks','mapping','stimulus_ids','stimulus_ticks'):
        assert len({seed_inputs[seed][name] for seed in SEEDS})==5,f'reused stochastic input: {name}'
    assert final['rows']==verified_rows,'aggregate rows differ from audited individual cases'
    for condition in CONDITIONS:
        group=[r for r in verified_rows if r['condition']==condition]
        for metric,declared in final['summary'][condition].items():
            values=[r[metric] for r in group]
            assert declared==dict(mean=float(np.mean(values)),min=float(min(values)),max=float(max(values)))
    output.mkdir(parents=True,exist_ok=False)
    (output/'completion_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    (output/'input_identity_audit.json').write_text(json.dumps(seed_inputs,indent=2)+'\n')
    for name in ('final_report.json','correctness.json','study_identity.json','execution_choice.json'):
        (output/name).write_bytes((study/name).read_bytes())
    rows=final['rows'];lines=[]
    def line(text=''):lines.append(text)
    line('# FlyWire 固定全脑 + 局部 KC→MBON 学习：CPU v1 结果')
    line();line('本报告来自完整的 139,255 神经元、15,091,983 条原始聚合连接：其中 2,560 条双侧 MBON01 输入边可塑，其余固定。25 个预设条件运行完成。')
    line();line('奖励采用外部合成教学信号，γ5/β′2a 合并为一个投影级学习单元。该模型没有实现空间分辨的多巴胺释放、真实果蝇行为或生理拟合。')
    line();line('## 预设主要结果')
    line();line('效应为同一 seed 中，学习条件相对冻结条件的 MBON01「B−A」测试放电率差，单位 Hz。它不是分类准确率。')
    line();line('| 条件 | 首次训练后均值 [最小,最大] | 第二训练后均值 [最小,最大] | 最终平均 gain |')
    line('| --- | ---: | ---: | ---: |')
    for condition in CONDITIONS:
        s=final['summary'][condition]
        def fmt(key):
            v=s[key];return f"{v['mean']:.4f} [{v['min']:.4f}, {v['max']:.4f}]"
        line(f"| {condition} | {fmt('post_effect_hz')} | {fmt('final_effect_hz')} | {s['final_gain_mean']['mean']:.6f} |")
    paired=[r for r in rows if r['condition']=='paired']
    all_silent=all(r['contrasts'][b]['A_hz']==r['contrasts'][b]['B_hz']==0 for r in rows for b in ('pre','post','final'))
    line()
    if all_silent:
        line('**负结果：所有预设测试中，选定 MBON01 输出均为 0 Hz。此实验未证明可读出的奖励关联或反转学习。** 即使局部增益发生变化，也不能用它替代输出层的关联证据。需要在后续独立研究中检验该简化模型的传入驱动与抑制平衡；本次不追加调参。')
    else:
        first=final['summary']['paired']['post_effect_hz'];last=final['summary']['paired']['final_effect_hz']
        reversal=final['summary']['reversal']['final_effect_hz']
        line(f"配对条件首次测试相对冻结对照的差值均值为 {first['mean']:.4f} Hz，第二训练后为 {last['mean']:.4f} Hz；反转条件最终差值均值为 {reversal['mean']:.4f} Hz。上表保留全部 seed 的正、零和负效应。5 个 seed 与每个测试块 4 次刺激限制了统计效力；这些描述性差异不能直接证明可靠关联学习、生物学习机制或真实拓扑优势。")
    line();line('### 原始读出与静默诊断')
    line();line('以下为描述性检查，不替换预设主要指标。A/B 分别是两种刺激的 MBON01 放电率，先在每个 seed 内汇总，再跨 5 个 seed 取均值。若学习后 A、B 同时归零，而冻结条件的 B−A 为负，相对冻结的效应仍可为正；这种情况不能作为刺激特异关联的证据。')
    line();line('| 条件 | 训练前 A/B Hz | 首次训练后 A/B Hz | 第二训练后 A/B Hz | 最终 A、B 均静默的 seed 数 |')
    line('| --- | ---: | ---: | ---: | ---: |')
    for condition in CONDITIONS:
        group=[r for r in rows if r['condition']==condition]
        rates=[]
        for block in ('pre','post','final'):
            a=np.mean([r['contrasts'][block]['A_hz'] for r in group])
            b=np.mean([r['contrasts'][block]['B_hz'] for r in group])
            rates.append(f'{a:.4f} / {b:.4f}')
        silent=sum(r['contrasts']['final']['A_hz']==r['contrasts']['final']['B_hz']==0 for r in group)
        line(f"| {condition} | {' | '.join(rates)} | {silent}/5 |")
    line();line('## 全部 seed')
    line();line('| seed | 条件 | 首次效应 Hz | 最终效应 Hz | 变化边数 | 最终平均 gain |')
    line('| ---: | --- | ---: | ---: | ---: | ---: |')
    for r in rows:line(f"| {r['seed']} | {r['condition']} | {r['post_effect_hz']:.4f} | {r['final_effect_hz']:.4f} | {r['changed_edges']} | {r['final_gain_mean']:.6f} |")
    line();line('## 正确性与恢复')
    line();line('独立逐时步参考、Brian2 NumPy/C++ 和 Rust 小回路验证通过；真实诱导子图全数组一致。完整全脑 Rust 1/4 线程与 C++ 对照通过；静态/可塑边分区逆向还原原图。全脑连续/分段、新进程 checkpoint 恢复、原生二进制重复执行均通过。')
    line();line(f"全脑跨后端数值门覆盖 1.1 生物秒；全脑连续/分段和新进程恢复比较采用 seed 11 的 paired 条件，覆盖完整 19.1 生物秒。比较范围包括所有神经元的最终状态和不应期状态、全部脉冲、全部可塑边的 gain/资格迹，以及 {len(config['recorded_indices'])} 个预选神经元的状态轨迹；没有记录所有神经元的逐时步电压。")
    line();line('小回路的中间突触样本和逐段禁学检查覆盖 Brian NumPy 与 Rust reference/AOT；C++ standalone 仅比较最终突触状态及神经轨迹/脉冲。历史 spiking_validation.json 中 C++ 的 learning_off_gain_exact=false 表示未采集中间样本，不是失败结果。')
    line();line('最终审计重新读取 25 组输入和权重数组：所有测试/休息时段学习与教学输入为零，测试前后 gain 逐位相同；冻结与教学关闭条件的 gain 始终为 1。完整输出索引、脉冲计数、有限状态、电导和增益边界检查通过。详细误差见 correctness.json。')
    line();line('同一 seed 的五个条件共用完全相同的非控制输入数组；五个 seed 的背景事件、刺激事件和随机映射均不同，散列记录见 input_identity_audit.json。')
    line();line('## 成本与复现')
    line();line(f"正式执行使用 Rust CPU {final['threads']} 线程。每个条件模拟 19.1 生物秒；原生循环耗时范围 {min(r['native_seconds'] for r in rows):.2f}–{max(r['native_seconds'] for r in rows):.2f} s，单次端到端范围 {min(r['end_to_end_seconds'] for r in rows):.2f}–{max(r['end_to_end_seconds'] for r in rows):.2f} s。")
    line();line(f"原生进程峰值 RSS 最大 {max(r['native_peak_rss_bytes'] for r in rows)/2**20:.1f} MiB，Python 前端峰值最大 {max(r['frontend_peak_rss_bytes'] for r in rows)/2**20:.1f} MiB。原生 RSS 由新建的小型 wait4 supervisor 测量；这些数值不等于并发进程树内存之和。编译、初始化和循环时间在逐阶段报告中分列，未作跨机器加速结论。")
    line();line('完整规则、时程、学习率和有限校准预算见 PROTOCOL.md；原始图与注释校验、边身份、输入数组、源码散列、编译制品及 checkpoint 均保存在研究输出目录。')
    line();line('```sh')
    line('python brian2-rust/experiments/flywire_learning/study.py --prepared /path/to/learning-prepared --output /path/to/new-study --amplitude 40')
    line('python brian2-rust/experiments/flywire_learning/report.py --prepared /path/to/learning-prepared --study /path/to/new-study --output /path/to/new-report')
    line('```')
    line();line('结构映射依据：[Li et al., eLife](https://elifesciences.org/articles/62576)。学习动机及与本实现的区别见 RESEARCH.md。FlyWire v783 数据：[Zenodo](https://zenodo.org/records/10676866)，CC BY 4.0。')
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(f'Completion artifact audit passed for {len(audits)} full-brain cases: {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();generate(a.study,a.prepared,a.output)
