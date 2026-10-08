"""Descriptive P12b tables from independently reconstructed archived metrics."""
import argparse
from pathlib import Path
import numpy as np
from .motion_refinement import read,sha
from .run_experiment import save


def build(root):
    verification=read(root/'verification.json')
    if not verification['all_passed'] or verification['report_sha256']!=sha(root/'report.json'):raise ValueError('independent verification must pass first')
    mutation=read(root/'verifier-mutation-tests.json')
    if not mutation['all_passed'] or not mutation['original_files_unchanged'] or len(mutation['mutations'])!=4 or mutation['verifier_sha256']!=sha(Path(__file__).with_name('verify_background_clamp.py')):raise ValueError('semantic mutation checks must pass against current verifier')
    metrics=read(root/'independent-metrics.json');p=read(root/'protocol.json');support={(r['background'],r['site']):r for r in verification['background_support']};by={(r['background'],r['condition'],r['site'],r['window_index']):r for r in metrics};pairs=[]
    for bg in (783,784):
        for sid in range(8):
            a=by[bg,'without_background',sid,0];b=by[bg,'with_background',sid,0];x=a['voltage_interaction_mean_abs_mv'];y=b['voltage_interaction_mean_abs_mv'];sa=a['voltage_interaction_mean_mv'];sb=b['voltage_interaction_mean_mv'];difference=y-x
            state='相同' if abs(difference)<1e-10 else ('增强' if difference>0 else '减弱')
            sign='反向' if sa*sb<0 and min(abs(sa),abs(sb))>1e-6 else ('同号' if sa*sb>0 else '近零/不定')
            pairs.append({'background':bg,'site':sid,'subtype':a['subtype'],**support[bg,sid],'without_mean_abs_mv':x,'with_mean_abs_mv':y,'difference_mean_abs_mv':difference,'without_signed_mean_mv':sa,'with_signed_mean_mv':sb,'without_peak_abs_mv':a['voltage_interaction_peak_abs_mv'],'with_peak_abs_mv':b['voltage_interaction_peak_abs_mv'],'amplitude_change':state,'sign_change':sign,'with_without_amplitude_ratio':y/x if x>1e-6 else None,
                'without_raw_order_mean_mv':a['voltage_order_mean_mv'],'with_raw_order_mean_mv':b['voltage_order_mean_mv'],'without_target_spikes':a['target_spikes'],'with_target_spikes':b['target_spikes'],'without_spike_interaction':a['spike_interaction'],'with_spike_interaction':b['spike_interaction']})
    windows=[]
    for bg in (783,784):
        for window in range(5):
            for group in ('all','early_background_support'):
                ids=[sid for sid in range(8) if group=='all' or support[bg,sid]['early_support']]
                means={condition:float(np.mean([by[bg,condition,sid,window]['voltage_interaction_mean_abs_mv'] for sid in ids])) for condition in ('without_background','with_background')}
                windows.append({'background':bg,'window_index':window,'group':group,'n_pairs':len(ids),**means})
    across={condition:{'same_sign_sites':[sid for sid in range(8) if by[783,condition,sid,0]['voltage_interaction_mean_mv']*by[784,condition,sid,0]['voltage_interaction_mean_mv']>0],
                       'opposite_sign_sites':[sid for sid in range(8) if by[783,condition,sid,0]['voltage_interaction_mean_mv']*by[784,condition,sid,0]['voltage_interaction_mean_mv']<0]} for condition in ('without_background','with_background')}
    result={'protocol_sha256':sha(root/'protocol.json'),'verification_sha256':sha(root/'verification.json'),'independent_metrics_sha256':sha(root/'independent-metrics.json'),'pairs':pairs,'window_means':windows,'across_background_signs':across,
        'amplitude_counts':{s:sum(r['amplitude_change']==s for r in pairs) for s in ('相同','增强','减弱')},'raw_order_same_sign_across_backgrounds':sum(by[783,'with_background',sid,0]['voltage_order_mean_mv']*by[784,'with_background',sid,0]['voltage_order_mean_mv']>0 for sid in range(8)),'sign_reversals':sum(r['sign_change']=='反向' for r in pairs),'empty_background_pairs':sum(r['events']==0 for r in pairs),'early_supported_pairs':sum(r['early_support'] for r in pairs)}
    save(root/'summary.json',result)
    lines=['# P12b：保留固定背景输出后的顺序交互','', '2026-09-11，`codex/flywire-vision`。本阶段复用 P11 八个目标、两个背景及冻结 P12 执行器，不调整内部权重、时间常数或分类器。','',
        f'完成 **{verification["native_attempts"]}/200 次新全图运行**：11 次前置背景回放、84 次新正式运行、3 次恢复。正式矩阵 256 条记录，其中 161 条复用 P12 身份一致的记录、11 条复用本阶段空白回放。复用不增加独立样本数。','',
        '两组均保留全网背景发生源；差异只在所选两源的传出列中是否回放固定的无刺激背景事件。详见 [METHODS.md](METHODS.md)。','',
        '## 主窗口结果','',
        '主窗口固定为 [152,202) ms。J = (AB − A早 − B晚 + 空白) − (BA − B早 − A晚 + 空白)。各条件使用自身空白及时间匹配的单独刺激。报告有符号时间平均 J，以及先逐时刻取绝对值后平均的幅度；单位均为 mV。','',
        '| 背景 | 目标 | 背景事件数 / 早期支持 | 无回放 / 有回放平均 |J| | 无回放 / 有回放有符号 J | 幅度 / 符号变化 |','| --- | --- | --- | --- | --- | --- |']
    # Escape the formula bar inside a Markdown table cell.
    lines[-2]=lines[-2].replace('平均 |J|','平均绝对 J')
    for r in pairs:lines.append(f'| {r["background"]} | {r["subtype"]} | {r["events"]} / {"有" if r["early_support"] else "无"} | {r["without_mean_abs_mv"]:.6f} / {r["with_mean_abs_mv"]:.6f} | {r["without_signed_mean_mv"]:.6f} / {r["with_signed_mean_mv"]:.6f} | {r["amplitude_change"]} / {r["sign_change"]} |')
    lines+=['',f'16 个配对组合中：{result["amplitude_counts"]["增强"]} 个增强、{result["amplitude_counts"]["减弱"]} 个减弱、{result["amplitude_counts"]["相同"]} 个数值上相同；{result["sign_reversals"]} 个有符号平均 J 反向。相同判据为绝对幅度差 < 1e-10 mV，反向计数排除任一有符号均值绝对值 ≤ 1e-6 mV；它们只是描述规则，不是生物显著性检验。','',
        f'有背景回放时，两个背景间有符号 J 同号的目标为 {len(across["with_background"]["same_sign_sites"])}/8；无回放匹配条件为 {len(across["without_background"]["same_sign_sites"])}/8。原始 AB−BA 电压均值跨背景同号仅 {result["raw_order_same_sign_across_backgrounds"]}/8；J 的符号一致不等于原始方向响应一致。此处只比较同一目标的 A/B 轴，没有将不同空间轴的符号合并成四方向准确率。','',
        '## 背景支持与时间窗口','',f'**{result["empty_background_pairs"]}/16 组的两源背景事件列为空，只有 {result["early_supported_pairs"]}/16 组有背景事件能在主窗口结束前到达。** 空背景或晚到背景造成的早期相同结果，不能当作额外的跨背景稳定性证据。分组依据空白事件时刻，在查看新响应前规定。','',
        '| 背景 | 窗口 | 全八目标：无 / 有背景回放平均幅度 | 有早期背景支持目标：无 / 有回放平均幅度 |','| --- | --- | --- | --- |']
    for bg in (783,784):
        for w in range(5):
            allrow=next(r for r in windows if r['background']==bg and r['window_index']==w and r['group']=='all');sub=next(r for r in windows if r['background']==bg and r['window_index']==w and r['group']=='early_background_support');label=['主窗口','到达对齐；末次到达后 20 ms','到达对齐；末次到达后 50 ms','到达对齐；末次到达后 100 ms','到达对齐；末次到达后 200 ms'][w]
            lines.append(f'| {bg} | {label} | {allrow["without_background"]:.6f} / {allrow["with_background"]:.6f} | {sub["without_background"]:.6f} / {sub["with_background"]:.6f}（n={sub["n_pairs"]}） |')
    lines+=['','补充窗口均从首个指定事件实际到达时刻开始，结束于末次到达后指定时长；不是只截取刺激结束后的响应。这里改变的是同一记录的观察窗口，没有改变刺激持续时间或速度。','',
        '## 冲突处理与匹配参照','',
        '背景 783 的 T4a 原背景事件在 150.4 ms，原首个刺激事件在 152.0 ms，间隔小于 2.2 ms 不应期。预先冻结的规则保留全部背景事件，选择最小共同非负偏移；该组所有刺激、顺序、单独与同时刺激以及无回放对照均后移 0.6 ms。其余 15 组偏移为零。','',
        '因此本阶段 T4a/783 的无回放刺激条件为新运行，不能直接复用 P12 原始未移位结果。主表比较的是时间匹配条件；原 P12 普通驱动及未移位数据保持冻结，可用于历史参照，不作为这个移位配对的唯一变量对照。','',
        '## 原始响应与脉冲','',
        '完整的原始 AB−BA、电压交互均值和峰值、目标全部八种条件脉冲、活源细胞 AB/BA 脉冲及计数交互见 `independent-metrics.json`，主窗口配对摘要见 `summary.json`。`trials/` 保留每探针目标与两源细胞的完整 6000 tick 电压/电导及自身脉冲，另存背景、刺激、合并传出事件与逐边实际写入。活源细胞的自然发放与最终受控传出事件分开解释。','',
        '## 验证和复算','',
        f'独立验证通过，覆盖 {verification["actual_edge_writes_checked"]:,} 条实际逐边写入记录检查（包括复用记录的再次核验），以及 {verification["voltage_steps_checked"]:,} 个有效膜电位更新步；最大方程误差 {verification["max_voltage_equation_error_V"]:.3g} V。','',
        '核验包括全部 16 组回放空白与普通空白一致、非空组前置全图哈希恢复、真实 CSR 边与权重、1.8 ms 延迟、无重复注入、背景原样保留、逐边扣除空白后的刺激预算配平、刺激前空白、背景事件到达前两干预一致、第二包到达前交互为零、三次末尾恢复、全部 160 条窗口指标独立重算、旧记录身份、冻结证据及 98 次运行逐项归属。','',
        '12 项 Python 测试通过。另用完整真实归档的临时副本执行四项语义错误检测：修改指标、只改单个顺序的背景、重复实际写入、删除恢复记录；均须由对应语义检查拒绝，原始归档保持不变。详见 `tests.log` 和 `verifier-mutation-tests.json`。','',
        '在仓库根目录执行（需既有冻结父模型与 Python 环境）：','',
        '```sh',f'PYTHONPATH=brian2-rust/python:brian2-rust/experiments:. OPENBLAS_NUM_THREADS=1 /private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.verify_background_clamp {root}',f'PYTHONPATH=brian2-rust/python:brian2-rust/experiments:. OPENBLAS_NUM_THREADS=1 /private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.background_clamp_report {root}','```','',
        '## 推断边界','',
        '- 回放的是无刺激时记录的固定背景输出。源细胞自然输出被替换，刺激后反馈不能更新它们的传出事件，因此不是完整恢复自然背景与反馈。',
        '- J 非零说明当前测量下存在非加性顺序交互，不自动等于方向选择性、目标自身计算或可用分类能力；原始顺序对比也须一起报告。',
        '- 比值不是某个机制的贡献百分比。仍未分离目标整合、直接权重不对称、周边反馈及阈值/复位效应。',
        '- 八个目标已用于开发，两个背景也已查看；来自同一连接组的目标和确定性重复不能充当独立生物重复。',
        '- 当前未提供真实图相对匹配重连的优势，没有证明稳定四方向识别。本阶段不调整模型参数或分类器，不开发可视化，不运行 P13。','',
        '阶段判读与是否进入 P13 的建议见 [DECISION.md](DECISION.md)。父证据见 [P12](../flywire-vision-event-clamp-v1/README.md)。']
    detail=['| 背景 | 目标 | 无 / 有回放原始 AB−BA 均值 (mV) | 无 / 有回放 J 绝对峰值 (mV) | 有回放 AB / BA 脉冲 | 计数交互 |','| --- | --- | --- | --- | --- | --- |']
    for r in pairs:
        counts=r['with_target_spikes'];detail.append(f'| {r["background"]} | {r["subtype"]} | {r["without_raw_order_mean_mv"]:.6f} / {r["with_raw_order_mean_mv"]:.6f} | {r["without_peak_abs_mv"]:.6f} / {r["with_peak_abs_mv"]:.6f} | {counts["AB200"]} / {counts["BA200"]} | {r["with_spike_interaction"]} |')
    at=lines.index('## 原始响应与脉冲')+2;lines[at:at]=detail+['','主窗口内，背景783的T4a出现AB=1、BA=0，但A早单独刺激也为1；背景784的T4c出现AB=0、BA=1，但B早单独刺激也为1。16组的目标计数交互均为0。膜电位包含阈值发放与复位的影响，不能把上述原始均值差直接当成独立新增的方向脉冲计算。','']
    (root/'README.md').write_text('\n'.join(lines)+'\n');print({k:v for k,v in result.items() if k not in ('pairs','window_means')},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();build(args.root)
