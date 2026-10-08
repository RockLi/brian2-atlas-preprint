"""P13 descriptive repeated-background analysis from independently rebuilt metrics."""
import argparse
from pathlib import Path
import numpy as np
from .motion_refinement import read,sha
from .run_experiment import save


def build(root):
    v=read(root/'verification.json');p=read(root/'protocol.json');m=read(root/'independent-metrics.json');mut=read(root/'verifier-mutation-tests.json')
    if not v['all_passed'] or v['report_sha256']!=sha(root/'report.json') or not mut['all_passed'] or mut['verifier_sha256']!=v['verifier_sha256']:raise ValueError('complete independent verification and mutation checks required')
    by={(r['background'],r['site'],r['window_index']):r for r in m};sign=lambda x:0 if abs(x)<=1e-8 else (1 if x>0 else -1);sites=[]
    for sid,s in enumerate(p['sites']):
        item={'site':sid,'subtype':s['subtype'],'target':s['target'],'signed_displacement':s['signed_displacement'],'windows':[]}
        for wi in range(5):
            rows=[by[bg,sid,wi] for bg in p['backgrounds']];j=[r['voltage_interaction_mean_mv'] for r in rows];raw=[r['voltage_order_mean_mv'] for r in rows];a=[r['voltage_interaction_mean_abs_mv'] for r in rows];js=[sign(x) for x in j];rs=[sign(x) for x in raw]
            jmatch=sum(js[0]!=0 and x==js[0] for x in js[1:]);rmatch=sum(rs[0]!=0 and x==rs[0] for x in rs[1:])
            item['windows'].append({'window_index':wi,'signed_mean_J_mv':j,'mean_abs_J_mv':a,'raw_order_mean_mv':raw,'J_signs':js,'raw_signs':rs,'J_held_matches':jmatch,'raw_held_matches':rmatch,'both_all_backgrounds':jmatch==rmatch==3,'mean_abs_J_min_max_ratio':min(a)/max(a) if max(a)>1e-8 else None,'spike_J':[r['spike_interaction'] for r in rows]})
        sites.append(item)
    means=[{'background':bg,'window_index':wi,'mean_abs_J_mv':float(np.mean([by[bg,sid,wi]['voltage_interaction_mean_abs_mv'] for sid in range(8)]))} for bg in p['backgrounds'] for wi in range(5)]
    result={'verification_sha256':sha(root/'verification.json'),'independent_metrics_sha256':sha(root/'independent-metrics.json'),'sites':sites,'window_means':means,'primary_J_all_backgrounds':sum(s['windows'][0]['J_held_matches']==3 for s in sites),'primary_raw_all_backgrounds':sum(s['windows'][0]['raw_held_matches']==3 for s in sites),'primary_both_all_backgrounds':sum(s['windows'][0]['both_all_backgrounds'] for s in sites),'primary_nonzero_spike_interactions':sum(r['spike_interaction']!=0 for r in m if r['window_index']==0),'early_background_support_pairs':sum(b['early_support'] for b in read(root/'background-records.json')),'empty_background_pairs':sum(b['event_count']==0 for b in read(root/'background-records.json'))}
    primary=[r for r in m if r['window_index']==0]
    result['primary_mean_abs_J_distribution_mv']={k:float(f([r['voltage_interaction_mean_abs_mv'] for r in primary])) for k,f in [('min',np.min),('median',np.median),('max',np.max)]}
    result['primary_pairs_with_any_target_spike']=sum(any(r['target_spikes'].values()) for r in primary)
    result['primary_pairs_with_raw_AB_BA_count_difference']=sum(r['target_spikes']['AB200']!=r['target_spikes']['BA200'] for r in primary)
    save(root/'summary.json',result)
    lines=['# P13：新目标与新背景的有限独立重复','', '本阶段完成265/300次新全图运行，256条正式记录。10项Python测试、3项原生测试、逐边检查、4次恢复、独立归档复算及5项语义错误检测通过。未调整权重、时间常数或分类器，未开发可视化或启动拓扑实验。','',
        f'以背景785固定每目标响应符号，再检验786/787/788：主窗口J在四背景均同号为 **{result["primary_J_all_backgrounds"]}/8** 个目标，原始AB−BA为 **{result["primary_raw_all_backgrounds"]}/8**，两指标都保持为 **{result["primary_both_all_backgrounds"]}/8**。这不是方向准确率。阶段建议见 [DECISION.md](DECISION.md)。','',
        '## 冻结设计与解释范围','',
        '八个新目标各对应一个T4/T5亚型，按固定空间位置及解剖规则选择，排除P11–P12b全部24个目标/输入监测细胞，且距每个旧目标至少0.15归一化单位。新目标及输入共24个细胞互不重复。A/B按各自输入空间坐标排序，保存有符号位移；不能跨不同轴平均符号或直接与旧亚型偏好对比。','',
        '四个新背景仅改变全网背景发生源事件列，初态与背景连接映射不变。所选两源自然输出被抑制，以普通空白记录的固定背景输出加逐源两次受控刺激替换。每背景/目标有空白、四个时间匹配单独刺激、同时刺激、AB20、BA20。四事件预算与真实出边、权重、1.8ms延迟不变，包内间隔10ms、两位置间隔20ms。','',
        '本轮复验固定背景回放条件，没有另跑无回放矩阵。跨背景差异包含全网背景变化，不能单独归因于所选源背景输出。背景785的T4d按预先冻结的冲突规则统一后移3ms，其余组合不移位；该目标跨背景比较还包含绝对刺激时刻差异，须结合到达时刻对齐的补充窗口解释。','',
        '主窗口为[152,202)ms。J=(AB−A早−B晚+空白)−(BA−B早−A晚+空白)。主指标为有符号时间平均J，另报平均绝对J、峰值、原始AB−BA和计数交互。符号判别容差1e-8mV仅用于数值区分，不是功能显著性门槛。本轮是新模型目标/背景的复验与波动摸底，不是独立生物个体的确认试验。','',
        '## 全部新目标的早期结果','',
        '下面每格按背景785 / 786 / 787 / 788排列，电压单位mV。后三背景只检验首背景的符号，不能重新指定偏好。参考规则在全部运行前冻结；reference-signs.json在后续背景已开始仿真时保存，但仅读取背景785的64条记录，独立验证器核对其全部来源哈希及符号。','',
        '| 目标 | 新神经元 | 有符号平均J | 平均绝对J | J符号保持 / 3 | 原始顺序符号保持 / 3 | 最小/最大绝对幅度 |','| --- | --- | --- | --- | --- | --- | --- |']
    for s in sites:
        w=s['windows'][0];fmt=lambda x:' / '.join(f'{a:.6f}' for a in x);ratio=w['mean_abs_J_min_max_ratio'];lines.append(f'| {s["subtype"]} | {s["target"]} | {fmt(w["signed_mean_J_mv"])} | {fmt(w["mean_abs_J_mv"])} | {w["J_held_matches"]} | {w["raw_held_matches"]} | {ratio:.4g} |' if ratio is not None else f'| {s["subtype"]} | {s["target"]} | {fmt(w["signed_mean_J_mv"])} | {fmt(w["mean_abs_J_mv"])} | {w["J_held_matches"]} | {w["raw_held_matches"]} | 不定 |')
    lines+=['','## 原始顺序响应与脉冲','',f'32个目标/背景组合中，主窗口非零目标脉冲计数交互为{result["primary_nonzero_spike_interactions"]}组。膜电位包含阈值发放和复位影响；原始AB/BA计数差应与单独刺激一起解释。','',
        '| 目标 | 原始AB−BA平均电压（四背景，mV） | 目标计数交互（四背景） |','| --- | --- | --- |']
    for s in sites:lines.append(f'| {s["subtype"]} | '+ ' / '.join(f'{x:.6f}' for x in s['windows'][0]['raw_order_mean_mv'])+' | '+' / '.join(map(str,s['windows'][0]['spike_J']))+' |')
    lines+=['', '| 目标 | AB / BA目标脉冲数：785 | 786 | 787 | 788 |', '| --- | --- | --- | --- | --- |']
    for s in sites:
        lines.append(f'| {s["subtype"]} | '+' | '.join(f'{by[bg,s["site"],0]["target_spikes"]["AB200"]} / {by[bg,s["site"],0]["target_spikes"]["BA200"]}' for bg in p['backgrounds'])+' |')
    dist=result['primary_mean_abs_J_distribution_mv']
    lines+=['',f'32组平均绝对J的最小值 / 中位数 / 最大值为 {dist["min"]:.6f} / {dist["median"]:.6f} / {dist["max"]:.6f}mV。{result["primary_pairs_with_any_target_spike"]}组至少一种条件有目标发放，{result["primary_pairs_with_raw_AB_BA_count_difference"]}组AB与BA目标计数不同。此分布及计数汇总仅作描述，不新增判定门槛。']
    lines+=['','全部八种刺激的目标计数、活源发放、完整电压/电导和逐边写入均保留，不只保存正结果。详见`independent-metrics.json`和`trials/`。','',
        '## 观察窗口与背景支持','',
        '| 背景 | 主窗口平均绝对J | 末次到达后20ms | 50ms | 100ms | 200ms |','| --- | --- | --- | --- | --- | --- |']
    for bg in p['backgrounds']:lines.append(f'| {bg} | '+' | '.join(f'{next(r["mean_abs_J_mv"] for r in means if r["background"]==bg and r["window_index"]==wi):.6f}' for wi in range(5))+' |')
    lines+=['',f'32组中，{result["empty_background_pairs"]}组所选源整段背景为空，{result["early_background_support_pairs"]}组有背景事件在主窗口结束前到达。此处“为空”不代表全网没有背景。补充窗口从首刺激实际到达起，结束于末刺激到达后指定时长；这是观察时长变化，不是改变刺激持续时间或速度。','',
        '## 验证与归档','',
        f'逐边独立核验覆盖{v["actual_edge_writes_checked"]:,}条实际写入检查，独立方程核验覆盖{v["voltage_steps_checked"]:,}个有效电压更新步，最大误差{v["max_voltage_equation_error_V"]:.3g}V。验证器独立重选解剖目标、重建四个背景发生器、核对最小共同偏移，重算全部160条窗口记录，确认所有265次运行归属。','',
        '原生源码只更改72处只读监测索引。旧背景空白的全网事件和终态精确复现；4个新普通空白供背景提取，32个逐探针空白回放均精确重现相应普通空白，224次正式刺激及4次预定恢复全部完成。预算在每次进程启动前记账，没有失败尝试或额外补跑。','',
        '归档v/ge/gi对各自空白按IEEE754位模式XOR无损压缩，保存6000tick三细胞轨迹；普通基线保存全部24个监测细胞。完整全网事件和终态保留摘要，用于回放/恢复比较。父模型与CSR按哈希复用，P10–P12b证据保持冻结。','',
        '在仓库根目录，用原Python环境和锁定父资产复算，无需新增全图运行：','',
        '```sh',f'PYTHONPATH=brian2-rust/python:brian2-rust/experiments:. /private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.verify_repeat_study {root}',f'PYTHONPATH=brian2-rust/python:brian2-rust/experiments:. /private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.repeat_report {root}','```','',
        '测试见`tests.log`、`executor/native/unit-tests.log`与`verifier-mutation-tests.json`；冻结规则见`protocol.json`，实际事件偏移见`background-records.json`，预算见`budget.json`。']
    (root/'README.md').write_text('\n'.join(lines)+'\n');print({k:v for k,v in result.items() if k not in ('sites','window_means')},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);a=parser.parse_args();build(a.root)
