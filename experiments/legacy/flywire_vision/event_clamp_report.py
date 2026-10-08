"""Deterministic P12 descriptive summary and reviewable stage report."""
import argparse
from pathlib import Path
import numpy as np
from .motion_refinement import read,sha
from .run_experiment import save


def summary(root):
    r=read(root/'formal-v1'/'report.json');v=read(root/'formal-v1'/'verification.json')
    if not v['all_passed'] or v['report_sha256']!=sha(root/'formal-v1'/'report.json'):raise ValueError('verified report required')
    data=r['per_site'];pairs=[];groups=[]
    for bg in (783,784):
        for window in ([1520,2020],[1538,2038],[1538,2338],[1538,2838],[1538,3838]):
            group=[]
            for site in range(8):
                old=next(a for a in data if a['background']==bg and a['site']==site and a['condition']=='ordinary' and a['window_ticks']==window)
                new=next(a for a in data if a['background']==bg and a['site']==site and a['condition']=='controlled' and a['window_ticks']==window)
                a=old['voltage_interaction_mean_mv'];b=new['voltage_interaction_mean_mv'];row={'background':bg,'site':site,'subtype':old['subtype'],'window_ticks':window,
                    'ordinary':old,'controlled':new,'signed_mean_difference_mv':b-a,'signed_mean_magnitude_ratio':abs(b)/abs(a) if abs(a)>=1e-6 else None,
                    'same_signed_mean_sign':bool(np.sign(a)==np.sign(b)),'mean_absolute_interaction_reduced':new['voltage_interaction_mean_abs_mv']<old['voltage_interaction_mean_abs_mv']}
                group.append(row);pairs.append(row)
            oldmean=float(np.mean([a['ordinary']['voltage_interaction_mean_abs_mv'] for a in group]));newmean=float(np.mean([a['controlled']['voltage_interaction_mean_abs_mv'] for a in group]))
            groups.append({'background':bg,'window_ticks':window,'ordinary_mean_absolute_interaction_mv':oldmean,'controlled_mean_absolute_interaction_mv':newmean,'magnitude_ratio':newmean/oldmean if oldmean>=1e-6 else None})
    primary=[a for a in pairs if a['window_ticks']==[1520,2020]];signs=[]
    for site in range(8):
        a=next(x for x in primary if x['background']==783 and x['site']==site)['controlled']['voltage_interaction_mean_mv'];b=next(x for x in primary if x['background']==784 and x['site']==site)['controlled']['voltage_interaction_mean_mv'];signs.append(bool(np.sign(a)==np.sign(b)))
    return {'report_sha256':sha(root/'formal-v1'/'report.json'),'scope':'descriptive paired magnitude and signed contrasts; ratios do not identify unique upstream contribution; 2 backgrounds and 8 previously inspected targets, no population CI',
        'native_attempts':read(root/'formal-v1'/'report.json')['cumulative_native_runs'],'primary_reduced_pairs':sum(a['mean_absolute_interaction_reduced'] for a in primary),'primary_same_sign_pairs':sum(a['same_signed_mean_sign'] for a in primary),'controlled_cross_background_same_sign_sites':sum(signs),'groups':groups,'pairs':pairs}


def markdown(root,s):
    v=read(root/'formal-v1'/'verification.json');primary=[a for a in s['pairs'] if a['window_ticks']==[1520,2020]];groups=[a for a in s['groups'] if a['window_ticks']==[1520,2020]]
    text=f'''# P12：实际传出事件配平后，早期顺序效应还剩多少

2026-09-11，`codex/flywire-vision`。**P12 完成：250/300 次全图运行，零失败尝试，全部正式条件与独立核验完成。** 本阶段未调整权重或神经时间常数，未优化分类器、搜索重连图或开发可视化。

**实际传出事件配平后，早期交互总体明显减弱，但没有全部消失，也没有形成稳定一致的方向结论。** 按八个目标的交互平均绝对幅度汇总，两个背景分别保留普通驱动的 **{groups[0]['magnitude_ratio']*100:.1f}% / {groups[1]['magnitude_ratio']*100:.1f}%**。16 个目标/背景组合中 {s['primary_reduced_pairs']} 个减弱；也存在反向或增强的例外。

这里的百分比是两种干预条件的描述性幅度比，**不能解释成“其余百分比唯一由上游贡献”**：受控条件还移除了两源细胞的原生背景输出，指定事件数与自然发放也不必相同。每个条件都用自己的空白与单独刺激扣除。

## 与 P11 的连接

复用 P11 八个目标、相邻输入、外部脉冲和主窗口 [152,202) ms。背景 783 的所有普通模式结果精确复现 P11，包括全网络事件/终态及所选细胞电压轨迹。背景 784 仅改变背景源事件列，初始状态和背景连接映射保持不变。

受控模式保留输入神经元自身动态与外部驱动，阻止其自然脉冲进入真实出边，改为两源各两次预定事件。AB/BA 逐真实出边的事件数、接触权重和 1.8 ms 延迟配平。日志取自实际突触写入位置，核验没有重复投递；并非只检查外部输入计划。

## 早期交互的汇总幅度

每个目标先计算 J = (AB − A早 − B晚 + 空白) − (BA − B早 − A晚 + 空白)，再计算主窗口中 |J| 的时间平均，最后对八个目标取平均。单位 mV；此汇总不将不同空间轴的有符号值相消。

| 背景 | 普通驱动平均幅度 | 事件受控平均幅度 | 幅度比 |
| --- | ---: | ---: | ---: |
'''
    for a in groups:text+=f"| {a['background']} | {a['ordinary_mean_absolute_interaction_mv']:.6f} | {a['controlled_mean_absolute_interaction_mv']:.6f} | {a['magnitude_ratio']*100:.1f}% |\n"
    text+='''
## 每个目标的主指标与实际目标发放

主指标为有符号平均 J；绝对峰值作为补充。电压单位 mV。完整差值、幅度比、原始 AB−BA 对比与其他窗口保存在 `summary.json` 和 `formal-v1/report.json`。比值分母接近零时尤其不稳定，优先比较下列绝对量。

| 背景 | 目标 | 普通 J 均值 | 受控 J 均值 | 普通 / 受控 J 绝对峰值 | 普通目标 AB/BA 脉冲 | 受控目标 AB/BA 脉冲 |
| --- | --- | ---: | ---: | --- | --- | --- |
'''
    for a in primary:
        o=a['ordinary'];c=a['controlled'];text+=f"| {a['background']} | {a['subtype']} | {o['voltage_interaction_mean_mv']:.6f} | {c['voltage_interaction_mean_mv']:.6f} | {o['voltage_interaction_peak_abs_mv']:.6f} / {c['voltage_interaction_peak_abs_mv']:.6f} | {o['target_spikes']['AB200']}/{o['target_spikes']['BA200']} | {c['target_spikes']['AB200']}/{c['target_spikes']['BA200']} |\n"
    text+=f'''
受控与普通 J 均值同号的组合为 {s['primary_same_sign_pairs']}/16；受控 J 均值在两个背景之间同号的目标为 {s['controlled_cross_background_same_sign_sites']}/8。这是对敏感性的描述，不是偏好方向显著性或群体泛化估计。A/B 仍是各自空间轴的坐标排序，不可直接合并为左右/上下准确率。

背景 784 的 T4c 在受控 BA 下发放一次、AB 下零次，但对应 B早单独刺激也发放一次，计数交互为零。不能把这一脉冲差直接视为新的方向计算证据。其他主窗口 AB/BA 目标组合均零发放；电压测量仍提供了单看计数看不到的信息。

## 活输入神经元发放与受控传出事件分开记录

下表列出主窗口内 A/B 活细胞的实际发放，每格按“AB 时 A/B；BA 时 A/B”表示。受控模式下这些活细胞脉冲被阻止沿所选真实出边传播；**真正用于投递的是预定传出事件，两种顺序每源都为两次**。因此活细胞发放仍不等量，不表示本轮逐出边配平失败。

| 背景 | 目标 | 普通模式活输入发放 | 受控模式活输入发放 | 受控投递源事件 AB；BA |
| --- | --- | --- | --- | --- |
'''
    for a in primary:
        def fmt(row):
            x=row['input_spikes'];return f"{x['AB']['a']}/{x['AB']['b']}；{x['BA']['a']}/{x['BA']['b']}"
        text+=f"| {a['background']} | {a['subtype']} | {fmt(a['ordinary'])} | {fmt(a['controlled'])} | 2/2；2/2 |\n"
    text+='''
## 对齐投递时刻及延长观察窗口

下表使用 [153.8 ms, 末次预定投递后指定时长) 的窗口，单位 mV，仍为八目标平均的时间平均 |J|。窗口都在正式运行前指定，没有按结果挑选。

| 背景 | 末次预定投递后观察 | 普通幅度 | 受控幅度 | 幅度比 |
| --- | --- | ---: | ---: | ---: |
'''
    for a in s['groups']:
        if a['window_ticks']==[1520,2020]:continue
        after=(a['window_ticks'][1]-1838)/10;text+=f"| {a['background']} | {after:g} ms | {a['ordinary_mean_absolute_interaction_mv']:.6f} | {a['controlled_mean_absolute_interaction_mv']:.6f} | {a['magnitude_ratio']*100:.1f}% |\n"
    text+='''
对齐后的早期幅度比约 19%，与 P11 同窗比较一致。但 100/200 ms 的较长窗口中，受控幅度反而超过普通条件。这说明“早期减弱”不能推广为“整个网络时序效应被消除”；晚期可能包含反馈、状态积累及阈值/复位效应，本阶段未将它们逐一隔离。

## 验证、预算与证据链

'''
    text+=f'''3 项原生小网络测试、4 项 Python 测试通过。9 次前置全图检查先于正式协议冻结，涵盖 P11 监测一致性、两背景的原生事件精确回放、空替换与独立 transmission=0 对照、恢复检查。正式阶段复用 19 条条件记录，新增 237 次运行；末尾 4 次恢复均精确重现，总数 **250**，无失败尝试或因实现错误而重跑。

独立验证器核验了 **{v['actual_edge_writes_checked']:,} 条逐边写入记录检查**（含复用记录的复核）以及 **{v['voltage_steps_checked']:,} 个有效膜电位更新步**，最大方程误差 {v['max_voltage_equation_error_V']:.3g} V。核验覆盖所有刺激计划、实际源事件、写入时刻/权重、重复投递、逐边 AB/BA 配平、刺激前空白、第二包到达前交互为数值零、所有指标和原 P11 普通结果。预算中的每次进程均能关联到验证、正式记录或恢复检查。

`budget.json` 是累计运行账本；`checkpoint.json` 是终点检查点。`validation-v1/` 保存前置协议、9 条原始记录、原生源码/二进制与检查结果；`formal-v1/` 保存冻结协议、256 条正式记录、4 条恢复记录及指标/核验。记录含每探针三细胞的完整 6000 tick v/ge/gi、自身脉冲、外部与替换事件、真实逐边写入和全图事件/状态摘要。无损归档逐数组哈希可复算，CSR 和父快照复用已有锁定文件，不重复复制整张图。

## 阶段结论与限制

P12 支持的结论是：**P11 的早期电压交互对实际预突触事件及背景传输状态很敏感；实施配平后多数目标的早期交互明显减弱，同时存在残余及例外。**

当前仍不能判断残余来自目标内在整合、直接权重不对称、周边反馈还是组合机制；也不能把幅度比解释为唯一上游贡献。仅有两个背景和八个已看过的目标，未提供生物群体置信区间；这里的变化范围、符号翻转和晚期反转就是需要保留的不确定性。没有证明稳定方向识别或真实拓扑优势。

本阶段按约定停止新增仿真，剩余 50 次预算不追加。是否进入 P13，或先补充保留背景事件的受控干预，需作为下一阶段的新协议决定；本轮没有启动这些工作。

方法详见 [METHODS.md](METHODS.md)，复算命令见 [实验 README](../../experiments/flywire_vision/README.md#presynaptic-event-clamp-p12)。前置证据：[P10](../flywire-vision-timing-probe-v1/README.md)、[P11](../flywire-vision-convergent-probe-v1/README.md)。
'''
    return text


def main(root,verify=False):
    s=summary(root);text=markdown(root,s)
    if verify:
        assert read(root/'summary.json')==s and (root/'README.md').read_text()==text
        save(root/'report-verification.json',{'all_passed':True,'summary_sha256':sha(root/'summary.json'),'markdown_sha256':sha(root/'README.md'),'reporter_sha256':sha(Path(__file__))})
    else:save(root/'summary.json',s);(root/'README.md').write_text(text)
    print({'report_verified' if verify else 'report_written':True},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--verify',action='store_true');a=ap.parse_args();main(a.root,a.verify)
