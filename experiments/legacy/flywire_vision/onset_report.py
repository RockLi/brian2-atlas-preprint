"""Render the descriptive response decomposition and frozen P13c diagnostic decision."""
import argparse
from pathlib import Path
import numpy as np
from .response_diagnostic import read,save,sha,decode,analyze,SPECS


def build(root):
    p=read(root/'protocol.json');v=read(root/'verification.json');comparison=read(root/'comparison.json');mut=read(root/'mutation-tests.json');source=Path(p['source'])
    assert v['all_passed'] and v['comparison_sha256']==sha(root/'comparison.json') and mut['all_passed'] and mut['verifier_sha256']==v['verifier_sha256']
    rows=read(root/'rows.json');index={(r['background'],r['site'],r['kind'],r['delay']):r for r in rows};new=[]
    for r in comparison:
        data={k:decode(root,index[r['background'],r['site'],*k]) for k in SPECS};new.append({'background':r['background'],'site':r['site'],'subtype':r['subtype'],**analyze(data,*r['window_ticks'])})
    assert all(np.isclose(a['raw_mean_mv'],b['raw_mean_mv'],atol=1e-11) and np.isclose(a['J_mean_mv'],b['J_mean_mv'],atol=1e-11) and np.isclose(a['relative_superposition_error'],b['relative_RMS_error'],atol=1e-11) for a,b in zip(new,comparison))
    save(root/'new-response-accounting.json',new)
    old=read(root/'offline/metrics.json');oldprimary=[r for r in old if r['window_index']==0];oldmatched=[r for r in old if r['window_index']==1 and r['background'] in p['backgrounds']];criterion=p['criteria'];correct=sum(r['raw_sign']==r['linear_sign']!=0 for r in comparison);preserved=sum(r['raw_sign']==r['old_raw_sign']!=0 for r in comparison);error=float(np.median([r['relative_RMS_error'] for r in comparison]));oldmedian=float(np.median([r['relative_superposition_error'] for r in oldmatched]));jpreserved=sum(r['J_sign']==r['old_J_sign']!=0 for r in comparison)
    early=lambda r:r['first_interaction_tick']['v'] is not None and all(t is None or t>r['first_interaction_tick']['v'] for k,t in r['first_interaction_tick'].items() if k!='v')
    result={'prediction_sign_correct':correct,'cases':16,'median_relative_waveform_error':error,'matched_old_median_relative_waveform_error':oldmedian,'raw_sign_preserved_after_shift':preserved,'J_sign_preserved_after_shift':jpreserved,'frozen_criteria':{'sign_prediction':correct>=criterion['sign_prediction_min_cases'],'waveform_approximation':error<=criterion['median_waveform_relative_RMS_error_max'],'onset_sign_stability':preserved>=criterion['old_to_new_raw_sign_min_cases']},'old_all32_raw_linear_sign_correct':sum(r['raw_sign']==r['linear_sign']!=0 for r in oldprimary),'old_nonspiking_cases':sum(not r['any_target_spike'] for r in oldprimary),'old_nonspiking_voltage_interaction_before_conductance_interaction':sum(not r['any_target_spike'] and early(r) for r in oldprimary),'new_spiking_cases':sum(r['any_target_spike'] for r in new),'comparison_sha256':sha(root/'comparison.json'),'verification_sha256':sha(root/'verification.json')}
    save(root/'summary.json',result)
    lines=['# P13c：响应分解与绝对起始时刻诊断','',f'本轮完成116/128次新全图运行。新单独刺激在双输入运行前产生固定预测，16组中符号预测正确 **{correct}/16**；波形相对RMS误差中位数 **{error:.2%}**。起始时刻后移50ms后原始顺序符号保持 **{preserved}/16**。阶段判读见[DECISION.md](DECISION.md)。','',
    '## 范围与先验规则','',
    '第一步只分析P13既有全部8目标×4背景，主窗口[152,202)ms和到达对齐的50ms窗口，共64条诊断记录，没有新增仿真。D=AB−BA；L=(A早+B晚)−(B早+A晚)；J=D−L。L是各自时间匹配的单独刺激叠加，未拟合参数，不等同于一个从视频端到端预测的分类器。','',
    '第二步使用全部8目标，选已有背景785和787，分别代表既有交互较强和较弱的背景。这是有意选取的开发对照，不是未见背景确认集。仅将所有视觉及受控源刺激事件从对应P13条件统一后移500tick（50ms）；包内10ms、两位置20ms间隔与每源两事件不变。全部背景事件保留在原时刻，权重、图、模型常数、初态与监测均不变。全体16组在运行前通过不应期冲突检查，没有额外移位。','',
    '主窗口固定为首刺激实际到达起，到末次到达后20ms，即[1538+shift,2038+shift)tick；旧、新条件的窗口相对各自到达时刻相同。P13原有T4d/785额外3ms在两个起始时刻都保留。后移改变了全网背景相位及经过的网络状态时间，不能只归因于某个抑制通路。','',
    '预算：2次旧刺激身份重现 + 112次新单独/同时/顺序刺激 + 2次恢复 = 116次；硬上限128包括失败及重跑，实际没有失败。16条空白复用对应P13已验证的回放空白，形成128条新时刻条件记录；旧时刻参照按哈希复用。','',
    '每组先记录A早、B早、A晚、B晚，再写入带时间和来源哈希的预测，之后才启动同时AB、AB20、BA20。预测公式、观察窗口和符号容差1e-8mV均不变。预先约定的探索性充分性标准为：符号正确至少14/16，波形相对RMS误差中位数不超过20%，旧→新原始符号保持至少14/16；三项分别判断，不代表生物显著性、统计功效或四方向识别。','',
    '## 冻结预测检验','',
    '| 指标 | 结果 | 预定标准 |','| --- | --- | --- |',f'| 新单独刺激预测新AB−BA符号 | {correct}/16 | ≥14/16 |',f'| RMS(J) / RMS(D)中位数 | {error:.4%} | ≤20% |',f'| 同一相对窗口的旧条件误差中位数 | {oldmedian:.4%} | 配对描述参照 |',f'| 旧→新原始符号保持 | {preserved}/16 | ≥14/16 |',f'| 旧→新交互J符号保持 | {jpreserved}/16 | 描述，不作为必要方向标准 |','',
    '| 背景 | 目标 | 旧D均值mV | 新D均值mV | 单独叠加预测mV | 相对RMS误差 | 旧/新J平均绝对值mV |','| --- | --- | --- | --- | --- | --- | --- |']
    for r in comparison:lines.append(f'| {r["background"]} | {r["subtype"]} | {r["old_raw_mean_mv"]:.6f} | {r["raw_mean_mv"]:.6f} | {r["linear_mean_mv"]:.6f} | {r["relative_RMS_error"]:.4f} | {r["old_J_mean_abs_mv"]:.6f} / {r["J_mean_abs_mv"]:.6f} |')
    lines+=['','## 既有全部32组的分解','',f'原始主窗口中，L与D符号一致{result["old_all32_raw_linear_sign_correct"]}/32；没有目标发放的{result["old_nonspiking_cases"]}组中，{result["old_nonspiking_voltage_interaction_before_conductance_interaction"]}组最早检测到的电压交互早于电导交互。电压检测容差1e-11V，电导1e-12，仅用于数值定位，不能把未检出等同于绝对不存在。','',
    '当前方程在可更新时为ΔV=0.005[−(V+0.052)−ge·V−gi·(V+0.070)]。即使电导按输入叠加，ge·V和gi·V仍可能产生电压交互。这为早期非加性提供了局部动力学解释；后来的反馈、电导交互与复位仍可能参与。','',
    '逐步会计分解保留实际不应期门控，将漏电、兴奋、抑制增量累加，发放时加入复位修正。各项相加重建J，但项间通过V相互依赖，不能当作独立因果份额，也不能只按最大的绝对项命名致因。','',
    '| 背景 | 目标 | 窗口起始距阈值mV | 相对RMS误差 | 目标发放 | J均值mV | 漏电 / 兴奋 / 抑制 / 复位会计项mV |','| --- | --- | --- | --- | --- | --- | --- |']
    for r in oldprimary:lines.append(f'| {r["background"]} | {r["subtype"]} | {r["baseline_threshold_distance_mv"]:.3f} | {r["relative_superposition_error"]:.4f} | {"有" if r["any_target_spike"] else "无"} | {r["J_mean_mv"]:.6f} | '+' / '.join(f'{r["component_J_mean_mv"][k]:.6f}' for k in ('leak','exc','inh','reset'))+' |')
    lines+=['','新条件的相同分解见`new-response-accounting.json`；新旧所有单条件发放、电压、电导与实际投递均保留或通过冻结父归档引用。','',
    '## 验证与复算','',f'独立验证核对全部来源、事件后移、背景保留、逐边写入、预测先后、恢复和116次账本，检查{v["actual_edge_writes_checked"]:,}条实际写入及{v["voltage_steps_checked"]:,}个电压步；最大误差{v["max_voltage_error_V"]:.3g}V。5项Python测试通过；使用P13完全相同的原生二进制，不重新编译或改动力学。语义错误注入检查见`mutation-tests.json`。','',
    '```sh','export MPLCONFIGDIR=/private/tmp/flywire-vision-mpl','export PYTHONPATH=brian2-rust/python:brian2-rust/experiments:.','export OPENBLAS_NUM_THREADS=1',f'/private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.verify_onset_diagnostic {root}',f'/private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.onset_report {root}','```','',
    'P13的全部证据保持冻结。本轮首次离线导出曾因NumPy整数的JSON序列化失败，修正后在独立v2目录重算；没有丢弃仿真尝试或更改原始数据。未新增可视化、拓扑重连、参数调优或视频分类。']
    (root/'README.md').write_text('\n'.join(lines)+'\n');print(result,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);build(a.parse_args().root)
