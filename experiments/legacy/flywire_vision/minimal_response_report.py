"""Report all prespecified minimal-model comparisons, including failures."""
import argparse
from pathlib import Path
import numpy as np
from .verify_minimal_response import read,save,sha

NAMES={'linear':'两连接线性模型','local':'单细胞+空白背景','single_superposition':'完整网络单独响应叠加'}

def build(root):
    p=read(root/'protocol.json');v=read(root/'verification.json');mut=read(root/'mutation-tests.json');m=read(root/'independent-metrics.json');rows=read(root/'prediction-rows.json')
    assert v['all_passed'] and mut['all_passed'] and v['verifier_sha256']==mut['verifier_sha256'] and v['metrics_sha256']==sha(root/'metrics.json')
    summaries=[]
    for group in (-1,0,1):
        for wi in range(5):
            a=[r for r in m if r['window_index']==wi and (group<0 or r['source_index']==group)]
            for model in NAMES:
                errors=[r['models'][model]['relative_RMS_error'] for r in a if r['models'][model]['relative_RMS_error'] is not None];correct=sum(r['actual_sign']==r['models'][model]['sign']!=0 for r in a);median=float(np.median(errors));summaries.append({'source_index':group,'window_index':wi,'model':model,'n':len(a),'sign_correct':correct,'median_relative_RMS_error':median,'max_relative_RMS_error':max(errors),'undefined_RMS_cases':len(a)-len(errors),'sign_adequacy':correct/len(a)>=p['criteria']['sign_fraction_min'],'waveform_adequacy':median<=p['criteria']['median_relative_RMS_error_max']})
    primary=[r for r in m if r['window_index']==0];full={r['key']:r for r in m if r['window_index']==4};ratios={model:[abs(full[r['key']]['models'][model]['integral_mv_ms'])/abs(r['models'][model]['integral_mv_ms']) for r in primary] for model in ('linear','local')};result={'summaries':summaries,'full_to_early_integral_ratio':{model:{'min':min(a),'median':float(np.median(a)),'max':max(a)} for model,a in ratios.items()},'local_target_spike_pairs_exact':sum(r['actual_spikes_AB_BA']==r['models']['local']['spikes_AB_BA'] for r in primary),'actual_spiking_cases':sum(any(r['actual_spikes_AB_BA']) for r in primary),'local_spiking_cases':sum(any(r['models']['local']['spikes_AB_BA']) for r in primary),'balanced_waveform_zero_all_models':all(r['models'][model]['balanced_peak_abs_mv']==0 for r in m for model in ('linear','local')),'verification_sha256':sha(root/'verification.json'),'independent_metrics_sha256':sha(root/'independent-metrics.json')};save(root/'summary.json',result)
    at=lambda g,wi,model:next(r for r in summaries if (r['source_index'],r['window_index'],r['model'])==(g,wi,model))
    lines=['# P13d：简化模型、输入强度和观察窗口','',
    '**早期顺序符号大部分可以由简单输入强度和时间响应解释；完整波形及较晚响应仍不能充分解释。** 本轮复用全部48组既有条件，生成240条固定窗口比较，新增完整网络仿真0次。判读和后续停止条件见[DECISION.md](DECISION.md)。','',
    '## 三层解释与冻结范围','',
    '| 模型 | 使用什么 | 不包含什么 |','| --- | --- | --- |','| 两连接线性模型 M0 | 两条真实直接边的接触数；冻结增益；5ms突触与20ms膜时间常数；静息电位−52mV | 空白轨迹、单独/双输入实测响应、其他网络节点、发放复位 |','| 单细胞模型 M1 | 相同两条直接边与导电LIF；空白目标ge/gi和初始V；固定阈值−45mV、复位−52mV、不应期2.2ms | 刺激引起的循环网络电导变化、任何刺激响应拟合 |','| 网络单独响应叠加 L | 同背景同时刻的四条单独刺激记录 | 这是含完整网络信息的校准参照，不是脱离网络的模型 |','',
    'P13的32组及P13c的16组全部纳入。这是已看过数据上的解释检验，不是新留出测试；相同目标和背景的重复时刻也不是独立生物样本。所有简化预测先按协议和空白记录生成、保存，再读取刺激响应比较；没有拟合参数或选择最佳窗口。M1保留完整网络产生的空白背景信息，不能据此断言完整网络无用。','',
    '每个源发出两次相同事件，包内10ms、两位置20ms、固定1.8ms投递延迟。实际电导增量为接触数×0.275/52，先写入电导，下一时步才影响膜电位。直接投递强度与实际逐边日志一致。M0是静息状态的线性化，未假定它可以重现有背景状态的绝对波形。','',
    '五个窗口均从首刺激实际到达起，结束于末次到达后20/50/100/200ms及记录结束600ms。主窗口为第一项，约50ms；这是到达对齐窗口，与早前P13的[152,202)ms主窗口不同，因此单独叠加符号数也可能不同，不能混用旧口径。短窗口结束前强输入贡献多少，是本轮要检验的竞争解释之一。','',
    '沿用上轮探索性充分性标准：符号一致至少87.5%，波形相对RMS误差中位数≤20%；按P13、P13c以及合并结果分别报告。符号计数不是方向分类准确率，数值符号容差1e-8mV不代表功能显著性。','',
    '## 主窗口：分别报告两批数据','',
    '| 数据 | 模型 | 符号一致 | 波形误差中位数 | 最大误差 | 符号/波形标准 |','| --- | --- | --- | --- | --- | --- |']
    for group in (-1,0,1):
        for model in NAMES:
            a=at(group,0,model);lines.append(f'| {"全部" if group<0 else "P13" if group==0 else "P13c"} | {NAMES[model]} | {a["sign_correct"]}/{a["n"]} | {a["median_relative_RMS_error"]:.2%} | {a["max_relative_RMS_error"]:.2%} | {"通过" if a["sign_adequacy"] else "未通过"} / {"通过" if a["waveform_adequacy"] else "未通过"} |')
    lines+=['','M1合并误差虽低于20%，P13c子组为21.55%，未达冻结的波形标准，不能只报合并通过。M0符号较准但波形幅度不足；L的误差较小也不能被解释成一个完全不需要网络的模型。','',
    '## 配平、交换与有限窗口','',
    '两路强度配平为各自均值，或互换强度，均保留总强度。简化模型内，配平后AB与BA轨迹逐点相同，交换强度后两条轨迹互换，差异精确反号。这是这些模型的结构性质与实现检查，不是完整网络已经实施的因果干预；真实图的边和输入没有改动。','',
    'M0可写成D=(wA−wB)×(h早−h晚)。每个输入具有相同滤波形状，短窗口对较早出现的输入积分更多，所以输入强度不对称即可产生稳定的正负差异。积分覆盖无限长响应时两个平移核面积相等，D积分为零；有限600ms记录也只剩很小残差。','',
    f'M0记录末尾累计面积的绝对值/主窗口面积最大为{result["full_to_early_integral_ratio"]["linear"]["max"]:.3g}，中位数{result["full_to_early_integral_ratio"]["linear"]["median"]:.3g}。M1相同比值中位数{result["full_to_early_integral_ratio"]["local"]["median"]:.3f}、最大{result["full_to_early_integral_ratio"]["local"]["max"]:.3f}，发放复位及随时间变化的背景下不要求面积相消。完整网络的较晚响应也不能由M0面积相消直接推出。','',
    '| 结束位置 | M0符号一致 / 误差中位数 | M1符号一致 / 误差中位数 | L符号一致 / 误差中位数 |','| --- | --- | --- | --- |']
    for wi,name in enumerate(('末次到达后20ms','50ms','100ms','200ms','记录结束600ms')):
        entries=[at(-1,wi,k) for k in NAMES];lines.append(f'| {name} | '+' | '.join(f'{a["sign_correct"]}/48 / {a["median_relative_RMS_error"]:.2%}' for a in entries)+' |')
    lines+=['','M0到记录结束的平均差异已落到数值符号容差内，表中的0/48表示没有非零符号预测，不是“分类准确率0%”。所有窗口保留，未用较好的窗口替换主窗口。','',
    '## 全部48组早期结果','',
    '| 数据/背景 | 目标 | 实际D均值mV | M0均值mV | M1均值mV | M0/M1相对误差 | 实际AB/BA脉冲 | M1脉冲 |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in primary:
        a=r['models']['linear'];b=r['models']['local'];lines.append(f'| {"P13" if r["source_index"]==0 else "P13c"}/{r["background"]} | {r["subtype"]} | {r["actual_mean_mv"]:.6f} | {a["mean_mv"]:.6f} | {b["mean_mv"]:.6f} | {a["relative_RMS_error"]:.4f} / {b["relative_RMS_error"]:.4f} | {r["actual_spikes_AB_BA"]} | {b["spikes_AB_BA"]} |')
    lines+=['',f'M1目标AB/BA计数与实际相同{result["local_target_spike_pairs_exact"]}/48组；实际两顺序任一发放的组合有{result["actual_spiking_cases"]}组，M1有{result["local_spiking_cases"]}组，需结合大量零发放解读，不能以计数匹配率代表方向能力。','',
    '## 验证与复算','',
    f'5项模型测试和3项语义错误注入检查通过。独立验证器用解析双指数公式重建M0，用34位十进制积分和另一种电导生成方法重建M1，并重算240条全部指标；最大预测误差分别为{v["max_linear_reconstruction_error_mv"]:.3g}和{v["max_local_reconstruction_error_mv"]:.3g}mV。全部48个空白重放、真实边强度及父证据哈希均通过。错误注入覆盖指标、强度元数据、预测波形。','',
    '```sh','export MPLCONFIGDIR=/private/tmp/flywire-vision-mpl','export PYTHONPATH=brian2-rust/python:brian2-rust/experiments:.','export OPENBLAS_NUM_THREADS=1',f'/private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.verify_minimal_response {root}',f'/private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.minimal_response_report {root}','```','',
    '冻结规则见`protocol.json`；预测数组、来源和封存记录见`predictions/`、`prediction-rows.json`、`prediction-seal.json`；完整指标见`independent-metrics.json`。原始刺激响应按哈希引用P13/P13c冻结归档，未复制或修改。无新增完整图仿真、分类器调参或可视化开发。']
    (root/'README.md').write_text('\n'.join(lines)+'\n');print({'primary':[at(-1,0,k) for k in NAMES],'integral_ratio':result['full_to_early_integral_ratio']},flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);build(a.parse_args().root)
