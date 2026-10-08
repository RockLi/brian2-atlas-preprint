"""Audit all ten matched-input warmup runs and report the frozen descriptive metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from study import compare

SEEDS = [11, 23, 47, 83, 131]
ARMS = [100, 10100]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def markdown(result, output, study):
    cases = result['cases']
    effects = result['primary_effects']
    lines = ['# 冻结网络预热对照：100 ms 与 10.1 s', '',
             '10/10 全脑运行完成；360 次试验读出独立重算通过。两组均无学习、无教学、无奖励，全部 gain 保持 1。', '',
             '长预热组在原输入前增加 10 s 背景活动；之后的背景和刺激事件逐项匹配。模型、初始状态、刺激顺序和参数保持相同。', '',
             '## 预设主要指标', '',
             '每种刺激分别取末次与首次测试放电率差的绝对值，再对 A/B 平均，单位 Hz。越小表示这两个测试时点的响应变化越小；它不衡量分类准确率，也不保证中间过程稳定。', '',
             '| seed | 100 ms 漂移 | 10.1 s 漂移 | 长减短 |',
             '| ---: | ---: | ---: | ---: |']
    for row in effects:
        lines.append(f"| {row['seed']} | {row['short_drift_hz']:.2f} | {row['long_drift_hz']:.2f} | {row['long_minus_short_hz']:+.2f} |")
    for key, label in [('short_drift_hz', '100 ms'), ('long_drift_hz', '10.1 s'), ('long_minus_short_hz', '长减短')]:
        values = [r[key] for r in effects]
        lines += ['', f'{label}：均值 {np.mean(values):.2f} Hz，范围 [{min(values):.2f}, {max(values):.2f}] Hz。']
    lines += ['', '## 全部原始测试读出', '', '表内为 A/B Hz，每种刺激每个测试块仅 2 次试验。', '',
              '| seed | 预热 ms | 早期 A/B | 中期 A/B | 晚期 A/B |',
              '| ---: | ---: | ---: | ---: | ---: |']
    for c in cases:
        cells = [f"{c['probes'][b]['A_hz']:.1f}/{c['probes'][b]['B_hz']:.1f}" for b in ['pre', 'post', 'final']]
        lines.append(f"| {c['seed']} | {c['warmup_ms']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## 五个种子均值：原始及基线校正读出', '',
              '基线为刺激前 50 ms；负的校正值表示刺激窗口放电率低于该短基线估计，不是负放电。', '',
              '| 预热 ms | 测试时点 | 原始 A/B Hz | 校正 A/B Hz | 原始 B−A Hz | 校正 B−A Hz | 双刺激静默 seeds |',
              '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        for block in ['pre', 'post', 'final']:
            group = [c['probes'][block] for c in cases if c['warmup_ms'] == arm]
            mean = lambda key: float(np.mean([r[key] for r in group]))
            lines.append(f"| {arm} | {block} | {mean('A_hz'):.2f}/{mean('B_hz'):.2f} | {mean('A_evoked_hz'):.2f}/{mean('B_evoked_hz'):.2f} | {mean('contrast_hz'):.2f} | {mean('evoked_contrast_hz'):.2f} | {sum(r['both_silent'] for r in group)}/5 |")
    lines += ['', '## 验证、资源及复现', '',
              '五个短预热组均与原 CPU v1 frozen 条件的完整 snapshot 逐项精确相同。十组的输入匹配、零学习/教学、全图大小、gain=1、有限状态、完整计数和模拟时长均已核对。', '',
              '每组用 Rust CPU 4 线程连续运行，短组模拟 19.1 s，长组模拟 29.1 s。', '']
    for arm in ARMS:
        group = [c for c in cases if c['warmup_ms'] == arm]
        wall = [c['end_to_end_seconds'] for c in group]
        native = [c['native']['native_wall_seconds'] for c in group]
        rss = max(c['native']['native_peak_rss_bytes'] for c in group) / 1024 ** 2
        lines.append(f'{arm} ms 组：原生墙钟 {min(native):.2f}–{max(native):.2f} s；端到端 {min(wall):.2f}–{max(wall):.2f} s；最大原生 RSS {rss:.1f} MiB。')
        lines.append('')
    lines += ['共享主机耗时受其他负载影响；长短组模拟时间不同，不作性能优劣结论。原生与前端峰值不相加；前端峰值保存在 results.json。', '',
              f'本机完整原始产物：`{study}`。本目录 results.json 保存全部试验、指标、资源及逐文件散列；progress.json 保存十个作业的命令、状态及退出码。', '',
              '```sh',
              'python brian2-rust/experiments/flywire_learning/warmup_study.py --prepared /path/to/learning-prepared --baseline /path/to/learning-study-compact --output /path/to/new-warmup-study',
              'python brian2-rust/experiments/flywire_learning/warmup_report.py --study /path/to/new-warmup-study --baseline /path/to/learning-study-compact --output /path/to/new-warmup-report',
              'python brian2-rust/experiments/flywire_learning/plot_warmup.py --report /path/to/new-warmup-report',
              '```', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))


def main(study, baseline, output):
    registry = json.loads((study / 'progress.json').read_text())
    assert len(registry) == 10 and all(r['status'] == 'complete' and r['exit_code'] == 0
                                     for r in registry.values()), 'study not complete'
    rows, cases, identities, short_equivalence = [], [], [], {}
    for seed in SEEDS:
        original = baseline / f'seed-{seed}-frozen'
        with np.load(original / 'input.npz') as f:
            original_inputs = {k: f[k] for k in f.files}
        old_cfg = json.loads((original / 'configuration.json').read_text())
        for warmup in ARMS:
            folder = study / f'seed-{seed}-warmup-{warmup}'
            cfg = json.loads((folder / 'configuration.json').read_text())
            report = json.loads((folder / 'report.json').read_text())
            identity = json.loads((folder / 'warmup_identity.json').read_text())
            shift = (warmup - 100) * 10
            total_ticks = 191000 + shift
            assert cfg['neurons'] == 139255 and cfg['static_edges'] == 15089423
            assert cfg['plastic_edges'] == 2560 and cfg['threads'] == 4
            assert cfg['scope'] == 'full' and cfg['condition'] == 'frozen'
            assert cfg['duration_ms'] == total_ticks / 10 and cfg['amplitude_mv'] == 40
            assert identity['wrapper_sha256'] == sha(Path(__file__).with_name('warmup_study.py'))
            assert identity['protocol_sha256'] == sha(Path(__file__).with_name('WARMUP_PROTOCOL.md'))
            assert identity['baseline_input_sha256'] == sha(original / 'input.npz')
            with np.load(folder / 'input.npz') as data:
                assert not data['learning'].any() and not data['teaching'].any()
                for key in ['mapping', 'stimulus_ids', 'stimulus_targets']:
                    np.testing.assert_array_equal(data[key], original_inputs[key])
                np.testing.assert_array_equal(data['stimulus_ticks'] - shift, original_inputs['stimulus_ticks'])
                matched = data['background_ticks'] >= shift
                np.testing.assert_array_equal(data['background_ticks'][matched] - shift, original_inputs['background_ticks'])
                np.testing.assert_array_equal(data['background_ids'][matched], original_inputs['background_ids'])
                assert len(data['learning']) == total_ticks and len(data['teaching']) == total_ticks
                input_hash = hashlib.sha256()
                for key in sorted(data.files):
                    input_hash.update(key.encode() + data[key].tobytes())
                assert input_hash.hexdigest() == cfg['input_sha256']
            with np.load(folder / 'snapshot.npz') as data:
                for key in ['v', 'ge', 'gi', 'gain', 'eligibility', 'neuron_trace']:
                    assert np.isfinite(data[key]).all(), key
                np.testing.assert_array_equal(data['gain'], np.ones(2560))
                assert np.all(data['ge'] >= 0) and np.all(data['gi'] >= 0)
                assert np.all((data['eligibility'] >= 0) & (data['eligibility'] <= 1))
                spike_i, spike_t = data['spike_i'], data['spike_t']
                tick = np.rint(spike_t / .0001).astype(np.int64)
                np.testing.assert_allclose(tick * .0001, spike_t, rtol=0, atol=1e-13)
                assert np.all(np.diff(tick) >= 0) and tick.min() >= 0 and tick.max() < total_ticks
                np.testing.assert_array_equal(np.bincount(spike_i, minlength=139255), data['counts'])
                trace_t = np.rint(data['trace_t'] / .0001).astype(np.int64)
                np.testing.assert_array_equal(trace_t, np.arange(total_ticks))
                selected = [cfg['recorded_indices'].index(i) for i in cfg['mbon_indices']]
                trace = data['neuron_trace'][:, selected, :]
            assert len(cfg['trials']) == len(report['trials']) == 36
            for trial_id, (trial, old, old_trial) in enumerate(zip(cfg['trials'], report['trials'], old_cfg['trials'], strict=True)):
                assert trial['start_ms'] == old_trial['start_ms'] + (warmup - 100)
                assert trial['cue'] == old_trial['cue'] and trial['block'] == old_trial['block']
                assert not trial['training'] and trial['reward'] == 0
                onset = int(round((trial['start_ms'] + 50) * 10))
                row = dict(seed=seed, warmup_ms=warmup, trial_id=trial_id, **trial)
                for window, begin, end in [('baseline', onset - 500, onset),
                                            ('response', onset, onset + 2500)]:
                    left, right = np.searchsorted(tick, [begin, end])
                    counts = np.bincount(spike_i[left:right], minlength=139255)
                    values = {}
                    for name, group in [('mbon', cfg['mbon_indices']), ('kc', cfg['kc_indices']),
                                        ('input', cfg['stimulus_groups'][trial['cue']])]:
                        rate = float(counts[group].sum() / (len(group) * (end - begin) * .0001))
                        values[name + '_hz'] = rate
                        if window == 'response':
                            np.testing.assert_allclose(rate, old[name + '_hz'], rtol=0, atol=1e-12)
                    v, ge, gi = trace[:, :, begin:end]
                    values.update(v_mean_mv=float(v.mean() * 1000), ge_mean=float(ge.mean()), gi_mean=float(gi.mean()))
                    row[window] = values
                row['evoked_mbon_hz'] = row['response']['mbon_hz'] - row['baseline']['mbon_hz']
                rows.append(row)
            probes = {}
            for block in ['pre', 'post', 'final']:
                group = [r for r in rows if r['seed'] == seed and r['warmup_ms'] == warmup and r['block'] == block]
                raw = [float(np.mean([r['response']['mbon_hz'] for r in group if r['cue'] == cue])) for cue in (0, 1)]
                bg = [float(np.mean([r['baseline']['mbon_hz'] for r in group if r['cue'] == cue])) for cue in (0, 1)]
                evoked = [a - b for a, b in zip(raw, bg)]
                probes[block] = dict(A_hz=raw[0], B_hz=raw[1], contrast_hz=raw[1] - raw[0],
                                     A_baseline_hz=bg[0], B_baseline_hz=bg[1],
                                     A_evoked_hz=evoked[0], B_evoked_hz=evoked[1],
                                     evoked_contrast_hz=evoked[1] - evoked[0],
                                     both_silent=bool(raw[0] == raw[1] == 0))
            drift = float(np.mean([abs(probes['final'][cue] - probes['pre'][cue]) for cue in ['A_hz', 'B_hz']]))
            native = json.loads((folder / 'project/native_metrics.json').read_text())
            timings = json.loads((folder / 'project/rust/summary.json').read_text())['timings']
            assert native['exit_code'] == 0
            cases.append(dict(seed=seed, warmup_ms=warmup, probes=probes, absolute_drift_hz=drift,
                              native=native, timings=timings, end_to_end_seconds=report['end_to_end_seconds'],
                              frontend_peak_rss_bytes=report['frontend_peak_rss_bytes']))
            if warmup == 100:
                short_equivalence[str(seed)] = compare(folder, original, exact=True)
            identities.append(dict(seed=seed, warmup_ms=warmup,
                                   hashes={name: sha(folder / name) for name in ['configuration.json', 'input.npz',
                                           'snapshot.npz', 'report.json', 'warmup_identity.json']}))
            print(f'AUDITED {seed} {warmup}', flush=True)
    effects = []
    for seed in SEEDS:
        short, long = [next(c for c in cases if c['seed'] == seed and c['warmup_ms'] == arm) for arm in ARMS]
        effects.append(dict(seed=seed, short_drift_hz=short['absolute_drift_hz'], long_drift_hz=long['absolute_drift_hz'],
                            long_minus_short_hz=long['absolute_drift_hz'] - short['absolute_drift_hz']))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(completed_cases=10, recounted_trials=len(rows), all_gates_zero=True,
                  all_gains_one=True, all_matched_inputs_exact=True, short_reference_equivalence=short_equivalence,
                  cases=cases, primary_effects=effects, trials=rows, artifact_identities=identities,
                  protocol_sha256=sha(Path(__file__).with_name('WARMUP_PROTOCOL.md')),
                  report_source_sha256=sha(__file__))
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for name in ['progress.json', 'study_identity.json', 'short_reference_equivalence.json']:
        (output / name).write_bytes((study / name).read_bytes())
    markdown(result, output, study)
    print(json.dumps(effects, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.study.resolve(), a.baseline.resolve(), a.output.resolve())
