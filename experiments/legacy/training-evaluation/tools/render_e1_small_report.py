#!/usr/bin/env python3
"""Render audited E1-small evidence without running any model or remote command.

Input is the standalone validator's full-mode JSON. Invalid or report-only
evidence produces an issues-only report and no ranking or performance figure.
Output is a new directory: no evidence or existing report is overwritten.
Matplotlib is imported only for a valid, explicitly requested render.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

SEEDS = (11, 23, 37, 51, 71)
VIEWS = ('atlas', 'sj-layerwise', 'snn-layerwise', 'sj-compile', 'snn-compile', 'spyx', 'brainstate')
STRICT = VIEWS[:5]
JAX = VIEWS[5:]
LABELS = {'atlas': 'Atlas', 'sj-layerwise': 'SpikingJelly layerwise',
          'snn-layerwise': 'snnTorch layerwise', 'sj-compile': 'SpikingJelly compile',
          'snn-compile': 'snnTorch compile', 'spyx': 'Spyx / JAX', 'brainstate': 'Brainstate / JAX'}
SCOPE = 'E1-small：128→128→10，全局 B16、T128、CPU FP64、完整 SG-BPTT、Adam 更新'
BOUNDARIES = [
    '每个独立进程先执行 10 次 warmup，再执行 50 次测量；每个视图使用 5 个预声明 seed（11、23、37、51、71）。先求每个进程的 50 步中位数，再报告这 5 个中位数的中位数、最小值和最大值。250 个更新不是 250 个独立实验。',
    'Atlas 与 4 个 Torch 视图属于 requested1thread 组。该名称表示请求了 1 个计算线程；macOS 没有设置同核 affinity，不能宣称固定同一物理核。',
    'Spyx 与 Brainstate 的 JAX worker pool 未被限制或证明为单线程，不满足 r2 首轮 1 线程资源门槛；单列为 thread-budget-unqualified host-config diagnostics。数值资格与资源资格分开记录，禁止 JAX 的严格性能比值。',
    '计时覆盖各自实际完整训练调用。Atlas 包含公共 execute 的 JSON、临时文件、native 子进程和返回解析；Torch 包含 forward/backward/Adam，JAX 使用同步后的实际更新路径。输入、模型和优化器准备的边界须结合冻结 runner 判断，不能把 API 延迟解释为整个作业耗时。',
    'RSS 是 50 ms 采样的整个作业进程树峰值，包含 import、oracle、JIT、资格检查及测量；采样可能漏过短峰值，共享页可能重复计入。它不是纯训练 RSS，也不能据此得出最大可训练规模。',
    '冷启动字段边界不同：Atlas 分开记录 constructor 与首次公共 API；竞品记录 oracle 之后 setup 加首次实际更新。二者都不覆盖完整进程冷启动，不能将这些字段直接排序；本报告也不证明缓存全冷。',
    '每 seed 的作业预算、退出原因和未完成步骤均保留。Atlas 在 E1-small 完整形状上校验第 1 次 Adam 更新，另有独立小 Q0 的多步校验；Torch/JAX 在完整形状的实际路径上校验 3 次 Adam 更新。运行后的 60 步 CE 轨迹审计单独列出，不能补全所有中间状态、梯度或参数的数值资格；证据验证器不会重新运行 oracle。',
    '这里只评价一个 E1-small 工作负载点。E1-large、其他模型/任务、A1/A2、其他线程预算及硬件需要独立证据；任何局部最低延迟都不构成全局冠军或整体引擎结论。',
]


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def near(a, b):
    return finite_positive(a) and finite_positive(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(evidence):
    """Cross-check reported aggregates without silently repairing their input."""
    issues = list(evidence.get('errors', []))
    if evidence.get('status') != 'passed_full_evidence_checks':
        issues.append('仅接受 passed_full_evidence_checks；当前 status=' + str(evidence.get('status')))
    if evidence.get('mode') != 'full' or evidence.get('raw_verified') is not True:
        issues.append('必须有 full 模式及 raw_verified=true；report-only 不能生成已验证排名。')
    denominator = evidence.get('denominator', {})
    for key, expected in [('expected_slots', 35), ('expected_warmup', 10), ('expected_measured', 50)]:
        if denominator.get(key) != expected:
            issues.append(f'分母 {key} 应为 {expected}，实际 {denominator.get(key)}。')
    if set(denominator.get('seeds', [])) != set(SEEDS):
        issues.append('正式 seed 集合与预声明的五个 seed 不一致。')
    rows = evidence.get('rows', [])
    by_key = {}
    for row in rows:
        key = (row.get('view'), row.get('seed'))
        if key in by_key:
            issues.append(f'重复 slot：{key}')
        by_key[key] = row
        for error in row.get('errors', []):
            issues.append(f'{key}: {error}')
    expected_keys = {(v, s) for v in VIEWS for s in SEEDS}
    if set(by_key) != expected_keys or len(rows) != 35:
        issues.append('必须完整保留 7 视图 × 5 seed 的 35 个 slot（包括失败/未完成）。')
    aggregates = {}
    for view in VIEWS:
        summary = evidence.get('views', {}).get(view)
        if not isinstance(summary, dict):
            issues.append(f'缺少视图汇总：{view}')
            continue
        selected = [by_key.get((view, seed), {}) for seed in SEEDS]
        complete = [r for r in selected if r.get('effective_status') == 'completed' and not r.get('errors')]
        if summary.get('completed_n') != len(complete) or summary.get('complete_five_seed_summary') is not (len(complete) == 5):
            issues.append(f'{view} 的完成数/五进程汇总标志不一致。')
        for row in complete:
            if not finite_positive(row.get('seed_median_s')):
                issues.append(f'{view}/{row.get("seed")} 缺有效进程中位数。')
            if row.get('numerical_qualification') is not True or row.get('raw_validation') != 'passed':
                issues.append(f'{view}/{row.get("seed")} 完成但数值或 raw 资格未通过。')
            if row.get('completed_measured_steps') != 50 or row.get('completed_warmup_steps') != 10:
                issues.append(f'{view}/{row.get("seed")} 完成但不是 10 warmup + 50 measured。')
        if view in JAX:
            if summary.get('resource_qualification') is not False or summary.get('strict_ranking_eligible') is not False:
                issues.append(f'{view} 必须 resource_qualification=false 且不进入严格排序。')
            if summary.get('ratios_validated') is not False:
                issues.append(f'{view} 不允许已验证的 strict 比值。')
            for row in selected:
                if row.get('resource_qualification') is not False or row.get('strict_ranking_eligible') is not False:
                    issues.append(f'{view}/{row.get("seed")} 错误进入严格资源组。')
        pairs = summary.get('paired_seed_ratios', [])
        pair_map = {p.get('seed'): p for p in pairs}
        if len(pairs) != 5 or set(pair_map) != set(SEEDS):
            issues.append(f'{view} 必须保留全部五个成对 ratio slot。')
        for seed in SEEDS:
            pair = pair_map.get(seed, {})
            row, atlas = by_key.get((view, seed), {}), by_key.get(('atlas', seed), {})
            eligible = (view in STRICT and row.get('effective_status') == 'completed'
                        and atlas.get('effective_status') == 'completed'
                        and not row.get('errors') and not atlas.get('errors')
                        and row.get('numerical_qualification') is True and atlas.get('numerical_qualification') is True
                        and row.get('resource_qualification') is True and atlas.get('resource_qualification') is True
                        and row.get('strict_ranking_eligible') is True and atlas.get('strict_ranking_eligible') is True)
            if pair.get('eligible') is not eligible:
                issues.append(f'{view}/{seed} 成对资格与逐行资格不符。')
            if eligible and finite_positive(row.get('seed_median_s')) and finite_positive(atlas.get('seed_median_s')):
                expected = atlas['seed_median_s'] / row['seed_median_s']
                if not near(pair.get('ratio_atlas_over_view'), expected):
                    issues.append(f'{view}/{seed} 成对 ratio 与进程中位数不符。')
            elif pair.get('ratio_atlas_over_view') is not None:
                issues.append(f'{view}/{seed} 无资格却存在 strict 比值。')
        if len(complete) == 5 and all(finite_positive(r.get('seed_median_s')) for r in complete):
            values = [r['seed_median_s'] for r in selected]
            result = dict(median=statistics.median(values), minimum=min(values), maximum=max(values), values=values)
            for field, key in [('median_of_process_medians_s', 'median'), ('min_process_median_s', 'minimum'), ('max_process_median_s', 'maximum')]:
                if not near(summary.get(field), result[key]):
                    issues.append(f'{view} 的 {field} 不等于五个独立进程中位数的重算值。')
            aggregates[view] = result
        if summary.get('strict_ranking_eligible') is True:
            if view not in STRICT or len(complete) != 5 or summary.get('ratios_validated') is not True:
                issues.append(f'{view} 的严格汇总资格不一致。')
            if any(r.get('strict_ranking_eligible') is not True for r in selected):
                issues.append(f'{view} 汇总进入严格排序，但逐行资格不全。')
    return list(dict.fromkeys(str(issue) for issue in issues)), by_key, aggregates


def fmt(value, digits=6):
    if value is None:
        return '—'
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, float):
        return format(value, f'.{digits}g')
    return str(value)


def flat(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value


def write_csv(path, columns, rows):
    with path.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({key: flat(row.get(key)) for key in columns} for row in rows)


def table(headers, rows):
    def clean(value):
        return fmt(value).replace('|', '&#124;').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(clean(v) for v in row) + ' |' for row in rows])


def html_table(headers, rows):
    def cell(value):
        return html.escape(fmt(value))
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + cell(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + cell(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def figures(output, evidence, aggregates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rendered = []
    for filename, candidates, title in [
        ('latency-requested1thread.png', STRICT, 'E1-small | requested 1 compute thread; no core affinity'),
        ('latency-jax-host-observations.png', JAX, 'E1-small | JAX host configuration observations; not strict 1-thread')]:
        names = [v for v in candidates if v in aggregates and
                 (v in JAX or evidence['views'][v].get('strict_ranking_eligible') is True)]
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(8.6, 1.6 + .56 * len(names)), layout='constrained')
        for index, name in enumerate(names):
            row = aggregates[name];values = [1000 * v for v in row['values']]
            ax.plot([row['minimum'] * 1000, row['maximum'] * 1000], [index, index], color='#718096', lw=2, zorder=1)
            ax.scatter(values, [index + (i - 2) * .048 for i in range(5)], color='#246880', marker='o', s=24, alpha=.8, label='5 process medians' if index == 0 else None)
            ax.scatter([row['median'] * 1000], [index], marker='D', s=36, color='#ae4d24', label='Median of 5 process medians' if index == 0 else None)
        ax.set_yticks(range(len(names)), [LABELS[n] for n in names]);ax.invert_yaxis()
        span = [v for n in names for v in aggregates[n]['values']]
        if max(span) / min(span) >= 20:
            ax.set_xscale('log');unit = 'ms per measured training API call (log axis)'
        else:
            ax.set_xlim(left=0);unit = 'ms per measured training API call'
        ax.set_xlabel(unit);ax.set_title(title, fontsize=10);ax.grid(axis='x', alpha=.2)
        ax.legend(fontsize=8, loc='best');ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(output / filename, dpi=150);fig.savefig(output / filename.replace('.png', '.svg'));plt.close(fig)
        rendered.append(filename)
    return rendered


def conclusion(evidence, aggregates):
    eligible = [v for v in STRICT if v in aggregates and evidence['views'][v].get('strict_ranking_eligible') is True]
    if not eligible:
        return '现有证据没有一个满足完整五进程和严格资源资格的视图，不能生成已验证的延迟排序。'
    smallest = min(aggregates[v]['median'] for v in eligible)
    names = [v for v in eligible if aggregates[v]['median'] == smallest]
    complete_scope = '这 5 个 requested1thread 视图' if len(eligible) == 5 else f'这 {len(eligible)} 个通过完整五进程资格的 requested1thread 视图'
    text = f'在{complete_scope}中，五进程中位数最低的观测为 {"、".join(LABELS[v] for v in names)}（{smallest * 1000:.6g} ms/更新）。这只是本 E1-small 点的描述统计，不是显著性结论。'
    if 'atlas' in eligible:
        peers = [v for v in eligible if v != 'atlas']
        lower = sum(aggregates['atlas']['median'] < aggregates[v]['median'] for v in peers)
        higher = sum(aggregates['atlas']['median'] > aggregates[v]['median'] for v in peers)
        equal = len(peers) - lower - higher
        text += f' Atlas 的该统计量相对 {lower} 个已合格 Torch 视图更低、{higher} 个更高、{equal} 个相同；各 seed 的配对情况见下表。'
    return text


def render(source, output):
    raw = source.read_bytes();evidence = json.loads(raw)
    issues, by_key, aggregates = inspect(evidence)
    if not issues:
        import matplotlib  # Fail before creating partial output if unavailable.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'validation-source.json').write_bytes(raw)
    provenance = {'schema': 'e1-small-report-provenance-r1', 'validation_sha256': hashlib.sha256(raw).hexdigest(),
                  'renderer_sha256': digest(__file__), 'validation_input': str(source),
                  'freeze_sha256': evidence.get('freeze_sha256'), 'run_reference': evidence.get('run'),
                  'evidence_status': evidence.get('status'), 'report_status': 'issues_only' if issues else 'verified_full_evidence',
                  'scope': SCOPE, 'issues': issues}
    if not issues:
        provenance['matplotlib_version'] = matplotlib.__version__
    sections = []
    markdown = ['# E1-small 技术报告', '', SCOPE, '']
    def paragraph(text):
        markdown.extend([text, '']);sections.append('<p>' + html.escape(text) + '</p>')
    def section(title):
        markdown.extend(['## ' + title, '']);sections.append('<h2>' + html.escape(title) + '</h2>')
    def add_table(headers, rows):
        markdown.extend([table(headers, rows), '']);sections.append(html_table(headers, rows))
    paragraph('验证文件 SHA256：' + provenance['validation_sha256'])
    paragraph('冻结配置 SHA256：' + str(evidence.get('freeze_sha256')) + '；运行引用：' + str(evidence.get('run')))
    if issues:
        paragraph('未生成已验证排名：输入未通过完整证据门槛或汇总一致性检查。以下问题必须先处理；本输出不包含性能排名或图表。')
        section('问题清单')
        add_table(['编号', '问题'], [[i + 1, issue] for i, issue in enumerate(issues)])
        write_csv(output / 'issues.csv', ['index', 'issue'], [{'index': i + 1, 'issue': issue} for i, issue in enumerate(issues)])
        status_rows = [{'view': row.get('view'), 'seed': row.get('seed'), 'effective_status': row.get('effective_status'),
                        'numerical_qualification': row.get('numerical_qualification'), 'resource_qualification': row.get('resource_qualification'),
                        'verified_ranking_eligible': False, 'errors': row.get('errors')} for row in evidence.get('rows', [])]
        write_csv(output / 'status-unverified.csv', ['view', 'seed', 'effective_status', 'numerical_qualification', 'resource_qualification', 'verified_ranking_eligible', 'errors'], status_rows)
        plots = []
    else:
        paragraph(conclusion(evidence, aggregates))
        rows = [by_key[(view, seed)] for view in VIEWS for seed in SEEDS]
        seed_columns = ['view','seed','effective_status','numerical_qualification','resource_qualification','strict_ranking_eligible',
                        'completed_warmup_steps','completed_measured_steps','seed_median_s','step_p10_s','step_p90_s',
                        'budget_s','worker_wall_s','supervisor_wall_s','exit_code','termination_reason','reason','array_sha256','raw_validation']
        write_csv(output / 'seed-results.csv', seed_columns, rows)
        summaries = []
        pairs = []
        resources = []
        for view in VIEWS:
            item = evidence['views'][view]
            summaries.append(dict(view=view, **{key: item.get(key) for key in ['scheduled_n','completed_n','independent_n','complete_five_seed_summary',
                'numerically_qualified_n','resource_qualification','strict_ranking_eligible','median_of_process_medians_s','min_process_median_s',
                'max_process_median_s','paired_complete_n','ratios_validated','timing_class']}))
            for p in item['paired_seed_ratios']:
                seed = p['seed']
                pairs.append(dict(view=view,seed=seed,eligible=p['eligible'],ratio_atlas_over_view=p['ratio_atlas_over_view'],
                                  atlas_process_median_s=by_key[('atlas',seed)].get('seed_median_s'),
                                  view_process_median_s=by_key[(view,seed)].get('seed_median_s'),
                                  reason='JAX strict ratios prohibited' if view in JAX else 'eligible' if p['eligible'] else 'incomplete/unqualified pair'))
        for row in rows:
            cold = row.get('cold', {})
            resources.append(dict(view=row['view'],seed=row['seed'],sampled_job_peak_rss_bytes=row.get('sampled_job_peak_rss_bytes'),rss_scope=row.get('rss_scope'),
                atlas_first_constructor_s=cold.get('first_constructor_s'),atlas_first_public_api_s=cold.get('first_public_api_s'),
                competitor_setup_and_first_update_s=cold.get('setup_and_first_update_s'),cold_scope=cold.get('scope'),
                root_thread_count_max=row.get('root_thread_count_max'),resource_samples=row.get('resource_samples'),
                worker_wall_s=row.get('worker_wall_s'),supervisor_wall_s=row.get('supervisor_wall_s'),budget_s=row.get('budget_s')))
        write_csv(output / 'summary.csv', list(summaries[0]), summaries)
        write_csv(output / 'paired-ratios.csv', list(pairs[0]), pairs)
        write_csv(output / 'cold-and-job-rss.csv', list(resources[0]), resources)
        section('请求单线程组：五个独立进程')
        headers = ['视图', '完成/计划', '五进程中位数 ms', '最小 ms', '最大 ms', '严格资格']
        def summary_row(view):
            item = evidence['views'][view]
            return [LABELS[view], f'{item["completed_n"]}/5', *[None if item.get(k) is None else 1000 * item[k] for k in
                ['median_of_process_medians_s','min_process_median_s','max_process_median_s']], item['strict_ranking_eligible']]
        add_table(headers, [summary_row(view) for view in STRICT])
        paragraph('表格使用固定视图顺序。min/max 是五个进程中位数的范围；不是单步范围或置信区间。缺任一正式 seed 的视图不生成完整五进程统计。')
        section('每个 seed 的进程中位数与成对比值')
        paragraph('ratio = Atlas 同 seed 进程中位数 ÷ 该视图同 seed 进程中位数。小于 1 表示该配对中 Atlas 延迟较低，大于 1 表示较高。保留全部五个比值，不把“中位数之比”替代为“比值中位数”。')
        seed_headers = ['视图', *[str(s) + ' ms' for s in SEEDS]]
        add_table(seed_headers, [[LABELS[v], *[None if by_key[(v,s)].get('seed_median_s') is None else by_key[(v,s)]['seed_median_s'] * 1000 for s in SEEDS]] for v in STRICT])
        pair_by_key = {(p['view'],p['seed']):p for p in pairs}
        add_table(['视图', *['seed ' + str(s) for s in SEEDS]], [[LABELS[v], *[pair_by_key[(v,s)]['ratio_atlas_over_view'] for s in SEEDS]] for v in STRICT])
        section('JAX 主机配置观测：未通过单线程资源门槛')
        paragraph('以下数据可描述该主机配置下的实际观测。它们不进入严格 requested1thread 排序，也没有 Atlas/JAX 严格比值。')
        add_table(headers, [summary_row(view) for view in JAX])
        add_table(seed_headers, [[LABELS[v], *[None if by_key[(v,s)].get('seed_median_s') is None else by_key[(v,s)]['seed_median_s'] * 1000 for s in SEEDS]] for v in JAX])
        section('作业 RSS 与不同边界的冷启动记录')
        paragraph('RSS 单位 MiB。三类冷启动时间分列，单位秒；“—”表示不适用或没有记录。这里没有跨边界的冷启动排名。')
        add_table(['视图','seed','作业采样峰 RSS MiB','Atlas constructor s','Atlas 首 API s','竞品 setup+首更新 s'],
                  [[LABELS[r['view']],r['seed'],None if r['sampled_job_peak_rss_bytes'] is None else r['sampled_job_peak_rss_bytes']/1024**2,
                    r['atlas_first_constructor_s'],r['atlas_first_public_api_s'],r['competitor_setup_and_first_update_s']] for r in resources])
        section('所有预声明 slot 的状态与预算')
        add_table(['视图','seed','状态','数值资格','资源资格','warmup/测量','预算 s','supervisor s','退出/原因'],
                  [[LABELS[r['view']],r['seed'],r.get('effective_status'),r.get('numerical_qualification'),r.get('resource_qualification'),
                    f'{r.get("completed_warmup_steps")}/{r.get("completed_measured_steps")}',r.get('budget_s'),r.get('supervisor_wall_s'),
                    f'{r.get("exit_code")} / {r.get("termination_reason") or r.get("reason") or "—"}'] for r in rows])
        section('运行后 60 步 CE 轨迹审计')
        trajectory = evidence.get('paired_loss_trajectory_audit', [])
        paragraph('此项是运行后补充审计；逐点容差为 1e-10 + 1e-8×|reference|。Atlas 的预声明资格是完整形状第 1 次 Adam 加独立小 Q0 多步校验，Torch/JAX 是完整形状实际路径 3 次 Adam。60 步 CE 一致不能替代所有中间状态、梯度或参数的资格验证，也不增加独立进程样本数或改变 JAX 的资源资格。')
        add_table(['视图','seed','比较步数','状态','不一致步骤'], [[LABELS.get(r.get('view'),r.get('view')),r.get('seed'),r.get('compared_steps'),r.get('status'),r.get('mismatching_step_indices')] for r in trajectory])
        write_csv(output / 'postrun-loss-audit.csv', ['view','seed','compared_steps','status','mismatching_step_indices','scope'], trajectory)
        plots = figures(output, evidence, aggregates)
        section('静态图')
        for filename in plots:
            markdown.extend([f'![{filename}]({filename})', ''])
            image_data = base64.b64encode((output / filename).read_bytes()).decode('ascii')
            sections.append('<figure><img alt="' + html.escape(filename) + '" src="data:image/png;base64,' + image_data + '"><figcaption>5 个进程中位数、其范围与中位数；固定视图顺序。</figcaption></figure>')
    section('解释边界')
    for boundary in BOUNDARIES:
        paragraph(boundary)
    for limitation in evidence.get('limitations', []):
        paragraph('验证器限制：' + str(limitation))
    section('可复核文件')
    paragraph('validation-source.json 保留输入验证文件完整字节；provenance.json 记录验证输入、冻结配置和 renderer 的 SHA256。CSV 保留数值原始精度；页面显示值为便于阅读而格式化。HTML 内嵌 PNG 和完整验证 JSON，不需要网络。')
    details = html.escape(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2))
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E1-small 技术报告</title><style>' + \
        'body{font-family:system-ui,sans-serif;line-height:1.65;color:#192d38;max-width:1180px;margin:36px auto;padding:0 24px}h1{font-size:30px}h2{margin-top:38px;font-size:22px}p{max-width:1080px;overflow-wrap:anywhere}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{border-bottom:1px solid #dbe3e7;padding:8px 10px;text-align:left;white-space:nowrap}th{background:#eef4f6}figure{margin:24px 0}img{max-width:100%;height:auto}figcaption{font-size:12px;color:#576775}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}details{margin:28px 0}' + \
        '</style><h1>E1-small 技术报告</h1><p>' + html.escape(SCOPE) + '</p>' + ''.join(sections) + \
        '<details><summary>完整验证 JSON</summary><pre>' + details + '</pre></details></html>'
    (output / 'report.html').write_text(body, encoding='utf-8')
    (output / 'report.zh.md').write_text('\n'.join(markdown), encoding='utf-8')
    provenance['figures'] = plots
    provenance['artifacts'] = {p.name: {'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='new directory; never overwrite')
    args = parser.parse_args()
    source = args.validation or args.root / 'evidence/e1-small-validation-full-r1.json'
    report = render(source.resolve(), args.output.resolve())
    print(json.dumps({'status':report['report_status'],'output':str(args.output.resolve()),'issues':report['issues']}, ensure_ascii=False))
    return 2 if report['issues'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
