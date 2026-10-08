"""Audit published MAM statistics without treating them as rerun acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path

REVISION = '11fa93a4427a0e4e4de307ca7a5455e80265053a'
NORMALIZATION_SHA = 'b9dda7098372ed14a94c8a3c4483657cb92d66272c532926e8e78787377f21bc'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('metric must be finite numeric data')
    return float(value)


def adapt(case, rates, lvr, corr, params, normalization):
    """Keep absent anatomical slots and unused statistic placeholders distinct."""
    expected = {'metastable100': (100500., 1.9, 2.), 'ground10': (10500., 1., 1.)}[case]
    if tuple(params[k] for k in ['T', 'cc_weights_factor', 'cc_weights_I_factor']) != expected:
        raise ValueError('published condition differs from declared case')
    window = rates['Parameters']
    if (window['t_min'], window['t_max'], window['compute_stat']) != (500., expected[0], False):
        raise ValueError('unexpected published rate window')
    groups = {}
    for p in normalization['populations']:
        area, pop = p['name'].removeprefix('mam_').rsplit('_', 1)
        if (area, pop) in groups:
            raise ValueError('duplicate population')
        groups[area, pop] = p
    areas = sorted({a for a, p in groups})
    if len(groups) != 254 or len(areas) != 32 or set(window['areas']) != set(areas):
        raise ValueError('requires complete 32-area / 254-population universe')
    for metric in [rates, lvr, corr]:
        if set(metric) - {'Parameters'} != set(areas):
            raise ValueError('incomplete metric area universe')
        metadata = metric.get('Parameters')
        if metadata is not None and (metadata['t_min'], metadata['t_max']) != (500., expected[0]):
            raise ValueError('conflicting metric window')
        for a in areas:
            real = {p for aa, p in groups if aa == a}
            extra = set(metric[a]) - real - {'total'}
            if extra - ({'4E', '4I'} if a == 'TH' else set()):
                raise ValueError('unknown population slots')
    rows = []
    placeholders = []
    for (area, pop), p in sorted(groups.items()):
        pair = rates[area][pop]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('expected historical [rate, unused-statistic] pair')
        rate = finite(pair[0])
        if rate < 0:
            raise ValueError('negative firing rate')
        unused = pair[1]
        if not isinstance(unused, (int, float)) or isinstance(unused, bool) or (not math.isnan(unused) and unused != 0.):
            raise ValueError('unexpected unused statistic')
        rows.append(dict(name=p['name'], area=area, population=pop,
            rate_hz=rate, unused_rate_statistic=None if math.isnan(unused) else unused,
            lvr=finite(lvr[area][pop]), correlation=finite(corr[area][pop]),
            modern_normalization_neurons=finite(p['official_normalization_neurons'])))
    for metric_name, metric in [('rate', rates), ('lvr', lvr), ('correlation', corr)]:
        for pop in ['4E', '4I']:
            if pop in metric['TH']:
                value = metric['TH'][pop]
                if value != ([0., 0.] if metric_name == 'rate' else 0.):
                    raise ValueError('nonzero value for absent TH layer 4')
                placeholders.append(dict(metric=metric_name, name='mam_TH_'+pop, value=value))
    area_rows = []
    for area in areas:
        selected = [r for r in rows if r['area'] == area]
        n = math.fsum(r['modern_normalization_neurons'] for r in selected)
        derived = math.fsum(r['rate_hz']*r['modern_normalization_neurons'] for r in selected)/n
        published = finite(rates[area]['total'])
        area_rows.append(dict(area=area, published_rate_hz=published,
            derived_rate_hz_using_modern_N=derived, delta_hz=derived-published,
            modern_normalization_neurons=n))
    total_n = math.fsum(r['modern_normalization_neurons'] for r in rows)
    weighted = math.fsum(r['rate_hz']*r['modern_normalization_neurons'] for r in rows)/total_n
    return dict(case=case, parameters=params, window_from_rates=window,
        metadata_by_metric={k: v.get('Parameters') for k, v in [('rates', rates), ('lvr', lvr), ('correlation', corr)]},
        populations=rows, areas=area_rows, excluded_absent_anatomy_placeholders=placeholders,
        summary=dict(population_count=len(rows), area_count=len(areas),
            neuron_weighted_hz_using_modern_N=weighted,
            rounded_to_one_decimal_hz=round(weighted, 1),
            unweighted_published_area_mean_hz=math.fsum(r['published_rate_hz'] for r in area_rows)/len(areas),
            unweighted_real_population_mean_hz=math.fsum(r['rate_hz'] for r in rows)/len(rows),
            maximum_area_consistency_error_hz=max(abs(r['delta_hz']) for r in area_rows),
            areas_with_consistency_error_above_1e_minus_10=[r['area'] for r in area_rows if abs(r['delta_hz']) > 1e-10]))


def audit(source, normalization, output):
    catalog = json.loads((source/'catalog.json').read_text())
    if catalog['revision'] != REVISION or digest(normalization) != NORMALIZATION_SHA:
        raise ValueError('unrecognized reference or normalization identity')
    for name, info in catalog['files'].items():
        if Path(name).name != name:
            raise ValueError('unsafe source catalog member')
        path = source/name
        if path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
            raise ValueError('source catalog integrity failure')
    norm = json.loads(normalization.read_text())
    def read(name):
        if name not in catalog['files']:
            raise ValueError('uncatalogued source')
        return json.loads((source/name).read_text())
    cases = {}
    for case in ['metastable100', 'ground10']:
        cases[case] = adapt(case, *(read(case+'-'+metric+'.json') for metric in ['pop_rates', 'pop_LvR', 'corrcoeff', 'params']), norm)
    result = dict(schema=1, source_revision=REVISION, source_catalog_sha256=digest(source/'catalog.json'),
        normalization_sha256=digest(normalization), cases=cases,
        reference_fetch_failures=catalog['failures'], scientific_acceptance=False,
        scope='Published processed statistics, with explicit historical pair-format adapter. Modern N weights provide a numerical consistency check, not proof of historical code, sampling, or raw spike reconstruction. Absent metric metadata remains null. No replicate uncertainty or current-simulator equivalence is inferred.')
    with output.open('x') as f:
        f.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'normalization', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.normalization, args.output)
    print(json.dumps({k: v['summary'] for k, v in result['cases'].items()}, indent=2))
