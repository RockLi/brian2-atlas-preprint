"""Audit nonzero model paths and frozen-background stimulus response counts."""
import argparse
import json
from pathlib import Path
import numpy as np
from .simulation import READOUT_TYPES
from .run_experiment import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from flywire_mnist.graph import load_graph
    from brian2_rust.binary_topology import inspect_csr, csr_arrays
    from brian2_rust.results import load_results
    from .pilot import event_features
    args.output.mkdir(parents=True, exist_ok=False)
    graph = load_graph(args.graph)
    data = json.loads((args.artifact / 'viewer-data.json').read_text())
    if graph.identity != data['manifest']['graph_identity']:
        raise ValueError('graph differs from recorded experiment')
    groups = json.loads((args.artifact / 'groups.json').read_text())
    offsets, target, values = csr_arrays(inspect_csr(graph.csr))
    n = len(graph.root_ids)
    group_lookup = np.full(n, -1, dtype=int)
    for j, name in enumerate(READOUT_TYPES):
        group_lookup[groups[name]] = j
    connectivity = {}
    for input_type in ('Mi1', 'Tm1'):
        positive, negative = np.zeros(9), np.zeros(9)
        pairs = np.zeros(9, dtype=int)
        input_signs = {'positive_only': 0, 'negative_only': 0, 'zero_only': 0, 'mixed': 0}
        reached = np.zeros(n, dtype=bool)
        reached[groups[input_type]] = True
        frontier = np.array(groups[input_type])
        hops = []
        for hop in (1, 2):
            following = np.zeros(n, dtype=bool)
            for node in frontier:
                a, b = int(offsets[node]), int(offsets[node + 1])
                weight, targets = values[0, a:b], target[a:b]
                following[targets[weight != 0]] = True
                if hop == 1:
                    pos, neg = np.any(weight > 0), np.any(weight < 0)
                    input_signs['mixed' if pos and neg else 'positive_only' if pos else 'negative_only' if neg else 'zero_only'] += 1
                    slots = group_lookup[targets]
                    mask = (slots >= 0) & (weight != 0)
                    np.add.at(pairs, slots[mask], 1)
                    np.add.at(positive, slots[mask], np.maximum(weight[mask], 0))
                    np.add.at(negative, slots[mask], np.maximum(-weight[mask], 0))
            frontier = np.flatnonzero(following & ~reached)
            reached |= following
            hops.append({name: int(reached[ids].sum()) for name, ids in groups.items() if name in READOUT_TYPES})
        connectivity[input_type] = {'input_neuron_output_signs': input_signs,
                                    'readouts_reachable_within_one_edge': hops[0],
                                    'readouts_reachable_within_two_edges': hops[1],
                                    'direct_edges': {name: {'pairs': int(pairs[j]), 'positive_contacts': float(positive[j]),
                                                            'negative_contacts_abs': float(negative[j])}
                                                     for j, name in enumerate(READOUT_TYPES)}}
    model = json.loads((args.artifact / 'model.json').read_text())
    ni = next(i for i, p in enumerate(model['definition']['populations']) if p['name'] == 'flywire_neurons')
    trials = {t['kind']: t for t in data['trials'] if t['condition'] == 'intact'}
    cells = np.concatenate([groups[k] for k in ('Mi1', 'Tm1', *READOUT_TYPES)])
    summaries, reference = {}, None
    for kind in ('blank', 'right', 'left', 'up', 'down', 'looming'):
        trial = trials[kind]
        pop = load_results(model, args.artifact / 'intact' / trial['id'])['populations'][ni]
        counts = event_features(pop['indices'], pop['spike_ticks'], cells, n).reshape(8, len(cells))
        if kind == 'blank':
            reference = counts.copy()
        cursor, summary = 0, {}
        for name in ('Mi1', 'Tm1', *READOUT_TYPES):
            k = len(groups[name])
            active = counts[:, cursor:cursor + k]
            blank = reference[:, cursor:cursor + k]
            summary[name] = {'cells': k, 'stimulus_window_spikes': int(active.sum()),
                             'active_cells': int(np.count_nonzero(active.sum(0))),
                             'cells_with_changed_50ms_counts_vs_blank': int(np.any(active != blank, axis=0).sum()),
                             'sum_abs_50ms_count_difference_vs_blank': int(np.abs(active - blank).sum())}
            cursor += k
        summaries[kind] = summary
    save(args.output / 'report.json', {'graph_identity': graph.identity, 'connectivity': connectivity,
                                     'recorded_response': summaries,
                                     'scope': 'nonzero signed model-edge reachability and exact spike counts; not a functional or physiological path validation'})
    print(json.dumps({'output': str(args.output), 'input_signs': {k: v['input_neuron_output_signs'] for k, v in connectivity.items()},
                      'LC4_two_hop_reachability': {k: v['readouts_reachable_within_two_edges']['LC4'] for k, v in connectivity.items()}}))


if __name__ == '__main__':
    main()
