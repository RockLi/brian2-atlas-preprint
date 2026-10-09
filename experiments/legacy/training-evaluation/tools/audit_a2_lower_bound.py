"""Reproduce A2's JSON lower bound without allocating its input tensor.

Only archived metadata is read. The minimized payload is a format proof, not a
replacement workload or an executed native request. No numerical dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    base = root/'evidence/data-preparation/atlas-a2-admission'
    if not base.exists():
        base = root/'evidence/data-preparation/remote-data-evidence/evidence/data-preparation/atlas-a2-admission'
    original = json.loads((base/'result.json').read_text())
    for filename in ['plan.json', 'optimizer-state.json']:
        assert digest(base/filename) == original['artifact_hashes'][filename]
    plan = json.loads((base/'plan.json').read_text())
    state = json.loads((base/'optimizer-state.json').read_text())
    counts = [len(w) for w in state['weights']]
    assert counts == [179200, 65536, 5120]
    for name in ['weights', 'first_moment', 'second_moment']:
        state[name] = [[0.0]*n for n in counts]
    state['rng'] = 0
    state['step'] = 0
    plan['seed'] = 0
    request = dict(plan=plan, state=state, operation='train', inputs=[], labels=[0]*32,
                   initial=[[0.0]*276 for _ in range(32)])
    # Each innermost row contains 700 one-digit integers and 699 commas;
    # every row/sample/container has two brackets and its separating commas.
    one_tick = 2 + 700 + 699
    one_sample = 2 + 1370*one_tick + 1369
    minimum_inputs = 2 + 32*one_sample + 31
    assert minimum_inputs == original['exact_serialized_inputs']['integer']['bytes']
    encoded = json.dumps(request, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    lower_bound = len(encoded)-2+minimum_inputs
    return dict(schema='atlas-a2-json-lower-bound-reproduction-r1',
                source_result_sha256=digest(base/'result.json'),
                script_sha256=digest(Path(__file__)),
                input_array_minimum_bytes=minimum_inputs,
                noninput_lower_bound_bytes=len(encoded)-2,
                request_lower_bound_bytes=lower_bound, input_limit_bytes=64*1024**2,
                rejection_proven=lower_bound>64*1024**2,
                applies_to_seeds=[11,23,37,51,71], native_invoked=False,
                performance_run=False,
                minimized_noninput_request_sha256=hashlib.sha256(encoded).hexdigest(),
                premises=['Nonnegative integer inputs need at least one digit.',
                          'Finite Python float parameter/moment tokens need at least three characters; 0.0 attains the minimum.',
                          'Topology/masks/keys/format remain fixed; explicit initial state stays zero.',
                          'Nonnegative labels/counters/seeds need at least one digit.'],
                meaning='A format lower bound for every complete B32 batch, not five executed or fabricated native trials.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result))
