"""Independent E2 shared-convolution fixtures; authorized remote generation only."""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
from generate_dense_large import ROOT, SEEDS, digest, write_json

CONTRACT = ROOT/'protocol/e2-convolution-r1.json'


def conv_projection(source_layer, target_layer, shape, out_channels, kernel, stride, padding):
    channels, height, width = shape
    oh = (height+2*padding-kernel)//stride+1
    ow = (width+2*padding-kernel)//stride+1
    sources, targets, ids = [], [], []
    for output in range(out_channels*oh*ow):
        oc, oy, ox = output//(oh*ow), (output//ow)%oh, output%ow
        for parameter_within_output in range(channels*kernel*kernel):
            ic = parameter_within_output//(kernel*kernel)
            ky, kx = (parameter_within_output//kernel)%kernel, parameter_within_output%kernel
            iy, ix = oy*stride+ky-padding, ox*stride+kx-padding
            if 0 <= iy < height and 0 <= ix < width:
                sources.append(ic*height*width+iy*width+ix)
                targets.append(output)
                ids.append(oc*channels*kernel*kernel+parameter_within_output)
    return dict(source_layer=source_layer, target_layer=target_layer, parameter_count=out_channels*channels*kernel*kernel,
                sources=sources, targets=targets, parameter_ids=ids), [out_channels, oh, ow]


def create(seed, spec, identity):
    import numpy as np
    rng = np.random.default_rng(seed)
    shapes = [spec['input_chw']]
    projections, operators, parameter_shapes = [], [], []
    for layer, channels in enumerate(spec['conv_channels']):
        p, output_shape = conv_projection(layer, layer+1, shapes[-1], channels, spec['kernel'], spec['stride'], spec['padding'])
        projections.append(p)
        parameter_shapes.append([channels, shapes[-1][0], spec['kernel'], spec['kernel']])
        operators.append(dict(type='conv2d', input_chw=shapes[-1], output_chw=output_shape,
                              kernel=spec['kernel'], stride=spec['stride'], padding=spec['padding']))
        shapes.append(output_shape)
    sizes = [int(np.prod(shape)) for shape in shapes]
    for count in spec['readout']:
        layer = len(sizes)-1
        source_count = sizes[-1]
        projections.append(dict(source_layer=layer, target_layer=layer+1, parameter_count=source_count*count,
                                sources=[i for j in range(count) for i in range(source_count)],
                                targets=[j for j in range(count) for i in range(source_count)],
                                parameter_ids=[i*count+j for j in range(count) for i in range(source_count)]))
        parameter_shapes.append([source_count, count])
        operators.append(dict(type='dense', source_count=source_count, target_count=count))
        sizes.append(count)
    inputs = rng.choice([0, 1, 2], size=(spec['B'], spec['T'], sizes[0]), p=[.9, .095, .005])
    weights = []
    for shape in parameter_shapes:
        fan_in = int(np.prod(shape[1:])) if len(shape) == 4 else shape[0]
        weights.append(rng.normal(0., np.sqrt(2/fan_in), size=shape).ravel().tolist())
    return dict(id=identity, seed=seed, sizes=sizes, inputs=inputs.tolist(), labels=(np.arange(spec['B'])%sizes[-1]).tolist(),
                weights=weights, projections=projections, operators=operators, parameter_shapes=parameter_shapes, beta=.95, theta=1.)


def prepare(scale, seed):
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('E2 fixture generation is restricted to 100.90.28.27')
    policy = json.loads(CONTRACT.read_text())
    if policy['status'] != 'frozen_before_fixture_generation_or_E2_execution':
        raise ValueError('E2 protocol is not frozen')
    directory = ROOT/'fixtures/e2-conv-r1'/scale
    directory.mkdir(parents=True, exist_ok=True)
    path, manifest_path = directory/f'seed-{seed}.json', directory/f'seed-{seed}.manifest.json'
    helper = ROOT/'snapshot/brian2-rust/python/brian2_rust/training_graph.py'
    identity = dict(contract_sha256=digest(CONTRACT), generator_sha256=digest(__file__), native_projection_builder_sha256=digest(helper))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity or digest(path) != manifest['arrays_sha256']:
            raise ValueError('Frozen E2 fixture differs; preserve prior evidence')
        return path
    if path.exists():
        raise FileExistsError('Prior partial E2 fixture must be preserved')
    case = create(seed, policy[scale], f'E2-conv-{scale}-seed-{seed}')
    # Check source helper equivalence without importing native dynamics/oracle.
    spec = importlib.util.spec_from_file_location('e2_projection_builder_reference', helper)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    for layer, operator in enumerate(case['operators'][:2]):
        p, shape = native.conv2d_training_projection(layer, layer+1, operator['input_chw'], operator['output_chw'][0],
                                                    operator['kernel'], stride=operator['stride'], padding=operator['padding'])
        if p != case['projections'][layer] or list(shape) != operator['output_chw']:
            raise ValueError('Independent frozen projection differs from native helper')
    counts = policy[scale+'_counts']
    if case['sizes'] != counts['sizes'] or [len(w) for w in case['weights']] != counts['bank_parameter_counts'] or [len(p['sources']) for p in case['projections']] != counts['projection_edge_counts']:
        raise ValueError('E2 graph shape/counts differ from pre-execution contract')
    write_json(path, case)
    write_json(manifest_path, dict(schema='e2-convolution-common-arrays-v1', scale=scale, seed=seed, identity=identity,
                                   arrays_sha256=digest(path), arrays_bytes=path.stat().st_size, counts=counts,
                                   native_projection_helper_agrees=True, parameter_tying='one OIHW parameter/optimizer slot per kernel coefficient'))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=['q0', 'full'], required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    path = prepare(args.scale, args.seed)
    print(json.dumps(dict(path=str(path), sha256=digest(path))), flush=True)
