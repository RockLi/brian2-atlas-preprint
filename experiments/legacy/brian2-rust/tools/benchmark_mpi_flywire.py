"""Serial, alternating before/after FlyWire MPI benchmark over Teleport.

Inputs must already exist on both hosts. Native results are audited separately;
launcher success alone is not a numerical or performance acceptance result.
"""
import argparse
import json
from pathlib import Path
import shlex

from mpi_teleport_launch import launch


def benchmark(args):
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    # repeat 0 warms each version/rank count; three measured rounds alternate.
    for repeat in range(4):
        versions = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
        for ranks in ((1, 2, 4) if repeat % 2 == 0 else (4, 2, 1)):
            for version in versions:
                jobs.append((version, 'rest', ranks, repeat))
    # Full numerical/control coverage for the other three sensory conditions.
    jobs += [('after', c, r, 0) for c in ('odor', 'cut_rest', 'cut') for r in (1, 2, 4)]
    completed = []
    for version, condition, ranks, repeat in jobs:
        label = f'{version}-{condition}-r{ranks}-{repeat}'
        root = args.before if version == 'before' else args.after
        project = f'{root}/{condition}/rank-{ranks}'
        result = f'{args.remote_base}/runs/{label}'
        metrics = f'{args.remote_base}/metrics/{label}-rank'
        nodes = args.nodes[:1] if ranks == 1 else args.nodes
        ips = args.ips[:1] if ranks == 1 else args.ips
        per_node = max(1, ranks // len(nodes))
        # Old runtime otherwise pins every local process to core 0. Give the
        # baseline distinct cores too; the new runtime selects them itself.
        cpus = f'"$((PMI_RANK % {per_node}))"' if version == 'before' else '0-15'
        application = ['sh', '-c', 'exec /usr/bin/time -v -o ' + shlex.quote(metrics)
                       + '"${PMI_RANK}.time" taskset -c ' + cpus + ' '
                       + shlex.join([project+'/b2-mpi', project+'/instance.bin', result])]
        print('START', label, flush=True)
        report = launch(nodes=nodes, ips=ips, ranks_per_node=per_node,
                        remote_base=args.remote_base, mpi_prefix=args.mpi_prefix,
                        application=application, output=args.output/label, timeout=180)
        completed.append({'label': label, 'ranks': ranks, 'version': version,
                          'condition': condition, 'repeat': repeat,
                          'returncodes': report['returncodes']})
        (args.output/'completed.json').write_text(json.dumps(completed, indent=2)+'\n')
        print('DONE', label, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nodes', nargs=2, required=True)
    p.add_argument('--ips', nargs=2, required=True)
    p.add_argument('--remote-base', required=True)
    p.add_argument('--mpi-prefix', required=True)
    p.add_argument('--before', required=True, help='Original models directory')
    p.add_argument('--after', required=True, help='Optimized models directory')
    p.add_argument('--output', type=Path, required=True)
    benchmark(p.parse_args())
