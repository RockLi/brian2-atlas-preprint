"""Pin one NEST MPI rank and its four OpenMP workers before importing NEST.

Layout is input, not resource admission. The launcher must separately enforce
memory, CPU, disk and time limits and verify the physical topology afresh.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def binding(layout, rank, host, allowed):
    if (layout.get('schema') != 'b2-mam-nest-affinity-v1'
            or layout.get('ranks_per_host') != 8 or layout.get('threads_per_rank') != 4
            or type(rank) is not int or not 0 <= rank < 8*len(layout['hosts'])):
        raise ValueError('invalid NEST rank layout')
    hosts = layout['hosts']
    if len({h['host'] for h in hosts}) != len(hosts):
        raise ValueError('duplicate layout host')
    entry = hosts[rank//8]
    if entry['host'] != host:
        raise ValueError('MPI rank placed on wrong host')
    groups = entry['rank_cpu_ids']
    if (len(groups) != 8 or any(len(g) != 4 for g in groups)
            or any(type(c) is not int or not 0 <= c < 4096 for g in groups for c in g)
            or len({c for g in groups for c in g}) != 32):
        raise ValueError('rank groups must contain 32 disjoint CPUs')
    cpus = groups[rank % 8]
    if not set(cpus) <= set(allowed):
        raise ValueError('rank CPUs outside launcher allowance')
    return cpus, dict(OMP_NUM_THREADS='4', OMP_DYNAMIC='FALSE',
                     OMP_PROC_BIND='close', OMP_PLACES=','.join('{'+str(c)+'}' for c in cpus),
                     OPENBLAS_NUM_THREADS='1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--layout-sha256', required=True)
    parser.add_argument('--receipt-directory', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.layout.stat().st_size > 2**20:
        parser.error('bounded layout and command required')
    raw = args.layout.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.layout_sha256:
        raise ValueError('layout hash mismatch')
    layout = json.loads(raw)
    # MPICH Hydra supplies global PMI_RANK/PMI_SIZE; launcher assigns eight
    # consecutive ranks to each host. Never guess rank zero when these are absent.
    rank, size = int(os.environ['PMI_RANK']), int(os.environ['PMI_SIZE'])
    if size != len(layout['hosts'])*8:
        raise ValueError('MPI size differs from layout')
    cpus, env = binding(layout, rank, os.uname().nodename, os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus)
    if sorted(os.sched_getaffinity(0)) != sorted(cpus):
        raise RuntimeError('rank affinity did not take effect')
    # Remove an inherited GNU affinity setting before defining explicit places.
    os.environ.pop('GOMP_CPU_AFFINITY', None)
    os.environ.update(env)
    args.receipt_directory.mkdir(parents=True, exist_ok=True)
    with (args.receipt_directory/f'rank{rank}.json').open('x') as output:
        output.write(json.dumps(dict(rank=rank, ranks=size, host=os.uname().nodename,
            initial_rank_cpu_ids=cpus, openmp_environment=env,
            layout_sha256=args.layout_sha256,
            scope='Pre-exec process affinity; actual OpenMP worker affinity requires separate observation.'), indent=2)+'\n')
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    main()
