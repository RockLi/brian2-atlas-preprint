"""Run the full FlyWire two-node matrix through Teleport, with per-rank RSS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex

from mpi_teleport_launch import launch


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)

    def one(item):
        condition, ranks = item
        project = f"{args.remote_base}/models/{condition}/rank-{ranks}"
        result = f"{args.remote_base}/runs/{condition}/rank-{ranks}"
        metric = f"{args.remote_base}/metrics/{condition}-r{ranks}-rank"
        # PMI_RANK is set by Hydra; each process writes its own GNU time file.
        command = ("exec /usr/bin/time -v -o " + shlex.quote(metric) + '"${PMI_RANK}.time" '
                   + shlex.join([project + "/b2-mpi", project + "/instance.bin", result]))
        label = f"{condition}-r{ranks}"
        print("START", label, flush=True)
        report = launch(nodes=args.nodes, ips=args.ips, ranks_per_node=ranks // 2,
                        remote_base=args.remote_base, mpi_prefix=args.mpi_prefix,
                        application=["/bin/sh", "-c", command], output=args.output / label,
                        interface=args.interface, timeout=args.timeout, login=args.login)
        print("DONE", label, report["returncodes"], flush=True)
        return label

    items = [(condition, ranks) for condition in ("rest", "odor", "cut_rest", "cut") for ranks in (2, 4)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(one, items))
    (args.output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs=2, required=True)
    parser.add_argument("--ips", nargs=2, required=True)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--mpi-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface", default="bond0")
    parser.add_argument("--login", default="rock")
    parser.add_argument("--timeout", type=int, default=1200)
    run(parser.parse_args())
