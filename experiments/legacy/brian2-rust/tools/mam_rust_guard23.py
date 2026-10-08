"""Apply the frozen brick2 reserve before entering the existing job guard."""
from pathlib import Path
import runpy
import sys


def arguments(args):
    args=list(args)
    if args.count('--volume')!=1 or args[args.index('--volume')+1]!='/data/brick2':
        raise ValueError('Rust leader requires brick2')
    if args.count('--min-free-gib')!=1 or args[args.index('--min-free-gib')+1]!='128':
        raise ValueError('unexpected base reserve')
    args[args.index('--min-free-gib')+1]='1280'
    return args


if __name__=='__main__':
    sys.argv=arguments(sys.argv)
    runpy.run_path(str(Path(__file__).with_name('mpi_resource_guard.py')),run_name='__main__')
