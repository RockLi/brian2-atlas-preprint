"""Prepare the existing Apple clang/Homebrew libomp compatibility wrapper."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler = Path("/usr/bin/clang++")
    libomp = Path("/opt/homebrew/opt/libomp")
    if not compiler.is_file() or not (libomp / "lib/libomp.dylib").is_file():
        raise RuntimeError("Apple clang and Homebrew libomp are required")
    wrapper = args.output.resolve()
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#!/bin/bash\n"
        "compile=0\n"
        'for arg in "$@"; do [[ "$arg" == -c ]] && compile=1; done\n'
        "args=()\n"
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == -fopenmp ]]; then\n'
        f'    if [[ $compile == 1 ]]; then args+=(-Xpreprocessor -fopenmp -I{libomp}/include); '
        f'else args+=(-L{libomp}/lib -Wl,-rpath,{libomp}/lib -lomp); fi\n'
        '  else args+=("$arg"); fi\n'
        "done\n"
        f'exec {compiler} "${{args[@]}}"\n')
    wrapper.chmod(0o755)
    print(json.dumps({"wrapper": str(wrapper),
                      "compiler": subprocess.check_output(
                          [str(compiler), "--version"], text=True).splitlines()[0],
                      "libomp": str(libomp / "lib/libomp.dylib")}, indent=2))


if __name__ == "__main__":
    main()
