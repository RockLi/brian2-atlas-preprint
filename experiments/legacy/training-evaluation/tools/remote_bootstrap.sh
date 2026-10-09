#!/bin/bash
set -euo pipefail
cd /atlas-home/0004/workspace/atlas-training-evaluation/20261004-r1
export UV_CACHE_DIR="$PWD/cache/uv"
export CARGO_HOME="$PWD/cache/cargo-home"
export CARGO_TARGET_DIR="$PWD/build/cargo"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PATH="/opt/homebrew/bin:/atlas-home/0004/.cargo/bin:$PATH"
mkdir -p evidence environment runtime cache build
test "$(shasum -a 256 source.tar.gz | cut -d ' ' -f 1)" = 9119b703b0739700f3351c8b25f17cc7615c25b88fc72ac79425658d0a9b72a5
if ! test -d snapshot; then tar -xzf source.tar.gz; fi
/opt/homebrew/bin/python3.12 - <<'PY'
import json,hashlib
from pathlib import Path
m=json.loads(Path('sources/snapshot-manifest.json').read_text())
bad=[p for p,h in m['source_hashes'].items() if hashlib.sha256((Path('snapshot')/p).read_bytes()).hexdigest()!=h]
assert not bad,bad
Path('evidence/source-verified.json').write_text(json.dumps(dict(coherent_copy=True,source_tree_sha256=m['source_tree_sha256'],checked_files=len(m['source_hashes'])),indent=2)+'\n')
PY
if ! test -d environment/cpu; then uv venv --python /opt/homebrew/bin/python3.12 environment/cpu; fi
uv pip install --python environment/cpu/bin/python 'numpy==2.5.2' 'Cython==3.3.0' 'sympy==1.14.0' 'pyparsing==3.3.2' 'jinja2==3.1.6' 'setuptools==84.0.0' setuptools_scm wheel packaging matplotlib psutil 'torch==2.14.0' 'snntorch==1.0.0' 'spikingjelly==2.0.0rc1'
uv pip freeze --python environment/cpu/bin/python > environment/cpu-lock.txt
if ! test -d build/brian-source; then cp -R snapshot build/brian-source; fi
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BRIAN2=2.10.1.post729
uv pip install --python environment/cpu/bin/python --no-build-isolation --no-deps ./build/brian-source
# CPU runner has the same Rust source; all native build output is private.
cargo build --manifest-path snapshot/brian2-rust/Cargo.toml --locked --release --bin b2-train -j 4
cp build/cargo/release/b2-train runtime/b2-train
shasum -a 256 runtime/b2-train > environment/runtime.sha256
if ! test -d environment/jax; then uv venv --python /opt/homebrew/bin/python3.12 environment/jax; fi
uv pip install --python environment/jax/bin/python 'spyx==1.0.0' 'brainstate==0.5.4' psutil
uv pip freeze --python environment/jax/bin/python > environment/jax-lock.txt
environment/cpu/bin/python - <<'PY'
from pathlib import Path
import json,platform,sys,psutil,torch,brian2,hashlib
x=dict(platform=platform.platform(),python=sys.version,torch=torch.__version__,brian2=brian2.__version__,cuda=torch.cuda.is_available(),mps=torch.backends.mps.is_available(),ram_bytes=psutil.virtual_memory().total,logical_cpu=psutil.cpu_count(),disk=psutil.disk_usage('.')._asdict(),runtime_sha256=hashlib.sha256(Path('runtime/b2-train').read_bytes()).hexdigest())
Path('environment/hardware.json').write_text(json.dumps(x,indent=2)+'\n')
Path('evidence/bootstrap-complete.json').write_text(json.dumps(dict(completed=True,hardware=x),indent=2)+'\n')
print(json.dumps(x,indent=2))
PY
