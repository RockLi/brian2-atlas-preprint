#!/bin/sh
# Isolated, pinned direct dependencies; resolved package sets are saved afterwards.
set -eu
export CUDA_PATH=/usr/local/cuda
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs
unset PYTHONPATH

uv python install 3.10.18
uv venv --python 3.10.18 /opt/brian2genn-env
uv pip install --python /opt/brian2genn-env/bin/python \
  numpy==1.26.4 brian2genn==1.7.0 brian2==2.5.4 cython==0.29.37 \
  setuptools==80.9.0 sympy==1.12 jinja2==3.1.6 packaging==25.0

uv venv --python /usr/local/bin/python /opt/brian2cuda-env
uv pip install --python /opt/brian2cuda-env/bin/python \
  brian2cuda==1.0b1 brian2==2.10.1 numpy==2.2.6

mkdir -p /opt/genn4 /opt/genn5
curl --fail --location https://codeload.github.com/genn-team/genn/tar.gz/a9155521a48c9a6c348308639703d7b5eae3508d \
  | tar xz --strip-components=1 -C /opt/genn4
make -C /opt/genn4 -j2

curl --fail --location https://codeload.github.com/genn-team/genn/tar.gz/dd258075263c4b2bcb6607d230add658bcc23127 \
  | tar xz --strip-components=1 -C /opt/genn5
uv venv --python /usr/local/bin/python /opt/genn5-env
uv pip install --python /opt/genn5-env/bin/python \
  numpy==2.2.6 setuptools==80.9.0 wheel==0.45.1 pybind11==2.13.6 psutil==7.0.0 pkgconfig==1.5.5
cd /opt/genn5
# GeNN setup uses host physical-core count for make. Limit build concurrency
# without modifying its source or any simulation/compiler policy.
/opt/genn5-env/bin/python - <<'PY'
import psutil,runpy,sys
psutil.cpu_count=lambda logical=True: 2
sys.argv=["setup.py","bdist_wheel"]
runpy.run_path("setup.py",run_name="__main__")
PY
uv pip install --python /opt/genn5-env/bin/python /opt/genn5/dist/*.whl
mkdir -p /opt/baseline-locks
for name in brian2cuda brian2genn genn5; do
  uv pip freeze --python /opt/$name-env/bin/python > /opt/baseline-locks/$name.txt
done
