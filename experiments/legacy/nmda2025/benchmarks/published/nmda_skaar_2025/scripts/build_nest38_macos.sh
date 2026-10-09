#!/usr/bin/env bash
# Build the paper's NEST 3.8 release on current arm64 macOS.
#
# AppleClang 21/libc++ cannot compile NEST 3.8's iterator-pair sort.  The
# isolated reference environment therefore uses native Homebrew GCC 15.  This
# is a build-compatibility choice only; no NEST or model source is patched.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EXTERNAL_WORK_ROOT" >&2
  exit 2
fi

work_root=$1
source_dir="$work_root/nest-simulator-v3.8-source"
build_dir="$work_root/nest-simulator-v3.8-gcc15-build"
install_dir="$work_root/nest-simulator-v3.8-gcc15"
venv_dir=${NMDA_NEST38_VENV:-/private/tmp/nmda-nest38-venv}
jobs=${NMDA_BUILD_JOBS:-8}
nest_commit=6dca9500e512da50fa730a7b62ce4b05c77e2bfd

for command_name in cmake git /opt/homebrew/bin/gcc-15 /opt/homebrew/bin/g++-15 /opt/homebrew/bin/python3.13; do
  command -v "$command_name" >/dev/null
done

mkdir -p "$work_root"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone https://github.com/nest/nest-simulator.git "$source_dir"
fi
git -C "$source_dir" fetch --tags origin
git -C "$source_dir" checkout --detach "$nest_commit"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$nest_commit"

/opt/homebrew/bin/python3.13 -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install \
  'numpy==2.1.3' 'Cython==3.0.11' 'h5py==3.12.1'

cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 \
  '-DCMAKE_CXX_FLAGS=-include cstddef' \
  -DPython_EXECUTABLE="$venv_dir/bin/python" \
  -Dwith-python=ON \
  -Dwith-openmp=ON \
  -Dwith-gsl=/opt/homebrew/opt/gsl \
  -Dwith-boost=/opt/homebrew/opt/boost \
  '-Dwith-models=iaf_bw_2001;iaf_bw_2001_exact;inhomogeneous_poisson_generator;poisson_generator;spike_recorder;static_synapse' \
  -Dwith-mpi=OFF \
  -Dwith-hdf5=OFF \
  -Dwith-readline=OFF \
  -Dwith-music=OFF \
  -Dwith-libneurosim=OFF \
  -Dwith-sionlib=OFF \
  -Dwith-userdoc=OFF \
  -Dwith-devdoc=OFF \
  -Dwith-cpp-std=c++17

cmake --build "$build_dir" --parallel "$jobs"
cmake --install "$build_dir"

echo "source $install_dir/bin/nest_vars.sh"
echo "$venv_dir/bin/python -c 'import nest; print(nest.__version__)'"
