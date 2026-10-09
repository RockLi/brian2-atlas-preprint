#!/usr/bin/env bash
set -euo pipefail

destination="${1:?usage: fetch_upstream.sh NEW_EXTERNAL_DIRECTORY}"
if [[ -e "$destination" ]]; then
  echo "Destination already exists: $destination" >&2
  exit 1
fi
git clone https://github.com/janskaar/approximate_NMDA_model "$destination"
actual="$(git -C "$destination" rev-parse HEAD)"
expected="68e6dd970cfc6bab26459fcb8c34ee0f16560d9e"
if [[ "$actual" != "$expected" ]]; then
  echo "Upstream HEAD has moved ($actual). Review and select the recorded commit before use." >&2
  exit 1
fi
echo "$actual"
