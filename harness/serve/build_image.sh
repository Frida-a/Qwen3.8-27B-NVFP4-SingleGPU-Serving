#!/bin/bash
# Build the patched vLLM serving image: upstream vLLM nightly f5d93296 + the QW38 patch series (patches/).
# Every QW38 change is Python / Triton / FlashInfer-JIT only (no csrc), so the patched Python tree is copied over
# the nightly image's installed package; the image's compiled extensions are reused unchanged.
#
# usage: [BASE_IMAGE=<image>] bash harness/serve/build_image.sh [image tag]   (needs git + docker, no GPU)
#
# BASE_IMAGE defaults to the amd64 image of nightly f5d93296 that every RTX PRO 6000 number was measured on.
# The GB300 (arm64) accuracy runs used the arm64 image of the same nightly,
#   BASE_IMAGE=vllm/vllm-openai@sha256:c68112d825ebdc6c856ff2e5d22bf514d278194a734baeb3b5f76d1ea35c6355
# with the measured build's patched vLLM Python package mounted on top (launched under Slurm with `vllm serve` and the flags of serve.sh, not with
# this script's docker image; building on arm64 with this script is untested). Build on a host of the target
# architecture. Any other base must be the same vLLM commit (BASE_COMMIT below), or the overlay will not match it.
set -euo pipefail
TAG=${1:-qw38/vllm:f5d93296-qw38}
BASE_IMAGE=${BASE_IMAGE:-vllm/vllm-openai@sha256:789bd3c0b35711ddf086a19f743a9423ab26670781e1453d8ec55a3766f791d7}   # nightly f5d93296, amd64
BASE_COMMIT=f5d93296c8ec82b7e7dabdb308d24d43cb1907bf
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
W=$(mktemp -d); trap 'rm -rf $W' EXIT

git clone -q https://github.com/vllm-project/vllm.git $W/vllm
git -C $W/vllm -c advice.detachedHead=false checkout -q $BASE_COMMIT
git -C $W/vllm -c user.name=qw38 -c user.email=qw38@localhost -c commit.gpgsign=false \
  am -q "$ROOT"/patches/[0-9]*/*.patch          # sorted glob: the series order
echo "applied $(git -C $W/vllm rev-list --count $BASE_COMMIT..HEAD) patches on $BASE_COMMIT"

# Only the files the series touches under vllm/ are copied (tests/ and benchmarks/ stay in the source tree).
git -C $W/vllm diff --name-only --diff-filter=AM $BASE_COMMIT HEAD -- 'vllm/*' > $W/files.txt
mkdir -p $W/ctx/overlay
(cd $W/vllm && xargs -a $W/files.txt cp --parents -t $W/ctx/overlay)
cat > $W/ctx/Dockerfile <<EOF
FROM $BASE_IMAGE
COPY overlay/vllm/ /tmp/qw38-overlay/vllm/
RUN python3 -c 'import vllm; v = vllm.__version__; assert "gf5d93296" in v, "base image has vLLM " + v + ", expected the f5d93296 nightly (0.34.0rc1.dev206+gf5d93296c)"' \\
 && SP=\$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))') \\
 && cp -r /tmp/qw38-overlay/vllm/. \$SP/vllm/ && rm -rf /tmp/qw38-overlay \\
 && python3 -m compileall -q \$SP/vllm/model_executor/layers/mamba \$SP/vllm/v1/attention && echo "QW38 overlay OK"
EOF
docker build -t $TAG $W/ctx
echo "built $TAG"
