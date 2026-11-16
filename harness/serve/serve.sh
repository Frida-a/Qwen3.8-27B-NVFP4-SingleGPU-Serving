#!/bin/bash
# Start one QW38 vLLM server (TP1) in docker on one GPU, with the exact flags of a serving profile.
#
# usage: PROFILE=default|latency|default-shipped|latency-shipped GPU=0 PORT=8000 \
#          MODEL_DIR=/path/to/a2gdn-lc32k-mtpa4 [DRAFTER_DIR=/path/to/Qwen3.8-27B-DFlash2-NVFP4-RTNcal] \
#          [IMAGE=qw38/vllm:f5d93296-qw38] [MAXLEN=131072] [MAX_SEQS=<profile default>] [SEED=<int>] \
#          [EXTRA_ARGS="--enable-prefix-caching ..."] [FI_TACTICS=<store>|none] [TRITON_SEED=<dir>|none] \
#          [SAVE_TACTICS=/dir] [COMPILE_CACHE=/dir] [NAME=<container name>] [DOCKER_ARGS="--cpuset-cpus 0-15"] \
#          [AS_USER=1] bash harness/serve/serve.sh
#
# Profiles (docs/FINAL_CONFIG.md):
#   default          throughput profile, release candidate: MTP head, probabilistic drafts,
#                    k3 while <= 32 requests run, k1 above; K8V4 KV cache; int16 GDN state; MNBT 8192;
#                    GDN recover spec state + deferred-commit fold (VLLM_GDN_SPEC_STATE=recover, VLLM_GDN_RECOVER_FOLD=1).
#   latency          low-concurrency profile (<= 16 concurrent requests; route more to "default"), release candidate:
#                    DFlash2-NVFP4 drafter, k7, probabilistic drafts; recover + fold + verify v2;
#                    prefill piecewise-graph capture pinned at 512 tokens.
#   default-shipped  / latency-shipped: the previously shipped profiles, i.e. without fold / verify v2.
#
# Overrides (the perf numbers use the defaults):
#   MAXLEN      --max-model-len (default 131072). The accuracy gate used 262144.
#   MAX_SEQS    --max-num-seqs (default 256 for default*, 32 for latency*). The accuracy gate ran latency at 256.
#   SEED        adds --seed SEED (the accuracy gate used 0).
#   EXTRA_ARGS  extra `vllm serve` flags, split on whitespace (no JSON values), e.g. "--enable-prefix-caching".
# Accuracy-gate servers (docs/FINAL_CONFIG.md, section 2):
#   MAXLEN=262144 SEED=0 EXTRA_ARGS=--enable-prefix-caching [MAX_SEQS=256 for latency] PROFILE=... bash serve.sh
#
# Tuning data (harness/serve/tuning/README.md; docs/FINAL_CONFIG.md, section 3). Used by default:
#   FI_TACTICS     FlashInfer GEMM tactic store (default: harness/serve/tuning/flashinfer, tuned on an RTX PRO 6000
#                  Blackwell Max-Q at 250 W with the image of build_image.sh). It is copied into the container at start
#                  (a server rewrites the store it is given), so the shipped files are never modified.
#                  Without a matching store the first boot autotunes during warmup and the tuner's memory is counted as
#                  activations: on the RTX PRO 6000 Max-Q (96 GB) the default profile loses ~24% of its KV cache and
#                  the latency profile does not boot.
#                  FI_TACTICS=none turns it off (cold autotune); FI_TACTICS=/dir uses your own store.
#   TRITON_SEED    directory with Triton `<hash>/<kernel>.autotune.json` picks (default: harness/serve/tuning/triton,
#                  the 6 FLA prefill kernels). Only the *.autotune.json files are copied, keeping their <hash>/ dirs,
#                  into the server's TRITON_CACHE_DIR: kernels recompile, the picks are kept. TRITON_SEED=none: off.
#   SAVE_TACTICS   host directory that keeps this boot's tuning state (the container runs with --rm):
#                  SAVE_TACTICS/flashinfer is the live FlashInfer store (seeded from FI_TACTICS if empty) and
#                  SAVE_TACTICS/triton the Triton cache. Reuse with FI_TACTICS=$SAVE_TACTICS/flashinfer
#                  TRITON_SEED=$SAVE_TACTICS/triton. This is how to make a store for another GPU or build.
#                  If COMPILE_CACHE is also set, SAVE_TACTICS/triton (not COMPILE_CACHE/triton) is the Triton cache.
# Other hooks:
#   COMPILE_CACHE  host directory kept across boots for VLLM_CACHE_ROOT (COMPILE_CACHE/vllm, torch.compile artifacts)
#                  and the Triton kernel cache (COMPILE_CACHE/triton), so later boots skip most compilation. Off by
#                  default (the measurements booted cold).
#   SAVE_TACTICS and COMPILE_CACHE hold files that a root container with host networking loads and executes (compiled
#   kernels, tactic choices). Put them on a local disk under a parent directory only you can enter (chmod 700). The
#   script creates the subdirectories (flashinfer/, triton/, vllm/) and, for the default root container, makes the ones
#   it creates world-writable so that root-squashed NFS also works; it never chmods a directory you named or one that
#   already exists. With AS_USER=1 nothing is chmodded and the container writes with umask 022.
#   NAME           container name (default qw38-$PROFILE-g$GPU).
#   DOCKER_ARGS    extra `docker run` flags, split on whitespace, e.g. "--cpuset-cpus 24-35".
#   AS_USER=1      run the container as the calling user (--user $(id -u):$(id -g), HOME=/tmp/qw38-home inside),
#                  for root-squashed NFS where container root cannot read your files. Not GPU-tested.
#
# Host directories (MODEL_DIR, DRAFTER_DIR, this repo's tuning/, SAVE_TACTICS, COMPILE_CACHE) are bind-mounted and, by
# default, read by root inside the container. On a root-squashed NFS mount root acts as "nobody": the directories must
# be world-traversable (o+x on every parent) and the files world-readable, MODEL_DIR must hold real files (a symlink
# pointing outside MODEL_DIR does not resolve inside the container). Otherwise copy the checkpoint to a local disk or use
# AS_USER=1. Relative paths are resolved against the current directory. No host temp dirs are used: the per-server
# copies of the tuning data live inside the container.
#
# Power: the measurements in docs/RESULTS.md ran an RTX PRO 6000 Blackwell Max-Q at a 250 W board limit:
#   sudo nvidia-smi -i $GPU -pl 250
set -euo pipefail
: ${MODEL_DIR:?MODEL_DIR}; GPU=${GPU:-0}; PORT=${PORT:-8000}; IMAGE=${IMAGE:-qw38/vllm:f5d93296-qw38}
PROFILE=${PROFILE:-default}; MAXLEN=${MAXLEN:-131072}
AS_USER=${AS_USER:-0}
TUNING=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tuning
FI_TACTICS=${FI_TACTICS-$TUNING/flashinfer}; [ "$FI_TACTICS" = none ] && FI_TACTICS=
TRITON_SEED=${TRITON_SEED-$TUNING/triton}; [ "$TRITON_SEED" = none ] && TRITON_SEED=
# Absolute host paths (a relative `-v name:/x` would silently become a docker named volume).
abspath() { (cd "$1" 2>/dev/null && pwd) || { echo "serve.sh: $2=$1 is not an existing directory" >&2; exit 2; }; }
MODEL_DIR=$(abspath "$MODEL_DIR" MODEL_DIR)
[ -n "${DRAFTER_DIR:-}" ] && DRAFTER_DIR=$(abspath "$DRAFTER_DIR" DRAFTER_DIR)
[ -n "$FI_TACTICS" ] && FI_TACTICS=$(abspath "$FI_TACTICS" FI_TACTICS)
[ -n "$TRITON_SEED" ] && TRITON_SEED=$(abspath "$TRITON_SEED" TRITON_SEED)

COMMON=(--served-model-name qw38 --host 0.0.0.0 --port $PORT --tensor-parallel-size 1
  --kv-cache-dtype fp8_e4m3 --max-model-len $MAXLEN
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --mm-encoder-tp-mode data --gpu-memory-utilization 0.85 --enable-chunked-prefill --async-scheduling
  --mamba-ssm-cache-dtype int16 --block-size 1600)
MTP_SCHED='{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","num_speculative_tokens_per_batch_size":[[1,32,3],[33,1024,1]]}'
DFLASH='{"method":"dflash","model":"/drafter","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'

ENVS=(); MOUNTS=(-v "$MODEL_DIR":/model:ro)
case $PROFILE in
  default|default-shipped)
    ARGS=("${COMMON[@]}" --max-num-seqs ${MAX_SEQS:-256} --max-num-batched-tokens 8192 --speculative-config "$MTP_SCHED")
    ENVS+=(-e VLLM_KV_K8V4=1)                                   # serve the fp8_e4m3 cache as K8V4 (FP8 K + NVFP4 V)
    [ $PROFILE = default ] && ENVS+=(-e VLLM_GDN_SPEC_STATE=recover -e VLLM_GDN_RECOVER_FOLD=1) ;;
  latency|latency-shipped)
    : ${DRAFTER_DIR:?DRAFTER_DIR (maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal)}
    MOUNTS+=(-v "$DRAFTER_DIR":/drafter:ro)
    ARGS=("${COMMON[@]}" --max-num-seqs ${MAX_SEQS:-32} --max-num-batched-tokens 32768 --speculative-config "$DFLASH"
          --compilation-config '{"max_prefill_cudagraph_capture_tokens":512}')
    ENVS+=(-e VLLM_GDN_SPEC_STATE=recover)
    [ $PROFILE = latency ] && ENVS+=(-e VLLM_GDN_RECOVER_FOLD=1 -e VLLM_GDN_VERIFY_V2=1) ;;
  *) echo "unknown PROFILE $PROFILE"; exit 2 ;;
esac
[ -n "${SEED:-}" ] && ARGS+=(--seed "$SEED")
if [ -n "${EXTRA_ARGS:-}" ]; then read -r -a XA <<< "$EXTRA_ARGS"; ARGS+=("${XA[@]}"); fi

# Host dirs the container writes to. The named dir is created if missing but never chmodded; the subdirs the script
# creates are made world-writable only for the root container (root may be "nobody" on root-squashed NFS).
hostdir() { mkdir -p "$1" && (cd "$1" && pwd); }
subdir() { [ -d "$1" ] && return 0; mkdir "$1"; if [ "$AS_USER" != 1 ]; then chmod 777 "$1"; fi; }

# In-container locations. Seeds are mounted read-only and copied at start, inside the container (no host temp dirs).
FI_DIR=/tmp/qw38/fi_tactics; TC_DIR=/tmp/qw38/triton
if [ -n "${COMPILE_CACHE:-}" ]; then
  COMPILE_CACHE=$(hostdir "$COMPILE_CACHE"); subdir "$COMPILE_CACHE/vllm"; subdir "$COMPILE_CACHE/triton"
  MOUNTS+=(-v "$COMPILE_CACHE":/compile_cache); ENVS+=(-e VLLM_CACHE_ROOT=/compile_cache/vllm); TC_DIR=/compile_cache/triton
fi
if [ -n "${SAVE_TACTICS:-}" ]; then
  SAVE_TACTICS=$(hostdir "$SAVE_TACTICS"); subdir "$SAVE_TACTICS/flashinfer"; subdir "$SAVE_TACTICS/triton"
  MOUNTS+=(-v "$SAVE_TACTICS":/save_tactics); FI_DIR=/save_tactics/flashinfer; TC_DIR=/save_tactics/triton
fi
if [ -n "$FI_TACTICS" ]; then
  [ -d "$FI_TACTICS/v2" ] || { echo "FI_TACTICS=$FI_TACTICS has no v2/ store (FI_TACTICS=none for a cold autotune)"; exit 2; }
  MOUNTS+=(-v "$FI_TACTICS":/seed/flashinfer:ro)
  # The shipped store is keyed to one GPU model + software build (tuning/flashinfer/v2/*/manifest.json).
  if [ "$FI_TACTICS" = "$TUNING/flashinfer" ] && command -v nvidia-smi >/dev/null 2>&1; then
    gname=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null) && gname=${gname%%$'\n'*} || gname=
    if [ -n "$gname" ] && ! grep -qF "\"gpu\": \"$gname\"" "$TUNING"/flashinfer/v2/*/manifest.json; then
      echo "WARNING: GPU '$gname' is not the GPU the shipped FlashInfer store was tuned on; it will autotune at boot." \
           "The latency profile may not boot (harness/serve/tuning/README.md)." >&2
    fi
  fi
fi
[ -n "$TRITON_SEED" ] && MOUNTS+=(-v "$TRITON_SEED":/seed/triton:ro)
ENVS+=(-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$FI_DIR -e TRITON_CACHE_DIR=$TC_DIR)

USER_ARGS=(); UMASK=0000
if [ "$AS_USER" = 1 ]; then USER_ARGS=(--user "$(id -u):$(id -g)"); ENVS+=(-e HOME=/tmp/qw38-home); UMASK=0022; fi
ENVS+=(-e QW38_UMASK=$UMASK)
DX=(); [ -n "${DOCKER_ARGS:-}" ] && read -r -a DX <<< "$DOCKER_ARGS"

# Runs inside the container before vllm: umask 0 for the root container so that what it writes to host mounts stays
# removable by you (022 with AS_USER=1, where the files are yours);
# seed the FlashInfer store (only if the target is empty, so a SAVE_TACTICS dir is resumed, not overwritten) and copy
# the Triton autotune picks with their <hash>/ dirs (overwriting, so pinned picks win over a persistent cache).
PRE='umask "$QW38_UMASK"; set -e; mkdir -p "$HOME" "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" "$TRITON_CACHE_DIR"
if [ -d /seed/flashinfer ] && [ -z "$(ls -A "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR")" ]; then
  cp -rf /seed/flashinfer/. "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"/; echo "qw38 serve.sh: FlashInfer store seeded"; fi
if [ -d /seed/triton ]; then n=0; for f in $(cd /seed/triton && find . -name "*.autotune.json"); do
  mkdir -p "$TRITON_CACHE_DIR/$(dirname $f)"; cp -f /seed/triton/$f "$TRITON_CACHE_DIR/$f"; n=$((n+1)); done
  echo "qw38 serve.sh: $n Triton autotune picks seeded"; fi
exec vllm "$@"'

exec docker run --rm --name "${NAME:-qw38-$PROFILE-g$GPU}" ${DX[@]+"${DX[@]}"} ${USER_ARGS[@]+"${USER_ARGS[@]}"} \
  --gpus device=$GPU --ipc=host --shm-size 32g --network host \
  -e HF_HUB_OFFLINE=1 "${ENVS[@]}" "${MOUNTS[@]}" --entrypoint bash $IMAGE \
  -c "$PRE" qw38-serve serve /model "${ARGS[@]}"
