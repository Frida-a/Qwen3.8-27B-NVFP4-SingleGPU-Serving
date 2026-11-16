#!/bin/bash
# Generic AIPerf concurrency sweep against ONE OpenAI-compatible server. Runs on the client host (AIPerf venv,
# harness/requirements-client.txt), not inside the server container.
# usage: sweep_aiperf.sh <port> <isl> <osl> <outdir> "<concurrencies>"
# env: MODEL (tokenizer path), SERVED=qw38, SEED=42, MIN_REQ=10, REQ_MULT=3
set -uo pipefail
PORT=$1; ISL=$2; OSL=$3; OUT=$4; CONCS=$5
MODEL=${MODEL:?set MODEL to the tokenizer / checkpoint dir}; SERVED=${SERVED:-qw38}; SEED=${SEED:-42}
MIN_REQ=${MIN_REQ:-10}; REQ_MULT=${REQ_MULT:-3}
AIPERF=${AIPERF:-aiperf}
mkdir -p "$OUT"
curl -s localhost:$PORT/v1/models > "$OUT/server_models.json"
for C in $CONCS; do
  N=$(( REQ_MULT*C > MIN_REQ ? REQ_MULT*C : MIN_REQ )); W=$C
  D=$OUT/c$C; mkdir -p "$D"
  curl -s localhost:$PORT/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft|drafts)_tokens_total|^vllm:spec_decode_num_drafts_total|^vllm:spec_decode_num_accepted_tokens_per_pos' > "$D/metrics_before.txt"
  echo "[$(date +%T)] port=$PORT isl=$ISL osl=$OSL c=$C n=$N warmup=$W"
  "$AIPERF" profile -m $SERVED --url http://localhost:$PORT --endpoint-type chat --streaming --apply-chat-template \
    --tokenizer "$MODEL" --isl $ISL --isl-stddev 0 --osl $OSL --osl-stddev 0 \
    --extra-inputs ignore_eos:true --extra-inputs min_tokens:$OSL \
    --concurrency $C --request-count $N --warmup-request-count $W --num-dataset-entries $((N+W)) \
    --random-seed $SEED --use-server-token-count --per-chunk-usage --ui none \
    --artifact-dir "$D" > "$D/aiperf.log" 2>&1
  echo "  rc=$?"
  curl -s localhost:$PORT/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft|drafts)_tokens_total|^vllm:spec_decode_num_drafts_total|^vllm:spec_decode_num_accepted_tokens_per_pos' > "$D/metrics_after.txt"
done
