#!/bin/bash
# Long-prompt TTFT under concurrent decode load ("mixload"): start a background AIPerf sweep (e.g. 1K/1K at c=32),
# wait DELAY s, then send NPROBE long-ISL probes at c=1 (osl 1) and report their TTFT; the background tok/s is the
# second number to watch. This is the recipe behind the mixload rows in docs/RESULTS.md.
# usage: MODEL=<tokenizer dir> bash mixload.sh <port> <bg_isl:bg_osl:bg_conc> <probe_isl:nprobe> <delay_s> <outdir>
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PORT=$1; IFS=: read BI BO BC <<< "$2"; IFS=: read PI PN <<< "$3"; DELAY=${4:-90}; OUT=$5
SEED=1001 REQ_MULT=8 MIN_REQ=10 bash $HERE/sweep_aiperf.sh $PORT $BI $BO ${OUT}_bg "$BC" > /dev/null 2>&1 & BG=$!
sleep $DELAY
SEED=3201 REQ_MULT=1 MIN_REQ=${PN:-3} bash $HERE/sweep_aiperf.sh $PORT $PI 1 ${OUT}_probe 1
kill -0 $BG 2>/dev/null && echo "background still running after the probes: OK" || echo "WARN: background finished before the probes ended"
wait $BG
