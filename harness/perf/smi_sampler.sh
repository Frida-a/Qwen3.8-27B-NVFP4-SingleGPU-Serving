#!/bin/bash
# Log power, SM clock and throttle reasons every 500 ms while a benchmark runs (clocks matched between arms is part of
# the A/B validity check at a power cap; docs/METHODS.md). Stop by creating the stop file.
# usage: bash smi_sampler.sh <gpu> <out.csv> <stopfile>
G=$1; OUT=$2; STOP=$3
nvidia-smi -i $G --query-gpu=timestamp,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active \
  --format=csv > $OUT
while [ ! -e $STOP ]; do
  nvidia-smi -i $G --query-gpu=timestamp,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader >> $OUT; sleep 0.5; done
