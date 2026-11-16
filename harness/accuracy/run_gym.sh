#!/bin/bash
# Run NeMo Gym benchmarks against an OpenAI-compatible endpoint serving the model as "qw38".
# Sampling = model card: temperature 1.0, top_p 0.95, thinking on, max 65,536 output tokens.
#
# usage: GYM=/path/to/Gym OUT=results/run1 MODEL_URL=http://127.0.0.1:8000/v1 CONC=1024 [LIMIT=n] [GYM_EXTRA=...] \
#          bash harness/accuracy/run_gym.sh mrcr_qwen38_128k:2 gpqa:4 ifbench:2 [hle_verified:1] [mrcr_qwen38_196k_gt128k:3]
#   bench:repeats[:max_output_tokens]. Built-in Gym benchmarks (gpqa, ifbench, hle_verified) are prepared by
#   `gym eval prepare`; mrcr_* are built locally (harness/README.md, section 3).
#   OUT may be relative to the directory you start from; it is made absolute before the script changes into $GYM.
#   LIMIT=<n>   passes `--limit n` to every benchmark, for smoke tests. Gym v0.7.0 applies it to the input rows
#               (tasks) before repeating them, so a run makes LIMIT x repeats rollouts.
#   GYM_EXTRA   extra `gym eval run` arguments / Hydra overrides, split on whitespace (e.g. "++some_key=value").
#   Exit status: non-zero if any benchmark was skipped (no config) or failed; the others still run.
#
# `++overwrite_metrics_conflicts=true`: Gym v0.7.0 keeps a metrics sidecar next to each built-in benchmark's data
# (e.g. gpqa_diamond_benchmark_metrics.json, computed with that benchmark's own num_repeats). The num_repeats-1 run copy
# below changes num_repeats, so Gym's data validation would abort with "Found conflicting aggregate metrics"; the
# override (a top-level Gym config field, default false) lets Gym rewrite the sidecar instead. Note that this modifies
# a file tracked in the Gym checkout (`git status` there shows it; `git checkout -- benchmarks` restores it).
#
# Paths (checked against NeMo Gym v0.7.0): `--search-dir X` makes Gym look for `X/benchmarks/<name>/config.yaml`
# (then the cwd and the Gym install), so X is this directory. Each benchmark runs from a generated copy,
# harness/accuracy/benchmarks/<name>_run/config.yaml (tools/repeat_overrides.py), that has every benchmark dataset's
# num_repeats set to 1 and every repo-relative jsonl_fpath of the local benchmarks made absolute. Gym reads a relative
# jsonl_fpath via its search roots but writes the prepared copy and the metrics sidecar next to it relative to the
# cwd ($GYM), so absolute paths keep the data and its sidecars together under harness/accuracy/benchmarks/<name>/data.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); EV=$HERE; T=$HERE/tools; ROOT=$(cd "$HERE/../.." && pwd)
: ${GYM:?GYM checkout (NVIDIA-NeMo/Gym v0.7.0)} ${OUT:?OUT}; MODEL_URL=${MODEL_URL:-http://127.0.0.1:8000/v1}; CONC=${CONC:-1024}
MAXTOK=${MAXTOK:-65536}; LIMIT=${LIMIT:-}; rc_all=0
XG=(); [ -n "${GYM_EXTRA:-}" ] && read -r -a XG <<< "$GYM_EXTRA"
GYM=$(cd "$GYM" && pwd); mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
cd "$GYM" && . .venv/bin/activate
for spec in "$@"; do
  IFS=: read -r b n mt <<< "$spec"; mt=${mt:-$MAXTOK}; d=$OUT/$b; mkdir -p "$d"
  cfg=""
  for c in "$EV/benchmarks/$b/config.yaml" "$GYM/benchmarks/$b/config.yaml"; do
    if [ -f "$c" ]; then cfg=$c; break; fi
  done
  if [ -z "$cfg" ]; then echo "$b: no config.yaml under $EV/benchmarks/$b or $GYM/benchmarks/$b; skipped"; rc_all=1; continue; fi
  # Gym multiplies a benchmark's own num_repeats with --num-repeats: run a num_repeats-1 copy (<name>_run).
  br=$(python "$T/repeat_overrides.py" "$cfg" "$EV" "$b" --abs-root "$ROOT") \
    || { echo "$b: repeat_overrides.py failed; skipped"; rc_all=1; continue; }
  gym eval run --benchmark "$br" --search-dir "$EV" --split benchmark \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml --config "$EV/configs/thinking_on.yaml" \
    --model-url "$MODEL_URL" --model qw38 --model-api-key dummy \
    --output "$d/rollouts.jsonl" --num-repeats "$n" --concurrency "$CONC" \
    --temperature 1.0 --top-p 0.95 --max-output-tokens "$mt" ++num_repeats_add_seed=true \
    ++reuse_existing_data_preparation=true ++overwrite_metrics_conflicts=true ${LIMIT:+--limit "$LIMIT"} \
    ${XG[@]+"${XG[@]}"} > "$d/gym.log" 2>&1 || { echo "$b failed, see $d/gym.log"; rc_all=1; continue; }
  GYM=$GYM python3 "$T/summarize.py" "$d" "$mt" || true
done
[ $rc_all = 0 ] || echo "run_gym.sh: at least one benchmark failed or was skipped" >&2
exit $rc_all
