# Harness: serving, performance and accuracy

Everything here runs on plain `docker run` / `vllm serve` / AIPerf / NeMo Gym. No benchmark data ships with this repo:
every dataset is downloaded (or exported by you) at run time. See `docs/METHODS.md` for the measurement rules
(power cap, run counts, A/B order, noise) that the numbers in `docs/RESULTS.md` follow.

```
harness/
  serve/build_image.sh       patched vLLM image = nightly f5d93296 image + patches/ (Python/Triton only)
  serve/serve.sh             one TP1 server with the exact flags of a profile (default | latency | *-shipped)
  serve/tuning/              FlashInfer GEMM tactic store + Triton FLA autotune picks (RTX PRO 6000 Max-Q, 250 W);
                             serve.sh uses them by default (tuning/README.md)
  perf/sweep_aiperf.sh       AIPerf concurrency sweep (synthetic ISL/OSL, ignore_eos) -> Pareto points
  perf/build_prompt_sets.py  real-prompt sets (MMLU-Pro, LCB v6, HumanEval, MT-Bench; GPQA from your own export)
  perf/sd_bench.py           real-prompt benchmark: tok/s/user, tok/s/GPU, TTFT/TPOT, acceptance length per domain
  perf/replay_agentic.py     multi-turn agentic replay of OpenHands SWE trajectories (real growing prefixes)
  perf/mixload.sh            long-prompt TTFT probes under a concurrent decode load
  perf/gate20.py             20-prompt greedy numerics gate (exact-match / first divergence / |dlogprob| vs a baseline)
  perf/smi_sampler.sh        500 ms power / SM clock log (clock matching between arms)
  accuracy/run_gym.sh        NeMo Gym runner (MRCR-128K, MRCR >128K, GPQA-Diamond, IFBench, HLE)
  accuracy/benchmarks/       Gym configs for the locally built MRCR slices; no data
  accuracy/tools/            run-copy configs, per-run summary, IFBench stripped re-score, pooled A/B statistics, MRCR by
                             needle count, MRCR >128K template-length filter, load balancer
  requirements-client.txt    pinned client-side packages (AIPerf, OpenAI client, ...); not needed for the server
```

Client side needs **Python >= 3.11** (aiperf 0.13.0 does not install on 3.10; measured with 3.12). With
[uv](https://docs.astral.sh/uv/), which also fetches the interpreter if the system Python is older:
`uv venv --python 3.12 .venv-client && . .venv-client/bin/activate && uv pip install -r harness/requirements-client.txt`
(versions as measured; plain `python3.12 -m venv` + `pip install` works too). `build_prompt_sets.py` needs its own
`datasets<4` venv, see the comment in `requirements-client.txt`.

## 1. Build and serve

```bash
bash harness/serve/build_image.sh qw38/vllm:f5d93296-qw38        # CPU only; clones vLLM, applies patches/, builds the overlay image

hf download frida404/Qwen3.8-27B-NVFP4-LC32K --local-dir models/a2gdn-lc32k-mtpa4   # or build it: recipes/
(cd models/a2gdn-lc32k-mtpa4 && sha256sum -c ../../recipes/SHA256SUMS)                # 7 files OK
hf download maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal --revision bd7a934213c47a9e7ef69eef36bb3325f47fd1f1 \
  --local-dir models/dflash2-nvfp4
(cd models/dflash2-nvfp4 && sha256sum -c ../../recipes/SHA256SUMS-drafter)          # 3 files OK

sudo nvidia-smi -i 0 -pl 250                                     # the measured configuration: Max-Q at 250 W

PROFILE=default GPU=0 PORT=8000 MODEL_DIR=$PWD/models/a2gdn-lc32k-mtpa4 bash harness/serve/serve.sh
PROFILE=latency GPU=0 PORT=8000 MODEL_DIR=$PWD/models/a2gdn-lc32k-mtpa4 DRAFTER_DIR=$PWD/models/dflash2-nvfp4 bash harness/serve/serve.sh
```

**Tuning data.** `serve.sh` copies `serve/tuning/flashinfer` (FlashInfer GEMM tactics) and `serve/tuning/triton` (6 Triton
FLA picks) into the container at start. They were tuned on an RTX PRO 6000 Blackwell Max-Q at 250 W with the image
above. Without a matching tactic store the first boot autotunes during warmup and the tuner's workspaces are counted
as activations by the KV sizing: on the RTX PRO 6000 Max-Q (96 GB) the default profile loses ~24% of its KV cache
(~1.52M instead of ~2.0M tokens) and the latency profile fails to boot ("No available memory for the cache blocks"). On another GPU or build, `serve.sh`
warns; make a store with `SAVE_TACTICS=/dir` (serve/tuning/README.md). `FI_TACTICS=none` / `TRITON_SEED=none` turn
them off.

**Docker, file access and other hooks** (details in the header of `serve.sh`):
- The container runs as root and bind-mounts `MODEL_DIR` read-only as `/model` (and `DRAFTER_DIR` as `/drafter`).
  On a root-squashed NFS mount, container root is "nobody": every parent directory must be world-traversable, the
  shards world-readable (`chmod -R a+rX`), and the directory must hold real files: a shard symlinked to a path outside
  `MODEL_DIR` does not resolve inside the container. Otherwise copy the checkpoint to a local disk, or try `AS_USER=1`
  (runs the container as you; not GPU-tested).
- No host temp dirs are used: the per-server copies of the tuning data are made inside the container. `SAVE_TACTICS`
  and `COMPILE_CACHE` are host dirs the container writes to and later loads compiled kernels from, as root with host
  networking: put them on a local disk under a parent directory only you can enter (`chmod 700`). `serve.sh` creates
  their `flashinfer/`, `triton/`, `vllm/` subdirectories and, for the root container, makes only the ones it creates
  world-writable (needed on root-squashed NFS); the container writes with umask 0 (022 with `AS_USER=1`), so you can
  delete what it leaves. Relative paths are resolved against the current directory.
- `NAME` sets the container name (default `qw38-$PROFILE-g$GPU`, which collides if two checkouts serve on one GPU id);
  `DOCKER_ARGS` adds `docker run` flags, e.g. `DOCKER_ARGS="--cpuset-cpus 24-35"`.
- `COMPILE_CACHE=/dir` keeps the Triton / JIT kernel cache across boots (the profiles run `CompilationMode.NONE`, so
  there is no torch.compile cache to keep). Default profile on the RTX PRO 6000: boot to `/health` 221 s cold, 111 s
  on the second boot with the cache. The measurements booted cold. If `SAVE_TACTICS` is also set, the Triton
  cache goes to `SAVE_TACTICS/triton` instead of `COMPILE_CACHE/triton`.
- GB300 / arm64: `serve.sh` is docker-only and `build_image.sh` defaults to the amd64 base. The GB300 accuracy runs used
  the arm64 image of the same nightly (`BASE_IMAGE` digest in the header of `build_image.sh`) under Slurm, running
  `vllm serve` with the flags of docs/FINAL_CONFIG.md section 2 and the same environment variables.

Check the proof lines in the server log before measuring anything (a flag that silently does nothing is the most
common way to get a "no gain" result):

| profile | log lines that must appear |
|---|---|
| default | `K8V4 KV cache active: FP8 keys + NVFP4 values`; `VLLM_GDN_SPEC_STATE=recover uses the Triton GDN verify`; `GDN recover fold (QW38): deferred commit`; `QW38 smallgemm: GDN in_proj_ba M=2..4` (SM120 only) |
| latency | `VLLM_GDN_SPEC_STATE=recover uses the Triton GDN verify`; `GDN recover fold (QW38)`; `GDN recover verify v2 (QW38 verify-v2): mode B fold`; `QW38 hostgap` (SM120 only); `max_prefill_cudagraph_capture_tokens` 512 in the non-default args |
| both | `FlashInfer prefill: SM120 FP8 FMHA (cute-dsl-prims)` (SM120 only); `CUDAGraphMode.FULL_AND_PIECEWISE`; `'async_scheduling': True` |

## 2. Performance

All perf numbers in this repo are tok/s/user (x) vs tok/s/GPU (y) Pareto points from a concurrency sweep, with CUDA
graphs and async scheduling on, GPU at the 250 W cap, measured in A B B A order with 2 repeats per arm (METHODS.md).

```bash
. .venv-client/bin/activate                         # Python >= 3.11, see above (aiperf 0.13.0)
export MODEL=$PWD/models/a2gdn-lc32k-mtpa4           # tokenizer for AIPerf

# Synthetic Pareto (ISL/OSL fixed, ignore_eos, 3 x c requests, c warmup requests, seed 42)
bash harness/perf/sweep_aiperf.sh 8000 1024 1024  results/default/1k1k   "1 16 64 128"
bash harness/perf/sweep_aiperf.sh 8000 8192 1024  results/default/8k1k   "1 16 64"
bash harness/perf/sweep_aiperf.sh 8000 32768 1024 results/default/32k1k  "1 16 32 64"

# Real prompts (thinking on, T=1.0, top_p 0.95, max 2,048 output tokens, natural finish); n per concurrency as measured.
# The headline real-prompt numbers are the mean over the gpqa, mmlu_pro and lcb domains, so they need a GPQA export:
# GPQA is gated and not downloaded by the script; pass NeMo Gym's prepared file. This needs the Gym checkout and
# `gym eval prepare` of section 3 first (GYM = that checkout; section 3 clones it to ./Gym).
GYM=${GYM:-$PWD/Gym}; GPQA=$GYM/benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl
.venv-prompts/bin/python harness/perf/build_prompt_sets.py --out data/prompt_sets.json \
  --gpqa-jsonl $GPQA                                   # datasets<4 venv (requirements-client.txt)
for cn in 1:8 4:16 16:32 32:64; do IFS=: read c n <<< $cn
  python3 harness/perf/sd_bench.py --port 8000 --conc $c --n $n --max-tokens 2048 --domains gpqa,mmlu_pro,lcb \
    --sets data/prompt_sets.json --out results/latency/real_c$c.json; done

# Agentic replay: SWE-Gym/OpenHands-SFT-Trajectories (MIT), 8 sessions, c=4, ignore_eos-only length mode
hf download SWE-Gym/OpenHands-SFT-Trajectories --repo-type dataset --local-dir data/openhands
python3 harness/perf/replay_agentic.py --base http://localhost:8000 \
  --parquet data/openhands/data/train.success.oss-00000-of-00001.parquet \
  --tokenizer $MODEL --sessions 8 --conc 4 --max-turns 20 --max-ctx 98304 --per-turn-metrics --len-mode ieos \
  --skip-sessions 4 --out results/latency/agentic_c4

# Long-prompt TTFT under load: 32K probes while a 1K/1K c=32 background runs
bash harness/perf/mixload.sh 8000 1024:1024:32 32768:3 90 results/default/mixload_32k

# Power / clock log alongside any run
bash harness/perf/smi_sampler.sh 0 results/smi.csv results/SMI_STOP & ...; touch results/SMI_STOP
```

Notes:
- `--len-mode matched` (min_tokens + ignore_eos) inflates DFlash2 acceptance by about 11% on agentic replay; the
  default `ieos` is what every agentic number in RESULTS.md after 2026-11-14 used.
- Spec-decode results always report acceptance length (AL); `sd_bench.py` and `sweep_aiperf.sh` read it from the
  server's `/metrics` counters.
- Numerics gate for kernel changes (greedy, 20 prompts): run `gate20.py` on the baseline server, then on the candidate
  with `--compare`. Bit-exact claims also need pinned Triton/FlashInfer autotune (METHODS.md; `FI_TACTICS` /
  `TRITON_SEED` in `serve.sh`). The data file is JSONL with the fields `question`, `options_text`, `uuid` and
  `expected_answer` per row (NeMo Gym's prepared GPQA-diamond file has exactly these; the first 20 rows are used):
  ```bash
  # $GPQA as above (needs the Gym checkout and `gym eval prepare` of section 3)
  python3 harness/perf/gate20.py --base http://localhost:8000 --data $GPQA --out results/gate20/baseline.json
  # restart with the candidate build / flags, then:
  python3 harness/perf/gate20.py --base http://localhost:8000 --data $GPQA \
    --out results/gate20/candidate.json --compare results/gate20/baseline.json
  ```
  It prints exact-match rate, first-divergence position and mean |dlogprob| over the common prefix.

## 3. Accuracy (NeMo Gym)

The accuracy gates ran on 4 GPUs per job (one TP1 server per GPU behind `tools/lb.py`), 1,024 concurrent requests,
model-card sampling (T=1.0, top_p 0.95, thinking on, 65,536 max output tokens), `--max-model-len 262144`.

```bash
git clone --branch v0.7.0 https://github.com/NVIDIA-NeMo/Gym.git && cd Gym && uv venv && . .venv/bin/activate && uv sync --frozen && cd ..
# Built-in benchmarks: data is fetched by Gym (GPQA and HLE are gated on Hugging Face: accept their terms first)
for b in gpqa ifbench hle_verified; do (cd Gym && gym eval prepare --config benchmarks/$b/config.yaml); done
# MRCR slices (openai/mrcr, Qwen3.8 tokenizer): commands in the header of each benchmarks/mrcr_*/config.yaml. Gym's
# mrcr prepare.py needs transformers + tiktoken, which the Gym venv lacks: install them into a side directory
# (pip install --target <extra deps dir> transformers==5.22.2 tiktoken==0.14.0) and pass it as PYTHONPATH.
# The >128K slice also needs tools/mrcr_template_filter.py (155 -> 139 rows; see that config's header).

# 4 servers + load balancer on one 4-GPU node (accuracy runs used --max-model-len 262144; the perf profiles use 131072)
# Gate settings: MAXLEN=262144 SEED=0 EXTRA_ARGS=--enable-prefix-caching (latency profile: also MAX_SEQS=256)
for i in 0 1 2 3; do PORT=$((8100+i)) GPU=$i PROFILE=default MODEL_DIR=$PWD/models/a2gdn-lc32k-mtpa4 \
  MAXLEN=262144 SEED=0 EXTRA_ARGS=--enable-prefix-caching bash harness/serve/serve.sh & done
python3 harness/accuracy/tools/lb.py 8000 8100 8101 8102 8103 &

# Every candidate: MRCR-128K x2 first (stop on a clear failure), then GPQA x4 and IFBench x2; HLE for recurrent-state
# (GDN) or fine-tune changes; the >128K slice for changes that act beyond 128K tokens.
GYM=$PWD/Gym OUT=$PWD/results/acc/default-r1 bash harness/accuracy/run_gym.sh mrcr_qwen38_128k:2
GYM=$PWD/Gym OUT=$PWD/results/acc/default-r1 bash harness/accuracy/run_gym.sh gpqa:4 ifbench:2 hle_verified:1
GYM=$PWD/Gym OUT=$PWD/results/acc/default-r1 bash harness/accuracy/run_gym.sh mrcr_qwen38_196k_gt128k:3
```

(`MAXLEN=262144` raises `--max-model-len` from the perf setting of 131,072 so MRCR >128K prompts plus 64K-token outputs
fit; `SEED`, `MAX_SEQS` and `EXTRA_ARGS` are the other `serve.sh` hooks, see its header. `OUT` is made absolute by
`run_gym.sh`, which runs Gym from inside the Gym checkout.)

Smoke test first: `LIMIT=16 GYM=$PWD/Gym OUT=$PWD/results/acc/smoke bash harness/accuracy/run_gym.sh gpqa:1:8192
mrcr_qwen38_128k:1:8192` (`LIMIT` = Gym `--limit`, which counts tasks: LIMIT x repeats rollouts; `GYM_EXTRA` passes further Gym arguments).
`run_gym.sh` exits non-zero if any benchmark failed or was skipped; each benchmark's log is `$OUT/<bench>/gym.log`.

**IFBench is reported on whitespace-stripped text.** The qwen3 reasoning parser leaves the `"\n\n"` that follows
`</think>` at the start of the message content, and Gym's strict IFBench checks run on that raw text, so instructions
about how the answer starts or is formatted can fail on the leading whitespace alone. After each IFBench run,
`tools/summarize.py` calls `tools/ifbench_rescore.py`, which re-runs Gym's own checker on the raw text (this equals
Gym's reward) and on the stripped text, and stores both in `$OUT/ifbench/summary.json` under `ifbench_rescore`
(`raw` / `stripped`, key `gym_reward(fraction,strict)`). RESULTS.md quotes "stripped / raw" or the stripped number.
It needs Gym's ifbench resources-server venv (`Gym/resources_servers/ifbench/.venv`, created by the first IFBench run;
CPU only). By hand: `(cd Gym/resources_servers/ifbench && .venv/bin/python
$REPO/harness/accuracy/tools/ifbench_rescore.py $OUT/ifbench/rollouts.jsonl)`.

Comparing configurations: run baseline and candidate jobs interleaved (A B A B ...) and compare against pooled
baseline runs. Pass rule (METHODS.md, "Accuracy standard"): MRCR-128K >= -1.5 and GPQA-Diamond >= -1.0 vs the pooled
baselines; IFBench acts only on a large loss; borderline results get more runs. The >128K slice and HLE are noisy job to
job, so judge them with the run-level SD over several jobs per arm:

```bash
python3 harness/accuracy/tools/slice_pool.py mrcr_qwen38_196k_gt128k base base=results/acc/b1,results/acc/b2,results/acc/b3 cand=results/acc/c1,results/acc/c2,results/acc/c3
python3 harness/accuracy/tools/mrcr_by_needle.py results/acc/b1/mrcr_qwen38_128k results/acc/c1/mrcr_qwen38_128k   # A, B
```

Benchmark licenses and access terms are the benchmark owners' (GPQA and HLE are gated; GPQA asks that examples not be
reposted). This repo contains only scores and these fetch commands.
