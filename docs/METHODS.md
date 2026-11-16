# Measurement methods

These rules were written down as the project hit the failure each one prevents. docs/PROCESS.md tells those stories;
this page is the checklist.

## Hardware and power

- **Perf target:** one RTX PRO 6000 Blackwell **Max-Q** (96 GB, SM120), board power limit **250 W** (the card's default
  is 300 W). Calibrated on the card: read bandwidth 1,634 GB/s; dense NVFP4 GEMM 913-956 TFLOPS, FP8 362-433,
  BF16 212-223 (M=8192) at the cap.
- **The card sits on the power cap in almost every serving cell, including c=1** (1,750-1,777 MHz at c=1, ~1,165-1,280
  MHz under heavy load; max boost 3,090). Consequences:
  - A kernel that gets faster but does not use less **energy per call (J/call)** mostly lowers the clock for every
    other kernel. Rank kernel ideas by J/call at the cap, then confirm end-to-end. Example: a K8V4 decode kernel cut
    time 14.5% but J/call only ~6%, and end-to-end was neutral.
  - Removing idle time (host gaps) does not raise throughput under the cap; it lowers the clock.
  - Every A/B logs power and SM clock every 500 ms (`harness/perf/smi_sampler.sh`); an A/B with mismatched clocks
    between arms is re-run.
- **Accuracy target:** GB300 (SM103), 4 GPUs per job, one TP1 server per GPU.

## Perf standard

1. **AIPerf concurrency-sweep Pareto**: x = tok/s/user, y = tok/s/GPU, at 1K/1K and 8K/1K (32K/1K and 124K for
   long-context items). Synthetic prompts, fixed ISL/OSL, `ignore_eos` + `min_tokens`, seed 42, 3 x c requests with c
   warmup requests (`sweep_aiperf.sh`).
2. **Real prompts** (`sd_bench.py`): thinking on, T=1.0, top_p 0.95, max 2,048 tokens, natural finish; GPQA /
   MMLU-Pro / LCB domains. Synthetic-prompt acceptance length differs from real text (it dropped 5% on synthetic
   prompts for a weight change that left real-text AL unchanged), so spec-decode claims rest on real prompts.
3. **Agentic replay** (`replay_agentic.py`): real OpenHands multi-turn prefixes, length mode `ieos` (ignore_eos
   only). The older `matched` mode (min_tokens + ignore_eos) inflated DFlash2 AL by ~11% and is retired.
4. **CUDA graphs and async scheduling confirmed in the server log**; the feature's own proof line present in the
   candidate arm only.
5. **Spec-decode results always report acceptance length** (from `/metrics`).
6. **Order and repeats:** A B B A (or A/B/A 4/3 for small effects), 2 repeats per arm minimum, same GPU, same session.
   A result counts as a gain only if the arms' repeat ranges are separated. Cross-GPU or cross-session comparisons
   are INDICATIVE.
7. **Numerics gate before any speedup counts:** 20 GPQA prompts, greedy (`gate20.py`); report pred_agree, exact
   matches, first divergence and mean |dlogprob| vs the baseline arm.
8. **Un-profiled runs only for perf conclusions.** nsys tracing slows the CPU-serial path and lowers power draw, so
   it overstated the host gap (1.7-2.2 ms/step) and showed c=1 as "not power-capped" when real runs are. Use nsys to
   attribute time, then confirm the effect without the profiler.
9. **ncu locks the SM clock** (default `--clock-control base`, ~1 GHz) and restores it only on a clean exit. After any
   ncu-wrapped run, run `ncu --clock-control reset` on that GPU and check that a load reaches >= 1.7 GHz before trusting
   timings on it.

## Autotune pinning

- **FlashInfer GEMM tactics** are re-tuned at every boot under 250 W clock swings: two fresh stores differed on 78 of
  138 FP8/FP4 entries (e.g. GDN in_proj_qkvz 83 us vs 162 us), up to ~15% step-time noise at c=128. Every server gets a
  private copy of one pinned reference store (`FI_TACTICS` in `serve.sh`), and runs record their tactics.
- **Triton autotune** is per package: a different `chunk_local_cumsum` num_warps pick made two otherwise identical
  builds diverge from token 0 (|dlogprob| ~4e-2). **Bit-exact claims across builds require pinned Triton autotune
  caches** (copy one build's autotune results into the other's cache; `TRITON_SEED` in `serve.sh`). The tactic store
  and Triton picks of the final runs ship in `harness/serve/tuning/` and `serve.sh` uses them by default
  (FINAL_CONFIG.md, section 3).
- **A tactic store is also needed to boot at all.** Without one the first boot autotunes during warmup, and the tuner's
  workspaces are counted as activations by the KV sizing. On the RTX PRO 6000 Max-Q (96 GB) the default profile loses
  ~24% of its KV cache and the latency profile (MNBT 32,768) fails to boot ("No available memory for the cache
  blocks"); on GB300 (288 GB) cold boots came up.
- Startup "session modes": the same build can start in one of two numerically distinct modes at concurrency 1 (seen in
  both profiles). Exactness claims need >= 2 server sessions per arm.

## Accuracy standard

- Sampling = model card: T=1.0, top_p 0.95, thinking on (`enable_thinking: true`), 65,536 max output tokens,
  `--max-model-len 262144`. NeMo Gym v0.7.0, 1,024 concurrent requests per 4-GPU job.
- Every run records: model sha256, server flags, the feature's proof line on every server, truncation rate, mean output
  tokens, no-answer rate, empties split into truncated vs premature EOS inside `<think>`, and AL for spec runs.
- **Gate, run for every candidate** (weights, KV cache, GDN state, spec-decode path):

  | benchmark | runs | pass rule (delta vs pooled baselines) |
  |---|---|---|
  | MRCR-128K (first; stop early on a clear failure) | x2 | >= -1.5 |
  | GPQA-Diamond | x4 | >= -1.0 |
  | IFBench (raw and whitespace-stripped scoring) | x2 | large-effect detector: act on <= -2.5; 2 more runs at -1.5..-2.5 |
  | HLE (hle_verified) | x1 | only for recurrent-state (GDN) or fine-tune changes |

  Borderline results (within about 0.5 of a limit) get more runs before a decision. The MRCR >128K slice (139 tasks,
  136-196K tokens; `harness/accuracy/benchmarks/mrcr_qwen38_196k_gt128k`) is the long-context check for changes that
  act beyond 128K; it is noisy job to job (see below), so compare it with the run-level SD over several interleaved
  jobs per arm.
- **Pools:** compare against pooled baselines (several runs of the baseline configuration), not a single same-day run,
  and pair by task where possible.
- **Noise lesson (>128K slice and HLE):** paired-by-task rollout SEs understate job-to-job variance. Two identical
  configurations differed by ~3 points on the >128K slice and by 2.5 on HLE (significant at 95% by the paired SE).
  Decisions there use the run-level SD pooled over all configurations (1.32 points on the slice).
- **Gate calibration ("benchmarking the benchmarks"; run once, on known-bad configurations):** each candidate
  benchmark was run on configurations known to be damaged (long-context quantization failures, GDN-state drift,
  fine-tune / format drift) to learn which benchmark detects which failure. MRCR caught all 13 long-context failures
  (most signal per node-hour), GPQA x4 caught state drift, IFBench caught fine-tune/format drift, HLE caught state drift
  strongly (z -7.6 via length inflation into the 64K cap). SciCode, MMMU-Pro and graphwalks-128K caught none of the
  known-bad configurations, so they are not part of the gate. To repeat it, run the gate benchmarks with `run_gym.sh`
  on a known-bad configuration (e.g. a bf16 GDN state or a 2K-calibrated NVFP4 checkpoint) next to its baseline and compare with `tools/pool_ab.py` / `tools/slice_pool.py`.

## Benchmark-data hygiene

- No benchmark text, answers, rollouts or derived prompts leave the eval cluster; only scores and fetch commands.
- Calibration and training data are decontaminated against every eval set (13-gram and 50-gram rule,
  recipes/README.md).
