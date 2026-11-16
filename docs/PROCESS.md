# How the work was done

This project ran from 2026-11-12 05:00 UTC to 2026-11-15 ~20:30 UTC by the git history (2026-11-11 evening to
2026-11-15, US Pacific), about three and a half days, as one human (the author) directing a lead agent and a set of
specialist agents. The results are in RESULTS.md. This page is about the method: how the benchmark settings changed and
why, how the two serving profiles came about, how the long investigations ran, and the mechanics that kept the claims
honest.

Evidence tags as in RESULTS.md: **[VERIFIED]**, **[INDICATIVE]**, **[HYPOTHESIS]**.

## 1. Organization

- **Lead agent** owns the plan, the evidence-tagged summary of conclusions, assignments and node budgets. It proposes;
  the human decides anything that ships, costs training compute, or changes scope.
- **Specialist agents**, one per workstream: serving perf, eval (accuracy gates), spec-decode, quantization,
  long-context calibration, GDN-state kernels, KV-cache kernel, QAD, and short-lived ones per
  candidate (ship-v4, ship-v5, verify-v2, small-GEMM, host-gap, kernel audit). Up to ~10 ran at once.
- **One append-only worklog per agent** (what ran, exact command and commit, result path, outcome, and an explicit
  NOT TESTED list). **One summary file of conclusions only**, edited by the lead, every line tagged
  [VERIFIED] / [INDICATIVE] / [HYPOTHESIS]. A tag is never upgraded without new evidence; wrong claims are struck
  through and marked RETRACTED, not deleted.
- **One vLLM branch per feature** off a fixed upstream nightly; integration branches ("ship-vN") are assembled only
  from features that passed both a perf A/B and the accuracy gate.
- **User-gated decisions:** what ships and when; training budgets (QAD); licenses (evaluation-only
  drafters, third-party fine-tunes); benchmark scope (which gates exist); compute allowance (4 then 8 GB300 nodes).
- **Standing instruction (11-13):** "when a trick does not show the expected gain, question the result first."
  Before concluding, check the serving flags, the proof line in the log, the right checkpoint and package, the
  baseline pairing, and the noise. This rule caught most of the stories in section 4.

## 2. How the benchmark settings evolved, and why

### 2.1 Accuracy gates

| when | change | why |
|---|---|---|
| 11-12 | GPQA-D x4 + IFBench x2 (raw and stripped), paired to a same-copy baseline, model sha256; no-go below -1 GPQA | GPQA reproduces the model card within 0.4; single IFBench runs vary +-2.4, so x2 and both scorings |
| 11-12 | report 64K-cap truncation rate and mean output length for state changes | a bf16 GDN state passed IFBench but failed GPQA by 3.66 with +14% output: the failure mode is drift into longer thinking |
| 11-12 | **MRCR-128K** joins the gate (limit -1.5 vs an FP8-KV pool) | uniform NVFP4 KV passed GPQA and IFBench but lost 2.3 on MRCR; short benchmarks are blind to long-context damage |
| 11-12 | graphwalks-128K and hle_verified added; RULER, LongBench v2 and an LLM-judged factuality set dropped | user scope decision: cheap, public, self-graded long-context and reasoning checks |
| 11-13 | **Gate v2:** per candidate = MRCR first (early kill), GPQA x4, IFBench x2, plus secondary checks later dropped by the gate calibration below (final form: METHODS.md) | MRCR had caught every NVFP4-extension failure, so run it first and stop early |
| 11-13 | borderline rule: MRCR within -1.0..-1.5 -> 2nd run; GPQA within 0.5 of -1 -> 3 runs; IFBench -1.5..-2.5 -> 2 more runs | single GPQA x4 has SE ~0.9; a DFlash2 "IFBench deficit" (-3.0) pooled to -1.17 +- 0.96 |
| 11-13 | **gate calibration** on known-bad configurations | to learn which benchmark detects which failure. MRCR caught all 13 long-context failures; GPQA caught state drift; IFBench caught fine-tune drift; HLE caught state drift strongly (z -7.6) via truncation; SciCode, MMMU and graphwalks caught nothing, so they are not part of the gate; HLE per candidate for state / fine-tune changes |
| 11-14 | **MRCR >128K slice** (139 tasks at 136-196K tokens, x3) | candidates now changed GDN state handling and drafting at very long context, which MRCR-128K does not reach |
| 11-15 | **>= 3-4 interleaved jobs per arm** on the >128K slice and HLE; run-level SD pooled over all configs | two identical configurations differed by ~3 on the slice and 2.5 on HLE (section 4.2) |

### 2.2 Perf settings

| when | change | why |
|---|---|---|
| 11-12 | perf target moved from GB300 to one RTX PRO 6000 Max-Q at 250 W (user); GB300 numbers kept as reference | deployment target; GB300 stays the accuracy platform |
| 11-12 | AIPerf concurrency-sweep Pareto (tok/s/user vs tok/s/GPU) at 1K/1K and 8K/1K, CUDA graphs + async confirmed in the log, a 20-prompt greedy numerics gate before any speedup counts, % of a calibrated roofline | one comparable picture across concurrency, not a single throughput number |
| 11-12 | **FlashInfer tactic pinning**: each server gets a private copy of one pinned GEMM-tactic store | two boots differed on 78 of 138 tactics, up to ~15% step-time noise at c=128 |
| 11-12 | **real prompts** (GPQA / MMLU-Pro / LCB, thinking on, 2K output) for every spec-decode claim | synthetic AIPerf text gives different acceptance lengths (5% AL drop on synthetic only, for a weight change) |
| 11-12 | **agentic replay** of OpenHands trajectories (real growing prefixes, prefix-cache hits) | the latency profile targets agent-style traffic; its `--max-ctx` filter was a no-op until fixed (no result changed) |
| 11-12 | port guard + per-GPU port ranges; never send load to a server you did not start | a benchmark hit a colleague's server on a shared port |
| 11-13 | perf scripts' leaked 1 Hz `nvidia-smi` sampler loops killed (82 of them) | marginal (<3%) results from before were re-checked |
| 11-13 | A/B/A with 3-4 reps, then **A B B A with 2 reps** and the "separated ranges" rule; kernel admission gate (unit tests, sanitizer, numerics, e2e A/B) | effects of 2-5% are the norm late in the project |
| 11-14 | agentic length mode **`ieos`** replaces `matched` (min_tokens + ignore_eos) | matched mode inflated DFlash2 AL by ~11% (MTP +-1%) and overstated the latency profile's agentic gain (same-session re-check: +22.7% -> +11.7% at c=1, +21.8% -> +7.9% at c=4) |
| 11-15 | **pinned Triton autotune** for bit-exact cross-build claims; >= 2 server sessions per arm | autotune picks and startup session modes changed numerics (section 4.4) |
| 11-15 | **un-profiled runs only** for perf conclusions; nsys only for attribution | the nsys host-gap reading (section 4.6) |
| 11-15 | **J/call rule**: rank kernel work by energy per call at the cap | K8V4 kernel -14.5% time converted to 0% e2e (section 4.7) |
| 11-15 | **ncu clock reset** after any ncu-wrapped server | a killed ncu left a GPU locked at ~1 GHz (section 4.8) |

## 3. How the two serving profiles came about

1. **Drafter choice depends on concurrency.** On GB300 and on the RTX PRO 6000, DFlash2 k7 was best at c=1 (2.2-2.6x)
   while MTP k3 was best at c >= 16, and DFlash2 dropped below no-spec at c=64 (0.69x) because 1+K full GDN states per
   request capped running requests at ~29. [VERIFIED]
2. **MTP depth sweep -> concurrency schedule.** k=4 was best at c=1, k=1 at c=64. A schedule (k3 while <= 32 requests
   run, k1 above) matched the best fixed k everywhere: c=64 2,152 tok/s/GPU vs MTP3 1,871. With probabilistic drafts,
   +13 / +15 / +8% more. It became the **default profile**. Raising the k3 threshold to 48, or adding a k2 band, raised AL
   25-36% but each step cost 25-43% more at the power cap: no gain. [VERIFIED]
3. **Recover spec state** (one GDN state page per request) removed DFlash2's capacity cap (c=64 1,029 -> 1,326,
   TTFT 19 -> 2.9 s) and made wide drafts viable at c=16-32. [VERIFIED]
4. **Head-to-head for the latency profile:** latency-k5 (MTP k5 at c=1 + reduced draft vocab) vs DFlash2-NVFP4 k7 +
   recover. DFlash2 won at every c <= 16 (+38.7% vs the then-default profile at c=1, +27.2% agentic) under a rule the user
   approved in advance. [INDICATIVE, n=2; earlier build, before the prefill-graph cap 512. The verified rerun with the
   cap gave +37.8% at c=1, RESULTS.md 2.1.]
5. **The c <= 16 cap.** DFlash2 vs the default profile crosses over between c=16 and c=32 (log-linear c ~= 18-21); at
   c=24/32 it is -10 / -16%. Traffic above 16 concurrent goes to the default profile. [INDICATIVE]
6. **The GB300 gate pulled it** (MRCR -9.8; section 4.1). A no-recover DFlash2 profile shipped in the meantime, then
   recover came back with the prefill graph-capture cap of 512 tokens. [VERIFIED]
7. **DFlash2 k-schedule:** a smaller fixed k never pays at c <= 16 (k5 -2 to -6%, k3 -3 to -24%); a per-batch k
   schedule gives +7.9 / +12.2% only at c=24 / 32, outside the profile's range. k7 stays. [VERIFIED]

## 4. Investigations

Each story: hypothesis -> control -> result -> what changed.

### 4.1 DFlash2 + recover loses 10 points of MRCR on GB300 only

- **Observation:** the latency-profile candidate passed every RTX PRO 6000 check but scored MRCR-128K -9.79 on GB300,
  with premature EOS inside `<think>` 157 vs 16. GPQA was within band. [INDICATIVE]
- **Isolation by controls** (one variable per GB300 job):
  - recover off: clean (-0.72) -> it follows recover x the a2gdn checkpoint; old package + a2gdn: still -10.1;
    max_num_seqs 16, 0 preemptions: still -10.6 -> not package, batch or preemption.
  - MTP (k3) + recover: clean -> the failing piece is the wide (8-token) verify.
  - prefix caching off: -8.9 -> not prefix caching.
  - **no recover, but forced onto the same Triton GDN path: -9.55** -> not recover itself; the generic Triton spec
    path at 8 tokens per row.
  - unit tests: generic and fused paths bit-identical at k=3 and k=7 -> not kernel math.
  - **`--enforce-eager`: +0.10** -> CUDA graphs / compile. FULL_DECODE_ONLY +0.80, **PIECEWISE -8.01** -> piecewise
    graphs.
- **Leads that were wrong and retracted:** b/a gating rows "changing" between projection and core op (they came only
  from warmup / capture dummy runs); draft tokens "changing" between propose and sample (request admissions). Every
  real step matched eager on SM120.
- **Key difference:** SM100 captures prefill piecewise graphs up to 8,192 tokens; SM120 caps them at 256-512. All
  failing steps were large padded mixed spec + prefill batches that SM120 never runs, hence no repro there.
- **Fix:** `--compilation-config {"max_prefill_cudagraph_capture_tokens": 512}` (pinned, so it does not depend on
  max_num_seqs). MRCR -0.31 (cap 128) and -0.94 (cap 512), GPQA +0.51, HLE +0.38. [VERIFIED]
- **Changed:** the latency profile pins the cap; the root mechanism inside those graphs is documented as open.

### 4.2 The ">128K fold failure" that was noise

- **Observation:** the default profile + recover + fold ("D3") scored -4.8 on the >128K slice vs K8V4 (t -3.8 at
  2 jobs). D2-only -2.9. NVFP4 KV -3.6. The shipped latency profile -2.9. [INDICATIVE at the time]
- **Controls:**
  - recover alone clean (+0.40) -> blame the fold or D2;
  - **bit-exact engine replay** of 175K-227K-token prompts on SM120 and SM103, with and without preemption: fold,
    D2 and the base all byte-identical to the shipped build (12/12, 16/16), while a no-recover arm diverged (the test
    can see ulp-level changes) [VERIFIED];
  - an exact repeat of the shipped latency profile moved -3.03 by itself -> single-run baselines cannot resolve
    2-3 point effects; a base change that is bit-identical on SM120 moved the score -1.4 to -1.9;
  - **powered A/B:** 6 interleaved jobs per arm. D3 vs its base -0.51 +- 0.77 (t -0.7); HLE -0.62 +- 0.75; an HLE A/A
    of identical configs differed by -2.50. [VERIFIED]
- **Changed:** the latency-profile ">128K loss" was retracted; D3 unblocked; the >=3-4 interleaved jobs rule.

### 4.3 The cap confound in recover's accuracy

- **Observation:** sampled MRCR on the RTX PRO 6000: recover -1.14 pp vs no recover (4 reps, ~1.5 SE, all reps negative).
- **Hypothesis:** the recover arm also carried the 512-token prefill graph cap; the no-recover arm ran the default.
- **Control:** cap 512 in both arms: recover vs no recover -0.28 +- 0.97; cap 512 vs 256 without recover -0.73.
  [VERIFIED]
- **Changed:** recover has no measurable accuracy cost at <= 128K; most of the deficit was the cap plus noise.

### 4.4 Autotune picks that change numerics

- **Observation:** a hardened fold build diverged from the shipped build from token 0 in a long-context replay.
- **Cause:** each package autotunes its Triton prefill kernels independently; one `chunk_local_cumsum` num_warps pick
  differed. With the autotune cache pinned, the builds were identical. A separate case: unpinned FlashInfer GEMM
  tactics (78/138 differed) gave ~15% step-time noise. [VERIFIED]
- **Changed:** pinned autotune for every bit-exact cross-build claim; every GB300 eval job autotunes fresh, which adds
  numeric variation between jobs (one reason for the job-level noise in 4.2).

### 4.5 SM103 fp32 FMA contraction

- **Observation:** the verify-v2 unit test passed on SM120 but failed 17 cases on GB300 (SM103), all stage A with an
  fp32 state; int16 cases passed.
- **Cause (PTX / SASS diff):** for the fp32 state, LLVM pairs and fuses the dot products differently (`fma.rn.f32x2`
  17 vs 25; FFMA2 vs FMUL2 + FADD2 = one vs two roundings) on sm_103a; SM120 has no FFMA2 and splits the pairs, so the
  results coincide there. The int16 path compiles identically. [VERIFIED]
- **Changed:** stage A guarded to int16 on SM10x; production uses int16 state and stage B, so the gate covered the
  shipped config; the final build's preflight passed 113/113 on SM103.

### 4.6 The host gap that was an nsys artifact

- **Observation (nsys):** 1.7-2.2 ms/step of host gap (7-10%) in the latency profile, and c=1 "not power-capped"
  (1,886 MHz, 220 W).
- **Fix found:** one per-step draft `seq_lens.cpu()` sync on the SM120 XQA path; removing it drops eager idle from
  1.0 to 0.04 ms and the CPU runs a step ahead. [VERIFIED mechanism]
- **But decode tok/s was unchanged** (0 / -0.9 / -1.1%). Un-profiled runs are capped at 250 W even at c=1
  (1,750-1,777 MHz); tracing slowed the step, lowered power and inflated the gap. Under the cap, removed idle becomes a
  lower clock. The fix still cut c=16 TTFT p50 by 31-40%. [VERIFIED]
- **Changed:** perf conclusions from un-profiled runs only; the fix shipped for its TTFT.

### 4.7 The K8V4 kernel win that did not convert

- **Observation:** a tuned K8V4 decode kernel: -14 to -20% kernel time at c=1/16, -14.5% at 32K c=16, bit-exact.
- **End-to-end (A B B A, clocks matched): 32K c=16 +0.7%, 32K c=64 -0.7%, 1K c=64 -0.7%.** The kernel cut time 14.5%
  but energy per call only ~6%; at the cap only J/call converts. An earlier "-8% J" from synthetic data was retracted
  with realistic KV data. [VERIFIED]
- **Same pattern:** int8 GDN state (kernel -15-20%, e2e +1.3%: the clock dropped 5% for the whole step); verify v2 (about
  1.5 points of its kernel saving lost to a 1.5% lower clock).
- **Changed:** the J/call rule; context-aware splits and a CuTe rewrite of that kernel were dropped.

### 4.8 A GPU left clock-locked at 1 GHz

- **Observation:** a kernel sweep ran unexpectedly slow on one GPU.
- **Cause:** ncu locks the SM clock to base while profiling and restores it only on a clean exit; the stop script
  removed ncu-wrapped servers with `docker rm -f`. The GPU stayed at ~1 GHz under load with no throttle reason.
- **Fix:** `ncu --clock-control reset`, load check at 1,702 MHz; the stop script now resets after any ncu run; the
  affected sweep was discarded. [VERIFIED]

### 4.9 The NVFP4 KV-cache verdict, twice revised

- 11-12: uniform NVFP4 KV on the published weights fails MRCR-128K (-2.32), growing with needle count; emulation shows
  the damage is **key** rounding (FP8 keys + NVFP4 values passes, -0.41) -> build the K8V4 kernel (shipped). [VERIFIED]
- 11-14: on the newer stack (different decode kernel), NVFP4 KV passed MRCR-128K x2 (-0.26) -> candidate again.
- 11-14: >128K slice -3.91 +- 1.64 (all 3 repeats below every baseline repeat) -> NO-GO.
- 11-15: the >128K baseline was a single run (section 4.2); in the pooled table NVFP4 KV is -1.38 vs the recover
  default (n.s.). Verdict downgraded to INDICATIVE; K8V4 stays because it is clean at every length and NVFP4 KV would
  also need an RTX PRO 6000-specific accuracy check (different decode kernel). [INDICATIVE]

### 4.10 QAD that taught the model to stop thinking

- Light distillation of the NVFP4 GDN projections toward BF16 (top-4096 logit KL) made MRCR worse with every step
  (-0.85 / -4.5 / -8.3 at 50/100/150 steps), with premature EOS inside `<think>` rising 7-16x.
- Teacher-forced P(<|im_end|> | mid-think): BF16 2.6e-8; student 85-780x higher. The teacher's END probability lies
  outside its top-4096, so only the residual mass was constrained and the student drifted onto END.
- Forcing special tokens into the KD support cut the drift 5-10x; lowering LR fixed the rest; the clean save then
  gave no gain (+0.18 MRCR). Closed as a miss, with the diagnostic kept. [VERIFIED mechanism]

## 5. Safety and hygiene rules, and the incidents behind them

| rule | incident |
|---|---|
| only GPUs assigned to the project, under a file lock held for the container's lifetime; kill processes by exact PID; remove only your own containers | shared multi-user host |
| port guard; never send load to a server you did not start | a sweep hit a colleague's server on a reused port (reported to the user) |
| CPU-only containers set `NVIDIA_VISIBLE_DEVICES=void` | the serving image exposes every GPU when started without `--gpus` |
| never `set -x` in scripts that read secrets; logs mode 600; tokens never on command lines | a traced prep script wrote a token into a log for under a minute (redacted, rotation recommended) |
| size scratch before enabling it; no per-layer activation checkpoints at full calibration size | a PTQ resumability option wrote 2.6 TB in an hour and filled a shared storage quota |
| long work runs as children of the job script, not overlap steps | three PTQs died when their job script exited |
| never edit a script a running shell has open (NFS) | queued runs died with "Stale file handle" |
| `ncu --clock-control reset` after ncu | section 4.8 |
| no benchmark text, rollouts or derived prompts outside the eval cluster; calibration and training data decontaminated (13-gram / 50-gram) | IFBench instruction text found inside a public training set during QAD data prep and excluded |
| evaluation-only and non-commercial artifacts (an EAGLE-4 drafter, a third-party fine-tune and its derivatives) stay internal | license terms |

## 6. Lessons

1. Under a power cap, energy per call, not time per call, decides whether a kernel matters.
2. Profilers change what they measure: attribute with nsys/ncu, conclude from un-profiled, clock-logged runs.
3. Short-context accuracy does not predict long-context accuracy; long-context calibration rescued a quantization
   layout that 2K calibration could not.
4. Paired per-task standard errors understate job-to-job variance on long, sampled benchmarks; repeat and interleave.
5. Bit-exactness is a property of the build and its autotune state, not only of the code.
6. Hardware differences (SM100 graph capture limits, SM103 FMA pairing) hide bugs from the platform you test on.
7. Write the rule down the day the failure happens.
