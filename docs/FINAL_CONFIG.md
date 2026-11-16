# Final serving configuration (Qwen3.8-27B-NVFP4-SingleGPU-Serving)

Qwen3.8-27B NVFP4 on one GPU. Perf target: **RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB, 250 W cap)**. Accuracy gate:
**GB300 (SM103)**. Status: **final release candidate, frozen 2026-11-15; final GB300 accuracy gate PASSED in both
profiles; soak and restart-determinism tests passed** (docs/RESULTS.md sections 1, 2.2, 3.2, 3.3).

Two profiles share one vLLM build and one checkpoint and differ only in flags and environment:

| | **DEFAULT (throughput)** | **LATENCY** |
|---|---|---|
| use | general serving, any concurrency up to 256 | interactive / agentic, **<= 16 concurrent requests** |
| speculative decoding | MTP head, probabilistic drafts, k3 below 33 running requests, k1 above (`[[1,32,3],[33,1024,1]]`) | DFlash2-NVFP4 drafter, probabilistic drafts, k7 |
| GDN spec state | recover + fold (`VLLM_GDN_SPEC_STATE=recover`, `VLLM_GDN_RECOVER_FOLD=1`) | recover + fold + verify v2 stage B (also `VLLM_GDN_VERIFY_V2=1`) |
| KV cache | K8V4 (FP8 keys + NVFP4 values) | fp8_e4m3 |
| max-num-seqs / max-num-batched-tokens | 256 / 8192 | 32 / 32768, prefill CUDA-graph capture cap 512 |
| KV capacity (tokens) | 2,006,295 (2,004,805 with the shipped tactic store, section 3; ~1.52M without any store) | 1,101,004 (1,091,472 with the shipped store; **fails to boot** without any store on the RTX PRO 6000 Max-Q, 96 GB) |

## 1. Exact versions

| item | value |
|---|---|
| vLLM base | nightly `f5d93296c8ec82b7e7dabdb308d24d43cb1907bf` (0.34.0rc1.dev206+gf5d93296c, torch 2.14.0+cu130, FlashInfer 0.7.3) |
| vLLM changes | `patches/` (46 patches). Measured build = this series plus two upstream mixed-KV PRs that the shipped profiles do not use; the series alone was verified bit-identical in serving (RESULTS.md 3.3) |
| image | `vllm/vllm-openai@sha256:789bd3c0b35711ddf086a19f743a9423ab26670781e1453d8ec55a3766f791d7` (amd64) with the patched Python files overlaid (`harness/serve/build_image.sh`) |
| GB300 image | arm64 build of the same nightly, `vllm/vllm-openai@sha256:c68112d825ebdc6c856ff2e5d22bf514d278194a734baeb3b5f76d1ea35c6355`, with the measured build's patched Python package on top, run under Slurm with `vllm serve` and the flags below (`BASE_IMAGE=` that digest in `build_image.sh`; that build path itself is untested on arm64) |
| tuning data | `harness/serve/tuning/` (FlashInfer GEMM tactic store + 6 Triton FLA picks, RTX PRO 6000 Max-Q at 250 W); used by `serve.sh` by default, see section 3 |
| checkpoint | `a2gdn-lc32k-mtpa4` (recipes/; output sha256 in `recipes/SHA256SUMS`) |
| drafter (latency) | `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal` @ `bd7a934213c47a9e7ef69eef36bb3325f47fd1f1` (sha256 of its files in `recipes/SHA256SUMS-drafter`; `model.safetensors` 2228b9b2…df26 = the Hub LFS sha256 at that revision) |

The overlay contains only Python and Triton changes; no rebuild of compiled extensions is needed.

## 2. Serve commands (one GPU, TP1)

`harness/serve/serve.sh` runs exactly these (`PROFILE=default|latency`).

**DEFAULT** (env `VLLM_KV_K8V4=1 VLLM_GDN_SPEC_STATE=recover VLLM_GDN_RECOVER_FOLD=1`; leave `VLLM_GDN_VERIFY_V2` unset):
```
vllm serve <a2gdn-lc32k-mtpa4> --served-model-name qw38 --tensor-parallel-size 1 \
  --kv-cache-dtype fp8_e4m3 --max-model-len 131072 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.85 --max-num-seqs 256 --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling \
  --mamba-ssm-cache-dtype int16 --block-size 1600 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","num_speculative_tokens_per_batch_size":[[1,32,3],[33,1024,1]]}'
```
(`VLLM_KV_K8V4=1` with `--kv-cache-dtype fp8_e4m3` selects the K8V4 cache; `--kv-cache-dtype k8v4` is equivalent.)

**LATENCY** (env `VLLM_GDN_SPEC_STATE=recover VLLM_GDN_RECOVER_FOLD=1 VLLM_GDN_VERIFY_V2=1`):
```
vllm serve <a2gdn-lc32k-mtpa4> --served-model-name qw38 --tensor-parallel-size 1 \
  --kv-cache-dtype fp8_e4m3 --max-model-len 131072 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.85 --max-num-seqs 32 --max-num-batched-tokens 32768 --enable-chunked-prefill --async-scheduling \
  --mamba-ssm-cache-dtype int16 --block-size 1600 \
  --speculative-config '{"method":"dflash","model":"<Qwen3.8-27B-DFlash2-NVFP4-RTNcal>","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}' \
  --compilation-config '{"max_prefill_cudagraph_capture_tokens":512}'
```
Route more than 16 concurrent requests to a default-profile server.

**Accuracy-gate servers (GB300)** used the same flags per profile with these differences: `--max-model-len 262144`,
`--seed 0`, `--enable-prefix-caching` passed explicitly, 4 TP1 servers per node behind `harness/accuracy/tools/lb.py`,
and the latency profile ran at `--max-num-seqs 256` for eval throughput (serve it at 32). With `serve.sh`:
`MAXLEN=262144 SEED=0 EXTRA_ARGS=--enable-prefix-caching` (plus `MAX_SEQS=256` for the latency profile). The default
profile's gate servers selected the cache as `--kv-cache-dtype k8v4`, which is equivalent to `VLLM_KV_K8V4=1` with
`fp8_e4m3`.

## 3. Switches and pinned caches

| switch | DEFAULT | LATENCY | effect / proof line |
|---|---|---|---|
| `VLLM_GDN_SPEC_STATE=recover` | on | on | one GDN state page per request in spec decode; `VLLM_GDN_SPEC_STATE=recover uses the Triton GDN verify` |
| `VLLM_GDN_RECOVER_FOLD=1` | on | on | deferred commit folded into the next verify; `GDN recover fold (QW38): deferred commit` |
| `VLLM_GDN_VERIFY_V2=1` | **unset** | on | split-tf32 verify stage B, only at >= 6 tokens per spec row (it loses at <= 4, i.e. MTP); `mode B fold (BV 64, warps 4, dot split)` |
| `VLLM_KV_K8V4=1` | on | off | FP8 keys + NVFP4 values; `K8V4 KV cache active` |
| small GEMM (automatic) | active at M=2-4 | not used | SM120 only; `QW38 smallgemm` |
| host gap (automatic) | not used | active | SM120 / XQA only; `QW38 hostgap` |

**Tuning data (shipped in `harness/serve/tuning/`, used by `serve.sh` by default):**

**On the RTX PRO 6000 Max-Q (96 GB), cold boots without a FlashInfer tactic store lose KV capacity (default profile) or
do not boot (latency profile).**
Without a store, the first boot autotunes the NVFP4/FP8 GEMMs during warmup. The tuner's workspaces stay in the PyTorch
caching allocator and vLLM's KV sizing counts them as transient activations: the default profile comes up with
~1.52M KV tokens instead of ~2.0M (-24%; ~15 GiB counted as activations instead of ~2 GiB), and the latency profile
(MNBT 32,768) runs the tuner out of memory at M up to 32,768 and then fails with "No available memory for the cache
blocks". A store saved from a default-profile boot does not fix the latency profile. On GB300 (288 GB) cold boots of
both profiles came up (accuracy-gate runs and the release end-to-end test). Boot with a store that matches
your GPU and build; `serve.sh` passes the shipped one unless `FI_TACTICS=none`.

1. **FlashInfer GEMM tactics** (`tuning/flashinfer`, `FI_TACTICS` in `serve.sh` -> `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`,
   copied into the container at start because a server rewrites its store). 251 entries, tuned on an RTX PRO 6000
   Blackwell Max-Q at 250 W with the image of section 1 (FlashInfer 0.7.3, torch 2.14.0, CUDA 13.0; the environment is
   in its `manifest.json`). A store tuned on the published checkpoint lacks the NVFP4 shapes this checkpoint adds (GDN
   in_proj_qkvz N16384 x K5120, out_proj N5120 x K6144 and others), so those re-tune at every boot and the picks vary.
   The shipped store adds fixed picks for all of them (modal pick over 174 boots; best-energy picks for qkvz /
   out_proj at M <= 256): 0 tactics re-tuned at boot (log: "Preloaded 251/251 managed cache entries").
2. **Triton (FLA prefill) autotune** (`tuning/triton`, `TRITON_SEED` in `serve.sh` -> `TRITON_CACHE_DIR`). The build's 6
   `<hash>/<kernel>.autotune.json` files (chunk_local_cumsum_scalar, chunk_scaled_dot_kkt_fwd,
   merge_16x16_to_64x64_inverse, recompute_w_u_fwd, chunk_gated_delta_rule_fwd_h_blockdim64, chunk_fwd_o). Kernels
   recompile; the picks are kept. Unpinned, two otherwise identical builds can differ at |dlogprob| ~4e-2 from about
   token 3. These do not affect memory.

KV capacities with this store: 2,004,805 tokens (default) and 1,091,472 (latency), in every determinism boot and again
in the release end-to-end test, both verified with the internal copy of this store (passed to the earlier `serve.sh`
as `FI_TACTICS`); the boot through the shipped default path is re-tested in the release re-test. The perf numbers in RESULTS.md were
measured with an earlier, 138-entry subset of this store that re-tuned only the new NVFP4 shapes at every boot; those
servers (warm compile caches) reported the KV capacities without parentheses in the table at the top. A cold-booted
default server (no store) measured within 0.3% of the published 1K/1K c=16 throughput in the end-to-end test.

The data only applies to the GPU and build it was tuned on; elsewhere FlashInfer and Triton are expected to ignore it
and tune at boot (`serve.sh` warns when the GPU name differs). To make your own: boot with `SAVE_TACTICS=/dir`, then
pass `FI_TACTICS=/dir/flashinfer TRITON_SEED=/dir/triton` to every later server (harness/serve/tuning/README.md). We
have no tested recipe for a store that lets the latency profile cold-boot on a GPU where it runs out of memory.

With both pinned (the shipped data), 3 cold restarts per profile were bit-identical (RESULTS.md 3.3).

## 4. Accuracy

Final GB300 gate vs the previous build: PASS in both profiles; table in RESULTS.md 3.2. Gym commands and data
preparation: harness/accuracy (NeMo Gym v0.7.0; T=1.0, top_p 0.95, thinking on, 65,536 max output tokens, 1,024
concurrent requests per 4-GPU job).

## 5. Perf headline

RESULTS.md section 1 (vs the model-card recipe + MTP, same session) and section 2.2 (vs the previous build). Method:
AIPerf synthetic (chat, streaming, ignore_eos, fixed ISL/OSL, per-cell seeds), real prompts (`sd_bench.py`), OpenHands
agentic replay (`--len-mode ieos`), interleaved arms on one GPU, 2 repeats, un-profiled, 500 ms power/clock sampling,
proof lines checked in every server log. CUDA graphs FULL_AND_PIECEWISE, async scheduling on, every cell at the 250 W
cap. Pareto plots: `docs/img/pareto_default.png`, `docs/img/pareto_latency.png`.

## 6. Known limits

- **Power cap.** The card sits at 250 W in every cell including c=1. Gains are energy-per-step gains; a kernel time cut
  that does not cut energy does not show end to end. Conclusions come from un-profiled runs.
- **Cold boots.** On the RTX PRO 6000 Max-Q (96 GB), without a matching FlashInfer tactic store the latency profile
  does not boot and the default profile loses ~24% of its KV cache (section 3). The shipped store matches only the RTX PRO 6000 Max-Q and the image above.
- **Startup session modes.** Without the extended tactic store, concurrency-1 outputs can fall into one of >= 2
  ulp-level startup modes; bit-exact claims then need >= 2 sessions per arm.
- **SM10x stage-A guard.** SM103 compiles the fp32-state verify-v2 stage A with different FMA pairing, so stage A is
  guarded to the int16 path on SM10x. Production uses int16 state and stage B (SM103 preflight PASS).
- **SM120-only pieces.** Small GEMM and host gap are no-ops on GB300.
- **Latency profile <= 16 concurrent.** It wins at c <= 16 and loses at c=32 against the default profile.
- **Default k-schedule cliff.** MTP drops k3 -> k1 above 32 running requests; at 32K c=64 per-user tok/s falls while
  tok/s/GPU rises. A k2 band did not pay at this cap.
- **MRCR-128K** is about 1 point below the published checkpoint (inside the -1.5 gate).
- **Untested:** TP > 1, multi-GPU, GB300 perf for this build, other GPUs, max-model-len > 131072 on the RTX PRO 6000
  (the GB300 gate ran at 262,144), agentic accuracy.
- **Other GPUs.** Only the RTX PRO 6000 Blackwell Max-Q (SM120, perf + serving) and GB300 (SM103, accuracy, Slurm with
  the arm64 image) were run. The SM120-only pieces (small GEMM, host gap, FP8 prefill FMHA) are no-ops elsewhere; the
  tuning data does not apply (cold boot, see above); the k-schedule, MNBT and the 16-request latency limit were tuned
  for this card at 250 W. Expect to re-tune, and check the proof lines (harness/README.md) on any other GPU.
