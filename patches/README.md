# vLLM patch series

**Upstream base:** vllm-project/vllm `f5d93296c8ec82b7e7dabdb308d24d43cb1907bf`
("[Bugfix][Frontend] Honor --response-role and echo in chat derender (#63357)", 2026-11-11), the nightly published as
`vllm/vllm-openai@sha256:789bd3c0b35711ddf086a19f743a9423ab26670781e1453d8ec55a3766f791d7`.

**Result:** the release-candidate serving build. 46 patches in 13 feature groups, applied in
directory order. All changes are Python / Triton / FlashInfer-JIT; nothing needs a CUDA build.

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout f5d93296c8ec82b7e7dabdb308d24d43cb1907bf
git am $(ls ../patches/[0-9]*/*.patch | sort)
```

What runs with no QW38 environment variable set and default CLI flags (the "how it is enabled" column below):
- **Always on:** group 03, a loader bugfix (it only changes behaviour for checkpoints with a quantized MTP head).
- **On by default, with an off switch:** 05 SM120 FP8 prefill (`VLLM_FLASHINFER_SM120_FP8_PREFILL=0`), 08 KV-grouping
  fix (`VLLM_KV_GROUP_FULL_PAD_FIX=0`) and 11 small GEMM (`VLLM_GDN_BA_SEQK=0`; it engages only on SM120 at M=2..4).
- **Automatic where the code path applies, no switch:** 12 host gap (SM120 trtllm-api/XQA non-causal draft batches,
  i.e. DFlash drafts).
- **Opt-in:** everything else, through a CLI flag (`--mamba-ssm-cache-dtype int16`, `--kv-cache-dtype k8v4`,
  `draft_vocab_ids`) or an environment variable (`VLLM_GDN_SPEC_STATE`, `VLLM_GDN_RECOVER_FOLD`, `VLLM_KV_K8V4`,
  `VLLM_GDN_VERIFY_V2`); groups 04, 09 and 13 only act together with the opt-in features they extend.
So on SM120 the build differs from upstream f5d93296 even with no variable set (FP8 prefill, KV grouping, small GEMM,
host gap); on other GPUs only 03 and 08 can change behaviour by default.

| group | patches | what it does | how it is enabled |
|---|---|---|---|
| 01 recover-spec-state | 0001-0006 | GDN speculative-decode state "recover" mode: verify writes small per-token records instead of K+1 full recurrent states per request, and a commit kernel advances the single state after acceptance. One state page per request instead of 1+K (lifts the running-request cap for wide drafts); prefix caching ("align" mode); records outside the Mamba page so the block size is unchanged; optional commit window W | `VLLM_GDN_SPEC_STATE=recover` (`VLLM_GDN_SPEC_STATE_W`) |
| 02 int16-gdn-state | 0007-0010 | int16 row-scaled GDN recurrent state (int16 codes + fp32 per-row scale, re-quantized every store with `cvt.rni`); packed decode, sigmoid-gating, Triton prefill paths and a Triton port of the fused MTP decode kernel. 0007 is rebased onto group 01 (its two conflict hunks resolved as in the measured build's merge) | `--mamba-ssm-cache-dtype int16` (serve with `--block-size 1600`) |
| 03 mtp-head-quant-load | 0011-0012 | Bugfix: Qwen3.5/3.8 MTP drafter lost the per-layer ModelOpt quant entries of `mtp.*` (the target's `hf_to_vllm_mapper` drops the prefix), so a quantized MTP head failed to load | always (bugfix) |
| 04 recover-spec-int16 | 0013-0014 | recover mode with the int16 state codec; offline test config | with 01 + 02 |
| 05 sm120-fp8-prefill | 0015-0018 | SM120 (RTX PRO 6000 / RTX 50) FlashInfer prefill with FP8 Q/K/V (cute-dsl-prims FMHA) and an HND KV layout; same E4M3 attention math as SM100's trtllm-gen path | **default on** for fp8 KV on SM120; `VLLM_FLASHINFER_SM120_FP8_PREFILL=0` turns it off |
| 06 k8v4-kv-cache | 0019-0031 | K8V4 KV cache: FP8 keys + NVFP4 values (Triton writer, decode/verify kernel, prefill gather to FP8 pages); exact FP16 dequant, register-broadcast V scales, hardware E2M1 conversion on SM100/SM120, SM120 split heuristic; ~1.28-1.30x KV tokens vs FP8 | `--kv-cache-dtype k8v4`, or `VLLM_KV_K8V4=1` with an fp8 cache dtype |
| 07 reduced-draft-vocab | 0032-0033 | reduced draft vocabulary for drafters that share the target lm_head (MTP, DFlash2): the draft lm_head runs over a token subset; lossless (the target verifies over the full vocabulary) | `speculative_config.draft_vocab_ids=<file>` (not used by the shipped profiles) |
| 08 kv-grouping-fix | 0034-0035 | hybrid KV-cache grouping: choose the group size by per-request block cost so a 5-layer drafter SWA bucket no longer pads the 16 full-attention layers (+10.6% KV tokens for the DFlash2 profile) | **default on**; `VLLM_KV_GROUP_FULL_PAD_FIX=0` restores upstream |
| 09 d2-commit-align-hint | 0036 | `tl.multiple_of` alignment hints on the recover commit kernel's address-table pointers (register pressure); bit-exact | with 01 |
| 10 recover-fold | 0037-0041 | deferred commit ("fold"): the recover commit runs inside the next verify instead of as its own kernel; pending records of non-spec rows are folded in every cache group; single-buffered u records halve the side buffer | `VLLM_GDN_RECOVER_FOLD=1` (with 01) |
| 11 smallgemm | 0042 | bit-exact sequential-K Triton GEMM for the BF16 GDN `in_proj_ba` at M=2..4 on SM120 (cuBLAS picked a single-warp wmma kernel, ~1.2 ms/step) | **default on** (`VLLM_GDN_BA_SEQK` defaults to 1): automatic on SM120 at M=2..4; `VLLM_GDN_BA_SEQK=0` disables |
| 12 hostgap | 0043-0044 | FlashInfer: skip the host `seq_lens` copy (a per-step device sync) for non-causal decode-only draft batches on the trtllm-api/XQA path | **automatic, no switch**, where the path applies (SM120 DFlash drafts) |
| 13 verify-v2 | 0045-0046 | GDN verify v2 stage B: chunked UT/WY verify with exact-operand split-tf32 MMAs on the fold path, used only at >= 6 tokens per spec row | `VLLM_GDN_VERIFY_V2=1` (`VLLM_GDN_VERIFY_V2B_MIN_T`, default 6) |

## Per-patch descriptions

| # | subject |
|---|---|
| 0001 | GDN spec-decode verify records + commit kernels (no per-draft state writes) |
| 0002 | `VLLM_GDN_SPEC_STATE=recover`: one state page per request in spec decode |
| 0003 | recover spec: cheaper verify checks, finite-input guards in tests |
| 0004 | recover spec: commit window W > 1 (`VLLM_GDN_SPEC_STATE_W`) |
| 0005 | recover spec: prefix caching (mamba_cache_mode "align"), W = 1 |
| 0006 | recover spec W=1: records outside the Mamba page (same block size) |
| 0007 | int16 row-scaled recurrent state (`--mamba-ssm-cache-dtype int16`) |
| 0008 | int16 state tests: allow FMA-contraction ulp differences |
| 0009 | int16 state store: single `cvt.rni` instead of rint + clamp |
| 0010 | int16 state on the fused MTP decode path (Triton port of the CUDA kernel) |
| 0011 | Bugfix: Qwen3.5 MTP keeps the per-layer quant entries of the `mtp.*` block |
| 0012 | Bugfix: same fix for the `vllm/models/qwen3_5/nvidia` MTP module |
| 0013 | recover spec: int16 row-scaled state (FMT=int16_row) |
| 0014 | recover spec caching test runs offline (synthetic local config) |
| 0015 | FlashInfer: opt-in SM120 FP8 prefill (cute-dsl-prims FMHA) |
| 0016 | FlashInfer: SM120 FP8 prefill flag selects an HND KV layout |
| 0017 | FlashInfer SM120 FP8 prefill: gather compact K/V planes |
| 0018 | FlashInfer: SM120 FP8 prefill on by default |
| 0019 | K8V4 KV cache ops: FP8 keys + NVFP4 values (Triton writer, decode, prefill gather) |
| 0020 | K8V4 tests: elementwise bf16 tolerance |
| 0021 | K8V4 tests: bound bf16 P rounding by sum p\|v\| |
| 0022 | K8V4 decode: FP16 exact dequant, split-half nibble packing, multi-page tiles |
| 0023 | K8V4 decode: optional register broadcast of V block scales |
| 0024 | FlashInfer: `--kv-cache-dtype k8v4` |
| 0025 | FlashInfer k8v4: log line when the K8V4 cache is active |
| 0026 | K8V4 decode: register-broadcast V scales by default |
| 0027 | `VLLM_KV_K8V4` env switch for the k8v4 KV cache |
| 0028 | K8V4 decode: optional hardware E2M1x2 -> f16x2 conversion |
| 0029 | K8V4 decode: hardware E2M1 conversion by default on SM100/SM120 |
| 0030 | K8V4 decode: opt-in FP8 Q.K (`VLLM_K8V4_QK_FP8=1`) |
| 0031 | K8V4 decode: split heuristic from an SM120 sweep |
| 0032 | Reduced draft vocabulary for drafters sharing the target lm_head |
| 0033 | Draft vocab: speculator class name fix; DFlash2 candidate top-k over the subset |
| 0034 | KV grouping: avoid padding full-attention buckets |
| 0035 | KV grouping v2: group size by per-request slot-block cost |
| 0036 | gdn recover commit: alignment hint on address-table pointers (D2) |
| 0037 | gdn recover fold: deferred commit into the next verify (`VLLM_GDN_RECOVER_FOLD=1`) |
| 0038 | fold test: zero recycled boundary blocks in both paths |
| 0039 | fold test: poison never-written blocks in the fold path only |
| 0040 | fold: fold (not drop) pending records of non-spec rows, in every cache group |
| 0041 | fold: single-buffered u records (halves the u side buffer) |
| 0042 | GDN in_proj_ba: bit-exact sequential-K Triton GEMM for M=2..4 on SM120 |
| 0043 | flashinfer: no host seq_lens read for non-causal trtllm-api decode-only (draft) batches |
| 0044 | hostgap: one-time proof log line |
| 0045 | verify v2 stage B on the fold branch (`VLLM_GDN_VERIFY_V2=1`), incl. the deferred-commit verify |
| 0046 | verify v2: stage B only at >= 6 tokens per spec row (`VLLM_GDN_VERIFY_V2B_MIN_T`) |

## What is not in the series

- **Not ours / not needed:** the measured integration branch also carried two open upstream PRs for ModelOpt
  layer-wise mixed KV caches (vllm-project/vllm #56156 and #56116). The shipped profiles do not use a mixed per-layer
  KV map, and the series applies and byte-compiles without them (the resulting tree equals the measured tree minus
  those PRs and a few comment / docstring lines scrubbed of internal names). Every measurement in docs/RESULTS.md was taken on a build that also contained them.
- **Net-zero pair dropped:** an int16 decode-tile retune and its revert (neutral in the bench).
- **Experimental branches, not shipped:** int8 GDN state with stochastic rounding, FlashInfer fused GDN decode port,
  lm_head autotune, sampler fast path, TRT-LLM small-M GEMV port, K8V4 stage-1 v2 tuning, CuTe DSL commit port,
  format-disaggregated NVFP4 decode (W4A16), MTP+suffix hybrid drafting, adaptive-k drafting, DFlash2-FP8 loader fix,
  and all debug/instrumentation branches.

## Tests

Unit tests ship with their features (`tests/kernels/mamba/test_gdn_*`, `tests/kernels/attention/test_k8v4_kv.py`,
`tests/v1/attention/test_gdn_recover_*`, `tests/v1/core/test_mamba_recover_spec_caching.py`). The release-candidate
gates were: recover/fold unit gate 25/25 (fold result == eager result bitwise, with poisoned never-written blocks),
fp64-referenced verify-v2 unit test, small-GEMM 8,640/8,640 bit-identical, and an SM103 (GB300) preflight of 113 cases.

All patches are against Apache-2.0 vLLM and are offered under Apache-2.0 (see LICENSE and NOTICE).
