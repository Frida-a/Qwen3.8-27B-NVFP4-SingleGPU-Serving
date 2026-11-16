# Tuning data for serve.sh

Kernel-selection data measured on the release hardware. `harness/serve/serve.sh` uses both by default. They are data
only (tactic ids, Triton launch configs and timings); no code.

| path | what | used as |
|---|---|---|
| `flashinfer/v2/<env hash>/` | FlashInfer GEMM autotune store: 251 entries (182 NVFP4 CUTLASS, 61 FP8 cuDNN, 8 FP8 cuBLAS), one JSON file per (op, runner, shape bucket) with the chosen `tactic`, plus `manifest.json` (the environment it is valid for) | `FI_TACTICS` (default) -> `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` |
| `triton/<hash>/<kernel>.autotune.json` | Triton autotune results of the 6 FLA prefill kernels used by the GDN layers (chunk_local_cumsum_scalar, chunk_scaled_dot_kkt_fwd, merge_16x16_to_64x64_inverse, recompute_w_u_fwd, chunk_gated_delta_rule_fwd_h_blockdim64, chunk_fwd_o). The `<hash>` directory names are Triton's cache keys and must be kept | `TRITON_SEED` (default) -> `TRITON_CACHE_DIR` |
| `flashinfer-picks.tsv` | how each of the 113 entries added for this checkpoint was picked: `modal` = most frequent pick over 174 boots, `energy` = best energy per step in a tactic sweep (GDN in_proj_qkvz / out_proj at M <= 256); with the observed pick counts | documentation only |

## Valid for

- **GPU:** NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (SM120, 96 GB) at a 250 W board limit.
- **Build:** the image from `harness/serve/build_image.sh` (vLLM nightly `f5d93296`, amd64): FlashInfer 0.7.3,
  torch 2.14.0, CUDA 13.0, cuBLAS 13.1.1, cuDNN 9.24 (92400), CUTLASS DSL 4.8.0, CPython 3.12 (all recorded in
  `flashinfer/v2/*/manifest.json`).
- **Checkpoint:** `a2gdn-lc32k-mtpa4` (the NVFP4 GDN projections of this checkpoint add 113 of the 251 entries).

FlashInfer keys the store to the environment in `manifest.json`, and Triton keys its files by kernel source, GPU
architecture and compiler, so on another GPU or build expect the data to be ignored and the server to autotune as if it
were absent (not tested). `serve.sh` prints a warning when the GPU name differs from the store's.

## Why it matters (not only for reproducibility)

Without a matching store, the first boot autotunes the NVFP4/FP8 GEMMs during warmup. The tuner's workspaces stay
in the PyTorch caching allocator, and vLLM's KV sizing counts them as transient activations. Measured on the RTX PRO
6000 Max-Q (96 GB); on GB300 (288 GB) cold boots of both profiles came up:

| profile | with this store | cold (no store) |
|---|---|---|
| default | 2,004,805 KV tokens, ~2 GiB counted as activations | ~1.52M KV tokens (-24%), ~15 GiB counted as activations |
| latency (MNBT 32768) | 1,091,472 KV tokens, ~8 GiB | **does not boot**: the tuner tries M up to 32,768, runs out of memory, then "No available memory for the cache blocks" |

A store saved from a default-profile boot does not fix the latency profile (it tunes larger M buckets). With the store,
0 entries are re-tuned at boot (log: "Preloaded 251/251 managed cache entries"), and with the Triton picks also pinned,
cold restarts are bit-identical (docs/RESULTS.md 3.3).

The Triton picks do not affect memory. They pin the FLA prefill launch configs: two otherwise identical builds that
autotuned differently diverged at |dlogprob| ~4e-2 from about token 3.

## Making a store for another GPU or build

`SAVE_TACTICS=/some/dir` makes `serve.sh` keep the boot's live FlashInfer store (`/some/dir/flashinfer`) and Triton
cache (`/some/dir/triton`) on the host; pass them back as `FI_TACTICS=/some/dir/flashinfer
TRITON_SEED=/some/dir/triton`; the second boot should then tune nothing and get its full KV cache (not yet tested end to
end with `SAVE_TACTICS`; the same steps done by hand with `docker cp` gave a default-profile store). For the
latency profile on a GPU where its cold boot runs out of memory, we have no tested recipe: the store must already cover
the M buckets up to 32,768, and a default-profile store does not. Untested on GPUs other than the RTX PRO 6000 Max-Q;
on GB300 (288 GB) cold boots of both profiles came up in the accuracy-gate runs.

Opt out: `FI_TACTICS=none` and/or `TRITON_SEED=none`.
