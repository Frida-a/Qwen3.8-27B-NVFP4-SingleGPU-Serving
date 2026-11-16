# Recipe: a2gdn-lc32k-mtpa4 (NVFP4 + FP8 PTQ of Qwen3.8-27B with long-context calibration)

`quantize_a2gdn_lc32k.sh` rebuilds the serving checkpoint from `Qwen/Qwen3.8-27B` (BF16, revision 37561816).

| | published `nvidia/Qwen3.8-27B-NVFP4` | **a2gdn-lc32k-mtpa4** (this recipe) |
|---|---|---|
| MLP gate/up/down (64 layers) | NVFP4 | NVFP4 |
| GDN in_proj_qkv / in_proj_z / out_proj (48 layers) | FP8 | **NVFP4** |
| full attention q/k/v/o (16 layers) | FP8 | FP8 |
| GDN in_proj_a/b, conv1d, embeddings, norms, vision | BF16 | BF16 |
| lm_head | NVFP4 | NVFP4 (grafted from a reproduction of the published recipe) |
| MTP head linears | BF16 | **NVFP4** ("a4" head) |
| weight calibration | Local-Hessian, 2,048 x 2,048 tokens Nemotron post-training v3 | Local-Hessian, **128 x 32,768 tokens of decontaminated long-context text** |
| size (safetensors) | 21.92 GB | **18.96 GB** (18.65 GB of trunk shards + a 0.31 GB NVFP4 MTP shard; -13.5%) |

Why long calibration: putting the GDN projections in NVFP4 with the standard 2K-token calibration cost 2.3 points on
MRCR-128K (8-needle -4.3). The same layout calibrated on 32K-token documents recovered most of it (MRCR-128K
-1.10 vs the published checkpoint's FP8-KV pool, inside the -1.5 gate) with no training. A 62%-long / 38%-short
mix lost the benefit entirely. See docs/RESULTS.md, sections 3.2 and 4.

## Inputs

- **ModelOpt:** NVIDIA/Model-Optimizer `main` at 9aeff41 plus the stacked scoped-calibration PRs #3095 -> #3096 ->
  #3097 (all three still open; each PR head contains the one before it). The measured tree is the three PR commits
  cherry-picked onto 9aeff41; the exported configs carry the producer string `0.48.0.dev0+qw38quant`:
  ```bash
  git clone https://github.com/NVIDIA/Model-Optimizer.git && cd Model-Optimizer && git checkout 9aeff41
  git fetch origin pull/3097/head                       # the whole stack: 49b050d1 <- 3d4a3544 <- 27471e4a
  git cherry-pick 49b050d1 3d4a3544 27471e4a            # = the measured tree (if the PR was force-pushed since,
                                                        #   these SHAs may no longer be fetchable)
  # simpler alternative, not byte-tested: the stack on its own (older) base a7cbbdf9
  #   git fetch origin pull/3097/head && git checkout FETCH_HEAD
  SETUPTOOLS_SCM_PRETEND_VERSION=0.48.0.dev0+qw38quant pip install -e ".[hf]"
  pip install -r examples/hf_ptq/requirements.txt
  ```
  The measured environment was a venv (`--system-site-packages`) inside the vLLM image below (torch 2.14.0,
  transformers 5.22.2) with `datasets>=3.0 scipy "pulp<4.0" "omegaconf>=2.3"` added and ModelOpt installed with
  `SETUPTOOLS_SCM_PRETEND_VERSION=0.48.0.dev0+qw38quant pip install --no-deps --no-build-isolation -e .`. This recipe
  does not use a scoped plan; whether `main` alone reproduces bit-identical weights has not been tested.
- **vLLM image for the PTQ environment:** the same nightly f5d93296 (arm64 build on GB300; any recent PyTorch +
  transformers 5 should work for ModelOpt's `hf_ptq.py`).
- **GPU:** one GB300 (288 GB) per PTQ. The 32K-token Local-Hessian pass (step 4) took 81 min; that GPU's job-level
  memory peak was 156 GB (the job also ran a short proxy evaluation on it afterwards, so the PTQ itself used at most
  that). Steps 3 and 5 use 2,048-token samples and need much less.
- **Calibration sources** (fetched by `tools/fetch_sources.py`; nothing is redistributed here):

  | records in the set | source | license (as declared on the source page) |
  |---|---|---|
  | 32 x 32K windows of public-domain books | `deepmind/pg19` (via the `emozilla/pg19` parquet mirror) | Apache-2.0 (deepmind/pg19) |
  | 32 x 32K repository concatenations | `princeton-nlp/prolong-data-64K`, `thestackv1_concat_by_repo-65536` (The Stack v1) | no license declared on the ProLong page; The Stack v1 terms of use apply |
  | 16 x 32K textbook chapters | `princeton-nlp/prolong-data-64K`, `textbooks` (TextbookChapters) | no license declared |
  | 32 x ~32K synthetic multi-turn recall chats | our generator `tools/gen_lcrecall.py`: haystack turns from `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2`, needles from PG-19 | ODC-By (Nemotron SFT IF chat v2) |
  | 16 x 32K agent trajectories | `nebius/SWE-rebench-openhands-trajectories` | CC-BY-4.0 |
  | lm_head and MTP head (short) | `nvidia/Nemotron-Post-Training-Dataset-v3` (ModelOpt's `nemotron-post-training-v3`) | gated; see the dataset page for its terms |
  | none (held-out proxy set `docs_v1` only) | `zai-org/LongAlign-10k` (`long.jsonl`), fetched by step 1 | no license tag on the dataset page |

- **Decontamination** (`tools/decontam.py`): lowercase word 13-gram and 50-gram hashes of the evaluation sets. A
  record is dropped if it shares any 50-word span, or 3 or more distinct 13-grams, with any indexed set. Held-out
  proxy sets (`lcrecall_v2`, `docs_v1`) are excluded from calibration by id. Long-context benchmark corpora
  (LongBench, RULER, SCROLLS, InfiniteBench, HELMET, BABILong, openai/mrcr, AA-LCR) and NeMo Gym rollouts were never
  used as sources.
  - **Indexed by default** (a missing set is an error; `--allow-missing` overrides): GPQA-D, IFBench, SciCode and
    LCB v6 from a NeMo Gym v0.7.0 checkout (`Gym/`, or `QW38_GYM`) with `gym eval prepare` run for gpqa, ifbench,
    scicode and livecodebench/v6_2408_2505 (commands in `tools/decontam.py`); MRCR-128K built under
    `harness/accuracy/benchmarks` (harness/README.md, section 3); MMMU-Pro full built with
    `tools/build_mmmu_pro_full.py` into `harness/accuracy/benchmarks/mmmu_pro_std10_full/data/`; plus Gym's AA-LCR and
    Terminal-Bench example rows when present. Run from the repo root (or set `QW38_GYM` / `QW38_EVALS`). SciCode,
    LCB, MMMU-Pro, MMLU-Pro and AA-LCR are indexed only to protect the calibration data; this repo does not report
    accuracy on them.
  - **Measured index:** the same sets, except MMMU-Pro was a 400-question development subset (the full set is a
    superset), plus a 1,400-question stratified MMLU-Pro subset and the AA-LCR documents and questions (added with
    `--extra NAME=GLOB`). The two subsets are not shipped.
  - **Not indexed:** HLE and the MRCR >128K slice (the slice comes from openai/mrcr, which was never a source).

## Determinism

The sources are streamed from the Hub, so a rebuild matches the measured set only if the sources are unchanged. The
decontamination index decides which records are dropped, so the sha256 also reproduces only with the same eval sets
indexed (see "Decontamination": the default index differs from the measured one).
`quantize_a2gdn_lc32k.sh` prints the sha256 of the calibration file; the measured one was
`177c5a94031610cf1984c471dd5d4722cdc15149eb7f31d829a90f2259f0f2c2` (128 records: pg19 32, prolong_code 32,
prolong_tbk 16, lccalib_chat 32, swe_traj 16). `SHA256SUMS` lists the measured output files.

## Checks after quantizing

`tools/verify_ckpt.py` prints the per-module format map, total size, MTP tensor dtypes and fused-group scale
consistency (q/k/v, gate/up, GDN qkvz, GDN ba). Expected: 48 x NVFP4 per GDN projection, 64 x NVFP4 per MLP
projection, 16 x FP8 per attention projection, 1 x NVFP4 lm_head, 0 fused-group mismatches, 19.50 GB. That is the
step-4 export before the MTP graft (it still holds the BF16 MTP block); the final checkpoint after step 5 is 18.96 GB
(sha256 of every shard in `SHA256SUMS`).
Then run the accuracy gate in harness/accuracy (at least MRCR-128K x2 and GPQA-D x4) before serving it.
