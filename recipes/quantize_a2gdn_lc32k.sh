#!/bin/bash
# Reproduce the a2gdn-lc32k-mtpa4 checkpoint from Qwen/Qwen3.8-27B (BF16) with NVIDIA Model Optimizer (ModelOpt) PTQ.
#
#   layout   NVFP4 (W4A4): MLP gate/up/down (64 layers) + GDN linear-attention in_proj_qkv / in_proj_z / out_proj (48 layers)
#            + lm_head; FP8 (W8A8, per-tensor): full-attention q/k/v/o (16 layers); BF16: embeddings, GDN in_proj_a/b,
#            conv1d, norms, vision tower. MTP head: NVFP4 linears (the "a4" head).
#   method   layerwise Local-Hessian weight calibration (recipe a2_gdn_nvfp4_lh_nolm.yaml), FP8 scale sweep,
#            shared global amax for fused projections (q/k/v, gate/up, in_proj_qkv/z).
#   data     128 x 32,768-token decontaminated long-context records (4.19M tokens; see README.md). The lm_head and
#            the MTP head are calibrated on short post-training data (steps 3 and 5).
#
# Steps 1-2 are CPU only; steps 3-5 need GPUs (the measured runs used one GB300 (288 GB) per PTQ: step 4 took 81 min).
# Step 2 needs the eval sets it decontaminates against (see tools/decontam.py and README.md, "Decontamination").
# usage (from the repo root): WORK=$PWD/work MO=/path/to/Model-Optimizer [QW38_GYM=$PWD/Gym] \
#          bash recipes/quantize_a2gdn_lc32k.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); T=$HERE/tools
: ${WORK:?WORK dir} ${MO:?ModelOpt checkout (see README.md for the exact commit)}
mkdir -p $WORK/{models,calib/raw,decontam,ckpt}
export PYTHONPATH=$T QW38_WORK=$WORK
export QW38_GYM=${QW38_GYM:-$PWD/Gym} QW38_EVALS=${QW38_EVALS:-$(cd "$HERE/../harness/accuracy/benchmarks" && pwd)}
BF16=$WORK/models/Qwen3.8-27B                    # local path: hf_ptq on a Hub id drops the MTP tensors
[ -d $BF16 ] || hf download Qwen/Qwen3.8-27B --revision 37561816 --local-dir $BF16

# ---- 1. long-context sources (200 records each) + token stats ------------------------------------------------
python $T/fetch_sources.py --out $WORK/calib/raw --src pg19,prolong_code,prolong_tbk,longalign,swe_traj --n 200
python $T/lc_stats.py --tok $BF16 --raw $WORK/calib/raw --out $WORK/calib/source_stats.json

# ---- 2. decontamination against every eval set, then the calibration set -------------------------------------
# Index: 13-gram + 50-gram hashes of GPQA-D, IFBench, SciCode, LCB v6 (prepared in $QW38_GYM), MRCR-128K and MMMU-Pro
# full (built under $QW38_EVALS: MRCR per harness/README.md section 3, MMMU-Pro with tools/build_mmmu_pro_full.py;
# commands in the docstring of tools/decontam.py), plus Gym's AA-LCR / Terminal-Bench example rows.
# A missing required set is an error. The measured index also held MMLU-Pro (1,400-question subset) and the AA-LCR
# documents and questions, added with --extra NAME=GLOB, and used the 400-question MMMU-Pro subset; the calibration
# sha256 below reproduces only with the same eval files indexed.
python $T/decontam.py build --out $WORK/decontam/index
# Proxy sets (held-out validation; their ids are excluded from calibration)
python $T/gen_lcrecall.py --tok $BF16 --pg19 $WORK/calib/raw/pg19.jsonl --out $WORK/calib/lcrecall_v2.jsonl
python $T/decontam.py check --index $WORK/decontam/index --report $WORK/decontam/report.json --data $WORK/calib/raw/*.jsonl $WORK/calib/lcrecall_v2.jsonl
python $T/build_docs.py --tok $BF16 --raw $WORK/calib/raw --report $WORK/decontam/report.json \
  --spec pg19:12,prolong_code:10,swe_traj:6,longalign:4 --out $WORK/calib/docs_v1.jsonl
# Calibration recall chats (haystack offset 6,000 from the proxy set)
python $T/gen_lcrecall.py --tok $BF16 --pg19 $WORK/calib/raw/pg19.jsonl --out $WORK/calib/lccalib_chat.jsonl \
  --lengths 32000 --needles 2,4,8 --n_per_cell 16 --seed 1 --skip_chat 6000 --id_prefix lcc
python $T/decontam.py check --index $WORK/decontam/index --report $WORK/decontam/report.json \
  --data $WORK/calib/lccalib_chat.jsonl $WORK/calib/raw/swe_traj.jsonl
python $T/build_calib.py --tok $BF16 --raw $WORK/calib/raw --report $WORK/decontam/report.json \
  --chat $WORK/calib/lccalib_chat.jsonl --mix books:32,code:32,tbk:16,chat:32,agent:16 --seq 32768 \
  --exclude $WORK/calib/docs_v1.jsonl $WORK/calib/lcrecall_v2.jsonl --out $WORK/calib/lc_calib_128x32k.jsonl
sha256sum $WORK/calib/lc_calib_128x32k.jsonl   # measured build: 177c5a94031610cf1984c471dd5d4722cdc15149eb7f31d829a90f2259f0f2c2

# ---- 3. lm_head donor: the published NVFP4 recipe reproduced from BF16 (2,048 x 2,048 Nemotron post-training v3) --
# Only its NVFP4 lm_head is used (step 4 grafts it; skips a ~38 min full-model lm_head pass). KL vs BF16 of this
# reproduction: 0.0286 (published checkpoint 0.0273).
python $MO/examples/hf_ptq/hf_ptq.py --pyt_ckpt_path $BF16 \
  --recipe $MO/modelopt_recipes/models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_local_hessian.yaml \
  --dataset nemotron-post-training-v3 --calib_size 2048 --calib_seq 2048 --batch_size 1 --export_path $WORK/ckpt/repro-pub

# ---- 4. a2gdn-lc32k: long-context Local-Hessian PTQ (lm_head quantizer off in the recipe, then grafted) ----------
python $MO/examples/hf_ptq/hf_ptq.py --pyt_ckpt_path $BF16 --recipe $HERE/a2_gdn_nvfp4_lh_nolm.yaml \
  --dataset $WORK/calib/lc_calib_128x32k.jsonl --calib_size 128 --calib_seq 32768 --batch_size 1 \
  --export_path $WORK/ckpt/a2gdn-lc32k
python $T/graft_lmhead.py --donor $WORK/ckpt/repro-pub --target $WORK/ckpt/a2gdn-lc32k
python $T/verify_ckpt.py $WORK/ckpt/a2gdn-lc32k --src $BF16 | tee $WORK/ckpt/a2gdn-lc32k/verify.txt

# ---- 5. NVFP4 MTP head ("a4"): quantize the BF16 MTP block alone (128 x 2,048 Nemotron post-training v3), graft it --
# mtp_quant.py writes the quantized head next to a base checkpoint (the published nvidia/Qwen3.8-27B-NVFP4 in the
# measured build); only its mtp.* tensors are used by graft_mtp.py, the base's other weights are discarded.
NV=$WORK/models/Qwen3.8-27B-NVFP4
[ -d $NV ] || hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir $NV
python $T/mtp_quant.py --fmt nvfp4 --bf16 $BF16 --base $NV --dst $WORK/ckpt/a4-mtp-nvfp4 --n 128 --seq 2048
python $T/graft_mtp.py --mode a4nvfp4 --target $WORK/ckpt/a2gdn-lc32k --a4-src $WORK/ckpt/a4-mtp-nvfp4 \
  --out $WORK/ckpt/a2gdn-lc32k-mtpa4
(cd $WORK/ckpt/a2gdn-lc32k-mtpa4 && sha256sum *.safetensors)
echo "done: $WORK/ckpt/a2gdn-lc32k-mtpa4 (compare with recipes/SHA256SUMS)"
