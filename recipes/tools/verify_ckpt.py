#!/usr/bin/env python3
"""Check an exported checkpoint: per-module-type formats, MTP tensors carried over in BF16, size, and that
the groups vLLM fuses into one GEMM share format and scales (q/k/v, gate/up, GDN in_proj_qkv+in_proj_z).
usage: verify_ckpt.py <ckpt_dir> [--src <bf16_dir>]"""
import argparse, collections, json, os, re
from safetensors import safe_open

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--src"); a = ap.parse_args()
idx = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
hq = json.load(open(os.path.join(a.ckpt, "hf_quant_config.json")))["quantization"]
ql = hq.get("quantized_layers", {})
cnt = collections.Counter()
for k, v in ql.items(): cnt[(re.sub(r"\.\d+\.", ".N.", k), v.get("quant_algo"))] += 1
print("quant_algo:", hq.get("quant_algo"), "| kv:", hq.get("kv_cache_quant_algo"), "| exclude:", hq.get("exclude_modules"))
for (p, f), n in sorted(cnt.items()): print(f"  {n:3d} x {f:6s} {p}")
size = sum(os.path.getsize(os.path.join(a.ckpt, f)) for f in set(idx.values()))
print(f"safetensors total: {size/1e9:.2f} GB ({size/2**30:.2f} GiB)")
handles = {}
def get(name):
    f = idx.get(name)
    if f is None: return None
    if f not in handles: handles[f] = safe_open(os.path.join(a.ckpt, f), "pt")
    return handles[f].get_tensor(name)
mtp = sorted(k for k in idx if k.startswith("mtp"))
dt = collections.Counter(str(get(k).dtype) for k in mtp)
print(f"mtp tensors: {len(mtp)} {dict(dt)}")
if a.src:
    sidx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    smtp = sorted(k for k in sidx if k.startswith("mtp"))
    print("mtp keys match source:", smtp == mtp)
groups = {"qkv": ("self_attn", ["q_proj", "k_proj", "v_proj"]), "gate_up": ("mlp", ["gate_proj", "up_proj"]),
          "gdn_qkvz": ("linear_attn", ["in_proj_qkv", "in_proj_z"]), "gdn_ba": ("linear_attn", ["in_proj_b", "in_proj_a"])}
layers = sorted({int(m.group(1)) for k in idx for m in [re.search(r"language_model\.layers\.(\d+)\.", k)] if m})
for g, (blk, mem) in groups.items():
    bad_fmt = bad_ws2 = bad_in = n = 0
    for L in layers:
        pre = f"model.language_model.layers.{L}.{blk}."
        if pre + mem[0] + ".weight" not in idx: continue
        n += 1
        fm = {ql.get(pre + m, {}).get("quant_algo") for m in mem}
        if len(fm) > 1: bad_fmt += 1
        for suf, ctr in (("weight_scale_2", "ws2"), ("input_scale", "in")):
            ts = [get(pre + m + "." + suf) for m in mem]
            if all(t is None for t in ts): continue
            if any(t is None for t in ts) or any(not (t.float() == ts[0].float()).all() for t in ts[1:]):
                if ctr == "ws2": bad_ws2 += 1
                else: bad_in += 1
    print(f"fused group {g:9s}: {n:2d} layers, format mismatch {bad_fmt}, weight_scale_2 mismatch {bad_ws2}, input_scale mismatch {bad_in}")
