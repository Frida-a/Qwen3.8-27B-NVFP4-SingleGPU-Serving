#!/usr/bin/env python3
"""Graft an MTP head onto a checkpoint whose MTP head is missing/incomplete (e.g. a checkpoint that indexes
only the 11 mtp.layers.0.* tensors, with mtp.fc / mtp.norm / mtp.pre_fc_norm_{embedding,hidden} absent).

Output dir: unchanged shards symlinked (relative), the shard(s) holding the target's mtp.* tensors rewritten without them when the
mode replaces them, one new shard model-mtp-graft.safetensors with the grafted tensors, patched index (+ quant configs for NVFP4).
Modes:
  hybrid   : keep target mtp.layers.0.*; add the 4 missing tensors (fc + 3 norms) from --bf16-src
  basebf16 : replace all mtp.* with the complete BF16 head from --bf16-src
  a4nvfp4  : replace all mtp.* with the a4 NVFP4 head from --a4-src (non-linear mtp.* + model-mtp-quant tensors); patch quant configs
usage: graft_mtp.py --target DIR --out DIR --mode M [--bf16-src DIR] [--a4-src DIR]
"""
import argparse, json, os, shutil, hashlib
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--mode", required=True, choices=["hybrid", "basebf16", "a4nvfp4"])
ap.add_argument("--bf16-src"); ap.add_argument("--a4-src")
a = ap.parse_args()
T, O = a.target.rstrip("/"), a.out.rstrip("/")
os.makedirs(O, exist_ok=False)
idx = json.load(open(f"{T}/model.safetensors.index.json")); wm = idx["weight_map"]
tgt_mtp = sorted(k for k in wm if k.startswith("mtp."))
MISSING = ["mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"]
MTP_LIN = ["mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj", "mtp.layers.0.self_attn.o_proj",
           "mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj", "mtp.layers.0.mlp.down_proj"]


def load_keys(src_dir, keys_filter):
    sidx = json.load(open(f"{src_dir}/model.safetensors.index.json"))["weight_map"]
    out = {}
    for f in sorted(set(sidx.values())):
        ks = [k for k, v in sidx.items() if v == f and keys_filter(k)]
        if not ks: continue
        with safe_open(f"{src_dir}/{f}", "pt") as s:
            for k in ks: out[k] = s.get_tensor(k).contiguous()
    return out


replace = a.mode in ("basebf16", "a4nvfp4")
if a.mode == "hybrid":
    graft = load_keys(a.bf16_src, lambda k: k in MISSING)
elif a.mode == "basebf16":
    graft = load_keys(a.bf16_src, lambda k: k.startswith("mtp."))
else:
    graft = load_keys(a.a4_src, lambda k: k.startswith("mtp."))
assert graft, "nothing to graft"
mtp_shards = sorted({wm[k] for k in tgt_mtp}) if replace else []
for f in sorted(set(wm.values())):
    if f in mtp_shards:
        keep = {}
        with safe_open(f"{T}/{f}", "pt") as s:
            meta = s.metadata()
            for k in s.keys():
                if not k.startswith("mtp."): keep[k] = s.get_tensor(k).contiguous()
        save_file(keep, f"{O}/{f}", metadata=meta)
    else:
        os.symlink(os.path.relpath(f"{T}/{f}", O), f"{O}/{f}")
save_file(graft, f"{O}/model-mtp-graft.safetensors", metadata={"format": "pt"})
nwm = {k: v for k, v in wm.items() if not (replace and k.startswith("mtp."))}
for k in graft: nwm[k] = "model-mtp-graft.safetensors"
idx["weight_map"] = dict(sorted(nwm.items()))
json.dump(idx, open(f"{O}/model.safetensors.index.json", "w"), indent=2)
skip = {"model.safetensors.index.json", "config.json", "hf_quant_config.json", "READY", ".verify.json"}
for f in os.listdir(T):
    if f.endswith(".safetensors") or f in skip or f.startswith("."): continue
    p = f"{T}/{f}"
    if os.path.isfile(p): shutil.copy2(p, f"{O}/{f}")
cfg = json.load(open(f"{T}/config.json")); hq = json.load(open(f"{T}/hf_quant_config.json"))
if a.mode == "a4nvfp4":
    qc = cfg["quantization_config"]
    lin_pat = lambda x: x.startswith("mtp") and not any(t in x for t in ("norm", "fc"))
    qc["ignore"] = [x for x in qc.get("ignore", []) if not lin_pat(x)] + ["mtp.fc*", "mtp.norm*", "mtp.pre_fc_norm*", "mtp.layers.0.*norm*"]
    if isinstance(qc.get("quantized_layers"), dict):
        for n in MTP_LIN: qc["quantized_layers"][n] = {"quant_algo": "NVFP4", "group_size": 16}
    grp = [g for g, v in qc.get("config_groups", {}).items() if v.get("weights", {}).get("num_bits") == 4]
    assert grp, "no NVFP4 config group"
    qc["config_groups"][grp[0]]["targets"] = list(qc["config_groups"][grp[0]]["targets"]) + MTP_LIN
    q = hq["quantization"]
    q["exclude_modules"] = [x for x in q.get("exclude_modules", []) if not lin_pat(x)]
    for n in MTP_LIN: q["quantized_layers"][n] = {"quant_algo": "NVFP4", "group_size": 16}
json.dump(cfg, open(f"{O}/config.json", "w"), indent=2); json.dump(hq, open(f"{O}/hf_quant_config.json", "w"), indent=2)
gsha = hashlib.sha256(open(f"{O}/model-mtp-graft.safetensors", "rb").read()).hexdigest()
with open(f"{O}/GRAFT.md", "w") as fh:
    fh.write(f"# MTP graft ({a.mode})\ntarget: {T}\nbf16-src: {a.bf16_src}\na4-src: {a.a4_src}\n"
             f"target mtp tensors ({len(tgt_mtp)}): {tgt_mtp}\nreplaced target mtp: {replace}; rewritten shards: {mtp_shards}\n"
             f"grafted ({len(graft)}): {sorted(graft)}\nmodel-mtp-graft.safetensors sha256 {gsha}\nscript: recipes/tools/graft_mtp.py\n")
print(open(f"{O}/GRAFT.md").read())
