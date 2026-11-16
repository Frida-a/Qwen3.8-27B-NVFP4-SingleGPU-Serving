#!/usr/bin/env python3
"""Graft the NVFP4 lm_head (packed weight + weight_scale + weight_scale_2 + input_scale) of a donor checkpoint
into a target checkpoint, so variants whose lm_head recipe equals the donor's skip ModelOpt's ~38 min
full-model lm_head calibration pass (their recipes disable the lm_head quantizer, e.g. recipes/a2_gdn_nvfp4_lh_nolm.yaml).

  graft_lmhead.py --donor <ckpt> --target <ckpt>               in place (rewrites the shard holding lm_head)
  graft_lmhead.py --donor <ckpt> --target <ckpt> --dst <dir>   new dir: untouched shards symlinked
"""
import argparse, json, os, shutil

from safetensors import safe_open
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--donor", required=True); ap.add_argument("--target", required=True); ap.add_argument("--dst")
a = ap.parse_args()
dst = a.dst or a.target
didx = json.load(open(os.path.join(a.donor, "model.safetensors.index.json")))["weight_map"]
dk = sorted(k for k in didx if k.startswith("lm_head."))
assert "lm_head.weight_scale_2" in dk, f"donor lm_head is not NVFP4: {dk}"
new = {}
for k in dk:
    with safe_open(os.path.join(a.donor, didx[k]), "pt") as h:
        new[k] = h.get_tensor(k)
tidx_all = json.load(open(os.path.join(a.target, "model.safetensors.index.json")))
tidx = tidx_all["weight_map"]
shards = sorted({tidx[k] for k in tidx if k.startswith("lm_head.")})
assert len(shards) == 1, shards
shard = shards[0]
if dst != a.target:
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(a.target):
        p = os.path.join(a.target, f)
        if not os.path.isfile(p) or f == shard:
            continue
        if f.endswith(".safetensors"):
            os.symlink(os.path.realpath(p), os.path.join(dst, f))
        else:
            shutil.copy2(p, os.path.join(dst, f))
sd = load_file(os.path.join(a.target, shard))
old = sorted(k for k in sd if k.startswith("lm_head."))
for k in old: del sd[k]
sd.update(new)
tmp = os.path.join(dst, shard + ".tmp")
save_file(sd, tmp, metadata={"format": "pt"})
if os.path.islink(os.path.join(dst, shard)): os.unlink(os.path.join(dst, shard))
os.replace(tmp, os.path.join(dst, shard))
for k in old: tidx.pop(k, None)
tidx.update({k: shard for k in new})
json.dump(tidx_all, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
# quant configs: lm_head NVFP4 group 16, and out of every exclude/ignore list
LM = {"quant_algo": "NVFP4", "group_size": 16}
hq = json.load(open(os.path.join(dst, "hf_quant_config.json")))
q = hq["quantization"]
q.setdefault("quantized_layers", {})["lm_head"] = LM
q["exclude_modules"] = [m for m in (q.get("exclude_modules") or []) if not m.startswith("lm_head")]
json.dump(hq, open(os.path.join(dst, "hf_quant_config.json"), "w"), indent=4)
cfg = json.load(open(os.path.join(dst, "config.json")))
qc = cfg["quantization_config"]
qc.setdefault("quantized_layers", {})["lm_head"] = LM
qc["ignore"] = [m for m in (qc.get("ignore") or []) if not m.startswith("lm_head")]
for g in qc.get("config_groups", {}).values():
    t = g.setdefault("targets", [])
    if "lm_head" in t: t.remove("lm_head")
    if g.get("weights", {}).get("num_bits") == 4 and g.get("weights", {}).get("group_size") == 16:
        t.append("lm_head")
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
with open(os.path.join(dst, "LMHEAD_GRAFT.md"), "w") as fh:
    fh.write(f"lm_head replaced by the NVFP4 lm_head of `{a.donor}` (tensors {dk}); previous lm_head tensors {old}. "
             f"Script: recipes/tools/graft_lmhead.py. Reason: skip the ~38 min full-model lm_head calibration pass.\n")
print(f"grafted {dk} from {a.donor} into {dst} (replaced {old})")
