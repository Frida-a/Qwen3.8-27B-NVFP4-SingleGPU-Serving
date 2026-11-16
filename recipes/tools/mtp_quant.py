#!/usr/bin/env python3
"""QW38 A4: quantize only the MTP head (mtp.layers.0 q/k/v/o + gate/up/down) and graft it onto an existing
checkpoint (the published one by default). mtp.fc, the MTP norms and everything else are left as they are.

HF transformers has no MTP forward, so the MTP block is rebuilt from HF classes exactly as vLLM's
Qwen3_5MultiTokenPredictor runs it: x = fc(cat(pre_fc_norm_embedding(embed(tok[t+1])),
pre_fc_norm_hidden(h[t]))) -> one full-attention decoder layer -> norm, where h is the target's final
(normed) hidden state (what vLLM passes to the drafter). Calibration inputs come from the BF16 model on
Nemotron post-training v3 samples. ModelOpt quantizes and packs the MTP linears; the new tensors go into
an extra shard and hf_quant_config.json / config.json gain per-layer entries for them.

formats: fp8   = FP8 W8A8 per-tensor (max) on all 7 MTP linears
         nvfp4 = NVFP4 W4A4 (static weights, MSE fp8-scale sweep; dynamic inputs) on all 7
         mixed = the main model's map: FP8 attention, NVFP4 MLP
usage: mtp_quant.py --fmt fp8|nvfp4|mixed --dst <dir> [--base <ckpt>] [--bf16 <dir>] [--n 128 --seq 2048]
"""
import argparse, copy, json, os, shutil

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import get_quantization_format, preprocess_linear_fusion
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

WORK = os.environ.get("QW38_WORK", ".")  # root holding models/ (see recipes/README.md)
ap = argparse.ArgumentParser()
ap.add_argument("--fmt", choices=["fp8", "nvfp4", "mixed"], required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--base", default=f"{WORK}/models/Qwen3.8-27B-NVFP4"); ap.add_argument("--bf16", default=f"{WORK}/models/Qwen3.8-27B")
ap.add_argument("--n", type=int, default=128); ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--dataset", default="nemotron-post-training-v3")
a = ap.parse_args()
dev = "cuda"

# ---- target model + MTP replica ------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(a.bf16)
model = AutoModelForImageTextToText.from_pretrained(a.bf16, dtype=torch.bfloat16, device_map={"": 0}).eval()
lm = model.model.language_model
tcfg = copy.deepcopy(model.config.text_config)
full_idx = tcfg.layer_types.index("full_attention")


class MTP(nn.Module):
    def __init__(self):
        super().__init__()
        H = tcfg.hidden_size
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.pre_fc_norm_embedding = q35.Qwen3_5RMSNorm(H, eps=tcfg.rms_norm_eps)
        self.pre_fc_norm_hidden = q35.Qwen3_5RMSNorm(H, eps=tcfg.rms_norm_eps)
        self.layers = nn.ModuleList([q35.Qwen3_5DecoderLayer(tcfg, full_idx)])
        self.norm = q35.Qwen3_5RMSNorm(H, eps=tcfg.rms_norm_eps)

    def forward(self, emb_next, hidden, pos_emb):
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(emb_next), self.pre_fc_norm_hidden(hidden)], -1))
        out = self.layers[0](x, position_embeddings=pos_emb, attention_mask=None)
        x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)


mtp = MTP().to(dev, torch.bfloat16).eval()
idx = json.load(open(os.path.join(a.bf16, "model.safetensors.index.json")))["weight_map"]
sd = {}
for k, f in idx.items():
    if k.startswith("mtp."):
        with safe_open(os.path.join(a.bf16, f), "pt") as h:
            sd[k[4:]] = h.get_tensor(k)
missing, unexpected = mtp.load_state_dict(sd, strict=False)
assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
print(f"MTP replica loaded: {len(sd)} tensors; decoder layer type {tcfg.layer_types[full_idx]}", flush=True)

# ---- calibration inputs ---------------------------------------------------------------------------------
dl = get_dataset_dataloader(a.dataset, tok, batch_size=1, num_samples=a.n, max_sample_length=a.seq, apply_chat_template=True)
cache = []
with torch.no_grad():
    for b in dl:
        ids = b["input_ids"].to(dev); m = b["attention_mask"].to(dev).bool()
        ids = ids[:, m[0]]  # drop padding (left-padded single rows)
        if ids.shape[1] < 16: continue
        h = lm(input_ids=ids).last_hidden_state                                   # normed final hidden state
        emb = lm.embed_tokens(ids)
        T = ids.shape[1] - 1
        pos = torch.arange(T, device=dev)[None]
        cache.append((emb[:, 1:].cpu(), h[:, :-1].cpu(), pos.cpu()))
print(f"calibration: {len(cache)} sequences, {sum(c[0].shape[1] for c in cache)} tokens", flush=True)
rot = lm.rotary_emb


def pos_emb(x, pos):
    try:
        return rot(x, pos.expand(3, -1, -1) if pos.dim() == 2 else pos)
    except Exception:
        return rot(x, pos)


def forward_loop(m):
    with torch.no_grad():
        for e, h, p in cache:
            e, h, p = e.to(dev), h.to(dev), p.to(dev)
            m(e, h, pos_emb(h, p))

# Reference (BF16) top-1 of the MTP head on the calibration tokens, to report agreement after quantization.
lm_head = model.lm_head
def top1(m):
    outs = []
    with torch.no_grad():
        for e, h, p in cache[:16]:
            e, h, p = e.to(dev), h.to(dev), p.to(dev)
            outs.append(lm_head(m(e, h, pos_emb(h, p))).argmax(-1).cpu())
    return torch.cat(outs, -1)
ref = top1(mtp)

# ---- quantize -------------------------------------------------------------------------------------------
FP8 = {"num_bits": (4, 3), "axis": None}
NV_W = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}}
NV_I = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
att, mlp = "*layers.0.self_attn.*_proj", "*layers.0.mlp.*_proj"
fmt_att = {"fp8": "fp8", "nvfp4": "nvfp4", "mixed": "fp8"}[a.fmt]
fmt_mlp = {"fp8": "fp8", "nvfp4": "nvfp4", "mixed": "nvfp4"}[a.fmt]
qc = [{"quantizer_name": "*", "enable": False}]
for pat, f in ((att, fmt_att), (mlp, fmt_mlp)):
    w, i = (FP8, FP8) if f == "fp8" else (NV_W, NV_I)
    qc += [{"quantizer_name": f"{pat}.weight_quantizer", "cfg": w}, {"quantizer_name": f"{pat}.input_quantizer", "cfg": i}]
algorithm = {"method": "mse", "fp8_scale_sweep": True} if "nvfp4" in (fmt_att, fmt_mlp) else "max"
mtq.quantize(mtp, {"quant_cfg": qc, "algorithm": algorithm}, forward_loop)
q1 = top1(mtp)
agree = (q1 == ref).float().mean().item()
print(f"quantized MTP ({a.fmt}): top-1 agreement with BF16 MTP on calibration tokens = {agree:.4f}", flush=True)

# ---- export the MTP linears -----------------------------------------------------------------------------
L0 = mtp.layers[0]
groups = [[L0.self_attn.q_proj, L0.self_attn.k_proj, L0.self_attn.v_proj], [L0.mlp.gate_proj, L0.mlp.up_proj]]
with torch.no_grad():
    for g in groups:
        preprocess_linear_fusion(g)           # one input scale (and NVFP4 global scale) per vLLM fused GEMM
names = {f"layers.0.self_attn.{n}": getattr(L0.self_attn, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")}
names.update({f"layers.0.mlp.{n}": getattr(L0.mlp, n) for n in ("gate_proj", "up_proj", "down_proj")})
new, layers_cfg = {}, {}
for n, mod in names.items():
    f = get_quantization_format(mod)
    with torch.no_grad():
        _export_quantized_weight(mod, torch.bfloat16)
    for k, v in mod.state_dict().items():
        if "quantizer" in k: continue
        new[f"mtp.{n}.{k}"] = v.detach().cpu().contiguous()
    layers_cfg[f"mtp.{n}"] = {"quant_algo": "NVFP4", "group_size": 16} if "nvfp4" in f.lower() else {"quant_algo": "FP8"}
    print(f"  mtp.{n}: {f} -> {sorted(k for k in mod.state_dict() if 'quantizer' not in k)}", flush=True)

# ---- graft onto a copy of the base checkpoint ---------------------------------------------------------
os.makedirs(a.dst, exist_ok=True)
bidx = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))
wm = {k: v for k, v in bidx["weight_map"].items() if not any(k.startswith(f"mtp.{n}.") for n in names)}
shard = "model-mtp-quant.safetensors"
save_file(new, os.path.join(a.dst, shard), metadata={"format": "pt"})
wm.update({k: shard for k in new})
# vLLM loads every tensor of every shard file (not only those the index names), so the shards that hold the
# base checkpoint's BF16 mtp.* linears are rewritten without them; the other shards are symlinked.
replaced = {k for k in bidx["weight_map"] if any(k.startswith(f"mtp.{n}.") for n in names)}
for f in sorted(set(bidx["weight_map"].values())):
    dstf = os.path.join(a.dst, f)
    if os.path.lexists(dstf):
        os.unlink(dstf)
    if any(bidx["weight_map"][k] == f for k in replaced):
        sd = load_file(os.path.join(a.base, f))
        for k in replaced: sd.pop(k, None)
        save_file(sd, dstf, metadata={"format": "pt"})
    else:
        os.symlink(os.path.realpath(os.path.join(a.base, f)), dstf)  # unchanged shards: symlink (copy with rsync -L)
bidx["weight_map"] = wm
json.dump(bidx, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)
for f in os.listdir(a.base):
    p = os.path.join(a.base, f)
    if os.path.isfile(p) and not f.endswith(".safetensors") and f not in ("model.safetensors.index.json",) and not f.startswith("."):
        shutil.copy2(p, os.path.join(a.dst, f))
excl = ["mtp.fc*", "mtp.norm*", "mtp.pre_fc_norm*", "mtp.layers.0.*norm*"]
hq = json.load(open(os.path.join(a.dst, "hf_quant_config.json")))
cfg = json.load(open(os.path.join(a.dst, "config.json")))
qcj = cfg["quantization_config"]
uniform = "quantized_layers" not in hq["quantization"]
if uniform:
    # Uniform header (e.g. a2-lh: quant_algo NVFP4 + exclude_modules, no per-layer map): the MTP linears become NVFP4
    # by dropping them from the exclude lists; mtp.fc, the MTP norms and everything else stay excluded.
    assert hq["quantization"]["quant_algo"] == "NVFP4" and fmt_att == fmt_mlp == "nvfp4", \
        "uniform-header bases support only --fmt nvfp4 on an NVFP4 header"
    drop = lambda lst: [e for e in (lst or []) if not any(e.startswith(f"mtp.{n}") for n in names)]
    hq["quantization"]["exclude_modules"] = drop(hq["quantization"].get("exclude_modules"))
    qcj["ignore"] = drop(qcj.get("ignore"))
    for g in qcj.get("config_groups", {}).values():
        if g.get("weights", {}).get("num_bits") == 4:
            g["targets"] = sorted(set(g.get("targets", [])) | set(layers_cfg))
else:
    # vLLM resolves per-layer formats (needed for the mtp.* entries) only under MIXED_PRECISION.
    hq["quantization"]["quant_algo"] = "MIXED_PRECISION"
    keep = [e for e in (hq["quantization"].get("exclude_modules") or []) if not e.startswith("mtp")]
    hq["quantization"]["exclude_modules"] = keep + excl
    hq["quantization"]["quantized_layers"].update(layers_cfg)
    qcj["quant_algo"] = "MIXED_PRECISION"
    qcj.setdefault("quantized_layers", dict(hq["quantization"]["quantized_layers"]))
    qcj["ignore"] = [e for e in (qcj.get("ignore") or []) if not e.startswith("mtp")] + excl
    qcj["quantized_layers"].update(layers_cfg)
    for gname, g in qcj.get("config_groups", {}).items():
        nb = g.get("weights", {}).get("num_bits")
        for m, v in layers_cfg.items():
            if (v["quant_algo"] == "FP8" and nb == 8) or (v["quant_algo"] == "NVFP4" and nb == 4):
                g.setdefault("targets", []).append(m)
json.dump(hq, open(os.path.join(a.dst, "hf_quant_config.json"), "w"), indent=4)
json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
size = sum(v.numel() * v.element_size() for v in new.values())
with open(os.path.join(a.dst, "MTP_QUANT.md"), "w") as fh:
    fh.write(f"# MTP-only quantization ({a.fmt})\n\nBase: `{a.base}` (unchanged shards symlinked). BF16 MTP source: `{a.bf16}`.\n"
             f"Script: recipes/tools/mtp_quant.py. Calibration: {len(cache)} x <= {a.seq} tokens of {a.dataset}.\n"
             f"Quantized MTP linears: {json.dumps(layers_cfg)}. New shard {shard}: {size/1e9:.3f} GB.\n"
             f"Top-1 agreement of the quantized MTP head with the BF16 MTP head on calibration tokens: {agree:.4f}.\n")
print(f"wrote {a.dst}: {len(new)} MTP tensors ({size/1e9:.3f} GB), top-1 agreement {agree:.4f}", flush=True)
