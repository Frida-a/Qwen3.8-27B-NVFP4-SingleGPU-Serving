#!/usr/bin/env python3
"""Spec-decode benchmark on real prompts against ONE vLLM server (run domains sequentially; server otherwise idle).
Per domain: acceptance (from vllm:spec_decode_* counter diffs; overall + per draft position), per-user decode tok/s,
aggregate output tok/s (= tok/s/GPU for a TP1 server), TTFT / TPOT / chunk-ITL p50/p99.
usage: sd_bench.py --port P --conc C --n N --max-tokens M [--domains gpqa,mmlu_pro,lcb,...] [--temperature 1.0 --top-p 0.95]
                   [--thinking on|off] [--save-text] --out out.json"""
import argparse, json, re, time, urllib.request, statistics as st, concurrent.futures as cf
from openai import OpenAI
ap = argparse.ArgumentParser()
for k, t, d in [("port", int, None), ("conc", int, None), ("n", int, None), ("max-tokens", int, 4096), ("domains", str, "gpqa,mmlu_pro,lcb"),
                ("temperature", float, 1.0), ("top-p", float, 0.95), ("thinking", str, "on"), ("out", str, None), ("seed", int, None),
                ("sets", str, "data/prompt_sets.json"), ("host", str, "localhost")]:
    ap.add_argument("--" + k, type=t, default=d, required=d is None and k not in ("seed",))
ap.add_argument("--save-text", action="store_true")
ap.add_argument("--profile", action="store_true", help="POST /start_profile before and /stop_profile after the run (one nsys window)")
a = ap.parse_args(); S = json.load(open(a.sets))
import os; os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)   # create the output dir before the long run
base = f"http://{a.host}:{a.port}"
c = OpenAI(base_url=base + "/v1", api_key="x", timeout=7200)
def counters():
    t = urllib.request.urlopen(base + "/metrics").read().decode(); d = {}
    for k in ["accepted_tokens", "draft_tokens", "drafts"]:
        m = re.search(rf"^vllm:spec_decode_num_{k}_total\{{[^}}]*\}} ([0-9.e+]+)", t, re.M); d[k] = float(m.group(1)) if m else 0.0
    d["pos"] = [float(x) for x in re.findall(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="\d+"\} ([0-9.e+]+)', t, re.M)]
    return d
def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None
def one(p):
    msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
    kw = dict(model="qw38", messages=msgs, max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p, stream=True,
              stream_options={"include_usage": True, "continuous_usage_stats": False},
              extra_body={"chat_template_kwargs": {"enable_thinking": a.thinking == "on"}})
    if a.seed is not None: kw["seed"] = a.seed
    t0 = time.time(); tf = None; n = 0; last = None; gaps = []; txt = []; rs = []
    for ch in c.chat.completions.create(**kw):
        now = time.time()
        if ch.choices:
            dl = ch.choices[0].delta; ex = dl.model_extra or {}; r = ex.get("reasoning_content") or ex.get("reasoning")
            if dl.content or r:
                if tf is None: tf = now
                else: gaps.append(now - last)
                last = now
                if a.save_text: txt.append(dl.content or ""); rs.append(r or "")
        if ch.usage: n = ch.usage.completion_tokens
    te = time.time()
    o = {"ttft": (tf or te) - t0, "out_tokens": n, "lat": te - t0,
         "tpot": (te - tf) / (n - 1) if tf and n > 1 else None, "itl_p50": pct(gaps, 50), "gaps": gaps}
    if a.save_text: o["content"] = "".join(txt); o["reasoning"] = "".join(rs)
    return o
def _post(p):
    try: urllib.request.urlopen(urllib.request.Request(base + p, method="POST"), timeout=600).read()
    except Exception as e: print("profile call", p, "failed:", e, flush=True)
if a.profile: _post("/start_profile")
res = {"args": vars(a), "server_models": json.load(urllib.request.urlopen(base + "/v1/models")), "domains": {}}
for dom in a.domains.split(","):
    prompts = S[dom][: a.n]; b = counters(); t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex: rr = list(ex.map(one, prompts))
    wall = time.time() - t0; e = counters()
    acc = e["accepted_tokens"] - b["accepted_tokens"]; drf = e["draft_tokens"] - b["draft_tokens"]; nd = e["drafts"] - b["drafts"]
    pos = [(y - x) / nd if nd else None for x, y in zip(b["pos"], e["pos"])]
    ntok = sum(r["out_tokens"] for r in rr); tpots = [r["tpot"] for r in rr]; gaps = [g for r in rr for g in r["gaps"]]
    D = {"n": len(rr), "accept_rate": acc / drf if drf else None, "accept_len": 1 + acc / nd if nd else None, "accept_by_pos": pos,
         "drafts": nd, "draft_tokens": drf, "accepted": acc,
         "user_tps_mean": st.mean([1 / t for t in tpots if t]) if any(tpots) else None,
         "user_tps_p50": 1 / pct(tpots, 50) if pct(tpots, 50) else None,
         "gpu_tps": ntok / wall, "mean_out_tokens": ntok / len(rr), "wall_s": wall,
         "ttft_p50": pct([r["ttft"] for r in rr], 50), "ttft_p99": pct([r["ttft"] for r in rr], 99),
         "tpot_p50": pct(tpots, 50), "tpot_p99": pct(tpots, 99), "itl_p50": pct(gaps, 50), "itl_p99": pct(gaps, 99)}
    if a.save_text: D["texts"] = [{"content": r["content"], "reasoning": r["reasoning"], "out_tokens": r["out_tokens"]} for r in rr]
    res["domains"][dom] = D
    print(dom, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in D.items() if k not in ("accept_by_pos", "texts")}),
          "pos", [round(x, 3) for x in pos if x is not None], flush=True)
if a.profile: _post("/stop_profile")
json.dump(res, open(a.out, "w"), indent=1)
