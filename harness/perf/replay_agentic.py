#!/usr/bin/env python3
"""Multi-turn agentic replay for spec-decode serving measurements: real OpenHands SWE prefixes with the turn structure
kept, per-turn output lengths taken from the trajectory, T=1 top_p 0.95, fixed tool gap. Options:
  --per-turn-metrics  snapshot /metrics around every turn -> per-turn acceptance length (exact only with --conc 1)
  --save-ids          store prompt token ids + output token ids per turn (return_token_ids) for the offline n-gram/suffix
                      draft simulators
Data: SWE-Gym/OpenHands-SFT-Trajectories (MIT).
usage: replay_agentic.py --base URL --parquet P --tokenizer T --sessions N --conc C --out DIR [--per-turn-metrics] [--save-ids]
"""
import argparse, json, os, re, time, random, statistics as st, urllib.request, concurrent.futures as cf, hashlib, collections, threading
import pandas as pd
from openai import OpenAI
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8000"); ap.add_argument("--parquet", required=True); ap.add_argument("--tokenizer", required=True)
ap.add_argument("--sessions", type=int, default=16); ap.add_argument("--conc", type=int, default=4); ap.add_argument("--out", required=True)
ap.add_argument("--max-ctx", type=int, default=98304); ap.add_argument("--max-turns", type=int, default=40); ap.add_argument("--min-out", type=int, default=16)
ap.add_argument("--max-out", type=int, default=2048); ap.add_argument("--tool-gap-s", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-sessions", type=int, default=0, help="skip the first N selected sessions (disjoint warm/measure sets)")
ap.add_argument("--free", action="store_true", help="greedy, natural finish (no min_tokens/ignore_eos)")
ap.add_argument("--len-mode", choices=["matched", "ieos"], default="ieos",
                help="non-free length forcing: ieos = ignore_eos only (default; no min_tokens, EOS not masked); matched = min_tokens+ignore_eos (historical; inflates acceptance)")
ap.add_argument("--per-turn-metrics", action="store_true"); ap.add_argument("--save-ids", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.tokenizer)
c = OpenAI(base_url=a.base + "/v1", api_key="x", timeout=3600)
KEYS = ["prefix_cache_queries", "prefix_cache_hits", "spec_decode_num_accepted_tokens", "spec_decode_num_draft_tokens", "spec_decode_num_drafts"]
def metrics():
    t = urllib.request.urlopen(a.base + "/metrics").read().decode(); D = {}
    for k in KEYS:
        m = re.search(rf"^vllm:{k}_total\{{[^}}]*\}} ([0-9.e+]+)", t, re.M); D[k] = float(m.group(1)) if m else 0.0
    pos = {}
    for m in re.finditer(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\} ([0-9.e+]+)', t, re.M):
        pos[int(m.group(1))] = float(m.group(2))
    D["per_pos"] = pos
    return D
def delta(M0, M1):
    d = {k: M1[k] - M0[k] for k in KEYS}
    nd = d["spec_decode_num_drafts"]
    d["accept_len"] = 1 + d["spec_decode_num_accepted_tokens"] / nd if nd else None
    d["accept_rate"] = d["spec_decode_num_accepted_tokens"] / d["spec_decode_num_draft_tokens"] if d["spec_decode_num_draft_tokens"] else None
    d["draft_len"] = d["spec_decode_num_draft_tokens"] / nd if nd else None  # < k when adaptive verification trims
    d["per_pos_rate"] = {p: (M1["per_pos"].get(p, 0) - M0["per_pos"].get(p, 0)) / nd for p in sorted(M1["per_pos"])} if nd else None
    return d
df = pd.read_parquet(a.parquet); rnd = random.Random(a.seed)
idx = list(range(len(df))); rnd.shuffle(idx)
def norm(msgs):
    return [{"role": ("user" if m["role"] == "tool" else m["role"]), "content": str(m["content"])} for m in msgs]
sessions = []
for i in idx:
    msgs = norm(list(df.iloc[i]["messages"]))
    turns = [j for j, m in enumerate(msgs) if m["role"] == "assistant"][: a.max_turns]
    if not turns: continue
    keep = []
    for j in turns:
        ids = tok.apply_chat_template(msgs[:j], tokenize=True, add_generation_prompt=True)
        ctx = len(ids["input_ids"] if hasattr(ids, "keys") else ids)  # BatchEncoding in newer transformers (perf driver measures len(dict)=2)
        if ctx + a.max_out > a.max_ctx: break
        keep.append((j, ctx, max(a.min_out, min(a.max_out, len(tok.encode(msgs[j]["content"]))))))
    if len(keep) >= 3: sessions.append((int(i), msgs, keep))
    if len(sessions) >= a.sessions + a.skip_sessions: break
sessions = sessions[a.skip_sessions:]
mlock = threading.Lock()
idsf = open(os.path.join(a.out, "turn_ids.jsonl"), "w") if a.save_ids else None
def run_session(s):
    sid, msgs, keep = s; rows = []; t0 = time.time()
    for (j, ctx, olen) in keep:
        ts = time.time(); tf = None; n = 0; last = None
        kw = (dict(max_tokens=a.max_out, temperature=0.0) if a.free else
              dict(max_tokens=olen, temperature=1.0, top_p=0.95, extra_body=({"min_tokens": olen, "ignore_eos": True} if a.len_mode == "matched" else {"ignore_eos": True})))
        if a.save_ids: kw.setdefault("extra_body", {})["return_token_ids"] = True
        fin = None; err = None; txt = []; pt = None; pids = None; oids = []
        Mb = metrics() if a.per_turn_metrics else None
        try:
          st_ = c.chat.completions.create(model="qw38", messages=msgs[:j], stream=True, stream_options={"include_usage": True}, **kw)
          for ch in st_:
            now = time.time(); cx = ch.model_extra or {}
            if pids is None and cx.get("prompt_token_ids"): pids = cx["prompt_token_ids"]
            if ch.choices:
                ch0 = ch.choices[0]
                if ch0.finish_reason: fin = ch0.finish_reason
                ox = ch0.model_extra or {}
                if ox.get("token_ids"): oids.extend(ox["token_ids"])
                d = ch0.delta; ex = d.model_extra or {}
                piece = (d.content or "") + (ex.get("reasoning_content") or ex.get("reasoning") or "")
                txt.append(piece)
                if piece:
                    if tf is None: tf = now
                    last = now
            if ch.usage: n = ch.usage.completion_tokens; pt = ch.usage.prompt_tokens
        except Exception as e:
          err = repr(e)[:300]
        te = time.time()
        r = dict(session=sid, turn=j, prompt_tokens=pt if pt is not None else ctx, out_tokens=n, finish=fin, error=err,
                 out_hash=hashlib.sha1("".join(txt).encode()).hexdigest()[:16], ttft_s=(tf or te) - ts, e2e_s=te - ts,
                 decode_tps=(n - 1) / (te - tf) if tf and n > 1 and te > tf else None, itl_ms_mean=1e3 * (te - tf) / max(n - 1, 1) if tf else None)
        if a.per_turn_metrics:
            dm = delta(Mb, metrics()); r.update(turn_accept_len=dm["accept_len"], turn_accept_rate=dm["accept_rate"],
                                              turn_drafts=dm["spec_decode_num_drafts"], turn_draft_len=dm["draft_len"], turn_prefix_hit=(dm["prefix_cache_hits"] / dm["prefix_cache_queries"]) if dm["prefix_cache_queries"] else None)
        if idsf is not None:
            with mlock:
                idsf.write(json.dumps(dict(session=sid, turn=j, prompt_ids=pids, out_ids=oids, out_text="".join(txt))) + "\n"); idsf.flush()
        rows.append(r)
        time.sleep(a.tool_gap_s)
    return rows, time.time() - t0
M0 = metrics(); T0 = time.time()
with cf.ThreadPoolExecutor(a.conc) as ex: res = list(ex.map(run_session, sessions))
wall = time.time() - T0; M1 = metrics(); D = delta(M0, M1)
turns = [r for rr, _ in res for r in rr]; sess_e2e = [e for _, e in res]
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else None
summ = dict(finish=dict(collections.Counter(str(r["finish"]) for r in turns)), errors=sum(1 for r in turns if r["error"]), free=a.free, sessions=len(res), turns=len(turns), conc=a.conc, wall_s=wall, tool_gap_s=a.tool_gap_s, max_ctx=a.max_ctx,
            prompt_tokens_mean=st.mean(r["prompt_tokens"] for r in turns), out_tokens_mean=st.mean(r["out_tokens"] for r in turns),
            ttft_p50_s=q([r["ttft_s"] for r in turns], .5), ttft_p99_s=q([r["ttft_s"] for r in turns], .99),
            decode_tps_mean=st.mean(r["decode_tps"] for r in turns if r["decode_tps"]),
            decode_tps_tokweighted=sum(r["out_tokens"] - 1 for r in turns if r["decode_tps"]) / sum((r["out_tokens"] - 1) / r["decode_tps"] for r in turns if r["decode_tps"]),
            itl_ms_p50=q([r["itl_ms_mean"] for r in turns if r["itl_ms_mean"]], .5),
            session_e2e_p50_s=q(sess_e2e, .5), session_e2e_max_s=max(sess_e2e), out_tok_per_s_gpu=sum(r["out_tokens"] for r in turns) / wall,
            prefix_hit_rate=D["prefix_cache_hits"] / D["prefix_cache_queries"] if D["prefix_cache_queries"] else None,
            spec_accept_rate=D["accept_rate"], spec_accept_len=D["accept_len"], spec_draft_len_mean=D["draft_len"], spec_per_pos_rate=D["per_pos_rate"])
if a.per_turn_metrics:
    al = [r["turn_accept_len"] for r in turns if r.get("turn_accept_len")]
    if al: summ.update(turn_al_p10=q(al, .1), turn_al_p50=q(al, .5), turn_al_p90=q(al, .9), turn_al_max=max(al))
json.dump(dict(summary=summ, args=vars(a), turns=turns), open(os.path.join(a.out, "replay_openhands.json"), "w"), indent=1)
pd.DataFrame(turns).to_csv(os.path.join(a.out, "turns.csv"), index=False)
print(json.dumps(summ, indent=1))
