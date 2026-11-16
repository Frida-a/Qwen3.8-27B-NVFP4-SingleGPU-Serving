#!/usr/bin/env python3
"""20-prompt numerics gate (harness/README.md): GPQA-diamond first 20 questions, thinking on, GREEDY (T=0) so runs are comparable.
Reports accuracy (n=20, coarse) and saves token-level output; --compare <baseline json> reports exact-match rate,
first-divergence position and mean |dlogprob| on the common prefix (numerics-preservation check for kernel changes).
usage: gate20.py --base http://localhost:8000 --data gpqa.jsonl --out out.json [--compare base.json] [--max-tokens 16384]"""
import argparse, json, re, time, concurrent.futures as cf
from openai import OpenAI
ap = argparse.ArgumentParser(); ap.add_argument("--base", default="http://localhost:8000"); ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--compare"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--max-tokens", type=int, default=16384); ap.add_argument("--conc", type=int, default=0, help="client concurrency (default: all n at once)")
a = ap.parse_args()
import os; os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)   # create the output dir before the long run
c = OpenAI(base_url=a.base + "/v1", api_key="x", timeout=7200)
rows = [json.loads(l) for l in open(a.data)][: a.n]
def one(r):
    p = r["question"].strip() + "\n\n" + r["options_text"].strip() + "\n\nPlease reason step by step, and put the letter of your final answer within \\boxed{}."
    x = c.chat.completions.create(model="qw38", messages=[{"role": "user", "content": p}], max_tokens=a.max_tokens, temperature=0.0,
                                  logprobs=True, top_logprobs=1, extra_body={"skip_special_tokens": False})
    ch = x.choices[0]; ct = ch.message.content or ""; rc = (ch.message.model_extra or {}).get("reasoning_content") or (ch.message.model_extra or {}).get("reasoning") or ""
    m = re.findall(r"\\boxed\{\s*(?:\\text\{)?\(?([ABCD])", ct) or re.findall(r"[Aa]nswer[:\s]*\**\(?([ABCD])\b", ct)
    lp = [(t.token, t.logprob) for t in (ch.logprobs.content or [])] if ch.logprobs else []
    return dict(uuid=r["uuid"], gold=r["expected_answer"].strip(), pred=m[-1] if m else None, out_tokens=x.usage.completion_tokens,
                finish=ch.finish_reason, text=rc + "\n<<content>>\n" + ct, logprobs=lp)
t0 = time.time()
with cf.ThreadPoolExecutor(a.conc or a.n) as ex: res = list(ex.map(one, rows))
for r in res: r["ok"] = r["pred"] == r["gold"]
S = dict(n=len(res), acc=sum(r["ok"] for r in res) / len(res), truncated=sum(r["finish"] == "length" for r in res),
         mean_out_tokens=sum(r["out_tokens"] for r in res) / len(res), wall_s=round(time.time() - t0))
if a.compare:
    B = {r["uuid"]: r for r in json.load(open(a.compare))["items"]}; ex_match = 0; divs = []; dl = []
    for r in res:
        b = B.get(r["uuid"]); 
        if not b: continue
        ta, tb = [t for t, _ in r["logprobs"]], [t for t, _ in b["logprobs"]]
        k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        if ta == tb: ex_match += 1
        divs.append(k); dl += [abs(x[1] - y[1]) for x, y in zip(r["logprobs"][:k], b["logprobs"][:k])]
    S.update(compare=a.compare, exact_match=ex_match / len(res), first_div_median=sorted(divs)[len(divs) // 2] if divs else None,
             first_div_min=min(divs) if divs else None, mean_abs_dlogprob_prefix=sum(dl) / len(dl) if dl else None,
             pred_agree=sum(r["pred"] == B[r["uuid"]]["pred"] for r in res if r["uuid"] in B) / len(res))
json.dump(dict(summary=S, items=res), open(a.out, "w"), indent=1); print(json.dumps(S, indent=1))
