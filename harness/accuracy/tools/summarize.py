"""Per-benchmark summary from a Gym output dir: Gym key metrics plus our own per-repeat breakdown.

usage: python3 summarize.py <bench_dir> [max_output_tokens]
Writes <bench_dir>/summary.json and prints one line per repeat. Per repeat (= _ng_rollout_index):
seed (from responses_create_params.metadata.extra_body), n rollouts, mean reward, no-answer rate
(mcqa: extracted_answer is None; otherwise empty final text), truncation rate (output_tokens >= max),
mean output tokens.
"""
import collections
import glob
import json
import os
import sys

d = sys.argv[1]
maxtok = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
rows = []
f = os.path.join(d, "rollouts.jsonl")
if os.path.exists(f):
    for line in open(f):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass


def final_text(r):
    out = (r.get("response") or {}).get("output") or []
    t = ""
    for o in out:
        if o.get("type") == "message":
            for c in o.get("content") or []:
                t += c.get("text") or ""
    return t


def seed_of(r):
    md = (r.get("responses_create_params") or {}).get("metadata") or {}
    try:
        return json.loads(md.get("extra_body") or "{}").get("seed")
    except Exception:
        return None


by_rep = collections.defaultdict(list)
for r in rows:
    by_rep[r.get("_ng_rollout_index", 0)].append(r)
reps = []
for k in sorted(by_rep):
    rs = by_rep[k]
    toks = [((r.get("response") or {}).get("usage") or {}).get("output_tokens") or 0 for r in rs]
    if any("extracted_answer" in r for r in rs):
        noans = sum(r.get("extracted_answer") in (None, "") for r in rs)
    else:
        noans = sum(not final_text(r).strip() for r in rs)
    rep = {"repeat": k, "seed": sorted({seed_of(r) for r in rs}, key=str), "n": len(rs),
           "mean_reward": sum(float(r.get("reward") or 0) for r in rs) / len(rs),
           "no_answer_rate": noans / len(rs), "truncated_rate": sum(t >= maxtok for t in toks) / len(rs),
           "mean_output_tokens": sum(toks) / len(rs)}
    reps.append(rep)
km = {}
for a in glob.glob(os.path.join(d, "*aggregate_metrics.json")):
    agg = json.load(open(a))
    for it in agg if isinstance(agg, list) else [agg]:
        km.update({k: v for k, v in (it.get("key_metrics") or {}).items() if isinstance(v, (int, float))})
fails = os.path.join(d, "rollouts_failures.jsonl")
nfail = sum(1 for _ in open(fails)) if os.path.exists(fails) else 0
allt = [((r.get("response") or {}).get("usage") or {}).get("output_tokens") or 0 for r in rows]
s = {"dir": d, "rollouts": len(rows), "tasks": len({r.get("_ng_task_index") for r in rows}), "repeats": len(reps),
     "failures": nfail, "key_metrics": km, "per_repeat": reps,
     "no_answer_rate": sum(x["no_answer_rate"] * x["n"] for x in reps) / max(len(rows), 1),
     "truncated_rate": sum(x["truncated_rate"] * x["n"] for x in reps) / max(len(rows), 1),
     "mean_output_tokens": sum(allt) / max(len(allt), 1)}
# IFBench: dual scoring with Gym's own checker, raw final text vs whitespace-stripped (the qwen3 reasoning
# parser leaves the "\n\n" after </think> in content; strict checks fail on it). Reported as "stripped (raw)".
if os.path.basename(d).startswith("ifbench") and rows:
    import subprocess
    srv = os.path.join(os.environ.get("GYM", "Gym"),
                       "resources_servers/ifbench")
    try:
        p = subprocess.run([os.path.join(srv, ".venv/bin/python"), os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ifbench_rescore.py"), os.path.abspath(f)], cwd=srv, capture_output=True, text=True, timeout=1800)
        s["ifbench_rescore"] = {l.split()[0]: {kv.split("=")[0]: float(kv.split("=")[1]) for kv in l.split()[2:]}
                                for l in p.stdout.splitlines() if l.startswith(("raw", "stripped"))}
        if not s["ifbench_rescore"]:
            s["ifbench_rescore"] = {"error": f"ifbench_rescore.py rc={p.returncode}: {p.stderr.strip()[-500:]}"}
    except Exception as e:  # never fail the summary on this
        s["ifbench_rescore"] = {"error": repr(e)}
    print("   ifbench rescore", s["ifbench_rescore"])
json.dump(s, open(os.path.join(d, "summary.json"), "w"), indent=1)
print(f"  {os.path.basename(d)}: {s['rollouts']} rollouts / {s['tasks']} tasks / {s['repeats']} repeats, "
      f"failures {nfail}, no-answer {s['no_answer_rate']:.3f}, truncated {s['truncated_rate']:.3f}, "
      f"mean out tok {s['mean_output_tokens']:.0f}")
print("   key", {k: round(v, 4) for k, v in km.items()})
for x in reps:
    print("   rep", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})
