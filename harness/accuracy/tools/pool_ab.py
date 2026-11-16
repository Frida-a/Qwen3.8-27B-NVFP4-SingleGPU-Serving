"""Pooled paired A/B: several runs per arm, paired by task (GPQA-style Gym rollouts with `reward`).

usage: python3 pool_ab.py --a <bench dir> [<bench dir> ...] --b <bench dir> [<bench dir> ...]
Per task: mean reward over every rollout of every run in an arm; delta = B - A, mean over tasks with the SE
across tasks (between-task variance cancels). Also prints per-run means, and the run-level t-style SE
(SD of run means / sqrt(n_runs)) for comparison.
"""
import argparse, collections, json, math, os

ap = argparse.ArgumentParser(); ap.add_argument("--a", nargs="+", required=True); ap.add_argument("--b", nargs="+", required=True)
args = ap.parse_args()


def arm(dirs):
    per_task = collections.defaultdict(list); runs = []; toks = []
    for d in dirs:
        rs = [json.loads(l) for l in open(os.path.join(d, "rollouts.jsonl"))]
        runs.append(100 * sum(float(r.get("reward") or 0) for r in rs) / len(rs))
        for r in rs:
            per_task[r["_ng_task_index"]].append(float(r.get("reward") or 0))
            toks.append(((r.get("response") or {}).get("usage") or {}).get("output_tokens") or 0)
    VERB.append(toks)
    return per_task, runs


VERB = []
ta, ra = arm(args.a); tb, rb = arm(args.b)
tasks = sorted(set(ta) & set(tb))
d = [100 * (sum(tb[t]) / len(tb[t]) - sum(ta[t]) / len(ta[t])) for t in tasks]
m = sum(d) / len(d); se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) / math.sqrt(len(d))
mean = lambda x: sum(x) / len(x)
sd = lambda x: math.sqrt(sum((v - mean(x)) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else float("nan")
print("A runs:", [round(x, 2) for x in ra], "mean %.2f sd %.2f" % (mean(ra), sd(ra)))
print("B runs:", [round(x, 2) for x in rb], "mean %.2f sd %.2f" % (mean(rb), sd(rb)))
print(f"pooled delta B-A = {m:+.2f} +- {se:.2f} (paired by task, {len(tasks)} tasks; 95% CI {m-1.96*se:+.2f}..{m+1.96*se:+.2f})")
print(f"run-level: {mean(rb)-mean(ra):+.2f} +- {math.sqrt(sd(ra)**2/len(ra)+sd(rb)**2/len(rb)):.2f} (Welch on run means)")
ma_, mb_ = (sum(t) / max(len(t), 1) for t in VERB)
tra, trb = (sum(x >= 65536 for x in t) / max(len(t), 1) for t in VERB)
print(f"verbosity: mean output tokens A {ma_:,.0f} B {mb_:,.0f} ({100 * (mb_ / ma_ - 1):+.1f}%); "
      f"truncation (64K) A {100 * tra:.1f}% B {100 * trb:.1f}%")
