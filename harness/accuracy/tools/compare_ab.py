"""Paired accuracy A/B between two Gym benchmark output dirs (same benchmark, same seeds).

usage: python3 compare_ab.py <baseline bench dir> <candidate bench dir>
Per repeat (seed): accuracy A, B, delta. Per task: the mean reward over repeats, paired by task index;
mean delta with its standard error (paired by task, so between-task variance cancels). Reward = Gym's
`reward` (for IFBench also reported on the whitespace-stripped re-score if summary.json has it).
"""
import collections, json, math, os, sys


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "rollouts.jsonl"))]
    by = {}
    for r in rows:
        by[(r["_ng_task_index"], r.get("_ng_rollout_index", 0))] = float(r.get("reward") or 0)
    return by


a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) & set(b))
print(f"paired rollouts: {len(keys)} (A {len(a)}, B {len(b)})")
for rep in sorted({k[1] for k in keys}):
    ks = [k for k in keys if k[1] == rep]
    ma = 100 * sum(a[k] for k in ks) / len(ks); mb = 100 * sum(b[k] for k in ks) / len(ks)
    print(f"  seed {rep}: A {ma:.2f}  B {mb:.2f}  delta {mb - ma:+.2f}  (n={len(ks)})")
ta, tb = collections.defaultdict(list), collections.defaultdict(list)
for k in keys:
    ta[k[0]].append(a[k]); tb[k[0]].append(b[k])
d = [100 * (sum(tb[t]) / len(tb[t]) - sum(ta[t]) / len(ta[t])) for t in ta]
m = sum(d) / len(d); sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)); se = sd / math.sqrt(len(d))
MA = 100 * sum(a[k] for k in keys) / len(keys); MB = 100 * sum(b[k] for k in keys) / len(keys)
print(f"overall: A {MA:.2f}  B {MB:.2f}  delta {m:+.2f} +- {se:.2f} (paired-by-task SE, {len(d)} tasks; 95% CI {m - 1.96 * se:+.2f}..{m + 1.96 * se:+.2f})")
for name, dd in (("A", sys.argv[1]), ("B", sys.argv[2])):
    s = os.path.join(dd, "summary.json")
    if os.path.exists(s):
        j = json.load(open(s))
        if "ifbench_rescore" in j:
            print(f"  {name} IFBench rescore:", j["ifbench_rescore"])
