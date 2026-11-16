"""Paired MRCR A/B by needle count.  usage: python3 mrcr_by_needle.py <A bench dir> <B bench dir>
Pairs rollouts by (_ng_task_index, _ng_rollout_index); reward = seq-match; prints delta B-A +- paired SE overall and per n_needles,
plus mean output tokens, truncation (incomplete), empties (no final answer text in the response output)."""
import json, math, os, sys, collections
def load(d):
    return {(r["_ng_task_index"], r.get("_ng_rollout_index", 0)): r for r in map(json.loads, open(os.path.join(d, "rollouts.jsonl")))}
A, B = load(sys.argv[1]), load(sys.argv[2]); keys = sorted(set(A) & set(B))
def stat(ks):
    d = [float(B[k]["reward"] or 0) - float(A[k]["reward"] or 0) for k in ks]; n = len(d); m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0
    a = 100 * sum(float(A[k]["reward"] or 0) for k in ks) / n; b = 100 * sum(float(B[k]["reward"] or 0) for k in ks) / n
    return n, a, b, 100 * m, 100 * se
tok = lambda R, ks: sum(R[k]["response"]["usage"]["output_tokens"] for k in ks) / len(ks)
def final_text(r):
    """Final answer text of one rollout, from the raw Gym (Responses API) output: the text of the "message" items
    (reasoning items excluded). Falls back to a precomputed final_text_len if a compacted file has no output items."""
    resp = r.get("response") or {}
    out = resp.get("output")
    if out is None:
        return "x" * int(resp.get("final_text_len") or 0)
    return "".join(c.get("text") or "" for o in out if isinstance(o, dict) and o.get("type") == "message"
                   for c in (o.get("content") or []) if isinstance(c, dict))
emp = lambda R, ks: sum(1 for k in ks if not final_text(R[k]).strip())
inc = lambda R, ks: sum(1 for k in ks if R[k]["response"].get("status") != "completed")
n, a, b, m, se = stat(keys)
print(f"paired n={n}  A {a:.2f}  B {b:.2f}  delta {m:+.2f} +- {se:.2f} (95% CI {m-1.96*se:+.2f}..{m+1.96*se:+.2f})  tok A {tok(A,keys):.0f} B {tok(B,keys):.0f}  empties A {emp(A,keys)} B {emp(B,keys)}  incomplete A {inc(A,keys)} B {inc(B,keys)}")
by = collections.defaultdict(list)
for k in keys: by[A[k].get("n_needles")].append(k)
for nn in sorted(by): n, a, b, m, se = stat(by[nn]); print(f"  needles {nn}: n={n} A {a:.2f} B {b:.2f} delta {m:+.2f} +- {se:.2f}")
