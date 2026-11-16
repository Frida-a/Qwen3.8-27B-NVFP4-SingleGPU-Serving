"""Pooled per-config comparison on one benchmark, with run-level and rollout-level SE.
usage: slice_pool.py <bench> <ref config> CONFIG=rundir,rundir ...   (each rundir = one eval job's output dir holding <bench>/rollouts.jsonl)
Run level: each job is one unit (its mean over all its rollouts). Between-job SD is pooled across all configs
with >=2 jobs (more df than per config); SE(delta) = s_pooled * sqrt(1/n_a + 1/n_b).
Rollout level: per-task means pooled over every rollout of every job in a config; delta paired by task, SE across tasks."""
import sys, json, glob, math, collections
bench, ref = sys.argv[1], sys.argv[2]
cfg = {}
for a in sys.argv[3:]:
    k, v = a.split('='); cfg[k] = v.split(',')
def load(j):
    d = j if glob.os.path.isdir(j) else glob.glob(f'eval-*-{j}')[0]
    return d, [json.loads(l) for l in open(f'{d}/{bench}/rollouts.jsonl')]
runs = {}; tasks = {}; needle = {}; eos = {}; nro = {}
for k, js in cfg.items():
    runs[k] = []; pt = collections.defaultdict(list); e = n = 0
    for j in js:
        d, rs = load(j)
        runs[k].append(100 * sum(float(r.get('reward') or 0) for r in rs) / len(rs))
        for r in rs:
            pt[r['_ng_task_index']].append(float(r.get('reward') or 0)); needle[r['_ng_task_index']] = r.get('n_needles')
            resp = r['response'] if isinstance(r['response'], dict) else {}
            ft = ''.join(c.get('text') or '' for o in resp.get('output') or [] if o.get('type') == 'message' for c in o.get('content') or [])
            u = (resp.get('usage') or {}).get('output_tokens') or 0; n += 1
            if not ft.strip() and 0 < u < 65536: e += 1
    tasks[k] = {t: sum(v) / len(v) for t, v in pt.items()}; eos[k] = e; nro[k] = n
ss = df = 0
for k, v in runs.items():
    if len(v) > 1:
        m = sum(v) / len(v); ss += sum((x - m) ** 2 for x in v); df += len(v) - 1
sp = math.sqrt(ss / df) if df else float('nan')
print(f'{bench}: pooled between-job SD = {sp:.2f} pts (df {df})')
print(f'{"config":8} {"jobs":>4} {"mean":>6}  job means              EOS        | vs {ref}: run-level d +- SE        | rollout-level d +- SE (paired tasks)  | needles 2/4/8 d')
R = sum(runs[ref]) / len(runs[ref])
for k in cfg:
    m = sum(runs[k]) / len(runs[k]); line = f'{k:8} {len(runs[k]):>4} {m:6.2f}  {" / ".join(f"{x:.2f}" for x in runs[k]):22} {eos[k]:>3}/{nro[k]:<5}'
    if k != ref:
        se_r = sp * math.sqrt(1 / len(runs[k]) + 1 / len(runs[ref]))
        dd = [tasks[k][t] - tasks[ref][t] for t in tasks[k] if t in tasks[ref]]
        md = sum(dd) / len(dd); se_t = math.sqrt(sum((x - md) ** 2 for x in dd) / (len(dd) - 1) / len(dd))
        bn = collections.defaultdict(list)
        for t in tasks[k]:
            if t in tasks[ref]: bn[needle[t]].append(tasks[k][t] - tasks[ref][t])
        nd = ' '.join(f'{100*sum(v)/len(v):+.1f}' for _, v in sorted(bn.items(), key=lambda x: str(x[0])))
        line += f' | {m-R:+6.2f} +- {se_r:4.2f} (t={(m-R)/se_r:+.1f})   | {100*md:+6.2f} +- {100*se_t:4.2f}                  | {nd}'
    print(line)
