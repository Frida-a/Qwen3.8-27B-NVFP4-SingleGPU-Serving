#!/usr/bin/env python3
"""Natural long-document proxy set: token ids (Qwen3.8 tokenizer) of decontaminated records >= --min tokens,
truncated to --max tokens. Records flagged by decontam.py (any 50-gram span hit, or >= --max13 distinct 13-gram
hits against any eval set) are skipped.
usage: build_docs.py --tok <model> --raw data/lc_calib/raw --report results/lcq/decontam/report.json
                     --spec pg19:12,prolong_code:12,swe_traj:8 --out data/lc_calib/docs_v1.jsonl"""
import argparse, json, os
from lc_stats import render


def flagged(rep, src, max13):
    bad = set()
    for h in rep.get(src, {}).get("hits", []):
        for k, v in h.items():
            if isinstance(v, list) and (v[1] > 0 or v[0] >= max13): bad.add(h["id"])
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min", type=int, default=32768); ap.add_argument("--max", type=int, default=102400)
    ap.add_argument("--max13", type=int, default=3)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    rep = json.load(open(a.report))
    n_out = 0
    with open(a.out, "w") as fh:
        for item in a.spec.split(","):
            src, n = item.split(":"); n = int(n); k = 0
            if src not in rep or not os.path.exists(f"{a.raw}/{src}.jsonl"):
                print(f"{src}: missing or not decontam-checked -> skipped"); continue
            bad = flagged(rep, src, a.max13)
            for line in open(f"{a.raw}/{src}.jsonl"):
                r = json.loads(line)
                if r["id"] in bad or r.get("tokens", 0) < a.min: continue
                ids = tok.encode(render(tok, r), add_special_tokens=False)[: a.max]
                fh.write(json.dumps({"id": r["id"], "src": src, "ids": ids}) + "\n"); k += 1
                if k >= n: break
            print(f"{src}: {k} docs (skipped {len(bad)} flagged)"); n_out += k
    print(f"{n_out} docs -> {a.out}")


if __name__ == "__main__":
    main()
