#!/usr/bin/env python3
"""QW38 lc-quant: n-gram decontamination of calibration / proxy data against every eval set we use.

Normalization: lowercase, words = regex \\w+ (unicode), so punctuation/whitespace/markup differences do not hide overlap.
  * 13-gram check (GPT-3 / Llama-style): fraction of a candidate's 13-grams that occur anywhere in the eval sets,
    and the number of distinct colliding 13-grams.
  * long-span check: 50-word exact spans (~300 chars) -> catches copied passages (e.g. MRCR haystack text, AA-LCR docs).
Hashes: crc32 per word -> 64-bit polynomial rolling hash over the window (collision rate negligible at these sizes).

  build: decontam.py build --out work/decontam/index [--extra NAME=GLOB ...] [--allow-missing]
  check: decontam.py check --index work/decontam/index --data a.jsonl [b.jsonl ...] --report r.json
         records need "text" or "messages"; adds nothing to the data, writes per-record hits to the report.

Prerequisites for `build` (run from the repo root, or set QW38_GYM / QW38_EVALS to absolute paths):
  * a NeMo Gym v0.7.0 checkout at ./Gym (or QW38_GYM) with these benchmarks prepared, from inside the checkout:
    `for b in gpqa ifbench scicode livecodebench/v6_2408_2505; do gym eval prepare --config benchmarks/$b/config.yaml; done`
    (SciCode and LiveCodeBench are indexed as contamination protection only; they are not reported benchmarks);
  * the locally built sets under harness/accuracy/benchmarks (or QW38_EVALS): mrcr_qwen38_128k (command in the header
    of its config.yaml; harness/README.md, section 3) and mmmu_pro_std10_full (index input only, built with
    recipes/tools/build_mmmu_pro_full.py).
Every set marked required below must be present; a missing one is an error unless --allow-missing is given (then it
is skipped and recorded as missing in meta.json). Optional sets (Gym example rows) are indexed when present.
The index decides which calibration records are dropped, so the calibration file's sha256 reproduces only with the
same eval files indexed. The measured index differed from this default in three ways: MMMU-Pro was the 400-question
development subset (the full set here is a superset), it also held a 1,400-question stratified MMLU-Pro subset, and
it held the AA-LCR documents and questions (both via --extra). HLE and the MRCR >128K slice were not indexed.
"""
import argparse, glob, json, os, re, sys, zlib
import numpy as np

# Eval files to index. QW38_GYM: NeMo Gym checkout with prepared benchmark data (default ./Gym); QW38_EVALS: the
# harness/accuracy/benchmarks dir with the locally built sets (default relative to the repo root).
G = os.environ.get("QW38_GYM", "Gym")
E = os.environ.get("QW38_EVALS", "harness/accuracy/benchmarks")
# name -> (files, required). A required set needs its first file; further files of a set are optional additions.
EVALS = {
    "gpqa_diamond": ([f"{G}/benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl"], True),
    "ifbench": ([f"{G}/benchmarks/ifbench/data/ifbench_benchmark.jsonl", f"{G}/benchmarks/ifbench/data/IFBench_test.jsonl"], True),
    "scicode": ([f"{G}/benchmarks/scicode/data/scicode_benchmark.jsonl"], True),
    "livecodebench": ([f"{G}/benchmarks/livecodebench/v6_2408_2505/data/livecodebench_v6_validation.jsonl"], True),
    "mrcr": ([f"{E}/mrcr_qwen38_128k/data/mrcr_qwen38_128k_benchmark.jsonl"], True),
    "mmmu_pro": ([f"{E}/mmmu_pro_std10_full/data/mmmu_pro_std10_full_benchmark.jsonl"], True),
    "aalcr_examples": ([f"{G}/resources_servers/aalcr/data/example.jsonl"], False),
    "terminal_bench_examples": ([f"{G}/resources_servers/terminal_bench_2_1/data/example.jsonl"], False),
}
SKIP_KEYS = {"image_url", "packed", "uuid", "id", "mmmu_id", "problem_id", "random_string_to_prepend"}
WRE = re.compile(r"\w+")
P = np.uint64(1000003)


def strings(o, key=None):
    """All natural-language strings in a json value (skips images / packed tests / ids)."""
    if key in SKIP_KEYS: return
    if isinstance(o, str):
        if o.startswith("data:") or (len(o) > 2000 and " " not in o[:2000]): return
        yield o
    elif isinstance(o, dict):
        for k, v in o.items(): yield from strings(v, k)
    elif isinstance(o, list):
        for v in o: yield from strings(v, key)


_wc = {}
def word_ids(text):
    ws = WRE.findall(text.lower())
    out = np.empty(len(ws), np.uint64)
    for i, w in enumerate(ws):
        h = _wc.get(w)
        if h is None:
            h = zlib.crc32(w.encode()) + 1
            if len(_wc) < 5_000_000: _wc[w] = h
        out[i] = h
    return out


def ngrams(ids, n):
    if len(ids) < n: return np.empty(0, np.uint64)
    m = len(ids) - n + 1
    h = np.zeros(m, np.uint64)
    with np.errstate(over="ignore"):
        for i in range(n): h = h * P + ids[i:i + m]
    return h


def rec_text(r):
    if "text" in r: return r["text"]
    if "messages" in r: return "\n".join(m["content"] for m in r["messages"])
    return "\n".join(strings(r))


def build(a):
    files = {k: (fl, req) for k, (fl, req) in EVALS.items()}
    for x in a.extra or []:
        name, path = x.split("=", 1); files[name] = (sorted(glob.glob(path, recursive=True)) or [path], True)
    missing = [(k, fl[0]) for k, (fl, req) in files.items() if req and not os.path.exists(fl[0])]
    if missing and not a.allow_missing:
        sys.exit("decontam.py build: required eval sets are missing (cwd " + os.getcwd() + "):\n"
                 + "".join(f"  {k}: {f}\n" for k, f in missing)
                 + "Prepare them (see the docstring of this file: Gym `gym eval prepare`, MRCR-128K, build_mmmu_pro_full.py), point\n"
                 + "QW38_GYM / QW38_EVALS at them, or pass --allow-missing to build a smaller index (the calibration\n"
                 + "set can then differ from the measured one).")
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for name, (fl, req) in files.items():
        g13, g50, nw, nd = [], [], 0, 0
        for fn in fl:
            if not os.path.exists(fn): print(f"MISSING {name}{' (required, --allow-missing)' if req and fn == fl[0] else ' (optional)'}: {fn}"); continue
            it = open(fn) if fn.endswith(".jsonl") else [json.dumps({"text": open(fn, errors="ignore").read()})]
            for line in it:
                r = json.loads(line); nd += 1
                for s in strings(r):
                    ids = word_ids(s); nw += len(ids)
                    g13.append(np.unique(ngrams(ids, 13))); g50.append(np.unique(ngrams(ids, 50)))
        g13 = np.unique(np.concatenate(g13)) if g13 else np.empty(0, np.uint64)
        g50 = np.unique(np.concatenate(g50)) if g50 else np.empty(0, np.uint64)
        np.save(f"{a.out}/{name}.13.npy", g13); np.save(f"{a.out}/{name}.50.npy", g50)
        meta[name] = {"files": [f for f in fl if os.path.exists(f)], "missing": [f for f in fl if not os.path.exists(f)],
                      "docs": nd, "words": nw, "uniq13": len(g13), "uniq50": len(g50)}
        print(name, meta[name], flush=True)
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)


def check(a):
    meta = json.load(open(f"{a.index}/meta.json"))
    idx = {k: (np.load(f"{a.index}/{k}.13.npy"), np.load(f"{a.index}/{k}.50.npy")) for k in meta}
    rep = json.load(open(a.report)) if os.path.exists(a.report) else {}
    for fn in a.data:
        src = os.path.basename(fn).rsplit(".", 1)[0]
        per, tot13, hit13 = [], 0, {k: 0 for k in idx}
        for line in open(fn):
            r = json.loads(line)
            ids = word_ids(rec_text(r))
            u13 = np.unique(ngrams(ids, 13)); u50 = np.unique(ngrams(ids, 50)); tot13 += len(u13)
            row = {"id": r.get("id"), "n13": int(len(u13))}
            for k, (e13, e50) in idx.items():
                h13 = int(np.isin(u13, e13, assume_unique=True).sum()); h50 = int(np.isin(u50, e50, assume_unique=True).sum())
                hit13[k] += h13
                if h13 or h50: row[k] = [h13, h50]
            per.append(row)
        flagged = {k: sum(1 for p in per if k in p) for k in idx}
        span = {k: sum(1 for p in per if k in p and p[k][1] > 0) for k in idx}
        rep[src] = {"file": fn, "records": len(per), "uniq13": tot13,
                    "overlap13_rate": {k: hit13[k] / max(tot13, 1) for k in idx},
                    "records_with_13gram_hit": flagged, "records_with_50gram_span": span,
                    "hits": [p for p in per if len(p) > 2]}
        print(src, {k: (round(rep[src]["overlap13_rate"][k] * 1e6, 2), flagged[k], span[k]) for k in idx}, flush=True)
    json.dump(rep, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--out", required=True); b.add_argument("--extra", nargs="*")
    b.add_argument("--allow-missing", action="store_true", help="skip missing required eval sets instead of failing")
    c = sp.add_parser("check"); c.add_argument("--index", required=True); c.add_argument("--data", nargs="+", required=True)
    c.add_argument("--report", required=True)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else check(a)
