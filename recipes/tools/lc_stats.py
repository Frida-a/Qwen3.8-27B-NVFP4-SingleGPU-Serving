#!/usr/bin/env python3
"""Token-length histogram of data/lc_calib/raw/<src>.jsonl with the Qwen3.8 tokenizer.
Texts are encoded raw; chats are rendered with the model's chat template (thinking off); falls back to a plain
role-tagged join if the template rejects a conversation (e.g. orphan tool turns).
Adds "tokens" to each record in place (rewrites the file) and prints/returns a summary.
usage: lc_stats.py --tok <model_dir> --raw data/lc_calib/raw [--src a,b] --out results/lcq/source_stats.json"""
import argparse, glob, json, os
import numpy as np

BINS = [0, 8192, 16384, 32768, 65536, 131072, 262144, 10 ** 9]
LABELS = ["<8K", "8-16K", "16-32K", "32-64K", "64-128K", "128-256K", ">256K"]


def render(tok, r):
    if "text" in r: return r["text"]
    try:
        return tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False)
    except Exception:
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in r["messages"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--src", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    summ = json.load(open(a.out)) if os.path.exists(a.out) else {}
    files = [f"{a.raw}/{s}.jsonl" for s in a.src.split(",")] if a.src else sorted(glob.glob(f"{a.raw}/*.jsonl"))
    for fn in files:
        src = os.path.basename(fn)[:-6]
        rows = [json.loads(l) for l in open(fn)]
        texts = [render(tok, r) for r in rows]
        lens = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
        for r, n in zip(rows, lens): r["tokens"] = n
        with open(fn + ".tmp", "w") as fh:
            for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(fn + ".tmp", fn)
        L = np.array(lens)
        h = np.histogram(L, BINS)[0].tolist()
        summ[src] = {"n": len(L), "tokens_total": int(L.sum()), "p10": int(np.percentile(L, 10)), "p50": int(np.median(L)),
                     "p90": int(np.percentile(L, 90)), "max": int(L.max()), "hist": dict(zip(LABELS, h)),
                     "n_ge_32K": int((L >= 32768).sum()), "n_ge_64K": int((L >= 65536).sum())}
        print(src, summ[src], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summ, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
