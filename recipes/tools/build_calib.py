#!/usr/bin/env python3
"""Long-context PTQ calibration set (hf_ptq --dataset <jsonl>, records {"text"}), from decontaminated sources only.
Each record is cut to exactly --seq tokens (Qwen3.8 tokenizer) so --calib_seq <seq> never pads or truncates mid-way:
  books   PG-19 windows (random offset inside a book >= seq tokens)
  code    ProLong The-Stack repo concatenations
  tbk     ProLong textbooks
  chat    lcrecall-style long multi-turn chats (Nemotron SFT haystack + PG-19 needles + our recall question +
          gold answer), i.e. the retrieval pattern MRCR stresses, with our own content
  agent   agent trajectories (chat-templated), if present
Records flagged by the decontam report (any 50-gram span, or >= 3 distinct 13-gram hits vs any eval) are skipped,
and records used by the proxy sets (docs_v1 ids, lcrecall_v2 ids) are excluded so calibration != validation.
usage: build_calib.py --tok <model> --raw data/lc_calib/raw --report results/lcq/decontam/report.json
         --chat data/lc_calib/lccalib_chat.jsonl --mix books:32,code:32,tbk:16,chat:32,agent:16 --seq 32768
         --exclude data/lc_calib/docs_v1.jsonl --out data/lc_calib/lc_calib_128x32k.jsonl
"""
import argparse, json, os, random
from build_docs import flagged
from lc_stats import render

SRC = {"books": "pg19", "code": "prolong_code", "tbk": "prolong_tbk", "agent": "swe_traj"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--chat", required=True); ap.add_argument("--mix", required=True); ap.add_argument("--seq", type=int, default=32768)
    ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok); rng = random.Random(a.seed)
    rep = json.load(open(a.report))
    excl = {json.loads(l)["id"] for f in a.exclude for l in open(f)}
    out, meta = [], {}
    for item in a.mix.split(","):
        kind, n = item.split(":"); n = int(n); got = 0
        if kind == "chat":
            src, fn = os.path.basename(a.chat)[:-6], a.chat
        else:
            src = SRC[kind]; fn = f"{a.raw}/{src}.jsonl"
        if not os.path.exists(fn) or src not in rep:
            print(f"{kind}: {src} missing or not decontam-checked -> 0"); meta[kind] = 0; continue
        bad = flagged(rep, src, 3)
        rows = [json.loads(l) for l in open(fn)]; rng.shuffle(rows)
        for r in rows:
            if got >= n: break
            if r["id"] in bad or r["id"] in excl: continue
            text = r["prompt"] + r["answer"] + "<|im_end|>\n" if kind == "chat" else render(tok, r)
            ids = tok.encode(text, add_special_tokens=False)
            if kind == "chat" and len(ids) <= a.seq: off, L = 0, len(ids)       # whole chat incl. question + answer
            elif len(ids) < a.seq: continue
            else: L = a.seq; off = rng.randrange(0, len(ids) - a.seq + 1) if kind in ("books", "code", "tbk") else len(ids) - a.seq
            out.append({"text": tok.decode(ids[off:off + L]), "src": src, "id": r["id"]}); got += 1
        meta[kind] = got; print(f"{kind}: {got}/{n} from {src} (flagged {len(bad)}, excluded {len(excl)})", flush=True)
    rng.shuffle(out)
    with open(a.out, "w") as fh:
        for r in out: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} x {a.seq} tokens -> {a.out}  {meta}")


if __name__ == "__main__":
    main()
