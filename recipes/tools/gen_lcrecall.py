#!/usr/bin/env python3
"""QW38 lc-quant: our own long-context multi-needle recall set ("lcrecall"), built without a GPU and without any
benchmark content. Used as (a) teacher-forced proxy (KL/NLL on the gold-answer span) and (b) accuracy proxy.

* Haystack: real multi-turn user/assistant exchanges from nvidia/Nemotron-SFT-Instruction-Following-Chat-v2 (ODC-By; streamed
  from the Hub; first turn pair of each conversation, assistant answers without reasoning).
* Needles: k in {2,4,8} turns "user: <request about LABEL>" -> "assistant: <passage>", where the passages are
  distinct PG-19 paragraphs (Apache-2.0; public-domain books) and LABEL is a made-up subject from our own word
  lists. The k requests use the same label, so the model must track order among k similar items.
* Final turn: "Quote passage number i ... about LABEL ... start with <code>" and the gold answer "<code>\\n<passage>".
  Wording is ours (not the openai/mrcr template); decontam.py checks the result against every eval set.
* Lengths: --lengths (default 32K/64K/120K tokens, Qwen3.8 chat template, thinking off).

usage: gen_lcrecall.py --tok <model_dir> --pg19 data/lc_calib/raw/pg19.jsonl --out data/lc_calib/lcrecall_v2.jsonl
                       [--n_per_cell 6 --seed 0 --chat nvidia/Nemotron-SFT-Instruction-Following-Chat-v2:reasoning_off]
"""
import argparse, json, random, re, string

ADJ = ["amber", "northern", "quiet", "copper", "hollow", "winter", "silver", "distant", "salt", "paper", "iron", "velvet",
       "orchard", "harbor", "midnight", "granite", "lantern", "meadow", "cobalt", "tidal"]
NOUN = ["ledger", "archive", "notebook", "folio", "dossier", "journal", "chronicle", "portfolio", "scrapbook", "register"]
ASK = ["Share an excerpt for the {L} collection.", "Add another passage to the {L} collection, please.",
       "I need one more excerpt for the {L} collection.", "Give me a passage for the {L} collection."]
FINAL = ("Look back through our conversation. Quote the {ord} passage you gave me for the {L} collection, word for "
         "word. Begin your reply with the code {H} on its own line, then the passage, and write nothing else.")
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}


def paragraphs(pg19_file, rng, tok, lo=120, hi=420, max_books=200, skip_books=0):
    out = []
    for i, line in enumerate(open(pg19_file)):
        if i < skip_books: continue
        if i >= skip_books + max_books: break
        t = json.loads(line)["text"]
        paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", t)]
        paras = [p for p in paras[len(paras) // 10: -len(paras) // 10 or None] if 600 < len(p) < 2600 and "gutenberg" not in p.lower()]
        rng.shuffle(paras)
        for p in paras[:30]:
            n = len(tok.encode(p, add_special_tokens=False))
            if lo <= n <= hi: out.append(p)
    rng.shuffle(out)
    return out


def chat_pairs(spec, n, tok, skip=0):
    from datasets import load_dataset
    path, split = spec.split(":")
    out = []
    for i, r in enumerate(load_dataset(path, split=split, streaming=True)):
        if i < skip: continue
        msgs = r.get("messages") or r.get("conversations") or []
        u = next((m for m in msgs if m.get("role") == "user"), None)
        a = next((m for m in msgs if m.get("role") == "assistant"), None)
        if not u or not a or not isinstance(u.get("content"), str) or not isinstance(a.get("content"), str): continue
        ac = re.sub(r"<think>.*?</think>", "", a["content"], flags=re.S).strip()
        if not ac or len(u["content"]) > 20000: continue
        nt = len(tok.encode(u["content"] + ac, add_special_tokens=False)) + 12
        if nt > 6000: continue
        out.append((u["content"], ac, nt))
        if len(out) >= n: break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True); ap.add_argument("--pg19", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_cell", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lengths", default="32768,65536,122880"); ap.add_argument("--needles", default="2,4,8")
    ap.add_argument("--chat", default="nvidia/Nemotron-SFT-Instruction-Following-Chat-v2:reasoning_off")
    ap.add_argument("--n_chat", type=int, default=6000)
    ap.add_argument("--skip_chat", type=int, default=0); ap.add_argument("--id_prefix", default="lcr")
    ap.add_argument("--book_offset", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    rng = random.Random(a.seed)
    paras = paragraphs(a.pg19, rng, tok, skip_books=a.book_offset)
    pairs = chat_pairs(a.chat, a.n_chat, tok, a.skip_chat)
    print(f"{len(paras)} passages, {len(pairs)} haystack pairs", flush=True)
    labels = [f"{x} {y}" for x in ADJ for y in NOUN]; rng.shuffle(labels)
    lengths = [int(x) for x in a.lengths.split(",")]; ks = [int(x) for x in a.needles.split(",")]
    pi = 0; s = 0
    with open(a.out, "w") as fh:
        for L in lengths:
            for k in ks:
                for _ in range(a.n_per_cell):
                    lab = labels[s % len(labels)]
                    needles = paras[pi:pi + k]; pi += k
                    ntok = sum(len(tok.encode(p, add_special_tokens=False)) + 40 for p in needles)
                    budget = L - 700 - ntok
                    turns, used = [], 0
                    for j in rng.sample(range(len(pairs)), len(pairs)):
                        if used + pairs[j][2] > budget: continue
                        turns.append(pairs[j][:2]); used += pairs[j][2]
                        if budget - used < 300: break
                    pos = sorted(rng.sample(range(len(turns) + 1), k))
                    for off, (p, nd) in enumerate(zip(pos, needles)):
                        turns.insert(p + off, (rng.choice(ASK).format(L=lab), nd))
                    i = rng.randrange(1, k + 1)
                    h = "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))
                    msgs = []
                    for q, r in turns: msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
                    msgs.append({"role": "user", "content": FINAL.format(ord=ORD[i], L=lab, H=h)})
                    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
                    answer = f"{h}\n{needles[i - 1]}"
                    n = len(tok.encode(prompt, add_special_tokens=False))
                    fh.write(json.dumps({"id": f"{a.id_prefix}{s}", "prompt": prompt, "answer": answer, "code": h, "needles": k,
                                         "target": i, "tokens": n, "len_bin": L, "turns": len(turns),
                                         "needle_turn_pos": [p + off for off, p in enumerate(pos)],
                                         "messages": msgs}, ensure_ascii=False) + "\n")
                    print(f"{a.id_prefix}{s}: {n} tok, k={k}, ask #{i}, {len(turns)} turns", flush=True); s += 1


if __name__ == "__main__":
    main()
