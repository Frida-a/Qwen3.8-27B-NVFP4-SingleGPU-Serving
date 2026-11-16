#!/usr/bin/env python3
"""QW38 lc-quant: fetch small subsets of candidate long-context calibration sources (streaming, no full downloads)
and write normalized jsonl to data/lc_calib/raw/<src>.jsonl: {"src", "id", "text"} or {"src", "id", "messages"}.

Sources (license / note):
  pg19          emozilla/pg19 train (parquet mirror of deepmind/pg19), Apache-2.0, public-domain books <1919
  prolong_code  princeton-nlp/prolong-data-64K thestackv1_concat_by_repo-65536 (The Stack v1, repo-concatenated),
                Llama-3-tokenized MDS -> decoded with the ProLong tokenizer
  prolong_tbk   princeton-nlp/prolong-data-64K textbooks (TextbookChapters)
  longalign     zai-org/LongAlign-10k long.jsonl (long-document instruction SFT, 8-64K)
  swe_traj      nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0), OpenHands agent trajectories
  nemo_swe      nvidia/Nemotron-SWE-v1 r2e_gym.jsonl (CC-BY-4.0), agent trajectories
usage: fetch_sources.py --out data/lc_calib/raw --src pg19,... [--n 200]
"""
import argparse, json, os, sys


def nd(b):
    """MDS NDArray with static dtype uint32, dynamic shape: [ndim<<2 | shape-dtype code][shape][values]."""
    import numpy as np
    h = b[0]; ndim = h >> 2; sw = (1, 2, 4, 8)[h & 3]
    shape = np.frombuffer(b[1:1 + ndim * sw], {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[sw])
    return np.frombuffer(b[1 + ndim * sw:], np.uint32).reshape(shape.astype(np.int64))


def w(fh, rec):
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def stream(path, n, **kw):
    from datasets import load_dataset
    ds = load_dataset(path, streaming=True, **kw)
    for i, r in enumerate(ds):
        if i >= n: break
        yield i, r


def to_msgs(conv):
    """Normalize OpenAI / ShareGPT style conversations to [{"role", "content"}] with string content."""
    out = []
    for m in conv:
        role = m.get("role") or {"human": "user", "gpt": "assistant"}.get(m.get("from"), m.get("from"))
        c = m.get("content", m.get("value"))
        if isinstance(c, list):
            c = "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
        if c is None: c = ""
        if role == "tool" or role == "function": role = "tool"
        tc = m.get("tool_calls")
        if tc: c = (c + "\n" if c else "") + json.dumps(tc, ensure_ascii=False)
        out.append({"role": role, "content": str(c)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--src", required=True); ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for src in a.src.split(","):
        fn = f"{a.out}/{src}.jsonl"
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            print(f"{src}: exists, skip"); continue
        k = 0
        try:
          with open(fn + ".tmp", "w") as fh:
            if src == "pg19":
                for i, r in stream("emozilla/pg19", a.n, split="train"):
                    w(fh, {"src": src, "id": r.get("url") or str(i), "text": r["text"]}); k += 1
            elif src in ("prolong_code", "prolong_tbk", "prolong_book"):
                from huggingface_hub import hf_hub_download
                from transformers import AutoTokenizer
                import numpy as np
                sub = {"prolong_code": "thestackv1_concat_by_repo-65536", "prolong_tbk": "textbooks",
                       "prolong_book": "book-65536"}[src]
                d = f"{sub}/proc00-64"
                loc = os.path.join(a.out, "_mds", src)
                idx = json.load(open(hf_hub_download("princeton-nlp/prolong-data-64K", f"{d}/index.json", repo_type="dataset", local_dir=loc)))
                info = idx["shards"][0]
                assert info["column_names"] == ["domain", "indices", "input_ids", "length"] and info["compression"] is None
                buf = open(hf_hub_download("princeton-nlp/prolong-data-64K", f"{d}/shard.00000.mds", repo_type="dataset", local_dir=loc), "rb").read()
                ltok = AutoTokenizer.from_pretrained("princeton-nlp/Llama-3-8B-ProLong-64k-Base")
                ns = int(np.frombuffer(buf, np.uint32, 1)[0]); offs = np.frombuffer(buf, np.uint32, ns + 1, 4)
                for i in range(min(a.n, ns)):   # MDS v2 sample: 3 uint32 sizes (str, ndarray, ndarray) + data + uint64
                    b = buf[offs[i]:offs[i + 1]]; sz = np.frombuffer(b, np.uint32, 3); p = 12
                    dom = b[p:p + sz[0]].decode(); p += sz[0]
                    spans = nd(b[p:p + sz[1]]).reshape(-1, 2).tolist(); p += sz[1]
                    ids = nd(b[p:p + sz[2]]).astype(np.int64).tolist()
                    spans = spans or [(0, len(ids))]
                    # one record per packed sequence; documents joined by blank lines (repo files stay contiguous)
                    text = "\n\n".join(ltok.decode(ids[x:y], skip_special_tokens=True) for x, y in spans)
                    w(fh, {"src": src, "id": f"{sub}/shard0/{i}", "domain": dom, "text": text, "n_docs": len(spans)}); k += 1
                import shutil; shutil.rmtree(loc, ignore_errors=True)   # keep only the decoded subset
            elif src == "longalign":
                for i, r in stream("zai-org/LongAlign-10k", a.n, split="train"):
                    w(fh, {"src": src, "id": str(i), "messages": to_msgs(r["messages"])}); k += 1
            elif src == "swe_traj":
                for i, r in stream("nebius/SWE-rebench-openhands-trajectories", a.n, split="train"):
                    conv = r.get("trajectory") or r.get("messages")
                    if isinstance(conv, str): conv = json.loads(conv)
                    w(fh, {"src": src, "id": str(r.get("instance_id", i)), "messages": to_msgs(conv),
                           "resolved": r.get("resolved")}); k += 1
            elif src == "nemo_swe":
                for i, r in stream("nvidia/Nemotron-SWE-v1", a.n, split="train", data_files="data/r2e_gym.jsonl"):
                    conv = r.get("messages") or r.get("conversations")
                    w(fh, {"src": src, "id": str(r.get("uuid", i)), "messages": to_msgs(conv)}); k += 1
            else:
                sys.exit(f"unknown src {src}")
          os.replace(fn + ".tmp", fn)
          print(f"{src}: {k} records -> {fn}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"{src}: FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
