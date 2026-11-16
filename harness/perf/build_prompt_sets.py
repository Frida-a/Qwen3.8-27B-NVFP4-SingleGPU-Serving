#!/usr/bin/env python3
"""Build the real-prompt sets used by sd_bench.py (64 prompts per domain) from public sources at run time.
No benchmark text ships with this repo; everything is downloaded (or read from a file you obtained yourself).

  gpqa      GPQA-diamond questions. GPQA is gated and its terms ask that examples not be posted online, so this
            script reads a local JSONL you export yourself (--gpqa-jsonl; rows {"question", "options_text"}), e.g. from
            NeMo Gym's prepared benchmarks/gpqa data (`gym eval prepare`). Omit it to skip the domain.
  mmlu_pro  TIGER-Lab/MMLU-Pro (test split)
  lcb       livecodebench/code_generation_lite, release v6, contest dates 2024-08..2025-05 (NeMo Gym's LCB v6 slice)
  code      openai/openai_humaneval
  chat      MT-Bench turn 1 (lmsys/mt_bench_human_judgments question set)

The measured sets were sampled with the same seeds (random.Random(0) / random_state=0) from locally staged copies
(LCB v6 from NeMo Gym's prepared livecodebench v6_2408_2505 file), so a rebuilt set can differ from the measured one if a
source changed; record the sha256 the script prints. The `lcb` loader needs datasets < 4 (harness/requirements-client.txt).
usage: build_prompt_sets.py --out data/prompt_sets.json [--gpqa-jsonl gpqa_diamond.jsonl] [--n 64]
"""
import argparse, hashlib, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--gpqa-jsonl"); ap.add_argument("--n", type=int, default=64)
a = ap.parse_args(); N = a.n; out = {}
import os; os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)   # create the output dir before the long run
from datasets import load_dataset

if a.gpqa_jsonl:
    g = [json.loads(l) for l in open(a.gpqa_jsonl)]
    out["gpqa"] = [r["question"].strip() + "\n\n" + r["options_text"].strip()
                   + "\n\nPlease reason step by step, and put the letter of your final answer within \\boxed{}."
                   for r in random.Random(0).sample(g, N)]

m = load_dataset("TIGER-Lab/MMLU-Pro", split="test").to_pandas().sample(N, random_state=0)
out["mmlu_pro"] = [r.question + "\n\n" + "\n".join(f"{'ABCDEFGHIJ'[i]}. {o}" for i, o in enumerate(r.options))
                   + "\n\nThink step by step, then give the final answer as 'The answer is (X)'." for r in m.itertuples()]

h = load_dataset("openai/openai_humaneval", split="test").to_pandas().sample(N, random_state=0)
out["code"] = ["Complete the following Python function. Return the full function in a ```python code block.\n\n```python\n"
               + p + "```" for p in h.prompt]

mt = load_dataset("lmsys/mt_bench_human_judgments", split="human").to_pandas().drop_duplicates("question_id")
out["chat"] = [t[0] for t in mt.sort_values("question_id").conversation_a.map(lambda c: [c[0]["content"]])][:N]

FM_STD = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
          "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the "
          "algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```")
FM_START = ("You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
            "\n```python\n{sc}\n```")
lcb_rows = [r for r in load_dataset("livecodebench/code_generation_lite", version_tag="release_v6", split="test",
                                    trust_remote_code=True)
            if "2024-08-01" <= r["contest_date"][:10] <= "2025-05-31"]
lcb = []
for r in random.Random(0).sample(lcb_rows, N):
    sc = (r.get("starter_code") or "").strip()
    lcb.append([{"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
                {"role": "user", "content": f"{r['question_content']}\n\n{FM_START.format(sc=sc) if sc else FM_STD}"}])
out["lcb"] = lcb

blob = json.dumps(out, indent=0)
open(a.out, "w").write(blob)
print({k: len(v) for k, v in out.items()}, "sha256", hashlib.sha256(blob.encode()).hexdigest()[:16])
