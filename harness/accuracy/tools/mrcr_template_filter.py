"""Keep the MRCR rows whose chat-templated prompt fits next to the output cap.

Gym's mrcr prepare.py filters on the raw message-content length (`n_tokens`). The served prompt is longer: the chat
template adds role markers and the generation prompt. For the >128K slice we keep a row only if its Qwen
chat-templated prompt (add_generation_prompt=True, thinking on as served) has <= --max-templated tokens, so that
prompt + 65,536 output tokens fit in --max-model-len 262144 (196,608 - 512 = 196,096 by default).

usage: python3 mrcr_template_filter.py <in.jsonl> <out.jsonl> --tokenizer <model dir or Hub id> [--max-templated 196096]
On the measured build this turned 155 rows into 139 (2/4/8 needles: 52/43/44), templated length 135,741..195,665.
"""
import argparse
import collections
import json

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--max-templated", type=int, default=196096)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer)
kept, lens, by_needles, n_in = [], [], collections.Counter(), 0
for line in open(a.inp):
    r = json.loads(line); n_in += 1
    ids = tok.apply_chat_template(r["responses_create_params"]["input"], add_generation_prompt=True, tokenize=True,
                                  enable_thinking=True)
    if not isinstance(ids, list):  # transformers 5 returns a BatchEncoding
        ids = ids["input_ids"]
    if len(ids) <= a.max_templated:
        kept.append(line if line.endswith("\n") else line + "\n"); lens.append(len(ids)); by_needles[r.get("n_needles")] += 1
with open(a.out, "w") as f:
    f.writelines(kept)
print(f"{n_in} -> {len(kept)} rows; needles {dict(sorted(by_needles.items()))}; "
      f"templated {min(lens) if lens else 0}..{max(lens) if lens else 0} (mean {sum(lens) / max(len(lens), 1):.0f})")
