"""MMMU-Pro full set, built only as an input to the calibration decontamination index (tools/decontam.py); it is not
one of the reported accuracy benchmarks. Every row of the `standard (10 options)` test split, including multi-image
questions, in NeMo Gym mcqa row format: a materialized responses_create_params.input with the image(s) as JPEG data
URIs (in image_1..image_7 order) followed by the mcq-10choices prompt, plus options / expected_answer (decontam.py
indexes the text fields and skips the images). Rows are written in dataset order (no shuffle); rows whose answer is
not a valid option letter are skipped.

usage (from the repo root; the path decontam.py reads by default):
  python recipes/tools/build_mmmu_pro_full.py \
    harness/accuracy/benchmarks/mmmu_pro_std10_full/data/mmmu_pro_std10_full_benchmark.jsonl
(needs HF access and `datasets`; CPU only)
"""

import ast
import base64
import collections
import io
import json
import sys

from datasets import load_dataset

out = sys.argv[1]
ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
letters = "ABCDEFGHIJ"
PROMPT = ("Answer the following multiple choice question. The last line of your response should be in the following "
          "format: 'Answer: A/B/C/D/E/F/G/H/I/J' (e.g. 'Answer: A').\n\n{problem}")
img_keys = [f"image_{j}" for j in range(1, 8)]
k, skipped, nimg = 0, collections.Counter(), collections.Counter()
with open(out, "w") as f:
    for r in ds:
        opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else list(r["options"])
        imgs = [r[c] for c in img_keys if c in r and r[c] is not None]
        if len(opts) > 10 or r["answer"] not in letters[: len(opts)]:
            skipped["bad_answer_or_options"] += 1
            continue
        if not imgs:
            skipped["no_image"] += 1
            continue
        parts = []
        for img in imgs:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=92)
            parts.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                          "detail": "high"})
        problem = r["question"] + "\n\n" + "\n".join(f"{letters[j]}) {o}" for j, o in enumerate(opts))
        parts.append({"type": "input_text", "text": PROMPT.format(problem=problem)})
        f.write(json.dumps({
            "responses_create_params": {"input": [{"role": "user", "content": parts}]},
            "options": [{letters[j]: str(o)} for j, o in enumerate(opts)],
            "expected_answer": r["answer"], "subject": r["subject"], "mmmu_id": r["id"], "problem": problem,
            "n_images": len(imgs)}) + "\n")
        k += 1
        nimg[len(imgs)] += 1
print(f"{k} rows of {len(ds)} -> {out}; images/row {dict(sorted(nimg.items()))}; skipped {dict(skipped)}")
