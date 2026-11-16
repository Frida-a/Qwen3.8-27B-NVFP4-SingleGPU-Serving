"""Write the run copy of a Gym benchmark config: benchmark datasets at `num_repeats: 1`, local data paths absolute.

Gym expands rows by a benchmark dataset's own `num_repeats` (GPQA ships 8, IFBench 5)
and then by `gym eval run --num-repeats`, so the two multiply.
A `--config` overlay does not help: the benchmark's own config is merged after it. So we
write the whole benchmark config again under a new name, with `num_repeats: 1`, into the
`--search-dir` tree (Gym v0.7.0 looks for `<search dir>/benchmarks/<name>/config.yaml`),
and run that name instead.

With `--abs-root DIR`, every relative benchmark `jsonl_fpath` that exists under DIR (the repo
root; the local configs here use repo-relative paths) is made absolute in the copy. Gym reads a
relative jsonl_fpath through its search roots but writes the prepared file and the metrics
sidecar next to it relative to the cwd (the Gym checkout), so an absolute path keeps them
together. Paths that do not exist under DIR (Gym's own benchmarks) are left for Gym to resolve.

usage: python repeat_overrides.py <benchmark config.yaml> <search dir> <benchmark name> [--abs-root DIR]
Writes <search dir>/benchmarks/<name>_run/config.yaml (slashes in the name -> _) and prints
`<name>_run`; prints the original name unchanged if nothing needed changing.
"""

import argparse
import os

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("search_dir"); ap.add_argument("name")
ap.add_argument("--abs-root", default=None)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.src))
changed = False


def walk(node):
    global changed
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "datasets" and isinstance(v, list):
                for d in v:
                    if not (isinstance(d, dict) and d.get("type") == "benchmark"):
                        continue
                    if d.get("num_repeats", 1) != 1:
                        d["num_repeats"] = 1
                        changed = True
                    p = d.get("jsonl_fpath")
                    if a.abs_root and isinstance(p, str) and not os.path.isabs(p):
                        full = os.path.join(a.abs_root, p)
                        if os.path.exists(full):
                            d["jsonl_fpath"] = os.path.abspath(full)
                            changed = True
            else:
                walk(v)
    elif isinstance(node, list):
        for v in node:
            walk(v)


walk(cfg)
if not changed:
    print(a.name)
    raise SystemExit(0)
new = a.name.replace("/", "_") + "_run"
out = os.path.join(a.search_dir, "benchmarks", new)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "config.yaml"), "w") as f:
    f.write(f"# Generated from {a.src}: benchmark datasets at num_repeats 1, local jsonl_fpath made absolute.\n")
    yaml.safe_dump(cfg, f, sort_keys=False)
print(new)
