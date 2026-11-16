"""Re-score NeMo Gym IFBench rollouts offline with Gym's own IFBench checker, on the raw final text and on the text
with leading/trailing whitespace stripped.

Why: the qwen3 reasoning parser leaves the "\\n\\n" that follows </think> at the start of the message content, and
Gym's strict checks run on that raw text, so instructions such as "start your answer with ..." fail on whitespace
alone. The stripped score is the one this repo reports; the raw score equals Gym's own reward.

usage: (cd $GYM/resources_servers/ifbench && .venv/bin/python <this> <rollouts.jsonl>)
  The ifbench resources-server venv is created by Gym the first time an IFBench run starts it. CPU only.
  summarize.py calls this for every IFBench output dir and stores the result in summary.json as
  {"raw": {...}, "stripped": {...}} with the keys printed below.
Prints one line per variant: prompt-level and instruction-level strict/loose accuracy, and
gym_reward(fraction,strict) = the mean per-prompt fraction of strictly followed instructions (Gym's IFBench reward).
"""
import json, sys
sys.path.insert(0, ".")
import setup_ifbench
setup_ifbench.ensure_ifbench()
from app import IFBenchResourcesServer  # noqa: E402
import instructions_registry  # noqa: E402


class _S:  # just enough of the server to call _check_instructions
    _instructions_registry = instructions_registry
    _check_instructions = IFBenchResourcesServer._check_instructions


s = _S()
rows = [json.loads(l) for l in open(sys.argv[1])]
for name, f in (("raw", lambda t: t), ("stripped", lambda t: t.strip())):
    ps = pl = 0; ins = inl = ni = 0; frac = 0.0
    for r in rows:
        out = (r.get("response") or {}).get("output") or []
        t = ""
        if out and out[-1].get("content"):
            t = out[-1]["content"][0].get("text") or ""
        st, lo = s._check_instructions(r["instruction_id_list"], r["kwargs"], r["prompt"], f(t))
        ps += all(st); pl += all(lo); ins += sum(st); inl += sum(lo); ni += len(st); frac += sum(st) / len(st)
    n = len(rows)
    print(f"{name:9s} n={n} prompt_strict={100*ps/n:.2f} prompt_loose={100*pl/n:.2f} "
          f"inst_strict={100*ins/ni:.2f} inst_loose={100*inl/ni:.2f} gym_reward(fraction,strict)={100*frac/n:.2f}")
