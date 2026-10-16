#!/usr/bin/env python3
"""
Heat-free end-to-end pipeline validation. No inference: a stub backend feeds
canned responses through EVERY runner with the REAL scorers/sandbox, then
aggregate + report + compare run over the records.

Why this exists: the 7h model run is the wrong place to discover a KeyError,
a schema mismatch, or a scorer regression. Every recent breakage in this
project (NameError fallback, aggregate KeyError on stubs, None+int crash,
whitespace-collapsed lines) would have been caught here in ~60s on a cold CPU.

Usage: python3 validate.py  (exit 0 = pipeline sound, exit 1 = defect)

It asserts PLUMBING (no exception, record schema, aggregate/report/compare
run), not model quality: canned answers are deliberately mediocre. Correctness
of the scorers themselves is covered by verify_*.py / negcontrols.py.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R
import agentic as A

HERE = os.path.dirname(os.path.abspath(__file__))
FAIL = []


def note(ok, label):
    print(f"  {'OK  ' if ok else 'BUG '} {label}")
    if not ok:
        FAIL.append(label)


def R_(content=""):
    return {"content": content, "tool_calls": [], "prompt_tokens": 10,
            "eval_tokens": 5, "prompt_tps": 5.0, "gen_tps": 3.0, "ttft_s": 0.5,
            "wall_s": 1.0, "load_s": 0.0, "prompt_cached_tokens": 0}


def T_(name, args):
    return {"content": "", "tool_calls": [{"id": "t1", "function": {"name": name, "arguments": args}}],
            "prompt_tokens": 10, "eval_tokens": 5, "prompt_tps": 5.0, "gen_tps": 3.0,
            "ttft_s": 0.5, "wall_s": 1.0, "load_s": 0.0, "prompt_cached_tokens": 0}


class Stub:
    """Canned backend. Agentic chains get exact-args tool calls + a final text."""

    def __init__(self):
        self.chain_script = []
        self.last_user = None

    def generate(self, messages, **kw):
        tools = kw.get("tools")
        if tools or any(m.get("tool_calls") for m in messages if isinstance(m, dict)):
            return self._agentic_next(messages)
        text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        low = text.lower()
        if "network engineer" in low:
            return R_("There is a fault. The faulty device is r1. The root cause is subnet mask.")
        if "dockerfile" in low or "compose" in low or "systemd" in low or "nginx" in low or "github actions" in low:
            return R_("```yaml\nservices:\n  web:\n    image: nginx\n    ports: [80]\n  db:\n    image: postgres\nvolumes:\n  data:\n```")
        if "rust" in low:
            return R_("```rust\npub fn double(x: i32) -> i32 {\n    x * 2\n}\n```")
        if "c++" in low or "arduino" in low or "setup()" in low:
            return R_("```cpp\nvoid setup() {}\nvoid loop() {}\n```")
        return R_("```python\nimport os\n\n\ndef f(x):\n    return x\n```")

    def _agentic_next(self, messages):
        users = [m for m in messages if m.get("role") == "user"]
        first = users[0]["content"] if users else ""
        if first != self.last_user:
            self.last_user = first
            chain = next((c for c in A.CHAINS if c["prompt"] == first), None)
            if chain is None:
                return R_("No tools needed. Answer: done.")
            self.chain_script = [T_(n, dict(a)) for n, a in chain["steps"]]
            if chain.get("expect_no_calls"):
                self.chain_script.append(R_(
                    "I cannot do that - none of my tools can look up that "
                    "information. I can help with orders, inventory, "
                    "customers, or shipping estimates."))
            elif "answer_regex" in chain:
                self.chain_script.append(R_("final answer: 12 units in stock"))
            else:
                self.chain_script.append(
                    R_("Yes, allowed." if chain.get("expect_yes") else "No, not allowed."))
        if self.chain_script:
            return self.chain_script.pop(0)
        return R_("Done.")

    def mc_logprobs(self, question, choices, system=None, prefill=""):
        # meta mirrors the real backend: the full generate dict (run_mc reads
        # meta["prompt_tokens"]/meta["eval_tokens"] off it).
        r = R_(choices[0] if choices else "")
        return (0, "A", {"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02},
                r, "top", 0.98)


def main():
    b = Stub()
    all_recs = []
    print("=== per-runner plumbing (1-2 items each, real scorers) ===")
    for name, (fn, cat) in R.RUNNERS.items():
        # agentic chains live in agentic.py, not a task file (same special-case
        # as run.py's main loop).
        tasks = ([{"id": c["id"]} for c in A.CHAINS] if name == "agentic"
                 else R.load_tasks(name)[:2])
        try:
            got = fn(b, tasks, None, 0)
            assert isinstance(got, list) and len(got) >= 1, f"{name}: {len(got)} records"
            for r in got:
                # run.py's main loop stamps these AFTER each runner returns
                # (RUNNERS table authoritative); replicate so schema asserts
                # check the real final shape.
                r["task_category"] = r.get("category")
                r["category"] = cat
                r["task"] = name
                # Agentic records carry per-call timing in `calls`, not a `speed`
                # dict (aggregate reads both defensively via .get). Uniformity
                # here would be cosmetic; the aggregate contract is what matters.
                assert ("speed" in r if cat != "agentic" else "calls" in r), \
                    f"{name}: record missing timing"
            all_recs.extend(got)
            note(True, f"{name}: {len(got)} records, schema ok")
        except Exception as e:  # noqa: BLE001
            import traceback as _tb
            note(False, f"{name}: {type(e).__name__}: {e}")
            print("   " + " | ".join(_tb.format_exc().strip().splitlines()[-4:]))

    print("=== aggregate over mixed records ===")
    try:
        agg = R.aggregate(all_recs)
        cats = {r["category"] for r in all_recs}
        missing = [c for c in cats if c not in agg]
        note(not missing, f"aggregate covers {len(agg)} categories"
             + (f" (missing {missing})" if missing else ""))
    except Exception as e:  # noqa: BLE001
        note(False, f"aggregate raised {type(e).__name__}: {e}")
        agg = {}

    print("=== known-good probes (guards scorer over-correction in-pipeline) ===")
    import scorers as S
    # use the task's own test with a trivially-correct stub is unreliable;
    # instead verify the pass PATH executes: a function returning the right
    # shape for MBPP item 0 is covered by verify_tasks. Here: run_python runs.
    r = S.run_python("def add(a, b):\n    return a + b\n", "assert add(2, 3) == 5")
    note(bool(r.get("passed")), "run_python pass-path executes")
    dt = R.load_tasks("devops")[0]
    art = ("FROM python:3.12\nEXPOSE 8000\nUSER appuser\n"
           "HEALTHCHECK CMD true\nWORKDIR /app\nCOPY . .\n")
    note(S.check_devops(art, dt["checks"])["score"] == 1.0,
         "genuine devops artifact scores 1.0 in-pipeline")
    nt = next(x for x in R.load_tasks("netdiag") if x.get("is_anomaly"))
    dev, cause = nt["faulty_device"], (nt["root_cause"] or "").replace("_", " ")
    note(S.score_netdiag(
        f"There is a fault. The faulty device is {dev}. The root cause is {cause}.",
        nt)["score"] == 1.0, "oracle netdiag scores 1.0 in-pipeline")
    # ERROR stubs must never enter scoring denominators (a dead server once
    # zeroed a model's code + agentic columns).
    stub_recs = [
        {"id": "code_humaneval#ERROR", "rep": 0, "category": "code",
         "task": "code_humaneval", "passed": False, "correct": False,
         "score": 0.0, "stopped": "error",
         "tokens": {"eval": 0}, "speed": {"gen_tps": None}},
        {"id": "h1", "rep": 0, "category": "code", "task": "code_humaneval",
         "passed": True, "tokens": {"eval": 0}, "speed": {"gen_tps": None},
         "lint": {"n_diagnostics": 0, "noqa_gaming": False}},
    ]
    agg2 = R.aggregate(stub_recs)
    note(agg2["code"]["accuracy"] == 1.0 and agg2["code"]["n_errors"] == 1,
         "ERROR stub excluded from scoring, counted in n_errors")

    print("=== report + compare render ===")
    import tempfile as _tf
    os.makedirs(os.path.join(HERE, "..", ".tmp-val"), exist_ok=True)
    fa = os.path.join(_tf.gettempdir(), "llmbench-val-A.json")
    fb = os.path.join(_tf.gettempdir(), "llmbench-val-B.json")
    cfg = {"label": "stub", "model": "stub", "started_unix": 0}
    json.dump({"schema": "llmbench/1", "config": cfg, "summary": agg,
               "records": all_recs}, open(fa, "w"))
    alt = copy.deepcopy(all_recs)
    for r in alt[:3]:
        if "correct" in r:
            r["correct"] = not r["correct"]
        if "passed" in r:
            r["passed"] = not r["passed"]
    json.dump({"schema": "llmbench/1", "config": {"label": "stub2", "model": "stub",
               "started_unix": 1}, "summary": R.aggregate(alt),
               "records": alt}, open(fb, "w"))
    import subprocess as _sp
    p1 = _sp.run([sys.executable, os.path.join(HERE, "compare.py"), fa, fb, "--holm"],
                 capture_output=True, text=True, timeout=60)
    note(p1.returncode == 0, "compare.py renders on pipeline records")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURES")
        raise SystemExit(1)
    print(f"VALIDATE OK: {len(all_recs)} records across "
          f"{len({r['category'] for r in all_recs})} categories, pipeline sound.")


if __name__ == "__main__":
    main()
