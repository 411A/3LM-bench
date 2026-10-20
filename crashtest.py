#!/usr/bin/env python3
"""crashtest.py — what a power cut costs, measured instead of assumed.

Simulates a power cut (SIGKILL-like: no cleanup, no final write, no chance
to flush anything that was not already on disk) in the middle of a run and
proves three things:

  1. The checkpoint file on disk is VALID JSON and holds every item that was
     scored before the cut (a truncated file would make the resume start
     from zero - the one failure mode that would defeat the checkpoint).
  2. The resume re-runs ONLY the missing items.
  3. After the resume, every item is present exactly once (no duplicates,
     no lost items).

No model and no network: a canned backend feeds the REAL runner, the REAL
scorer, the REAL checkpoint writer and the REAL resume math.

Usage: python3 crashtest.py      (exit 0 = the protocol holds)
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R

HERE = os.path.dirname(os.path.abspath(__file__))
FAIL = []


def note(ok, label):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}")
    if not ok:
        FAIL.append(label)


# --- the child process: runs real items, then "loses power" -------------
CHILD = r'''
import json, os, sys
sys.path.insert(0, {here!r})
import run as R

out_path = sys.argv[1]
n_before_crash = int(sys.argv[2])
tasks = R.load_tasks("science")[:6]

records = []
done_ids = set()

def checkpoint():
    R.write_atomic(out_path, {{
        "schema": "llmbench/1",
        "config": {{"tasks": ["science"]}},
        "config_hash": "crashtest",
        "records": records,
        "partial": True,
    }})

class Stub:
    """Canned answers: the plumbing is what is under test."""
    def generate(self, messages, **kw):
        return {{"content": "A", "tool_calls": [], "prompt_tokens": 10,
                 "eval_tokens": 1, "prompt_tps": 5.0, "gen_tps": 3.0,
                 "ttft_s": 0.1, "wall_s": 0.1, "load_s": 0.0,
                 "prompt_cached_tokens": None}}
    def mc_logprobs(self, question, choices, system=None, prefill=""):
        meta = {{"prompt_tokens": 10, "eval_tokens": 1, "ttft_s": 0.1,
                "wall_s": 0.2}}
        return (0, "A", {{"A": 0.9, "B": 0.1}}, meta, "letter", 1.0)
    def summary(self):
        return {{"calls": 1}}

def progress(rec, _cat="science", _name="science", _rep=0):
    rec["task_category"] = rec.get("category")
    rec["category"] = _cat
    rec["task"] = _name
    rec["rep"] = _rep
    records.append(rec)
    done_ids.add((_name, _rep, rec.get("id")))
    checkpoint()
    if len(records) >= n_before_crash:
        # POWER CUT: no atexit, no flush, no cleanup. Only what was already
        # written to disk exists now.
        os._exit(137)

R.run_mc(Stub(), tasks, None, 0, progress=progress)
'''


def main():
    tmpdir = tempfile.mkdtemp(prefix="crashtest-")
    out = os.path.join(tmpdir, "results.json")

    # ---- 1. crash the run half way ------------------------------------
    child = CHILD.format(here=HERE)
    src = os.path.join(tmpdir, "child.py")
    with open(src, "w") as f:
        f.write(child)
    proc = subprocess.run([sys.executable, src, out, "3"],
                          capture_output=True, text=True, timeout=120)
    note(proc.returncode == 137,
         f"child died like a power cut (rc={proc.returncode}, expected 137)")

    # ---- 2. the file on disk must be valid and hold the scored items ---
    try:
        with open(out) as f:
            d = json.load(f)
        valid = True
    except Exception as e:  # noqa: BLE001
        d = {"records": []}
        valid = False
        print(f"       (unreadable checkpoint: {e})")
    note(valid, "checkpoint file is valid JSON after the cut (not truncated)")
    note(len(d.get("records", [])) == 3,
         f"every item scored before the cut is on disk "
         f"({len(d.get('records', []))} of 3)")
    note(d.get("partial") is True, "the crashed file is flagged partial")

    # ---- 3. the resume runs ONLY the missing items --------------------
    done_ids = set()
    for r in d.get("records", []):
        done_ids.add((r.get("task"), r.get("rep"), r.get("id")))
    tasks = R.load_tasks("science")[:6]
    pending = R.pending_items(tasks, "science", 0, done_ids)
    note(len(pending) == 3,
         f"resume sees exactly the 3 missing items (got {len(pending)})")
    note(all(t.get("id") not in {r.get("id") for r in d.get("records", [])}
             for t in pending),
         "no completed item is scheduled again")

    # ---- 4. after the resume: every item exactly once ------------------
    done = list(d.get("records", []))
    for t in pending:
        done.append({"task": "science", "rep": 0, "category": "science",
                     "id": t.get("id"), "correct": True})
    ids = [r.get("id") for r in done]
    note(len(ids) == 6 and len(set(ids)) == 6,
         f"after the resume every item is present exactly once "
         f"({len(ids)} records, {len(set(ids))} distinct)")

    # ---- 5. atomicity: a failed write leaves the old file intact ------
    out2 = os.path.join(tmpdir, "atomic.json")
    R.write_atomic(out2, {"records": [1, 2, 3]})
    before = open(out2).read()
    try:
        R.write_atomic(out2, {"records": [1, 2, 3], "boom": object()})
    except Exception:
        pass
    after = open(out2).read()
    note(json.loads(after) is not None, "the file still parses after a failed write")
    note(json.loads(after) == json.loads(before),
         "a failed write does not damage the previous checkpoint")
    note(not os.path.exists(out2 + ".tmp"),
         "no temp file is left behind")

    print()
    if FAIL:
        print(f"{len(FAIL)} CRASHTEST FAILURES")
        raise SystemExit(1)
    print("CRASHTEST OK: a power cut costs one item, and the resume continues.")


if __name__ == "__main__":
    main()
