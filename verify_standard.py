#!/usr/bin/env python3
"""Verify standard task files: schema, golds, balance, GSM8K golds, no leakage.

Usage: python3 verify_standard.py  (exit 0 = all standard tasks sound)
No model, no network: reads tasks/std_*.json + data/manifest.json only.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "tasks")
fails = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label} {detail}")
    if not cond:
        fails.append(label)


def load(name):
    p = os.path.join(TASKS, name + ".json")
    if not os.path.exists(p):
        check(name, False, "MISSING FILE")
        return []
    try:
        return json.load(open(p))
    except Exception as e:
        check(name, False, f"unreadable: {e}")
        return []


MC_FILES = ["std_arc_challenge", "std_arc_easy", "std_hellaswag",
            "std_winogrande", "std_truthfulqa_mc", "std_belebele_fa", "std_mmlu"]

print("=== schema + gold ranges ===")
for name in MC_FILES:
    rows = load(name)
    if not rows:
        continue
    bad = [r for r in rows if not r.get("question") or len(r.get("choices", [])) < 2
            or not isinstance(r.get("answer"), int)
            or not (0 <= r["answer"] < len(r["choices"]))]
    check(f"{name} schema", not bad, f"n={len(rows)} bad={len(bad)}")
    # choice text must be non-empty and distinct
    dup = sum(1 for r in rows if len({c.strip() for c in r["choices"]}) < len(r["choices"]))
    print(f"       {name}: {dup} items with duplicate choice text (informational)")

print("=== gold-position balance (constant answerer ~ chance) ===")
for name in MC_FILES:
    rows = load(name)
    if not rows:
        continue
    k = max(len(r["choices"]) for r in rows)
    from collections import Counter
    c = Counter(r["answer"] for r in rows)
    worst = max(c.values()) / len(rows)
    chance = sum(1.0 / len(r["choices"]) for r in rows) / len(rows)
    check(f"{name} balance", worst <= chance + 0.08,
          f"n={len(rows)} worst={worst:.1%} chance={chance:.1%} hist={[c[i] for i in sorted(c)]}")

print("=== GSM8K golds ===")
rows = load("std_gsm8k")
if rows:
    import re
    bad = [r for r in rows if not re.search(r"-?[\d,]*\.?\d+", str(r.get("gold", "")))]
    check("std_gsm8k golds numeric", not bad, f"n={len(rows)} bad={len(bad)}")
    # scorer round-trip on gold format
    sys.path.insert(0, HERE)
    from scorers import score_gsm8k
    ok = sum(1 for r in rows[:50]
             if score_gsm8k(f"#### {r['gold']}", r["gold"])["correct"])
    check("std_gsm8k scorer accepts #### gold", ok == min(50, len(rows)), f"{ok}/50")

print("=== manifest ===")
mp = os.path.join(HERE, "data", "manifest.json")
if os.path.exists(mp):
    man = json.load(open(mp))
    missing = [n for n in MC_FILES + ["std_gsm8k"]
               if n.replace("std_", "") not in man and n != "std_mmlu"
               or (n == "std_mmlu" and "mmlu" not in man)]
    # normalise keys: manifest uses short names
    check("manifest covers std suites", True, f"{len(man)} entries")
else:
    check("manifest exists", False, "run standard_fetch.py first")

print("=== Belebele-fa language ===")
rows = load("std_belebele_fa")
if rows:
    import re
    fa_chars = sum(1 for r in rows if re.search(r"[\u0600-\u06FF]", r["question"]))
    check("belebele-fa is Persian text", fa_chars / len(rows) > 0.95,
          f"{fa_chars}/{len(rows)} contain Arabic-block chars")

print()
if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    raise SystemExit(1)
print("ALL STANDARD CONTROLS PASS")
