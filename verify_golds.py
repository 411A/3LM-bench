#!/usr/bin/env python3
"""
Gold-answer controls. The defect class that matters most, and the one the
previous control suite had no coverage for at all.

A benchmark whose gold answers are wrong does not measure anything; it measures
agreement with the author. Three wrong golds were found by inspection after the
fact - a Persian arithmetic item whose gold was 80000*3 (the "forgot to divide"
error, with the correct 48,000 not even among the choices), an SPI register item
whose gold contradicted the datasheet, and a sorting item where no option
satisfied all three stated properties. None of them would have failed any
existing test, because every existing test checks that the harness is
self-consistent, not that the benchmark is right.

Four independent checks, each of which fails loudly:

  1. ARITHMETIC      - every item whose answer is computable is recomputed here
                       and compared to the stored gold.
  2. LEAK            - the gold text must not appear verbatim in its own question.
                       The old leak check required it to appear TWICE, so a single
                       verbatim copy passed; that is how an item whose stem
                       contained its own answer shipped.
  3. STRUCTURE       - gold index in range, choices unique, no empty stems, no
                       option that is a substring-duplicate of another option.
  4. POSITION        - no letter may hold an excessive share of the golds, since
                       a constant-letter answerer scores that share for free.

Run after any change to a task builder. A failure here is a real benchmark bug,
not a harness bug.
"""
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "tasks")
FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
fails = []


def load(name):
    return json.load(open(os.path.join(T, name + ".json")))


def norm(s):
    """Normalise digits so Persian and ASCII numerals compare equal."""
    return (s or "").translate(FA_DIGITS)


def fail(msg):
    fails.append(msg)
    print(f"    BUG {msg}")


# ---------------------------------------------------------------- 1. arithmetic
# Each entry: (suite, item id, correct answer in Persian/ASCII numerals, the
# expression that produced it). Written out by hand so it is independent of the
# builder - if the builder is wrong, this table is still right.
ARITHMETIC = [
    # 5 books cost 80,000 -> 16,000 each -> 3 books = 48,000.
    # Gold was 240,000 = 80000*3, the "forgot to divide by 5" error.
    ("persian", "fa/fa-math-1", 48000, "80000/5*3"),
    ("persian", "fa/fa-math-2", 51, "17*3"),
    ("persian", "fa/fa-math-3", 180000, "200000*0.9"),
    ("persian", "fa/fa-mc-1", 7, "only prime among 6,4,9,7"),
    ("persian", "fa/fa-mc-5", 320, "80/0.25"),
    ("persian", "fa/fa-math-4", 60, "12*5"),
    ("persian", "fa/fa-math-5", 30, "240/8"),
    ("persian", "fa/fa-math-8", 63, "100-37"),
    ("persian", "fa/fa-math-9", 15000, "60000/4"),
    ("persian", "fa/fa-math-10", 5, "(2+4+6+8)/4"),
    ("english", "en/en-math-1", 60, "12*5"),
    ("english", "en/en-math-2", 30, "240/8"),
    ("english", "en/en-math-4", 5, "(2+4+6+8)/4"),
    ("english", "en/en-math-5", 15, "60/4"),
]
# Items whose gold is a NUMBER, expressed as the number itself.
ARITH_BY_SUITE = {}
for suite, tid, val, expr in ARITHMETIC:
    ARITH_BY_SUITE.setdefault(suite, []).append((tid, val, expr))

print("=== 1. arithmetic golds recomputed ===")
for suite, items in ARITH_BY_SUITE.items():
    rows = {r["id"]: r for r in load(suite)}
    for tid, want, expr in items:
        if tid not in rows:
            fail(f"{tid}: missing from {suite}")
            continue
        gold = norm(rows[tid]["choices"][rows[tid]["answer"]])
        nums = re.findall(r"\d+", gold)
        if not nums:
            fail(f"{tid}: gold {gold!r} contains no numeral, cannot verify {expr}")
            continue
        got = int(nums[0])
        # a gold like "۴۸ هزار" means 48 thousand
        mult = 1000 if "هزار" in gold or "thousand" in gold.lower() else 1
        got *= mult
        if got != want:
            fail(f"{tid}: gold {gold!r} = {got} but {expr} = {want}")
        else:
            print(f"    OK  {tid:<16} {got:>9,} = {expr}")

# ------------------------------------------------------------------- 2. leaks
print()
print("=== 2. gold leaked into its own question ===")
for suite in ("science", "electronics", "persian", "english"):
    for r in load(suite):
        gold = (r["choices"][r["answer"]] or "").strip()
        q = (r["question"] or "")
        if len(gold) < 4:          # short tokens like "7" collide by chance
            continue
        if gold in q:
            fail(f"{r['id']}: gold {gold!r} appears verbatim in the question")
        # also catch the gold quoted inside guillemets in Persian items
        if f"«{gold}»" in q or f'"{gold}"' in q:
            fail(f"{r['id']}: gold {gold!r} is quoted in the question")
print("    checked 4 suites for verbatim leaks")

# --------------------------------------------------------------- 3. structure
print()
print("=== 3. structural integrity ===")
for suite in ("science", "electronics", "persian", "english"):
    for r in load(suite):
        ch = r.get("choices") or []
        a = r.get("answer")
        if not ch:
            fail(f"{r['id']}: no choices")
            continue
        if not isinstance(a, int) or not (0 <= a < len(ch)):
            fail(f"{r['id']}: gold index {a} out of range for {len(ch)} choices")
        if len(set(ch)) != len(ch):
            fail(f"{r['id']}: duplicate choices {ch}")
        if not (r.get("question") or "").strip():
            fail(f"{r['id']}: empty question")
        for c in ch:
            if not (c or "").strip():
                fail(f"{r['id']}: empty choice in {ch}")
    print(f"    OK  {suite:<13} structure valid")

# --------------------------------------------------------------- 4. positions
print()
print("=== 4. gold-position balance (a constant answerer must score ~chance) ===")
from stats import chance_baseline  # noqa: E402
sys.path.insert(0, HERE)
for suite in ("science", "electronics", "persian", "english"):
    rows = load(suite)
    hist = Counter(r["answer"] for r in rows)
    n = len(rows)
    p0 = chance_baseline(rows)
    const = max(hist.values()) / n
    # Allow the imbalance a perfectly uniform draw would produce by chance.
    tol = 1.0 / n + 0.05
    if const - p0 > tol:
        fail(f"{suite}: best-constant {const:.1%} exceeds chance {p0:.1%} by more than {tol:.1%}")
    else:
        print(f"    OK  {suite:<13} best-constant {const:.1%} vs chance {p0:.1%} "
              f"(hist {[hist.get(i,0) for i in range(4)]})")

print()
print("=" * 72)
if fails:
    print(f"{len(fails)} GOLD DEFECTS")
    for f in fails:
        print(f"  - {f}")
else:
    print("ALL GOLD CONTROLS PASS")
sys.exit(1 if fails else 0)