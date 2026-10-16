#!/usr/bin/env python3
"""
Deterministically de-bias the answer position of every multiple-choice task.

Why this file exists: the hand-authored MC tuples were written with the correct
answer at whichever index felt natural, which turned out to be overwhelmingly
index 1 ('B'). Measured before this fix:

    electronics  n=58   A 21%  B 69%  C 10%  D  0%   -> always-B scores 0.69
    english      n=15   A  7%  B 67%  C 27%  D  0%   -> always-B scores 0.67
    science      n=35   A 34%  B 46%  C 14%  D  6%
    persian      n=14   A  7%  B 43%  C 43%  D  7%

A model that always answers "B" - no knowledge at all - would have outscored
most real models on the knowledge categories. Worse, the argmax-over-logprobs
scorer has a mild but real preference for early letters, so the two biases
compound in the same direction.

The fix is a deterministic re-ordering of each item's choices with the gold
index remapped to follow. The gold is placed by least-used round-robin and the
remaining distractors are shuffled with a seed derived from the item id, so:

  * gold positions are balanced to within ONE item of perfectly uniform
    (a random shuffle is only approximately balanced - persian landed at 43%
    on one letter, which is still a free 43% for an answerer that always says B);
  * the same item always lands the same way, so the pairing across models is
    preserved;
  * the build is reproducible byte-for-byte.

Why round-robin rather than shuffle: the requirement is not "looks random", it is
"a model with no knowledge scores as close to 1/k as possible". Shuffling leaves
residual imbalance that a constant-answerer can still exploit. Round-robin
removes it by construction, and the distractor shuffle keeps the distractors from
becing predictable.

This module is imported by the MC builders, so the de-biased files ARE the build
output. It used to be a separate manual step that nothing called, which meant a
rebuild regenerated the 69%-B suite exactly.
"""
import hashlib
import json
import os
import random
import sys

# Resolve relative to this file, NOT to ~/llmbench. The hardcoded path meant that
# running this from a copy of the tree rewrote the real benchmark - the single
# most dangerous thing a de-biasing pass can do.
TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
FILES = ["science", "electronics", "persian", "english"]


def item_seed(tid, salt="mcpos-v1"):
    h = hashlib.sha256(f"{salt}:{tid}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def rebalance(rows):
    """Place each gold by least-used round-robin; shuffle the distractors.

    Returns the number of items whose stored ordering changed.
    """
    changed = 0
    used = {}            # n_choices -> list of times each gold position is used
    # Deterministic iteration order regardless of how the builder emitted them.
    for r in sorted(rows, key=lambda x: x["id"]):
        choices = list(r["choices"])
        k = len(choices)
        gold_text = choices[r["answer"]]
        if k < 2:
            continue
        if k not in used:
            used[k] = [0] * k
        # least-used position, ties broken low -> imbalance never exceeds 1 item
        slot = min(range(k), key=lambda i: (used[k][i], i))
        used[k][slot] += 1
        rest = [c for c in choices if c != gold_text]
        random.Random(item_seed(r["id"])).shuffle(rest)
        new = rest[:slot] + [gold_text] + rest[slot:]
        new_gold = new.index(gold_text)
        if new != choices or new_gold != r["answer"]:
            changed += 1
        r["choices"] = new
        r["answer"] = new_gold
    return changed


def report(rows):
    from collections import Counter
    n = len(rows)
    c = Counter(r["answer"] for r in rows)
    hist = [c.get(i, 0) for i in range(4)]
    return hist, max(hist) / n if n else 0


def main():
    for name in FILES:
        p = os.path.join(TASKS, name + ".json")
        rows = json.load(open(p))
        before, _ = report(rows)
        changed = rebalance(rows)
        json.dump(rows, open(p, "w"), ensure_ascii=False, indent=1)
        after, best = report(rows)
        L = "ABCD"
        print(f"  {name:<14} n={len(rows):<3} changed={changed:<3} "
              f"before[{L[0]}{before[0]} {L[1]}{before[1]} {L[2]}{before[2]} {L[3]}{before[3]}] "
              f"-> after[{L[0]}{after[0]} {L[1]}{after[1]} {L[2]}{after[2]} {L[3]}{after[3]}] "
              f"max={best:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())