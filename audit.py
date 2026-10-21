#!/usr/bin/env python3
"""Independent audit: recompute every published cell from the raw records.

The report is only as trustworthy as its weakest cell, so this script
recomputes each category's headline number straight from the records in the
results files and compares it with the number the file claims. It knows
nothing about the harness internals: it only reads the JSON.

Metric families (each verified against the family the summary actually
uses - a mismatch means a corrupted or mis-aggregated cell):
  code families  -> accuracy = passed / n        (Wilson CI)
  MC + gsm8k     -> accuracy = correct / n       (Wilson CI)
  lint           -> clean_rate = clean / n       (Wilson CI on clean)
  netdiag/devops -> mean_score (partial credit, no Wilson on the mean)
  agentic        -> pass_hat_k_reliability (chains that always succeed)
"""
import glob
import json
import math
import sys

Z = 1.959963985


def wilson(k, n):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = (Z / d) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


CODE = {"code", "bugfix", "algo", "refactor", "rust", "embedded_cpp"}
GSM = {"std_gsm8k", "std_gsm8k_s30", "gsm8k"}
MC = {"science", "electronics", "persian", "english",
      "std_arc_c", "std_arc_e", "std_arc_challenge_s30", "std_arc_easy_s30",
      "std_hellaswag", "std_hellaswag_s30", "std_winogrande",
      "std_winogrande_s30", "std_truthfulqa", "std_truthfulqa_mc_s30",
      "std_belebele_fa", "std_belebele_fa_s30", "std_mmlu", "std_mmlu_s30",
      "std_mmlu_cs"}
# The second MC mechanism (text match): the same metric on the same items,
# so the audit checks it exactly like the letter families.
MC |= {c + "_txt" for c in MC}


def is_stub(r):
    return ("ERROR" in str(r.get("id", ""))
            or r.get("stopped") in ("error", "exception"))


def close(a, b, tol=5e-4):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def main():
    bad, checked, cells = [], 0, 0
    for f in sorted(glob.glob("results/*/*.json")):
        if "/attic/" in f:
            continue
        d = json.load(open(f))
        summ_all = d.get("summary", {})
        for cat, summ in summ_all.items():
            recs = [r for r in d.get("records", [])
                    if r.get("category") == cat and not is_stub(r)]
            if not recs:
                continue
            n = len(recs)
            cells += 1
            if cat in CODE:
                k = sum(1 for r in recs if r.get("passed"))
                checked += 1
                if not close(summ.get("accuracy"), round(k / n, 4)):
                    bad.append((f, cat, "accuracy", summ.get("accuracy"), k / n))
                if summ.get("ci95_wilson") and list(summ["ci95_wilson"]) != list(wilson(k, n)):
                    bad.append((f, cat, "ci", summ.get("ci95_wilson"), wilson(k, n)))
            elif cat in GSM or cat in MC:
                k = sum(1 for r in recs if r.get("correct"))
                checked += 1
                if not close(summ.get("accuracy"), round(k / n, 4)):
                    bad.append((f, cat, "accuracy", summ.get("accuracy"), k / n))
                if summ.get("ci95_wilson") and list(summ["ci95_wilson"]) != list(wilson(k, n)):
                    bad.append((f, cat, "ci", summ.get("ci95_wilson"), wilson(k, n)))
            elif cat == "lint":
                k = sum(1 for r in recs if r.get("n_diagnostics") == 0)
                checked += 1
                if not close(summ.get("clean_rate"), round(k / n, 4)):
                    bad.append((f, cat, "clean_rate", summ.get("clean_rate"), k / n))
                if summ.get("ci95_wilson_clean") and list(summ["ci95_wilson_clean"]) != list(wilson(k, n)):
                    bad.append((f, cat, "ci_clean", summ.get("ci95_wilson_clean"), wilson(k, n)))
            elif cat in ("netdiag", "devops"):
                scores = [float(r.get("score", 0)) for r in recs
                          if isinstance(r.get("score"), (int, float))]
                checked += 1
                if scores and not close(summ.get("mean_score"), round(sum(scores) / len(scores), 4)):
                    bad.append((f, cat, "mean_score", summ.get("mean_score"),
                                round(sum(scores) / len(scores), 4)))
            elif cat == "agentic":
                # pass^k = fraction of chains where EVERY repeat solved the
                # chain. "Solved" follows the documented definition: the model
                # used the tools and completed every expected step (trace
                # complete) and got the answer right; for the no-call
                # (irrelevance) chains the correct behaviour is a grounded
                # refusal with zero calls.
                def solved(r):
                    if r.get("expect_no_calls"):
                        return bool(r.get("outcome_ok")
                                    and (r.get("n_tool_calls") or 0) == 0)
                    return bool(r.get("outcome_ok")
                                and r.get("trace_ok") == r.get("trace_total")
                                and (r.get("n_tool_calls") or 0) > 0)
                chains = {}
                for r in recs:
                    chains.setdefault(r.get("chain_id"), []).append(solved(r))
                n_chains = len(chains)
                k_chains = sum(1 for v in chains.values() if all(v))
                val = round(k_chains / n_chains, 4) if n_chains else 0
                checked += 1
                if not close(summ.get("pass_hat_k_reliability"), val):
                    bad.append((f, cat, "pass_hat_k", summ.get("pass_hat_k_reliability"), val))
    print(f"cells seen      : {cells}")
    print(f"numbers verified: {checked}")
    print(f"mismatches      : {len(bad)}")
    for b in bad[:12]:
        print("   ", b)
    if bad:
        sys.exit(1)
    print("AUDIT OK: every published number reproduces from its raw records.")


if __name__ == "__main__":
    main()
