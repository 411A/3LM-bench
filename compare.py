#!/usr/bin/env python3
"""
Paired A-vs-B comparison on shared items (McNemar exact test).

Point-score gaps at n<100 are noise: at 70% accuracy a 5pp gap needs n~400-600
paired items for 80% power. This tool is the only sanctioned way to claim
"model A > model B" in llmbench:

  python3 compare.py results/nemotron-3-nano.json results/qwen3.5-4b.json

It pairs records by (task, id, rep) - the design is paired (same items + seed
across models), so an unpaired test would waste that - and runs an exact
McNemar test per category on the discordant pairs (b = A right/B wrong,
c = A wrong/B right). Concordant pairs contribute nothing.

Outcome per category is one of: A>B (p<0.05), B>A (p<0.05), or TIE. With 14
categories, per-cell p-values are Holm-corrected on request (--holm); without
it they are exploratory. Binary outcome per item is `passed` for execution
tasks, `correct` for MC, score==1.0 for devops/netdiag/lint-clean, and the
agentic chain verdict.

Usage:
  python3 compare.py A.json B.json [--holm] [--json]
"""
import argparse
import json
import math
import sys


def binom_cdf(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** n for i in range(k + 1))


def mcnemar_exact(b, c):
    """Two-sided exact p-value. Returns 1.0 when there are no discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(1.0, 2.0 * binom_cdf(m, n))


def outcome(r):
    if r.get("error") and "ERROR" in str(r.get("id", "")):
        return None
    cat = r.get("category")
    if cat in ("code", "bugfix", "algo", "refactor", "rust", "embedded_cpp"):
        return bool(r.get("passed"))
    if "correct" in r:
        return bool(r.get("correct"))
    if "score" in r and isinstance(r["score"], (int, float)):
        return r["score"] == 1.0
    if r.get("passed") is not None:
        return bool(r.get("passed"))
    return None


def key(r):
    return (r.get("task"), r.get("id"), r.get("rep"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--holm", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    da = json.load(open(args.a))
    db = json.load(open(args.b))
    ra = {key(r): outcome(r) for r in da.get("records", [])}
    rb = {key(r): outcome(r) for r in db.get("records", [])}
    shared = {k for k in ra if k in rb and ra[k] is not None and rb[k] is not None}

    by_cat = {}
    for t, i, rep in shared:
        rec = next(r for r in da["records"]
                   if (r.get("task"), r.get("id"), r.get("rep")) == (t, i, rep))
        by_cat.setdefault(rec.get("category"), []).append(((t, i, rep)))

    rows = []
    for cat, keys in sorted(by_cat.items()):
        b = sum(1 for k in keys if ra[k] and not rb[k])
        c = sum(1 for k in keys if not ra[k] and rb[k])
        p = mcnemar_exact(b, c)
        na = sum(1 for k in keys if ra[k])
        nb = sum(1 for k in keys if rb[k])
        rows.append({"category": cat, "n_shared": len(keys),
                     "a_acc": round(na / len(keys), 4), "b_acc": round(nb / len(keys), 4),
                     "discord_b": b, "discord_c": c, "p": round(p, 4)})

    if args.holm:
        m = len(rows)
        ordered = sorted(range(m), key=lambda i: rows[i]["p"])
        adj, prev = [0.0] * m, 0.0
        for rank, i in enumerate(ordered):
            adj[i] = max(prev, min(1.0, rows[i]["p"] * (m - rank)))
            prev = adj[i]
        for i, r in enumerate(rows):
            r["p_holm"] = round(adj[i], 4)
            r["verdict"] = ("A>B" if r["a_acc"] > r["b_acc"] else "B>A") \
                if adj[i] < 0.05 else "TIE"
    else:
        for r in rows:
            r["verdict"] = ("A>B" if r["a_acc"] > r["b_acc"] else "B>A") \
                if r["p"] < 0.05 else "TIE"

    la = da.get("config", {}).get("label", args.a)
    lb = db.get("config", {}).get("label", args.b)
    if args.json:
        print(json.dumps({"A": la, "B": lb, "rows": rows}, indent=1))
        return
    print(f"A={la}  vs  B={lb}  (paired on {len(shared)} shared items)")
    print(f"{'category':<14} {'n':>4} {'A':>6} {'B':>6} {'b':>4} {'c':>4} "
          f"{'p_holm' if args.holm else 'p':>8}  verdict")
    for r in rows:
        p = r.get("p_holm", r["p"])
        print(f"{r['category']:<14} {r['n_shared']:>4} {r['a_acc']:>6.3f} "
              f"{r['b_acc']:>6.3f} {r['discord_b']:>4} {r['discord_c']:>4} "
              f"{p:>8.4f}  {r['verdict']}")
    if not args.holm:
        print("(exploratory: re-run with --holm before claiming any winner)")


if __name__ == "__main__":
    sys.exit(main())
