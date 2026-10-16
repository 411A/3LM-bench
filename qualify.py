#!/usr/bin/env python3
"""World-Cup referee: rank qualifier runs, CUT the hopeless, ADVANCE the rest.

Reads results/*-q.json (written by the `qualifier` profile: identical
front-biased subsets for every model, so the ranking is fair). Qualifier
numbers are NEVER compared against full-split scores — only q-vs-q.

Verdict logic (signal checklist, no single magic number):
  CUT  = no significant MC category AND zero code passes AND zero solved
         chains (nothing measured but noise), OR >50% empty outputs.
  ELSE = ranked by qualifier macro-mean; top half ADVANCE, rest CUT.
         With <3 runs everything not hard-cut ADVANCEs (nothing to rank against).

Usage:
  python3 qualify.py
  python3 qualify.py --top 2        # advance at most 2
  python3 qualify.py --json
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

MC_KEYS = ("accuracy",)
CODE_CATS = {"code"}
AGENT_CATS = {"agentic"}


def headline(summary, cat):
    s = summary.get(cat)
    if not s:
        return None
    for k in ("accuracy", "pass_hat_k_reliability", "mean_score", "clean_rate"):
        if isinstance(s.get(k), (int, float)):
            return s[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RESULTS, "*-q.json")))
    if not files:
        raise SystemExit("no qualifier runs in results/ (looked for *-q.json)")

    rows = []
    for f in files:
        d = json.load(open(f))
        s = d.get("summary", {})
        mc_sig = sum(1 for c, v in s.items()
                     if isinstance(v, dict) and v.get("significant_vs_chance"))
        code_pass = sum(v.get("passed", 0) for c, v in s.items()
                        if c in CODE_CATS and isinstance(v, dict))
        chains = sum(v.get("pass_hat_k_reliability", 0) * v.get("n_chains", 0)
                     for c, v in s.items() if c in AGENT_CATS and isinstance(v, dict))
        vals = [v for c in s for v in [headline(s, c)] if isinstance(v, (int, float))]
        macro = sum(vals) / len(vals) if vals else 0.0
        empty = max([v.get("empty_output_rate", 0) for v in s.values()
                     if isinstance(v, dict)] + [0])
        rows.append({"label": os.path.basename(f)[:-7], "file": f,
                     "macro": round(macro, 4), "cats": len(vals),
                     "mc_significant": mc_sig,
                     "code_passed": code_pass,
                     "chains_solved": round(chains, 2),
                     "empty_rate": empty})

    for r in rows:
        r["no_signal"] = (r["mc_significant"] == 0 and r["code_passed"] == 0
                          and r["chains_solved"] == 0) or r["empty_rate"] > 0.5

    ranked = sorted([r for r in rows if not r["no_signal"]],
                    key=lambda r: -r["macro"])
    if len(ranked) >= 3:
        adv_n = args.top or max(1, len(ranked) // 2)
        for i, r in enumerate(ranked):
            r["verdict"] = "ADVANCE" if i < adv_n else "CUT"
    else:
        for r in ranked:
            r["verdict"] = "ADVANCE"
    for r in rows:
        if r["no_signal"]:
            r["verdict"] = "CUT"

    if args.json:
        print(json.dumps(rows, indent=1))
        return
    print("=== QUALIFIER TABLE (q-vs-q only; never vs full splits) ===\n")
    print(f"{'model':<22} {'macro':>6} {'MCsig':>6} {'code':>5} "
          f"{'chains':>7} {'verdict':>8}")
    for r in sorted(rows, key=lambda r: (r["verdict"] != "ADVANCE", -r["macro"])):
        print(f"{r['label']:<22} {r['macro']:>6.3f} {r['mc_significant']:>6} "
              f"{r['code_passed']:>5} {r['chains_solved']:>7} {r['verdict']:>8}")
    print("\nADVANCE = goes to screening/full. CUT = out of the tournament.")


if __name__ == "__main__":
    main()
