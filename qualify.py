#!/usr/bin/env python3
"""World-Cup referee for qualifier runs.

Ranks only categories that are both qualified and measured on identical items
for every complete run. If fewer than three common categories survive those
checks, it reports NOT RANKABLE instead of manufacturing a cross-model macro.
Qualifier numbers are never compared against full-split scores.

Verdict logic (signal checklist, no single magic number):
  CUT  = no significant MC category AND zero code passes AND zero solved
         chains (nothing measured but noise), OR >50% empty outputs.
  ELSE = if >=3 comparable categories remain, rank their common-item macro;
          otherwise report NOT RANKABLE. With <3 runs, don't rank.

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

MC_CATS = {
    "science", "electronics", "persian", "english", "std_arc_easy_s30",
    "std_winogrande_s30", "std_truthfulqa_mc_s30", "std_hellaswag_s30",
    "std_belebele_fa_s30", "std_mmlu_s30",
}
CODE_CATS = {"code"}
AGENT_CATS = {"agentic"}
MIN_SHARED_CATEGORIES = 3


def headline(summary, cat):
    s = summary.get(cat)
    if not s:
        return None
    for k in ("accuracy", "pass_hat_k_reliability", "mean_score", "clean_rate"):
        if isinstance(s.get(k), (int, float)):
            return s[k]
    return None


def item_signature(data, cat):
    """Stable item identity for a qualifier category, independent of scores."""
    def belongs(r):
        if cat == "code":
            return r.get("task_category") == "code"
        if cat == "std_gsm8k_s30":
            return r.get("task") == cat or r.get("task_category") == "gsm8k"
        if cat == "agentic":
            return r.get("task_category") == "agentic"
        return r.get("task") == cat or r.get("task_category") == cat

    items = []
    for r in data.get("records", []):
        if not belongs(r):
            continue
        ident = r.get("id") or r.get("chain_id")
        if ident is not None:
            items.append((r.get("task"), r.get("task_category"), ident,
                          r.get("prompt")))
    return tuple(sorted(items))


def analyze_files(files):
    """Analyze qualifier files without pooling failed or mismatched cells."""
    rows = []
    for f in sorted(files):
        d = json.load(open(f))
        s = d.get("summary", {})
        be = d.get("config", {}).get("backend", "?")
        probes = ((d.get("provenance") or {}).get("backend_probes") or {})
        mc_ok = (probes.get("mc_logprob") or {}).get("ok") is True
        tools_ok = (probes.get("tools") or {}).get("ok") is True
        scores, signatures, invalid = {}, {}, []
        for c in s:
            v = headline(s, c)
            if not isinstance(v, (int, float)):
                continue
            if c in MC_CATS and not mc_ok:
                invalid.append(c)
                continue
            if c in AGENT_CATS and not tools_ok:
                invalid.append(c)
                continue
            sig = item_signature(d, c)
            if not sig:
                invalid.append(c)
                continue
            scores[c] = v
            signatures[c] = sig

        mc_sig = sum(1 for c in MC_CATS if c in scores
                     and (s.get(c) or {}).get("significant_vs_chance"))
        code_pass = sum(v.get("passed", 0) for c, v in s.items()
                        if c in CODE_CATS and c in scores and isinstance(v, dict))
        chains = sum(v.get("pass_hat_k_reliability", 0) * v.get("n_chains", 0)
                     for c, v in s.items()
                     if c in AGENT_CATS and c in scores and isinstance(v, dict))
        empty = max([v.get("empty_output_rate", 0) for v in s.values()
                     if isinstance(v, dict)] + [0])
        n_errors = sum(v.get("n_errors", 0) for v in s.values()
                       if isinstance(v, dict))
        rows.append({
            "label": os.path.basename(f)[:-7] + f" ({be})", "file": f,
            "partial": bool(d.get("partial")), "scores": scores,
            "signatures": signatures, "invalid_categories": sorted(invalid),
            "mc_significant": mc_sig, "code_passed": code_pass,
            "chains_solved": round(chains, 2), "empty_rate": empty,
            "n_errors": n_errors,
        })

    complete = [r for r in rows if not r["partial"]]
    common = set.intersection(*(set(r["scores"]) for r in complete)) if complete else set()
    common = [c for c in common if all(
        r["signatures"].get(c) == complete[0]["signatures"].get(c)
        for r in complete)]
    # Stable ordering makes output and tests reproducible.
    common.sort()
    rankable = len(common) >= MIN_SHARED_CATEGORIES and len(complete) >= 3
    for r in rows:
        vals = [r["scores"][c] for c in common if c in r["scores"]]
        r["macro"] = round(sum(vals) / len(vals), 4) if vals else None
        r["cats"] = len(vals)
        r["rankable"] = rankable
        r["common_categories"] = common
        r["no_signal"] = (r["mc_significant"] == 0 and r["code_passed"] == 0
                          and r["chains_solved"] == 0) or r["empty_rate"] > 0.5
    return rows, common, rankable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RESULTS, "**", "*-q.json"),
                                 recursive=True))
    if not files:
        raise SystemExit("no qualifier runs in results/ (looked for **/*-q.json)")

    rows, common, rankable = analyze_files(files)
    eligible = [r for r in rows if not r["no_signal"] and not r["partial"]]
    if rankable:
        ranked = sorted(eligible, key=lambda r: -(r["macro"] or 0))
        adv_n = args.top or max(1, len(ranked) // 2)
        for i, r in enumerate(ranked):
            r["verdict"] = "ADVANCE" if i < adv_n else "CUT"
    else:
        for r in eligible:
            r["verdict"] = "NOT RANKABLE"
    for r in rows:
        if r["no_signal"]:
            r["verdict"] = "CUT"
        if r["partial"]:
            r["verdict"] = "PARTIAL"

    if args.json:
        print(json.dumps(rows, indent=1))
        return
    print("=== QUALIFIER TABLE (q-vs-q only; never vs full splits) ===\n")
    print("shared qualified, identical-item categories: "
          + (", ".join(common) if common else "none"))
    if not rankable:
        print(f"NOT RANKABLE: {len(common)} shared categories; need "
              f"{MIN_SHARED_CATEGORIES}. Probe failures and/or different item "
              "sets prevent a like-for-like macro.\n")
    print(f"{'model':<24} {'shared μ':>8} {'cats':>5} {'MCsig':>6} "
          f"{'code':>5} {'chains':>7} {'errs':>5} {'verdict':>13}")
    order = {"ADVANCE": 0, "PARTIAL": 1, "NOT RANKABLE": 2, "CUT": 3}
    for r in sorted(rows, key=lambda x: (order[x["verdict"]], -(x["macro"] or 0))):
        macro = f"{r['macro']:.3f}" if r["macro"] is not None else "—"
        print(f"{r['label']:<24} {macro:>8} {r['cats']:>5} "
              f"{r['mc_significant']:>6} {r['code_passed']:>5} "
              f"{r['chains_solved']:>7} {r['n_errors']:>5} "
              f"{r['verdict']:>13}")
    print("\nThe shared mean is descriptive when NOT RANKABLE; it is not a CUT/ADVANCE score. "
          "PARTIAL runs need completion before any ranking.")


if __name__ == "__main__":
    main()
