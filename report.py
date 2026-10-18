#!/usr/bin/env python3
"""
Build the cross-model comparison table from results/*.json.

Reads only what run.py wrote, so a table can be regenerated at any time without
touching a model. Every cell that came from a Wilson interval is rendered with its
interval; a single number with no interval at n<100 is a claim the data does not
support.

Usage:
  python3 report.py                     # markdown table to stdout
  python3 report.py --json              # machine-readable
  python3 report.py --csv out.csv
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

CATS = ["code", "bugfix", "algo", "refactor", "rust", "embedded_cpp", "science",
        "electronics", "persian", "english", "netdiag", "devops", "lint", "agentic",
        "std_arc_c", "std_arc_e", "std_hellaswag", "std_winogrande",
        "std_truthfulqa", "std_belebele_fa", "std_mmlu", "std_gsm8k",
        "std_arc_challenge_s30", "std_arc_easy_s30", "std_hellaswag_s30",
        "std_winogrande_s30", "std_truthfulqa_mc_s30", "std_belebele_fa_s30",
        "std_mmlu_s30", "std_gsm8k_s30", "std_mmlu_cs"]

HEAD = {
    "code": "HumanEval+MBPP", "bugfix": "BugFix", "algo": "Algo",
    "refactor": "Refactor", "rust": "Rust", "embedded_cpp": "Embedded C++",
    "science": "Science (custom)", "electronics": "Electronics (custom)",
    "persian": "Persian (custom)", "english": "English (custom)",
    "netdiag": "NetDiag", "devops": "DevOps",
    "lint": "Lint", "agentic": "Agentic",
    "std_arc_c": "ARC-Challenge", "std_arc_e": "ARC-Easy",
    "std_hellaswag": "HellaSwag", "std_winogrande": "WinoGrande",
    "std_truthfulqa": "TruthfulQA-MC1", "std_belebele_fa": "Belebele-fa",
    "std_mmlu": "MMLU-6", "std_gsm8k": "GSM8K",
    "std_arc_challenge_s30": "ARC-C (s30)", "std_arc_easy_s30": "ARC-E (s30)",
    "std_hellaswag_s30": "HellaSwag (s30)",
    "std_winogrande_s30": "WinoGrande (s30)",
    "std_truthfulqa_mc_s30": "TruthfulQA (s30)",
    "std_belebele_fa_s30": "Belebele-fa (s30)",
    "std_mmlu_s30": "MMLU-6 (s30)", "std_gsm8k_s30": "GSM8K (s30)",
    "std_mmlu_cs": "MMLU-CS (100)",
}

# What "higher is better" means per category, so the composite can be signed.
HIGHER_BETTER = {
    "code": "accuracy", "bugfix": "accuracy", "algo": "accuracy",
    "refactor": "accuracy", "rust": "accuracy", "embedded_cpp": "accuracy",
    "science": "accuracy", "electronics": "accuracy", "persian": "accuracy",
    "english": "accuracy", "netdiag": "mean_score", "devops": "mean_score",
    "lint": "clean_rate", "agentic": "pass_hat_k_reliability",
    "std_arc_c": "accuracy", "std_arc_e": "accuracy",
    "std_hellaswag": "accuracy", "std_winogrande": "accuracy",
    "std_truthfulqa": "accuracy", "std_belebele_fa": "accuracy",
    "std_mmlu": "accuracy", "std_gsm8k": "accuracy",
    "std_arc_challenge_s30": "accuracy", "std_arc_easy_s30": "accuracy",
    "std_hellaswag_s30": "accuracy", "std_winogrande_s30": "accuracy",
    "std_truthfulqa_mc_s30": "accuracy", "std_belebele_fa_s30": "accuracy",
    "std_mmlu_s30": "accuracy", "std_gsm8k_s30": "accuracy",
    "std_mmlu_cs": "accuracy",
}


def val(s, key):
    return s.get(key) if isinstance(s, dict) else None


def ci(s, key, default=None):
    # Mean partial-credit scores (netdiag/devops) have NO Wilson interval:
    # ci95_wilson_perfect is a CI on the *perfect-rate*, pairing it with the
    # mean once put a point estimate outside its own bracket. Show nothing
    # rather than something misleading.
    if key == "mean_score":
        return default
    for k in (key.replace("accuracy", "ci95_wilson"),
              key.replace("clean_rate", "ci95_wilson_clean"),
              key.replace("pass_hat_k_reliability", "ci95_wilson")):
        if k in s:
            return s[k]
    return default


def fmt(v, w=6):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--filter", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RESULTS, "**", "*.json"),
                                 recursive=True))
    # Backend separation is structural: results/ollama/ vs results/llamacpp/.
    # Rows are namespaced "label (backend)" so cross-backend rows can never
    # be mistaken for same-setup comparisons. speed.json (a dict of probes,
    # not a run) is skipped.
    runs = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if "summary" not in d:
            continue
        name = os.path.basename(f)[:-5]
        if args.filter and args.filter not in name:
            continue
        runs.append(d)
    if not runs:
        raise SystemExit(f"no results in {RESULTS}")
    runs.sort(key=lambda d: d["config"].get("started_unix", 0))

    rows = {}
    for d in runs:
        base = (d["config"].get("label")
                or os.path.basename(d["config"]["model"])
                .replace("/", "_").replace(":", "_"))
        be = d["config"].get("backend", "?")
        rows[f"{base} ({be})"] = d

    if args.json:
        print(json.dumps({k: {"config": v["config"], "summary": v["summary"],
                             "backend_totals": v.get("backend_totals"),
                             "host": v.get("host")}
                         for k, v in rows.items()}, ensure_ascii=False, indent=1))
        return

    if args.csv:
        import csv
        keys = [HEAD[c] for c in CATS]
        w = csv.writer(open(args.csv, "w", newline=""))
        w.writerow(["model"] + keys + ["gen_tps_mean", "wall_min"])
        for name, d in rows.items():
            s = d["summary"]
            w.writerow([name] + [fmt(val(s.get(c), HIGHER_BETTER[c])) for c in CATS]
                       + [fmt(val(s.get("code"), "gen_tps_mean")),
                          round(d.get("elapsed_s", 0) / 60, 1)])
        print(f"wrote {args.csv}")
        return

    # ------------------------------------------------------------ markdown
    print("## llmbench results\n")
    print("Accuracy cells show the point estimate with its 95% Wilson interval. "
          "Intervals are wide by design - every task here has n < 100, where the "
          "normal approximation is invalid.\n")
    names = list(rows)
    # PRIMARY (pre-registered): macro-mean of the per-category headline
    # scores. This is the only number on which a "model A > model B" claim may
    # rest, and even then only with a paired test (see compare.py). Every
    # per-category cell below is EXPLORATORY: with 8 models x 22 cells, reading
    # winners off individual cells is multiple-comparisons roulette.
    # Standard (published) and custom (hand-made) suites are reported in
    # separate primaries so the composite never mixes incomparable protocols:
    # custom MC uses hand-written items, standard uses official splits.
    print("\n### PRIMARY-STD: macro-mean over standard suites (pre-registered)\n")
    print("| model | macro-mean-std | standard suites present |")
    print("|---|---|---|")
    std_cats = [c for c in CATS if c.startswith("std_")]
    for n in names:
        s = rows[n]["summary"]
        vals = [val(s.get(c), HIGHER_BETTER[c]) for c in std_cats]
        vals = [v for v in vals if isinstance(v, (int, float))]
        macro = sum(vals) / len(vals) if vals else None
        print(f"| {n} | {fmt(macro)} | {len(vals)}/{len(std_cats)} |")
    print("\n### PRIMARY-CUSTOM: macro-mean over custom suites (pre-registered)\n")
    print("| model | macro-mean-custom | custom suites present |")
    print("|---|---|---|")
    custom_cats = [c for c in CATS if not c.startswith("std_")]
    for n in names:
        s = rows[n]["summary"]
        vals = [val(s.get(c), HIGHER_BETTER[c]) for c in custom_cats]
        vals = [v for v in vals if isinstance(v, (int, float))]
        macro = sum(vals) / len(vals) if vals else None
        print(f"| {n} | {fmt(macro)} | {len(vals)}/{len(custom_cats)} |")
    print("\n_Cells below are exploratory. Do not declare per-cell winners without "
          "a Holm-corrected paired test (compare.py)._\n")
    for c in CATS:
        if not any(c in rows[n]["summary"] for n in names):
            continue
        key = HIGHER_BETTER[c]
        print(f"\n### {HEAD[c]}\n")
        print("| model | score | 95% CI | n | extra |")
        print("|---|---|---|---|---|")
        for n in names:
            s = rows[n]["summary"].get(c)
            if not s:
                continue
            v = val(s, key)
            i = ci(s, key)
            ncell = s.get("n_items") or s.get("n") or s.get("n_chains") or "-"
            extra = []
            if c in ("code", "bugfix", "algo", "refactor", "rust", "embedded_cpp"):
                if s.get("parse_fail_rate"):
                    extra.append(f"parse-fail {s['parse_fail_rate']:.0%}")
            if c == "rust":
                for k, lbl in (("clippy_clean_rate", "clippy-clean"),
                               ("wrote_fn_main_rate", "wrote-main"),
                               ("rustfmt_already_formatted_rate", "fmt-ok")):
                    if k in s:
                        extra.append(f"{lbl} {s[k]:.0%}")
            if c == "embedded_cpp":
                for k, lbl in (("compile_error_rate", "compile-err"),
                               ("assert_fail_rate", "assert-fail"),
                               ("warning_free_rate", "no-warn")):
                    if k in s:
                        extra.append(f"{lbl} {s[k]:.0%}")
            if c == "lint":
                extra.append(f"mean {s.get('mean_diagnostics')} diags")
                if s.get("noqa_gaming_rate"):
                    extra.append(f"noqa {s['noqa_gaming_rate']:.0%}")
                for k in ("clippy_pedantic_mean", "clippy_nursery_mean"):
                    if k in s:
                        extra.append(f"{k.split('_')[1]} {s[k]}")
            if c == "netdiag":
                extra.append(f"false-alarm {s.get('false_alarm_rate', 0):.0%}")
                extra.append(f"detect {s.get('real_fault_detected_rate', 0):.0%}")
            if c == "agentic":
                extra.append(f"pass@1 {s.get('pass_at_1')}")
                extra.append(f"trace {s.get('per_trial_trace_rate')}")
                extra.append(f"fabricated {s.get('fabrication_rate'):.0%}")
                extra.append(f"calls {s.get('mean_tool_calls')}")
            if c in ("science", "electronics", "persian", "english",
                     "std_arc_c", "std_arc_e", "std_hellaswag", "std_winogrande",
                     "std_truthfulqa", "std_belebele_fa", "std_mmlu",
                     "std_arc_challenge_s30", "std_arc_easy_s30",
                     "std_hellaswag_s30", "std_winogrande_s30",
                     "std_truthfulqa_mc_s30", "std_belebele_fa_s30",
                     "std_mmlu_s30", "std_mmlu_cs"):
                extra.append(f"conf {s.get('mean_confidence')}")
                extra.append(f"{s.get('tokens_per_item')} tok/item")
            print(f"| {n} | {fmt(v)} | {('[' + ', '.join(fmt(x,3) for x in i) + ']') if i else '-'} "
                  f"| {ncell} | {', '.join(extra)} |")

    print("\n### Throughput and conditions\n")
    print("| model | backend | mean tok/s | calls | prompt tok | eval tok | wall min | mem avail end | swap free end | reps | cfg |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        d = rows[n]
        bt = d.get("backend_totals") or {}
        h = d.get("host") or {}
        print(f"| {n} | {d['config'].get('backend')} | {fmt(bt.get('mean_gen_tps'))} "
              f"| {bt.get('calls')} | {bt.get('prompt_tokens')} | {bt.get('eval_tokens')} "
              f"| {round(d.get('elapsed_s',0)/60,1)} | {h.get('mem_available_end_mib')} MiB "
              f"| {h.get('swap_free_end_mib')} MiB | {d['config'].get('reps')} | `{d.get('config_hash')}` |")

    print("\n### Per-model detail\n")
    for n in names:
        d = rows[n]
        s = d["summary"]
        for cat in CATS:
            if cat in s and s[cat].get("by_subject"):
                subs = s[cat]["by_subject"]
                print(f"- **{n} / {HEAD[cat]}**: " +
                      ", ".join(f"{k} {v['acc']:.0%} (n={v['n']})" for k, v in subs.items()))


if __name__ == "__main__":
    main()