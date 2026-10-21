#!/usr/bin/env python3
"""Build a self-contained HTML comparison report (ECharts, vendored offline).

Reads results/llamacpp/*.json screening runs (skips *-q qualifier files),
renders capability radar, accuracy bars with Wilson CIs, heatmap, efficiency
scatter, bilingual gaps, MC-vs-chance, MMLU subjects, agentic detail,
throughput, and a full data table into reports/<name>.html.

Usage:
  python3 charts.py
  python3 charts.py --out reports/screening-2026-10-18.html --title "llmbench screening"
"""
import argparse
import datetime
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "llamacpp")
REPORTS = os.path.join(HERE, "reports")

HEADLINE = {"code": "accuracy", "bugfix": "accuracy", "algo": "accuracy",
            "refactor": "accuracy", "rust": "accuracy", "embedded_cpp": "accuracy",
            "science": "accuracy", "electronics": "accuracy", "persian": "accuracy",
            "english": "accuracy", "netdiag": "mean_score", "devops": "mean_score",
            "lint": "clean_rate", "agentic": "pass_hat_k_reliability"}
CIKEY = {"code": "ci95_wilson", "bugfix": "ci95_wilson", "algo": "ci95_wilson",
         "refactor": "ci95_wilson", "rust": "ci95_wilson", "embedded_cpp": "ci95_wilson",
         "science": "ci95_wilson", "electronics": "ci95_wilson", "persian": "ci95_wilson",
         "english": "ci95_wilson",
         # netdiag/devops deliberately absent: their mean partial-credit scores
         # have no Wilson interval (ci95_wilson_perfect belongs to the perfect
         # rate, and pairing them once put a point outside its own bracket).
         "lint": "ci95_wilson_clean",
         "agentic": "ci95_wilson"}
CUSTOM_CATS = ["code", "bugfix", "algo", "refactor", "rust", "embedded_cpp",
               "science", "electronics", "persian", "english",
               "netdiag", "devops", "lint", "agentic"]
# The s30 instrument only. MMLU-CS-100 is a DIFFERENT instrument (a decider
# suite with its own section and its own runs in <label>-cs.json); leaving it
# in this axis rendered an empty column for every model.
STD_CATS = ["std_arc_challenge_s30", "std_arc_easy_s30", "std_hellaswag_s30",
            "std_winogrande_s30", "std_truthfulqa_mc_s30", "std_belebele_fa_s30",
            "std_mmlu_s30", "std_gsm8k_s30"]
SHORT = {"code": "Code", "bugfix": "BugFix", "algo": "Algo", "refactor": "Refactor",
         "rust": "Rust", "embedded_cpp": "Embed-C++", "science": "Science",
         "electronics": "Electronics", "persian": "Persian", "english": "English",
         "netdiag": "NetDiag", "devops": "DevOps", "lint": "Lint", "agentic": "Agentic",
         "std_arc_challenge_s30": "ARC-C", "std_arc_easy_s30": "ARC-E",
         "std_hellaswag_s30": "HellaSwag", "std_winogrande_s30": "WinoGrande",
         "std_truthfulqa_mc_s30": "TruthfulQA", "std_belebele_fa_s30": "Belebele-fa",
         "std_mmlu_s30": "MMLU-6", "std_gsm8k_s30": "GSM8K",
         "std_mmlu_cs": "MMLU-CS (100)"}
COLORS = ["#5470C6", "#91CC75", "#FAC858", "#EE6666", "#73C0DE", "#3BA272"]
# Stable per-family colors: the same model keeps its color in every section
# (file-order colors made granite change identity between renders).
FAMILY_COLORS = {
    "nemotron-3-nano": "#91CC75",
    "qwen35-2b": "#FAC858",
    "granite4-3b": "#5470C6",
    "minicpm5-2b": "#B57EDC",
    "k2-horizon": "#3BA272",
    "medgemma-4b": "#EE6666",
    "qwen35-4b": "#9aa4b2",
}
FAMILIES = ["nemotron-3-nano", "qwen35-2b", "granite4-3b", "minicpm5-2b",
            "k2-horizon", "medgemma-4b", "qwen35-4b"]


def famkey(label):
    for fam in FAMILIES:
        if label.startswith(fam):
            return fam
    return label


def famcolor(label):
    return FAMILY_COLORS.get(famkey(label), "#8b96a3")


def cell(summary, cat):
    s = summary.get(cat)
    if not s:
        return None
    v = s.get(HEADLINE.get(cat, "accuracy"))
    # Mean partial-credit scores carry no interval (see report.py ci()).
    ci = None if cat in ("netdiag", "devops") \
        else s.get(CIKEY.get(cat, "ci95_wilson"), [None, None])
    n = s.get("n_items") or s.get("n") or s.get("n_chains")
    return {"v": v, "lo": ci[0] if ci else None, "hi": ci[1] if ci else None,
            "n": n, "chance": s.get("chance_baseline"),
            "range": s.get("accuracy_range")}


# The llama-server build swap: ~/ollama-work/run-llama.sh pointed at
# llama-b11558 until this moment, llama-b11714 afterwards.
UPGRADE_TS = datetime.datetime(2026, 10, 17, 11, 6).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPORTS, "screening.html"))
    ap.add_argument("--title", default="llmbench screening — model comparison")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(RESULTS, "*.json"))
                   if not f.endswith("-q.json") and not f.endswith("-cs.json")
                   and not f.endswith("-agentic.json")
                   and not f.endswith("-gsm8k.json")
                   and not f.endswith("-speed.json")
                   and not f.endswith("-txt.json")
                   and not f.endswith("-medical.json")
                   and not f.endswith("-coder.json")
                   and not f.endswith("-embedded.json"))
    # Cross-backend agentic evidence (e.g. nemotron via Ollama, whose blob
    # cannot call tools): dedicated rows with ONLY agentic populated. Every
    # other cell renders missing, so these rows compare on exactly one suite.
    oll_dir = os.path.join(HERE, "results", "ollama")
    extra = sorted(glob.glob(os.path.join(oll_dir, "*-agentic.json")))
    if not files:
        raise SystemExit(f"no screening runs in {RESULTS}")
    models = []

    # Isolated throughput for EVERY model that has it, so the report treats
    # all six models equally even when one has no screening row.
    speed_rows = []
    for sp in sorted(glob.glob(os.path.join(RESULTS, "*-speed.json"))):
        try:
            sd = json.load(open(sp))
        except Exception:
            continue
        speed_rows.append({
            "label": sd.get("model", os.path.basename(sp)[:-10]),
            "tps": sd.get("tps_median"),
            "p25": sd.get("tps_p25"), "p75": sd.get("tps_p75"),
            "min": sd.get("tps_min"), "max": sd.get("tps_max"),
            "n": sd.get("n_prompts"),
            "foreign_cpu": sd.get("foreign_cpu_pct_at_start"),
            "server": sd.get("server_version"),
        })

    def build_row(f, label=None, agentic_pref=True):
        d = json.load(open(f))
        s = d.get("summary", {})
        deep_src = {}
        deep_data = {}
        # Deepest-available samples: dedicated <label>-agentic.json /
        # <label>-gsm8k.json runs supersede the screening subset for those
        # cells only (chain/item counts stay visible, never silently mixed).
        for suffix in ("-agentic.json", "-gsm8k.json"):
            fp = os.path.join(RESULTS, os.path.basename(f)[:-5] + suffix)
            if os.path.exists(fp):
                try:
                    dd = json.load(open(fp))
                    deep_data[suffix] = dd
                    deep_src[suffix] = dd.get("summary", s)
                except Exception:
                    pass
        ag_sum = deep_src.get("-agentic.json", s) if agentic_pref else s
        gs_sum = deep_src.get("-gsm8k.json", s)
        cats = {c: cell(ag_sum if c == "agentic" else
                        gs_sum if c == "std_gsm8k_s30" else s, c)
                for c in CUSTOM_CATS + STD_CATS}
        # A cell can be REAL but not COMPARABLE: if the backend's tool probe
        # failed, the agentic chains cannot call tools, so the score measures a
        # broken pairing rather than the model. It must not sit silently next
        # to models whose backend does call tools.
        broken = {}
        probes = (d.get("provenance") or {}).get("backend_probes") or {}
        tools_ok = (probes.get("tools") or {}).get("ok")
        if tools_ok is False:
            alt = ""
            oll = os.path.join(HERE, "results", "ollama",
                               f"{os.path.basename(f)[:-5]}-agentic.json")
            if os.path.exists(oll):
                try:
                    av = (json.load(open(oll)).get("summary", {})
                          .get("agentic", {}).get("pass_hat_k_reliability"))
                    if av is not None:
                        alt = (f" The same weights score {av:.3f} through Ollama, "
                               f"where the tool-calling pairing works.")
                except Exception:
                    pass
            broken["agentic"] = (
                "NOT COMPARABLE: this backend cannot emit a parseable tool call "
                "(tools probe FAILED), so the chains scored zero by construction."
                + alt)
        vals_c = [cats[c]["v"] for c in CUSTOM_CATS
                  if cats[c] and isinstance(cats[c]["v"], (int, float))]
        vals_s = [cats[c]["v"] for c in STD_CATS
                  if cats[c] and isinstance(cats[c]["v"], (int, float))]
        dg = d.get("provenance", {}).get("model_digest", {}) or {}
        lab = label or os.path.basename(f)[:-5]
        # Server build, per file: captured in the provenance when the run
        # happened, otherwise inferred from the file's completion date (the
        # launcher served b11558 until 2026-10-17 11:06, b11714 after). The
        # qualifier instrument is entirely b11558; screening/CS-100/agentic are
        # b11714. Instruments are never compared across eras.
        ver = (d.get("provenance", {}) or {}).get("versions", {}) or {}
        server = ver.get("llama_server")
        # Dedicated isolated throughput pass (speed.py): one model on a quiet
        # box, fresh server, warmed up, median of 12 identical prompts. This
        # is the number the efficiency chart publishes - a tok/s taken as a
        # by-product of a long benchmark run is not a speed measurement.
        speed = None
        sp = os.path.join(RESULTS, f"{os.path.basename(f)[:-5]}-speed.json")
        if os.path.exists(sp):
            try:
                speed = json.load(open(sp))
            except Exception:
                speed = None
        if not server:
            server = ("b11558 (inferred)" if os.path.getmtime(f) < UPGRADE_TS
                      else "b11714 (inferred)")
        # Coverage gaps of every file that feeds this row: an incomplete run
        # must be visible on the page, not hidden behind a missing cell. A
        # gap in a cell that is DISPLAYED from a deeper, complete file is not
        # a gap of this row (the deeper file already answered it).
        deep_by_task = {}
        for suf, dd in deep_data.items():
            deep_by_task["agentic" if suf == "-agentic.json" else "std_gsm8k_s30"] = dd
        base_name = os.path.basename(f)
        gaps = []
        for src_name, dd in [(base_name, d)] + list(deep_data.items()):
            g = dd.get("coverage_gaps")
            for x in (g or []):
                task = x.split("/")[0]
                if src_name == base_name:
                    deep = deep_by_task.get(task)
                    if deep is not None and not (deep.get("coverage_gaps")
                                                 or deep.get("partial")):
                        continue
                gaps.append(f"{src_name}: {x}")
            if not g and dd.get("partial"):
                gaps.append(f"{src_name}: incomplete run (stubs or unrun tasks)")
        return {
            "label": lab,
            "color": famcolor(lab),
            "fileGB": round((dg.get("size_bytes") or 0) / 1e9, 2),
            "digest": str(dg.get("digest", ""))[-12:],
            "server": server,
            "macro_custom": round(sum(vals_c) / len(vals_c), 4) if vals_c else None,
            "macro_std": round(sum(vals_s) / len(vals_s), 4) if vals_s else None,
            "mean_tps": (speed or {}).get("tps_median"),
            "tps_source": "isolated" if speed else "in-run",
            "speed": speed,
            "mean_tps_inrun": (d.get("backend_totals") or {}).get("mean_gen_tps"),
            "elapsed_h": round(d.get("elapsed_s", 0) / 3600, 2),
            "n_recs": len(d.get("records", [])),
            "cats": cats,
            "gaps": gaps,
            "broken": broken,
            "mmlu": (s.get("std_mmlu_s30") or {}).get("by_subject", {}),
            "agentic": {k: ag_sum.get("agentic", {}).get(k) for k in
                        ("pass_hat_k_reliability", "per_trial_trace_rate",
                         "final_answer_accuracy_no_tools_required",
                         "mean_tool_calls", "fabrication_rate", "n_chains")},
            "tps": {c: ((gs_sum if c == "std_gsm8k_s30" else s).get(c) or {}).get("gen_tps_mean")
                    for c in ("code", "std_gsm8k_s30")},
        }

    for f in files:
        models.append(build_row(f))
    # Cross-backend rows are collected here for the deep-dive section only.
    deep = []
    for f in extra:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        s = d.get("summary", {})
        a = s.get("agentic")
        if not a:
            continue
        lab = os.path.basename(f)[:-5] + " (ollama)"
        deep.append({
            "label": lab, "color": famcolor(lab),
            "n_chains": a.get("n_chains"), "n_trials": None,
            "agentic": {k: a.get(k) for k in
                        ("pass_hat_k_reliability", "per_trial_trace_rate",
                         "final_answer_accuracy_no_tools_required",
                         "mean_tool_calls", "fabrication_rate", "n_chains")},
        })
    # ---- the second MC mechanism (text match), a SEPARATE instrument ----
    # A model that answers with the option text instead of the letter cannot
    # be scored by the letter-logprob mechanism. Its numbers live in their own
    # categories and their own section: two mechanisms, two instruments, never
    # one ranking.
    txt_rows = []
    TXT_CATS = ["science_txt", "electronics_txt", "persian_txt", "english_txt",
                "std_arc_challenge_s30_txt", "std_arc_easy_s30_txt",
                "std_hellaswag_s30_txt", "std_winogrande_s30_txt",
                "std_truthfulqa_mc_s30_txt", "std_belebele_fa_s30_txt",
                "std_mmlu_s30_txt", "std_mmlu_cs_txt"]
    for f in sorted(glob.glob(os.path.join(RESULTS, "*-txt.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        s = d.get("summary", {})
        if not any(c in s for c in TXT_CATS):
            continue
        lab = os.path.basename(f)[:-len("-txt.json")]
        if lab.endswith("-cs"):
            lab = lab[:-3]
        txt_rows.append({
            "label": lab, "color": famcolor(lab),
            "cats": {c: cell(s, c) for c in TXT_CATS},
            "n_recs": len(d.get("records", [])),
            "unparsed": {c: (s.get(c) or {}).get("unparsed_rate")
                         for c in TXT_CATS},
            "gaps": list(d.get("coverage_gaps") or [])
                    + (["incomplete run (stubs or unrun tasks)"]
                       if d.get("partial") and not d.get("coverage_gaps") else []),
        })
    # ---- qualifier section: identical s30-era subsets for all six models ----
    qcats = ["science", "electronics", "persian", "english",
             "std_arc_easy_s30", "std_winogrande_s30", "std_truthfulqa_mc_s30",
             "std_hellaswag_s30", "std_belebele_fa_s30", "std_mmlu_s30",
             "code", "std_gsm8k_s30", "agentic"]
    qualifier = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "*-q.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        s = d.get("summary", {})
        lab = os.path.basename(f)[:-5]
        qver = (d.get("provenance", {}) or {}).get("versions", {}) or {}
        qserver = qver.get("llama_server") or (
            "b11558 (inferred)" if os.path.getmtime(f) < UPGRADE_TS
            else "b11714 (inferred)")
        qualifier.append({
            "label": lab, "color": famcolor(lab),
            "cats": {c: cell(s, c) for c in qcats},
            "n_recs": len(d.get("records", [])),
            "server": qserver,
        })
    # ---- agentic deep-dive: dedicated 19-chain files (both backends) plus
    # screening subsets for models without one. The ollama rows live ONLY
    # here — never in the screening charts (they would flatten every axis).
    for f in sorted(glob.glob(os.path.join(RESULTS, "*-agentic.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a = (d.get("summary", {}) or {}).get("agentic")
        if not a:
            continue
        lab = os.path.basename(f)[:-5]
        if "ollama" in f:
            lab += " (ollama)"
        deep.append({
            "label": lab, "color": famcolor(lab),
            "n_chains": a.get("n_chains"), "n_trials": a.get("n_trials"),
            "agentic": {
                "pass_hat_k_reliability": a.get("pass_hat_k_reliability"),
                "per_trial_trace_rate": a.get("per_trial_trace_rate"),
                "final_answer_accuracy_no_tools_required":
                    a.get("final_answer_accuracy_no_tools_required"),
                "mean_tool_calls": a.get("mean_tool_calls"),
                "fabrication_rate": None,
                "n_chains": a.get("n_chains"),
                "neg_refusal_rate": a.get("neg_refusal_rate")},
        })
    # screening agentic subsets, for depth contrast (chain counts shown)
    for m in models:
        if not any(r["label"].startswith(m["label"]) for r in deep):
            deep.append({
                "label": m["label"] + " (screen-subset)", "color": m["color"],
                "n_chains": m["agentic"].get("n_chains"),
                "n_trials": None,
                "agentic": dict(m["agentic"]),
            })
    deep.sort(key=lambda r: (r["label"].endswith("(screen-subset)"),
                             r["label"]))
    # ---- CS-100 decider ----
    cs100 = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "*-cs.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        s = (d.get("summary", {}) or {}).get("std_mmlu_cs") or {}
        if s.get("accuracy") is None:
            continue
        lab = os.path.basename(f)[:-5]
        ci = s.get("ci95_wilson") or [None, None]
        cs100.append({"label": lab, "color": famcolor(lab),
                      "cats": {"std_mmlu_cs": {
                          "v": s.get("accuracy"), "lo": ci[0], "hi": ci[1],
                          "n": s.get("n_items") or s.get("n"),
                          "chance": s.get("chance_baseline")}},
                      "n_recs": len(d.get("records", []))})
    # ---- coverage matrix: who ran what (plan vs reality) ----
    cov_cells = []
    for fam in FAMILIES:
        row = {"model": fam, "color": famcolor(fam), "cells": []}
        scr = [f for f in files if famkey(os.path.basename(f)[:-5]) == fam]
        row["cells"].append(sum(len(json.load(open(f)).get("records", []))
                                for f in scr) or None)
        qf = [f for f in glob.glob(os.path.join(RESULTS, "*-q.json"))
              if famkey(os.path.basename(f)[:-5]) == fam]
        row["cells"].append(sum(len(json.load(open(f)).get("records", []))
                                for f in qf) or None)
        ag = [f for f in (glob.glob(os.path.join(RESULTS, "*-agentic.json"))
                          + glob.glob(os.path.join(HERE, "results", "ollama",
                                                   "*-agentic.json")))
              if famkey(os.path.basename(f)[:-5]) == fam]
        row["cells"].append(sum(len(json.load(open(f)).get("records", []))
                                for f in ag) or None)
        cs = [f for f in glob.glob(os.path.join(RESULTS, "*-cs.json"))
              if famkey(os.path.basename(f)[:-5]) == fam]
        row["cells"].append(sum(len(json.load(open(f)).get("records", []))
                                for f in cs) or None)
        tx = [f for f in glob.glob(os.path.join(RESULTS, "*-txt.json"))
              if famkey(os.path.basename(f)[:-len("-txt.json")]) == fam]
        row["cells"].append(sum(len(json.load(open(f)).get("records", []))
                                for f in tx) or None)
        if any(row["cells"]):
            cov_cells.append(row)
    payload = {"models": models,
               "custom": CUSTOM_CATS, "std": STD_CATS, "short": SHORT,
               "speeds": speed_rows,
               "txt": txt_rows, "txt_cats": TXT_CATS,
               "q": qualifier, "q_cats": qcats,
               "deep": deep, "cs": cs100, "coverage": cov_cells,
               "meta": {"title": args.title, "runs": len(models)}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    html = PAGE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    open(args.out, "w").write(html)
    print(f"wrote {args.out} ({len(html) // 1024} KB, {len(models)} models)")


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llmbench screening</title>
<script src="assets/echarts.min.js"></script>
<style>
body{background:#11161d;color:#e6e8eb;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;padding:0 18px 60px}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:26px;margin:26px 0 4px} .sub{color:#9aa4b2;font-size:13px;margin-bottom:6px}
.chip{display:inline-block;background:#1d2630;border:1px solid #2c3947;border-radius:12px;padding:2px 10px;margin:2px 4px 2px 0;font-size:12px}
.chart{background:#161d26;border:1px solid #232e3b;border-radius:10px;margin:16px 0;padding:12px 8px 4px}
.chart h2{font-size:16px;margin:6px 12px;color:#cfd6de}
.chart .note{font-size:12px;color:#8b96a3;margin:0 12px 8px}
.chart>div[id]{width:100%;height:420px}
table{border-collapse:collapse;width:100%;font-size:12px;background:#161d26}
th,td{border:1px solid #232e3b;padding:5px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#1d2630;color:#cfd6de}
tr:nth-child(even) td{background:#131922}
.foot{color:#8b96a3;font-size:12px;margin-top:18px;line-height:1.6}
code{background:#1d2630;padding:1px 6px;border-radius:6px}
.warn{background:#3a2a12;border:1px solid #7a5a1e;color:#f0c674;border-radius:9px;padding:0 7px;font-size:11px;cursor:help}
</style></head><body><div class="wrap">
<h1 id="t"></h1><div class="sub" id="chips"></div>
<div class="sub" style="margin:2px 0 10px"><b>How to read the marks:</b>
<span class="warn">⚠ incomplete</span> = the run behind that row is missing items (never a score) ·
<span class="warn">⚠ n/c</span> = the cell is real but <i>not comparable</i> (broken tool-calling backend) ·
every bar is an accuracy with an exact 95% Wilson interval, and every comparison on this page stays inside one instrument.</div>
<h2 style="margin-top:20px">Coverage: who ran what</h2>
<p class="note" style="font-size:12px;color:#8b96a3">Record counts per instrument. Every section below compares only within one instrument — cross-section ranks are never computed.</p>
<div style="overflow-x:auto"><table id="cov"></table></div>
<div class="chart"><h2>Capability radar (headline metric per suite, higher is better)</h2>
<p class="note">One axis per suite. Lint = clean rate, NetDiag/DevOps = mean score, Agentic = pass^k reliability, rest = accuracy. One cell is a broken-backend artifact and is marked everywhere it appears: nemotron's agentic score comes from a llama.cpp build that cannot emit a parseable tool call, so its chains could not use tools (the same weights reach 0.895 through Ollama). It is drawn here as measured, but it is not a like-for-like comparison — see the ⚠ n/c marks below.</p><div id="c-radar"></div></div>
<div class="chart"><h2>Custom suites — accuracy with 95% Wilson intervals</h2>
<p class="note">Whiskers are exact 95% Wilson intervals (never the normal approximation). Hover for n and chance baseline.</p><div id="c-custom" style="height:460px"></div></div>
<div class="chart"><h2>Standard s30 suites — accuracy with 95% Wilson intervals</h2>
<p class="note">Stratified 30-item official-split instances. Same-interval semantics as above. MMLU-6, HellaSwag and WinoGrande are POSITION-DEBIASED: the upstream option order put 10-18 points of the answer in one letter, so the gold positions were permuted deterministically (same questions, same distractors) and those cells were re-measured. Every cell also publishes the score a constant picker would get, so the remaining confound is visible.</p><div id="c-std" style="height:440px"></div></div>
<div class="chart"><h2>Capability heatmap</h2>
<p class="note">Darker = stronger. Missing = suite absent for that model.</p><div id="c-heat" style="height:380px"></div></div>
<div class="chart"><h2>Efficiency: capability vs decode speed</h2>
<p class="note">Bubble area = model file GB. Faster + higher is better; top-right wins. Speed here is the DEDICATED isolated pass: one model on an otherwise idle box, fresh server, two warm-up prompts discarded, then the median decode rate of 12 identical prompts - not a by-product of a long benchmark run. Server builds are named per model above: the qualifier instrument is entirely b11558, screening/CS-100/agentic are b11714; no comparison on this page spans two builds.</p><div id="c-eff" style="height:400px"></div></div>
<div class="chart"><h2>English vs Persian (custom + Belebele-fa standard)</h2>
<p class="note">Gap = English accuracy − Persian accuracy, in percentage points.</p><div id="c-lang" style="height:380px"></div></div>
<div class="chart"><h2>Multiple-choice vs chance baseline</h2>
<p class="note">Bars = accuracy; ✕ = chance (mean 1/k over items, not flat 25%). A bar at chance level knows nothing.</p><div id="c-chance" style="height:440px"></div></div>
<div class="chart"><h2>MMLU-6 by subject (n=5 each)</h2>
<p class="note">Tiny-n: ±35pp intervals. Read rankings, not gaps.</p><div id="c-mmlu" style="height:380px"></div></div>
<div class="chart"><h2>Agentic: reliability, trace, answer-only</h2>
<p class="note">pass^k requires tool use in order; answer-only ignores tools. Their gap = "knew but would not use tools". Cells use each model's deepest run (chain count shown); depths differ, so compare shapes, not just ranks.</p><div id="c-agent" style="height:360px"></div></div>
<div class="chart"><h2>In-run decode rate (historical, under background load)</h2>
<p class="note">Kept for the record: these per-suite rates were taken as a by-product of long benchmark runs while a runaway desktop thumbnailer held half the CPU (REVIEW5), so they UNDERSTATE true speed. Any speed claim on this page uses the isolated pass instead.</p><div id="c-tps" style="height:340px"></div></div>
<div class="chart"><h2>Isolated throughput, every model</h2>
<p class="note">One model on an otherwise idle box, fresh server, two warm-up prompts discarded, then the median of 12 identical prompts. The spread (p25-p75, min-max) is in the tooltip; the foreign-CPU column is what the desktop was using at measurement time. These replace the in-run rates above, which were taken while a runaway thumbnailer held half the CPU.</p><div id="c-speed" style="height:300px"></div></div>
<div class="chart"><h2>Text-match MC: the second mechanism (separate instrument)</h2>
<p class="note">A model that answers with the option's TEXT instead of the letter ("4" rather than "B") puts no probability mass on the letters, so the letter-logprob mechanism cannot score it - that model is unmeasured there, not wrong. This is the other mechanism: the generated answer is read and matched to the option. It is a DIFFERENT instrument: its categories are separate and are never ranked against the letter-logprob numbers above. Hover for the unparsed rate (answers that matched no option - counted as incorrect, never guessed).</p><div id="c-txt" style="height:380px"></div></div>
<div class="chart"><h2>Qualifier: identical fast subsets, all models</h2>
<p class="note">Same items for everyone (deterministic subsets); ranks models for screening, never against full splits. All six qualifier runs share ONE server build (b11558, 2026-10-17) so the ranking is internally consistent; the screening instrument was measured on b11714. The two are never compared with each other. See <code>python3 qualify.py</code> for CUT/ADVANCE.</p><div id="c-qual" style="height:440px"></div></div>
<div style="overflow-x:auto"><table id="tbl-q"></table></div>
<div class="chart"><h2>Agentic deep-dive: every model with 19-chain evidence</h2>
<p class="note">Deep files (19 chains) plus screening subsets (8 chains, chain count shown) for models without one. pass^k requires tool use; compare within equal chain counts.</p><div id="c-deep" style="height:380px"></div></div>
<div class="chart"><h2>CS-100 decider: full official CS subject</h2>
<p class="note">100 items, ±10pp — the instrument that acquitted qwen's s30 gap (all pairs TIE). Whiskers are Wilson intervals.</p><div id="c-cs" style="height:360px"></div></div>
<div style="overflow-x:auto"><table id="tbl-cs"></table></div>
<h2 style="margin-top:26px">Full data table</h2>
<div style="overflow-x:auto"><table id="tbl"></table></div>
<div class="foot" id="foot"></div>
</div>
<script>
const D = __DATA__;
document.title = D.meta.title;
document.getElementById('t').textContent = D.meta.title;
document.getElementById('chips').innerHTML =
  D.models.map(m=>`<span class="chip"><b style="color:${m.color}">●</b> ${m.label} · ${m.fileGB} GB · <code>…${m.digest}</code> · ${m.elapsed_h} h · server ${m.server}${m.gaps&&m.gaps.length?' <span class="warn" title="'+m.gaps.join('&#10;')+'">⚠ incomplete</span>':''}</span>`).join('');
document.getElementById('foot').innerHTML =
  `Scores: per-suite headline with 95% Wilson intervals; MC figures carry chance baselines and exact binomial p-values in the JSON. ` +
  `PRIMARY-STD and PRIMARY-CUSTOM macro-means are shown as ★ AVG in the table. ` +
  `s30 = stratified 30-item official-split instances (never full-leaderboard numbers). ` +
  `Reproduce: <code>bash run_llama.sh all qualifier</code> → <code>python3 qualify.py</code> → screening → <code>python3 charts.py</code>. ` +
  `Charts: Apache ECharts ${echarts.version}, vendored offline in <code>reports/assets</code>.`;
const AX = {textStyle:{color:'#cfd6de'},splitLine:{lineStyle:{color:'#232e3b'}},axisLine:{lineStyle:{color:'#3a4756'}}};
const CHARTS = [];
function mk(id){ const c = echarts.init(document.getElementById(id), null, {renderer:'canvas'}); CHARTS.push(c); return c; }
// ?only=c-radar etc: render a single section (deep-linking + debugging)
const ONLY = new URLSearchParams(location.search).get('only');
if (ONLY) {
  const _mk = mk;
  mk = function(id){ if(!id.includes(ONLY)) return {setOption(){},resize(){}}; return _mk(id); };
}
function pct(v){ return v==null?'—':(v*100).toFixed(1)+'%'; }
function tipOf(m, c, cell){
  if(!cell || cell.v==null) return `${m.label}<br/>${D.short[c]}: — (not run)`;
  if(m.broken && m.broken[c]) return `${m.label}<br/><b>${D.short[c]}: ${pct(cell.v)}</b><br/>`+
    `<span style="color:#f0c674">⚠ ${m.broken[c]}</span>`;
  if(m.gaps&&m.gaps.length) { /* gap badge shown next to the model name */ }
  let t = `${m.label}<br/><b>${D.short[c]}: ${pct(cell.v)}</b>`;
  if(cell.lo!=null) t += `<br/>95% CI ${pct(cell.lo)} – ${pct(cell.hi)} · n=${cell.n}`;
  else if(cell.n!=null) t += `<br/>n=${cell.n}`;
  if(cell.chance!=null) t += `<br/>chance ${pct(cell.chance)}`;
  if(c==='agentic' && m.agentic && m.agentic.n_chains!=null)
    t += `<br/>${m.agentic.n_chains} chains`;
  return t;
}
/* 1. radar */
(function(){
  const axes = ['code','rust','embedded_cpp','science','electronics','persian',
    'english','netdiag','devops','agentic','std_mmlu_s30','std_gsm8k_s30'];
  mk('c-radar').setOption({radar:{indicator:axes.map(c=>({name:D.short[c],max:1})),
    axisName:{color:'#9aa4b2',fontSize:11},splitArea:{areaStyle:{color:['#161d26','#141a22']}}},
    legend:{textStyle:{color:'#cfd6de'},bottom:0},
    tooltip:{formatter:p=>`${p.seriesName}<br/>${p.name}: <b>${pct(p.value)}</b>`}, series:D.models.map(m=>({name:m.label,type:'radar',
      itemStyle:{color:m.color},lineStyle:{color:m.color},areaStyle:{color:m.color,opacity:0.12},
      data:[{value:axes.map(c=>{const x=m.cats[c];return x&&x.v!=null?x.v:0;}),name:m.label}]}))});
})();
/* shared grouped-bar builder with CI whiskers (custom series) */
function barsWithCI(id, cats, models, rotate, fs){
  rotate = rotate==null?28:rotate; fs = fs==null?11:fs;
  const ch = mk(id), M = models.length;
  const series = [];
  models.forEach((m,mi)=>{
    series.push({name:m.label,type:'bar',itemStyle:{color:m.color},
      data:cats.map(c=>{const x=m.cats[c];
        if(!x||x.v==null) return '-';
        if(m.broken&&m.broken[c]) return {value:x.v,itemStyle:{
          color:'#39424e',borderColor:'#f0c674',borderWidth:1.5,borderType:'dashed'}};
        return x.v;}),
      barGap:'18%', barCategoryGap:'32%'});
  });
  // whisker data: [catIndex, modelIndex, lo, hi]
  const wd = [];
  models.forEach((m,mi)=>cats.forEach((c,ci)=>{
    const x=m.cats[c];
    if(x&&x.v!=null&&x.lo!=null) wd.push([ci,mi,x.lo,x.hi]);
  }));
  const CATGAP=0.32, BARGAP=0.18;
  series.push({type:'custom',name:'95% CI',
    // NOTE (ECharts 5.6): renderItem params carry NO .value — read dimensions
    // via api.value(i). Tooltip uses the dataIndex into wd for the same reason.
    renderItem:(p,api)=>{
      const ci=api.value(0), mi=api.value(1), lo=api.value(2), hi=api.value(3);
      const band=api.size([1,0])[0];
      const usable=band*(1-CATGAP*2);
      const w=usable/(M+BARGAP*(M-1));
      const start=api.coord([ci,0])[0]-usable/2;
      const cx=start+w*(mi*(1+BARGAP)+0.5);
      const yLo=api.coord([0,lo])[1], yHi=api.coord([0,hi])[1];
      return {type:'group',children:[
        {type:'line',shape:{x1:cx,y1:yLo,x2:cx,y2:yHi},style:{stroke:'#e6e8eb',lineWidth:1.5}},
        {type:'line',shape:{x1:cx-4,y1:yLo,x2:cx+4,y2:yLo},style:{stroke:'#e6e8eb',lineWidth:1.5}},
        {type:'line',shape:{x1:cx-4,y1:yHi,x2:cx+4,y2:yHi},style:{stroke:'#e6e8eb',lineWidth:1.5}}]};
    },data:wd,
    tooltip:{formatter:p=>{const w=wd[p.dataIndex];const m=D.models[w[1]],c=cats[w[0]];
      return tipOf(m,c,m.cats[c]);}},z:5});
  ch.setOption({legend:{textStyle:{color:'#cfd6de'},top:0},
    tooltip:{trigger:'item',formatter:p=>{
      if(p.seriesName==='95% CI') return p.data&&p.data.tooltip||'';
      const m=models.find(x=>x.label===p.seriesName);
      return tipOf(m,cats[p.dataIndex],m.cats[cats[p.dataIndex]]);}},
    grid:{bottom:70,top:40}, xAxis:{type:'category',data:cats.map(c=>D.short[c]),
      axisLabel:{color:'#9aa4b2',rotate:rotate,fontSize:fs},axisLine:{lineStyle:{color:'#3a4756'}}},
    yAxis:{type:'value',max:1,...AX}, series});
}
/* wire whisker tooltip via data */
barsWithCI('c-custom', D.custom, D.models); barsWithCI('c-std', D.std, D.models);
/* 3. heatmap */
(function(){
  const cats=[...D.custom,...D.std], data=[];
  D.models.forEach((m,mi)=>cats.forEach((c,ci)=>{
    const x=m.cats[c];
    const brk = m.broken && m.broken[c];
    data.push([ci,mi,x&&x.v!=null?Math.round(x.v*1000)/1000:null,brk?1:0]);
  }));
  mk('c-heat').setOption({tooltip:{formatter:p=>{
      const m=D.models[p.value[1]],c=cats[p.value[0]];return tipOf(m,c,m.cats[c]);}},
    grid:{height:'78%',top:10}, xAxis:{type:'category',data:cats.map(c=>D.short[c]),
      axisLabel:{color:'#9aa4b2',rotate:30,fontSize:10},splitArea:{show:true}},
    yAxis:{type:'category',data:D.models.map(m=>m.label),axisLabel:{color:'#cfd6de'}},
    visualMap:{min:0,max:1,calculable:true,orient:'vertical',right:0,top:'middle',
      textStyle:{color:'#9aa4b2'},inRange:{color:['#2b3a4a','#3d6b8e','#5470C6','#91CC75','#FAC858']}},
    series:[{type:'heatmap',data,label:{show:true,fontSize:9,color:'#fff',
      formatter:p=>p.value[2]==null?'—':(p.value[3]?'⚠'+p.value[2]:p.value[2])},
      itemStyle:{borderColor:'#11161d',borderWidth:1}}]});
})();
/* 4. efficiency */
(function(){
  mk('c-eff').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{formatter:p=>`${p.seriesName} · ${p.data[3]}<br/>macro ${pct(p.data[1])} · ${p.data[0]} tok/s · ${p.data[2]} GB`},
    xAxis:{name:'decode tok/s (isolated pass)',...AX}, yAxis:{name:'macro-mean',max:1,...AX},
    series:['macro_custom','macro_std'].map((k,ki)=>({name:ki?'PRIMARY-STD':'PRIMARY-CUSTOM',
      type:'scatter',symbolSize:d=>18+Math.sqrt(d[2])*14,
      itemStyle:{color:ki?'#EE6666':'#5470C6',opacity:0.9},
      data:D.models.filter(m=>m[k]!=null&&m.mean_tps!=null&&m.tps_source==='isolated')
        .map(m=>[m.mean_tps,m[k],m.fileGB,m.label])}))});
})();
/* 5. bilingual */
(function(){
  const cats=['english','persian','std_belebele_fa_s30'];
  mk('c-lang').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>{const m=D.models.find(x=>x.label===p.seriesName);
      return tipOf(m,cats[p.dataIndex],m.cats[cats[p.dataIndex]]);}},
    xAxis:{type:'category',data:['English','Persian','Belebele-fa'],axisLabel:{color:'#cfd6de'}},yAxis:{max:1,...AX},
    series:D.models.map(m=>({name:m.label,type:'bar',itemStyle:{color:m.color},
      data:cats.map(c=>{const x=m.cats[c];return x&&x.v!=null?x.v:'-';})}))});
})();
/* 6. MC vs chance */
(function(){
  const cats=[...D.custom.filter(c=>['science','electronics','persian','english'].includes(c)),...D.std.filter(c=>c!=='std_gsm8k_s30')];
  const series=D.models.map(m=>({name:m.label,type:'bar',itemStyle:{color:m.color},
    data:cats.map(c=>{const x=m.cats[c];return x&&x.v!=null?x.v:'-';})}));
  series.push({name:'chance',type:'scatter',symbol:'x',symbolSize:12,
    itemStyle:{color:'#8b96a3'},data:cats.map(c=>{
      const xs=D.models.map(m=>m.cats[c]).filter(x=>x&&x.chance!=null).map(x=>x.chance);
      return xs.length?Math.max(...xs):'-';})});
  mk('c-chance').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>{const m=D.models.find(x=>x.label===p.seriesName);
      if(!m) return 'chance baseline';return tipOf(m,cats[p.dataIndex],m.cats[cats[p.dataIndex]]);}},
    grid:{bottom:70},xAxis:{type:'category',data:cats.map(c=>D.short[c]),
      axisLabel:{color:'#9aa4b2',rotate:26,fontSize:11}},yAxis:{max:1,...AX},series});
})();
/* 7. MMLU subjects */
(function(){
  const subs=[...new Set(D.models.flatMap(m=>Object.keys(m.mmlu)))].sort();
  mk('c-mmlu').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>`${p.seriesName}<br/>${p.name}: <b>${pct(p.value)}</b> (n=5)`},
    xAxis:{type:'category',data:subs.map(s=>s.split('/')[1]||s),
      axisLabel:{color:'#9aa4b2',rotate:22,fontSize:10}},yAxis:{max:1,...AX},
    series:D.models.map(m=>({name:m.label,type:'bar',itemStyle:{color:m.color},
      data:subs.map(s=>{const x=m.mmlu[s];return x?x.acc:'-';})}))});
})();
/* 8. agentic */
function agenticChart(id, models){
  const rows=[['pass_hat_k_reliability','pass^k (tools required)'],
    ['per_trial_trace_rate','trace rate'],
    ['final_answer_accuracy_no_tools_required','answer-only']];
  mk(id).setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>{const m=models.find(x=>x.label===p.seriesName);
      return `${m.label}<br/>${rows[p.dataIndex][1]}: <b>${pct(p.value)}</b><br/>${m.agentic.mean_tool_calls} calls/trial · ${m.agentic.n_chains} chains`;}},
    xAxis:{type:'category',data:rows.map(r=>r[1]),axisLabel:{color:'#cfd6de'}},yAxis:{max:1,...AX},
    series:models.map(m=>({name:m.label,type:'bar',itemStyle:{color:m.color},
      data:rows.map(r=>m.agentic[r[0]]!=null?m.agentic[r[0]]:'-')}))});
}
agenticChart('c-agent', D.models);
/* 8b. isolated throughput table */
(function(){
  const rows=D.speeds||[];
  if(!rows.length){mk('c-speed').setOption({title:{show:true,text:'no isolated measurements yet'}});return;}
  mk('c-speed').setOption({
    tooltip:{trigger:'item',formatter:p=>{const r=rows[p.dataIndex];
      return `<b>${r.label}</b><br/>median ${r.tps} tok/s<br/>p25 ${r.p25} - p75 ${r.p75}<br/>min ${r.min} - max ${r.max}<br/>${r.n} prompts · foreign CPU ${r.foreign_cpu}%`;}},
    grid:{left:120,right:40,top:20,bottom:30},
    xAxis:{type:'value',name:'decode tok/s (median)',nameTextStyle:{color:'#8b96a3'},...AX},
    yAxis:{type:'category',data:rows.map(r=>r.label),axisLabel:{color:'#cfd6de'},...AX},
    series:[{type:'bar',itemStyle:{color:'#5470C6'},
      data:rows.map(r=>({value:r.tps,itemStyle:{color:r.tps>=8?'#91CC75':r.tps>=5?'#5470C6':'#FAC858'}})),
      label:{show:true,position:'right',color:'#e6e8eb',formatter:p=>p.value+' tok/s'}}]});
})();
/* 8c. text-match MC (a separate instrument) */
(function(){
  const rows=D.txt||[];
  if(!rows.length){mk('c-txt').setOption({title:{show:true,text:'no text-match runs yet',
    left:'center',top:'middle',textStyle:{color:'#8b96a3',fontSize:13}}});return;}
  const cats=D.txt_cats;
  mk('c-txt').setOption({legend:{textStyle:{color:'#cfd6de'},top:0},
    tooltip:{trigger:'item',formatter:p=>{const r=rows.find(x=>x.label===p.seriesName||x.label+' ⚠'===p.seriesName);
      const c=cats[p.dataIndex];const cell=r.cats[c];
      const un=r.unparsed&&r.unparsed[c]!=null?`<br/>unparsed ${(r.unparsed[c]*100).toFixed(1)}%`:'';
      const rng=cell&&cell.range?`<br/>range ${pct(cell.range[0])}-${pct(cell.range[1])} (an unparsed answer could have been right)`:'';
      const gap=r.gaps&&r.gaps.length?`<br/><span style="color:#f0c674">⚠ ${r.gaps.join('; ')}</span>`:'';
      return `${r.label}<br/><b>${D.short[c]||c}: ${pct(cell&&cell.v)}</b>${rng}<br/>n=${cell&&cell.n}${un}${gap}`;}},
    grid:{bottom:110,top:40},xAxis:{type:'category',data:cats.map(c=>(D.short[c]||c).replace('_txt','')),
      axisLabel:{color:'#9aa4b2',rotate:30,fontSize:10},axisLine:{lineStyle:{color:'#3a4756'}}},
    yAxis:{max:1,...AX},
    series:rows.map(r=>({name:r.label+(r.gaps&&r.gaps.length?' ⚠':''),
      type:'bar',itemStyle:{color:r.color},
      data:cats.map(c=>{const x=r.cats[c];return x&&x.v!=null?x.v:'-';})}))});
})();
/* 9. throughput */
(function(){
  const rows=[['code','Code tok/s'],['std_gsm8k','GSM8K tok/s']];
  mk('c-tps').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>`${p.seriesName}<br/>${rows[p.dataIndex][1]}: <b>${p.value}</b>`},
    xAxis:{type:'category',data:rows.map(r=>r[1]),axisLabel:{color:'#cfd6de'}},yAxis:{...AX},
    series:D.models.map(m=>({name:m.label,type:'bar',itemStyle:{color:m.color},
      data:rows.map(r=>m.tps[r[0]]!=null?m.tps[r[0]]:'-')}))});
})();
/* table */
(function(){
  const cats=[...D.custom,...D.std];
  let h='<tr><th>suite</th>'+D.models.map(m=>`<th>${m.label}</th>`).join('')+'</tr>';
  const macC=D.models.map(m=>m.macro_custom), macS=D.models.map(m=>m.macro_std);
  h+=`<tr><td>★ PRIMARY-CUSTOM</td>${macC.map(v=>`<td><b>${v!=null?pct(v):'—'}</b></td>`).join('')}</tr>`;
  h+=`<tr><td>★ PRIMARY-STD</td>${macS.map(v=>`<td><b>${pct(v)}</b></td>`).join('')}</tr>`;
  cats.forEach(c=>{
    h+=`<tr><td>${D.short[c]}</td>`+D.models.map(m=>{
      const x=m.cats[c];
      if(!x||x.v==null) return '<td>—</td>';
      let s=`<b>${pct(x.v)}</b>${(m.broken&&m.broken[c])?' <span class="warn" title="'+m.broken[c]+'">⚠ n/c</span>':''}`;
      if(x.lo!=null) s+=`<br/><span style="color:#8b96a3">${pct(x.lo)}–${pct(x.hi)} · n=${x.n}</span>`;
      else if(x.n!=null) s+=`<br/><span style="color:#8b96a3">n=${x.n}</span>`;
      return `<td>${s}</td>`;}).join('')+'</tr>';
  });
  document.getElementById('tbl').innerHTML=h;
})();
/* coverage matrix */
(function(){
  let h='<tr><th>model</th><th>screening recs</th><th>qualifier recs</th><th>agentic-deep recs</th><th>CS-100 recs</th><th>text-MC recs</th></tr>';
  D.coverage.forEach(r=>{
    h+='<tr><td><b style="color:'+r.color+'">●</b> '+r.model+'</td>'+
      r.cells.map(c=>'<td>'+(c==null?'—':c)+'</td>').join('')+'</tr>';
  });
  document.getElementById('cov').innerHTML=h;
})();
/* qualifier section */
(function(){
  const QC=['science','electronics','persian','english','std_arc_easy_s30','std_winogrande_s30','std_truthfulqa_mc_s30','std_hellaswag_s30','std_belebele_fa_s30','std_mmlu_s30','code','std_gsm8k_s30','agentic'];
  barsWithCI('c-qual', QC, D.q, 42, 10);
  let h='<tr><th>model</th><th>qual macro</th><th>code</th><th>agentic rel</th></tr>';
  D.q.forEach(m=>{
    const vals=QC.map(c=>m.cats[c]).filter(x=>x&&x.v!=null).map(x=>x.v);
    const macro=vals.length?(vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(3):'—';
    const code=m.cats.code, ag=m.cats.agentic;
    const ci=x=>x&&x.lo!=null?` [${x.lo},${x.hi}]`:'';
    h+=`<tr><td><b style="color:${m.color}">●</b> ${m.label}${m.gaps&&m.gaps.length?' <span class="warn" title="'+m.gaps.join('&#10;')+'">⚠</span>':''}</td><td><b>${macro}</b></td>`+
      `<td>${code&&code.v!=null?pct(code.v)+ci(code):'—'}</td><td>${ag&&ag.v!=null?pct(ag.v)+ci(ag):'—'}</td></tr>`;
  });
  document.getElementById('tbl-q').innerHTML=h;
})();
agenticChart('c-deep', D.deep);
/* cs100 section */
(function(){
  barsWithCI('c-cs', ['std_mmlu_cs'], D.cs);
  let h='<tr><th>model</th><th>CS-100</th><th>95% CI</th></tr>';
  D.cs.forEach(m=>{
    const x=m.cats.std_mmlu_cs;
    h+=`<tr><td><b style="color:${m.color}">●</b> ${m.label}</td><td><b>${pct(x.v)}</b></td><td>${pct(x.lo)}–${pct(x.hi)} · n=${x.n}</td></tr>`;
  });
  h+='<tr><td colspan="3" style="text-align:left;color:#8b96a3">Paired McNemar on shared items, all three pairs: TIE (closest p=0.053). The s30 gap did not replicate.</td></tr>';
  document.getElementById('tbl-cs').innerHTML=h;
})();
window.addEventListener('resize',()=>{CHARTS.forEach(c=>c.resize());});
</script></body></html>
"""


if __name__ == "__main__":
    main()
