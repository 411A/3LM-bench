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
        # The text-mechanism cells for the same model, if any. A model whose
        # backend cannot score answer LETTERS still has multiple-choice
        # numbers - measured by the other mechanism. They belong in the row,
        # visibly marked as a different instrument, because a row that looks
        # empty is a false statement about the model.
        txt_cells = {}
        ts = {}
        tf = os.path.join(RESULTS, f"{os.path.basename(f)[:-5]}-txt.json")
        if os.path.exists(tf):
            try:
                ts = json.load(open(tf)).get("summary", {})
            except Exception:
                ts = {}
            for c, e in ts.items():
                if c.endswith("_txt") and e.get("accuracy") is not None:
                    ci = e.get("ci95_wilson") or [None, None]
                    txt_cells[c[:-4]] = {
                        "v": e["accuracy"], "lo": ci[0], "hi": ci[1],
                        "n": e.get("n_items"),
                        "chance": e.get("chance_baseline"),
                        "src": "text_match",
                        "unparsed": e.get("unparsed_rate"),
                        "range": e.get("accuracy_range"),
                    }
        vals_c = []
        c_has_txt = False
        for c in CUSTOM_CATS:
            if cats[c] and isinstance(cats[c]["v"], (int, float)):
                vals_c.append(cats[c]["v"])
            elif c in txt_cells and isinstance(txt_cells[c]["v"], (int, float)):
                vals_c.append(txt_cells[c]["v"])
                c_has_txt = True
        vals_s = []
        s_has_txt = False
        for c in STD_CATS:
            if cats[c] and isinstance(cats[c]["v"], (int, float)):
                vals_s.append(cats[c]["v"])
            elif c in txt_cells and isinstance(txt_cells[c]["v"], (int, float)):
                vals_s.append(txt_cells[c]["v"])
                s_has_txt = True
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
        mmlu_sub = (s.get("std_mmlu_s30") or {}).get("by_subject") or (
            ts.get("std_mmlu_s30_txt") or {}).get("by_subject", {})
        return {
            "label": lab,
            "color": famcolor(lab),
            "fileGB": round((dg.get("size_bytes") or 0) / 1e9, 2),
            "digest": str(dg.get("digest", ""))[-12:],
            "server": server,
            "macro_custom": round(sum(vals_c) / len(vals_c), 4) if vals_c else None,
            "macro_std": round(sum(vals_s) / len(vals_s), 4) if vals_s else None,
            "macro_custom_txt": c_has_txt,
            "macro_std_txt": s_has_txt,
            "mean_tps": (speed or {}).get("tps_median"),
            "tps_source": "isolated" if speed else "in-run",
            "speed": speed,
            "mean_tps_inrun": (d.get("backend_totals") or {}).get("mean_gen_tps"),
            "elapsed_h": round(d.get("elapsed_s", 0) / 3600, 2),
            "n_recs": len(d.get("records", [])),
            "cats": cats,
            "txt_cats": txt_cells,
            "gaps": gaps,
            "broken": broken,
            "mmlu": mmlu_sub,
            "mmlu_txt": bool(not (s.get("std_mmlu_s30") or {}).get("by_subject") and mmlu_sub),
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
    txt_by_lab = {}
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
        row = txt_by_lab.setdefault(lab, {
            "label": lab, "color": famcolor(lab),
            "cats": {c: None for c in TXT_CATS},
            "n_recs": 0,
            "unparsed": {c: None for c in TXT_CATS},
            "gaps": [],
        })
        for c in TXT_CATS:
            v = cell(s, c)
            if v is not None:
                row["cats"][c] = v
                row["unparsed"][c] = (s.get(c) or {}).get("unparsed_rate")
        row["n_recs"] += len(d.get("records", []))
        row["gaps"] += list(d.get("coverage_gaps") or []) + (
            ["incomplete run (stubs or unrun tasks)"]
            if d.get("partial") and not d.get("coverage_gaps") else [])
    txt_rows = [txt_by_lab[k] for k in sorted(txt_by_lab)]
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
    for f in sorted(glob.glob(os.path.join(RESULTS, "*-cs.json"))
                    + glob.glob(os.path.join(RESULTS, "*-cs-txt.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        is_txt = f.endswith("-cs-txt.json")
        key = "std_mmlu_cs_txt" if is_txt else "std_mmlu_cs"
        s = (d.get("summary", {}) or {}).get(key) or {}
        if s.get("accuracy") is None:
            continue
        lab = os.path.basename(f)[:-5]
        ci = s.get("ci95_wilson") or [None, None]
        c_entry = {
            "v": s.get("accuracy"), "lo": ci[0], "hi": ci[1],
            "n": s.get("n_items") or s.get("n"),
            "chance": s.get("chance_baseline"),
        }
        if is_txt:
            c_entry["src"] = "text_match"
            c_entry["unparsed"] = s.get("unparsed_rate")
            c_entry["range"] = s.get("accuracy_range")
        cs100.append({"label": lab, "color": famcolor(lab),
                      "cats": {"std_mmlu_cs": None if is_txt else c_entry},
                      "txt_cats": {"std_mmlu_cs": c_entry} if is_txt else {},
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
<span class="warn">⚠ txt</span> / lighter bar = multiple-choice cell measured by the <i>text-match</i> mechanism (generated answer matched to option, not single-token letter logprob) ·
bars show accuracy; hover any bar for its exact 95% Wilson interval (also printed in the full data table).</div>
<h2 style="margin-top:20px">Coverage: who ran what</h2>
<p class="note" style="font-size:12px;color:#8b96a3">Record counts per instrument. Every section below compares only within one instrument — cross-section ranks are never computed.</p>
<div style="overflow-x:auto"><table id="cov"></table></div>
<div class="chart"><h2>Capability radar (headline metric per suite, higher is better)</h2>
<p class="note">One axis per suite. Lint = clean rate, NetDiag/DevOps = mean score, Agentic = pass^k reliability, rest = accuracy. One cell is a broken-backend artifact and is marked everywhere it appears: nemotron's agentic score comes from a llama.cpp build that cannot emit a parseable tool call, so its chains could not use tools (the same weights reach 0.895 through Ollama). It is drawn here as measured, but it is not a like-for-like comparison — see the ⚠ n/c marks below.</p><div id="c-radar"></div></div>
<div class="chart"><h2>Custom suites — accuracy</h2>
<p class="note">Hover any bar for its exact 95% Wilson interval (never the normal approximation), n and chance baseline. Lighter bars marked ⚠ txt are the text-match instrument, not letter-logprob peers.</p><div id="c-custom" style="height:460px"></div></div>
<div class="chart"><h2>Standard s30 suites — accuracy</h2>
<p class="note">Stratified 30-item official-split instances. Hover for the Wilson interval, n and chance baseline. MMLU-6, HellaSwag and WinoGrande are POSITION-DEBIASED: the upstream option order put 10-18 points of the answer in one letter, so the gold positions were permuted deterministically (same questions, same distractors) and those cells were re-measured. Every cell also publishes the score a constant picker would get, so the remaining confound is visible.</p><div id="c-std" style="height:440px"></div></div>
<div class="chart"><h2>Capability heatmap</h2>
<p class="note">Cells outlined in amber were measured with the OTHER multiple-choice mechanism (text match) - a different instrument, not a peer of the cells around it. Lighter = stronger (0 to 1). "\u2014" = the suite was not measured for that model <i>in this instrument</i>. A cell marked with an amber \u26a0 is real but not comparable (a broken tool-calling backend). k2-horizon's multiple-choice cells come from the TEXT-MATCH instrument - its llama.cpp build cannot score answer letters - so they are a different measurement and are never ranked against the letter-logprob cells of the other models. Hover any cell for its instrument, n and unparsed rate.</p><div id="c-heat" style="height:380px"></div></div>
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
<p class="note">100 items, ±10pp — the instrument that acquitted qwen's s30 gap (all pairs TIE). Hover for Wilson intervals.</p><div id="c-cs" style="height:360px"></div></div>
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
function cellOf(m, c){
  if(!m) return null;
  const x = m.cats && m.cats[c];
  if(x && x.v!=null) return x;
  const t = m.txt_cats && m.txt_cats[c];
  if(t && t.v!=null) return t;
  return null;
}
function tipOf(m, c, cell){
  cell = cell || cellOf(m, c);
  const sname = D.short[c] || c;
  if(!cell || cell.v==null) return `${m.label}<br/>${sname}: — (not run)`;
  if(m.broken && m.broken[c]) return `${m.label}<br/><b>${sname}: ${pct(cell.v)}</b><br/>`+
    `<span style="color:#f0c674">⚠ ${m.broken[c]}</span>`;
  let t = `${m.label}<br/><b>${sname}: ${pct(cell.v)}</b>`;
  if(cell.src==='text_match'){
    t += `<br/><span style="color:#f0c674">⚠ text-match instrument (generated answer matched to option; not single-token letter logprob)</span>`;
    if(cell.range) t += `<br/>range ${pct(cell.range[0])}–${pct(cell.range[1])} (if unparsed answers were right)`;
    if(cell.unparsed!=null) t += `<br/>unparsed ${(cell.unparsed*100).toFixed(1)}%`;
  }
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
      // Missing is null, never 0: a gap in the polygon, not a false zero.
      data:[{value:axes.map(c=>{const x=cellOf(m,c);return x&&x.v!=null?x.v:null;}),name:m.label}]}))});
})();
/* shared grouped-bar builder: clean bars, no whiskers, no gridlines.
   Every bar's accuracy + exact 95% Wilson interval lives in the hover
   tooltip and the full data table. Drawing 48-84 whiskers on one axis
   (all overlapping at n=30) added 150+ line segments and a phantom
   '95% CI' legend entry without adding information, so they are gone.
   Text-match cells keep their model color at reduced opacity (no dashed
   outline); the tooltip + table badge carry the instrument marking. */
function barsWithCI(id, cats, models, rotate, fs){
  rotate = rotate==null?28:rotate; fs = fs==null?11:fs;
  const ch = mk(id);
  const series = models.map(m=>({
    name:m.label, type:'bar',
    itemStyle:{color:m.color, borderRadius:[3,3,0,0]},
    barGap:'20%', barCategoryGap:'30%',
    data:cats.map(c=>{
      const x=cellOf(m,c);
      if(!x||x.v==null) return '-';
      if(m.broken&&m.broken[c]) return {value:x.v,
        itemStyle:{color:'#39424e', borderRadius:[3,3,0,0]}};
      if(x.src==='text_match') return {value:x.v,
        itemStyle:{color:m.color, opacity:0.7, borderRadius:[3,3,0,0]}};
      return x.v;})
  }));
  ch.setOption({legend:{textStyle:{color:'#cfd6de'},top:0},
    tooltip:{trigger:'axis', axisPointer:{type:'shadow'},
      formatter:ps=>{
        if(!ps||!ps.length) return '';
        const c=cats[ps[0].dataIndex];
        let h=`<b>${D.short[c]||c}</b>`;
        ps.forEach(p=>{
          if(p.value==='-'||p.value==null) return;
          const m=models.find(x=>x.label===p.seriesName);
          if(m) h+='<br/>'+tipOf(m,c,cellOf(m,c));
        });
        return h;}},
    grid:{bottom:70,top:40,containLabel:true},
    xAxis:{type:'category',data:cats.map(c=>D.short[c]||c),
      axisLabel:{color:'#9aa4b2',rotate:rotate,fontSize:fs},
      axisLine:{lineStyle:{color:'#3a4756'}},axisTick:{show:false}},
    yAxis:{type:'value',min:0,max:1,
      axisLabel:{color:'#cfd6de',formatter:v=>Math.round(v*100)+'%'},
      splitLine:{show:false},axisLine:{show:false},axisTick:{show:false}},
    series});
}
/* wire whisker tooltip via data */
barsWithCI('c-custom', D.custom, D.models); barsWithCI('c-std', D.std, D.models);
/* 3. heatmap */
(function(){
  // The value dimension MUST be the third element: a fourth element made
  // ECharts colour every cell by that flag instead of by the score, which
  // painted the whole map in the minimum colour and highlighted exactly
  // one cell. The flags now travel in a side table.
  const cats=[...D.custom,...D.std], data=[], flags={}, srcs={};
  D.models.forEach((m,mi)=>cats.forEach((c,ci)=>{
    const x=m.cats[c];
    const t=(m.txt_cats&&m.txt_cats[c])||null;
    const cell=(x&&x.v!=null)?x:((t&&t.v!=null)?t:null);
    const isText=!((x&&x.v!=null))&&!!(t&&t.v!=null);
    const val = cell&&cell.v!=null?Math.round(cell.v*1000)/1000:null;
    if(isText){
      // A visibly different cell: the value was measured with the OTHER
      // multiple-choice mechanism, so it must not be read as a peer of the
      // letter-logprob cells around it.
      data.push({value:[ci,mi,val],itemStyle:{borderColor:'#f0c674',borderWidth:2}});
      srcs[mi+':'+ci]='text';
    } else {
      data.push([ci,mi,val]);
      if(m.broken&&m.broken[c]) flags[mi+':'+ci]='brk';
    }
  }));
  mk('c-heat').setOption({tooltip:{formatter:p=>{
      const mi=p.value[1],ci=p.value[0],m=D.models[mi],c=cats[ci];
      const x=m.cats[c];
      const t=(m.txt_cats&&m.txt_cats[c])||null;
      const cell=(x&&x.v!=null)?x:((t&&t.v!=null)?t:null);
      if(!cell||cell.v==null) return `${m.label}<br/>${D.short[c]}: \u2014 (not measured in this instrument)`;
      let s=`${m.label}<br/><b>${D.short[c]}: ${pct(cell.v)}</b>`;
      if(srcs[mi+':'+ci]==='text')
        s+=`<br/><span style="color:#f0c674">text-match instrument: this model's backend cannot score answer letters, so its MC was measured by reading the generated answer. NOT comparable with the letter-logprob cells of the other models.</span>`;
      if(cell.range) s+=`<br/>range ${pct(cell.range[0])}-${pct(cell.range[1])} (an unparsed answer could have been right)`;
      if(cell.unparsed!=null) s+=`<br/>unparsed ${(cell.unparsed*100).toFixed(1)}%`;
      if(flags[mi+':'+ci]==='brk') s+=`<br/><span style="color:#f0c674">\u26a0 ${m.broken[c]}</span>`;
      s+=`<br/>n=${cell.n}`;
      return s;}},
    grid:{height:'78%',top:10}, xAxis:{type:'category',data:cats.map(c=>D.short[c]),
      axisLabel:{color:'#9aa4b2',rotate:30,fontSize:10},axisTick:{show:false}},
    yAxis:{type:'category',data:D.models.map(m=>m.label),axisLabel:{color:'#cfd6de'}},
    visualMap:{min:0,max:1,calculable:true,orient:'vertical',right:0,top:'middle',
      textStyle:{color:'#8b96a3'},inRange:{color:['#2b3a4a','#3d6b8e','#5470C6','#91CC75','#FAC858']}},
    series:[{type:'heatmap',data,label:{show:true,fontSize:9,color:'#fff',
      textBorderColor:'#000',textBorderWidth:2,
      formatter:p=>{const k=p.value[1]+':'+p.value[0];
        if(p.value[2]==null) return '\u2014';
        const pc=Math.round(p.value[2]*100)+'%';
        if(flags[k]==='brk') return '\u26a0'+pc;
        if(srcs[k]==='text') return '\u25d0'+pc;
        return pc;}},
      itemStyle:{borderColor:'#11161d',borderWidth:1},
      emphasis:{itemStyle:{borderColor:'#e6e8eb',borderWidth:2}},
      encode:{x:0,y:1,value:2}}]});
})();
/* 4. efficiency */
(function(){
  mk('c-eff').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{formatter:p=>{
      const v=(p.value||(p.data&&p.data.value)||[]);
      const hasTxt = v[4];
      return `${p.seriesName} · ${v[3]}<br/>macro ${pct(v[1])}${hasTxt?' <span style="color:#f0c674">(⚠ MC via text-match)</span>':''} · ${v[0]} tok/s · ${v[2]} GB`;}},
    xAxis:{name:'decode tok/s (isolated pass)',...AX}, yAxis:{name:'macro-mean',max:1,...AX},
    series:['macro_custom','macro_std'].map((k,ki)=>({name:ki?'PRIMARY-STD':'PRIMARY-CUSTOM',
      type:'scatter',symbolSize:d=>18+Math.sqrt(d[2])*14,
      itemStyle:{color:ki?'#EE6666':'#5470C6',opacity:0.9},
      data:D.models.filter(m=>m[k]!=null&&m.mean_tps!=null&&m.tps_source==='isolated')
        .map(m=>{
          const hasTxt = m[k+'_txt'];
          return {value:[m.mean_tps,m[k],m.fileGB,m.label,hasTxt?1:0],
                  itemStyle:hasTxt?{borderColor:'#f0c674',borderWidth:2}: {}};})}))});
})();
/* 5. bilingual */
(function(){
  const cats=['english','persian','std_belebele_fa_s30'];
  mk('c-lang').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>{const m=D.models.find(x=>x.label===p.seriesName);
      return tipOf(m,cats[p.dataIndex],cellOf(m,cats[p.dataIndex]));}},
    xAxis:{type:'category',data:['English','Persian','Belebele-fa'],axisLabel:{color:'#cfd6de'},axisTick:{show:false}},
    yAxis:{type:'value',min:0,max:1,
      axisLabel:{color:'#cfd6de',formatter:v=>Math.round(v*100)+'%'},
      splitLine:{show:false},axisLine:{show:false},axisTick:{show:false}},
    series:D.models.map(m=>({name:m.label,type:'bar',itemStyle:{color:m.color,borderRadius:[3,3,0,0]},
      data:cats.map(c=>{const x=cellOf(m,c);
        if(!x||x.v==null) return '-';
        if(x.src==='text_match') return {value:x.v,itemStyle:{color:m.color,opacity:0.7,borderRadius:[3,3,0,0]}};
        return x.v;})}))});
})();
/* 6. MC vs chance */
(function(){
  const cats=[...D.custom.filter(c=>['science','electronics','persian','english'].includes(c)),...D.std.filter(c=>c!=='std_gsm8k_s30')];
  const series=D.models.map(m=>({name:m.label,type:'bar',
    itemStyle:{color:m.color,borderRadius:[3,3,0,0]},
    data:cats.map(c=>{const x=cellOf(m,c);
      if(!x||x.v==null) return '-';
      if(x.src==='text_match') return {value:x.v,itemStyle:{color:m.color,opacity:0.7,borderRadius:[3,3,0,0]}};
      return x.v;})}));
  series.push({name:'chance',type:'scatter',symbol:'x',symbolSize:12,
    itemStyle:{color:'#8b96a3'},data:cats.map(c=>{
      const xs=D.models.map(m=>cellOf(m,c)).filter(x=>x&&x.chance!=null).map(x=>x.chance);
      return xs.length?Math.max(...xs):'-';})});
  mk('c-chance').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>{const m=D.models.find(x=>x.label===p.seriesName);
      if(!m) return 'chance baseline';return tipOf(m,cats[p.dataIndex],cellOf(m,cats[p.dataIndex]));}},
    grid:{bottom:70,containLabel:true},xAxis:{type:'category',data:cats.map(c=>D.short[c]),
      axisLabel:{color:'#9aa4b2',rotate:26,fontSize:11},axisTick:{show:false}},
    yAxis:{type:'value',min:0,max:1,
      axisLabel:{color:'#cfd6de',formatter:v=>Math.round(v*100)+'%'},
      splitLine:{show:false},axisLine:{show:false},axisTick:{show:false}},series});
})();
/* 7. MMLU subjects */
(function(){
  const subs=[...new Set(D.models.flatMap(m=>Object.keys(m.mmlu)))].sort();
  mk('c-mmlu').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>{
      const m=D.models.find(x=>x.label===p.seriesName);
      const tag=m&&m.mmlu_txt?'<br/><span style="color:#f0c674">⚠ text-match instrument</span>':'';
      return `${p.seriesName}<br/>${p.name}: <b>${pct(p.value)}</b> (n=5)${tag}`;}},
    xAxis:{type:'category',data:subs.map(s=>s.split('/')[1]||s),
      axisLabel:{color:'#9aa4b2',rotate:22,fontSize:10},axisTick:{show:false}},
    yAxis:{type:'value',min:0,max:1,
      axisLabel:{color:'#cfd6de',formatter:v=>Math.round(v*100)+'%'},
      splitLine:{show:false},axisLine:{show:false},axisTick:{show:false}},
    series:D.models.map(m=>({name:m.label,type:'bar',itemStyle:{color:m.color,borderRadius:[3,3,0,0]},
      data:subs.map(s=>{const x=m.mmlu[s];
        if(!x) return '-';
        if(m.mmlu_txt) return {value:x.acc,itemStyle:{color:m.color,opacity:0.7,borderRadius:[3,3,0,0]}};
        return x.acc;})}))});
})();
/* 8. agentic */
function agenticChart(id, models){
  const rows=[['pass_hat_k_reliability','pass^k (tools required)'],
    ['per_trial_trace_rate','trace rate'],
    ['final_answer_accuracy_no_tools_required','answer-only']];
  mk(id).setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>{const m=models.find(x=>x.label===p.seriesName);
      return `${m.label}<br/>${rows[p.dataIndex][1]}: <b>${pct(p.value)}</b><br/>${m.agentic.mean_tool_calls} calls/trial · ${m.agentic.n_chains} chains`;}},
    xAxis:{type:'category',data:rows.map(r=>r[1]),axisLabel:{color:'#cfd6de'},axisTick:{show:false}},
    yAxis:{type:'value',min:0,max:1,
      axisLabel:{color:'#cfd6de',formatter:v=>Math.round(v*100)+'%'},
      splitLine:{show:false},axisLine:{show:false},axisTick:{show:false}},
    series:models.map(m=>({name:m.label,type:'bar',itemStyle:{color:m.color,borderRadius:[3,3,0,0]},
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
    grid:{bottom:110,top:40,containLabel:true},xAxis:{type:'category',data:cats.map(c=>(D.short[c]||c).replace('_txt','')),
      axisLabel:{color:'#9aa4b2',rotate:30,fontSize:10},axisLine:{lineStyle:{color:'#3a4756'}},axisTick:{show:false}},
    yAxis:{type:'value',min:0,max:1,
      axisLabel:{color:'#cfd6de',formatter:v=>Math.round(v*100)+'%'},
      splitLine:{show:false},axisLine:{show:false},axisTick:{show:false}},
    series:rows.map(r=>({name:r.label+(r.gaps&&r.gaps.length?' ⚠':''),
      type:'bar',itemStyle:{color:r.color,borderRadius:[3,3,0,0]},
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
  const txtTip='Includes multiple-choice suites measured via the text-match mechanism (generated answer matched to option, not single-token letter logprob).';
  h+=`<tr><td>★ PRIMARY-CUSTOM</td>${D.models.map(m=>`<td><b>${m.macro_custom!=null?pct(m.macro_custom):'—'}</b>${m.macro_custom_txt?' <span class="warn" title="'+txtTip+'">⚠ txt</span>':''}</td>`).join('')}</tr>`;
  h+=`<tr><td>★ PRIMARY-STD</td>${D.models.map(m=>`<td><b>${m.macro_std!=null?pct(m.macro_std):'—'}</b>${m.macro_std_txt?' <span class="warn" title="'+txtTip+'">⚠ txt</span>':''}</td>`).join('')}</tr>`;
  cats.forEach(c=>{
    h+=`<tr><td>${D.short[c]}</td>`+D.models.map(m=>{
      const x=cellOf(m,c);
      if(!x||x.v==null) return '<td>—</td>';
      let badge='';
      if(m.broken&&m.broken[c]) badge=` <span class="warn" title="${m.broken[c]}">⚠ n/c</span>`;
      else if(x.src==='text_match'){
        const u=x.unparsed!=null?`unparsed ${(x.unparsed*100).toFixed(1)}%`:'';
        const r=x.range?` · range ${pct(x.range[0])}–${pct(x.range[1])}`:'';
        badge=` <span class="warn" title="text-match instrument (generated answer matched to option) · ${u}${r}">⚠ txt</span>`;
      }
      let s=`<b>${pct(x.v)}</b>${badge}`;
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
    const x=cellOf(m,'std_mmlu_cs');
    if(!x||x.v==null) return;
    const badge=x.src==='text_match'?` <span class="warn" title="text-match instrument · unparsed ${(x.unparsed*100).toFixed(1)}% · range ${pct(x.range[0])}–${pct(x.range[1])}">⚠ txt</span>`:'';
    h+=`<tr><td><b style="color:${m.color}">●</b> ${m.label}${badge}</td><td><b>${pct(x.v)}</b></td><td>${pct(x.lo)}–${pct(x.hi)} · n=${x.n}</td></tr>`;
  });
  h+='<tr><td colspan="3" style="text-align:left;color:#8b96a3">Paired McNemar on shared items, all three pairs: TIE (closest p=0.053). The s30 gap did not replicate.</td></tr>';
  document.getElementById('tbl-cs').innerHTML=h;
})();
window.addEventListener('resize',()=>{CHARTS.forEach(c=>c.resize());});
</script></body></html>
"""


if __name__ == "__main__":
    main()
