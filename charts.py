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


def lighten(hexcolor, f=0.4):
    """Mix a hex colour towards white by factor f (0 = unchanged).

    Used for the cross-backend rows: nemotron measured through Ollama and
    through llama.cpp is the SAME model and must keep the same identity, but
    two series in the same chart cannot wear the same colour either - the
    reader could not tell which bar was which.
    """
    h = hexcolor.lstrip("#")
    if len(h) != 6:
        return hexcolor
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % (
        round(r + (255 - r) * f),
        round(g + (255 - g) * f),
        round(b + (255 - b) * f))


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


# Categories scored by the letter-logprob mechanism. If a run's mc_logprob
# probe FAILED for that model (the answer letters carry too little
# probability mass), the numbers in those cells were produced by a
# mechanism that was never qualified: they stay visible, but they are
# marked NOT COMPARABLE instead of being drawn as peer scores.
MC_MECH_CATS = ["science", "electronics", "persian", "english",
                "std_arc_challenge_s30", "std_arc_easy_s30",
                "std_hellaswag_s30", "std_winogrande_s30",
                "std_truthfulqa_mc_s30", "std_belebele_fa_s30",
                "std_mmlu_s30"]


def mc_probe_broken(d, summary, cats):
    """Mark the letter-logprob cells of a run whose MC probe failed.

    Only the multiple-choice categories are affected: code execution and
    free-text math never touch the answer-letter mechanism, so a failed
    letter probe says nothing about them.
    """
    probes = (d.get("provenance") or {}).get("backend_probes") or {}
    mc = probes.get("mc_logprob") or {}
    if mc.get("ok") is not False:
        return {}
    mass = mc.get("mean_mass")
    out = {}
    for c in [x for x in cats if x in MC_MECH_CATS]:
        if (summary.get(c) or {}).get("accuracy") is None:
            continue
        out[c] = ("NOT COMPARABLE: the answer-letter probe FAILED for this "
                  "model (probability mass on the option letters "
                  + (f"{mass:.3f}" if mass is not None else "below the floor")
                  + " against a 0.50 floor), so the letter-logprob numbers "
                    "below come from a mechanism that was never qualified. "
                    "The same model's multiple-choice scores that WERE "
                    "qualified are the text-match ones, shown in the "
                    "text-match section.")
    return out


def build_payload(title="llmbench screening — model comparison"):
    """Read every result file and return the report payload.

    Split out of main() so the tests can assert on the DATA the report
    publishes (no NaN, no missing-as-zero, instruments never merged)
    without writing a file.
    """
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
            "color": famcolor(sd.get("model", os.path.basename(sp)[:-10])),
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
        broken.update(mc_probe_broken(d, s, MC_MECH_CATS))
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
            "agentic_ci": (ag_sum.get("agentic", {}) or {}).get("ci95_wilson"),
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
            "label": lab, "color": lighten(famcolor(lab), 0.45),
            "n_chains": a.get("n_chains"), "n_trials": None,
            "backend": "ollama",
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
            "broken": mc_probe_broken(d, s, qcats),
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
        # The cross-backend row is the same weights measured through a
        # different runtime. Same family colour, lightened so the two rows
        # are distinguishable in one chart.
        col = famcolor(lab)
        # A run that never made a tool call cannot be scored like one that
        # did: its pass^k is an artifact of a broken pairing, not a model
        # property. The signature is explicit in the run itself.
        broken = None
        if a.get("mean_tool_calls") == 0 and \
                (a.get("halt_without_first_tool_call_rate") or 0) == 1.0:
            alt = ""
            oll = os.path.join(HERE, "results", "ollama",
                               f"{os.path.basename(f)[:-len('-agentic.json')]}"
                               "-agentic.json")
            if os.path.exists(oll):
                try:
                    av = (json.load(open(oll)).get("summary", {})
                          .get("agentic", {}).get("pass_hat_k_reliability"))
                    if av is not None:
                        alt = (f" The same weights score {av:.3f} through "
                               f"Ollama, where the tool-calling pairing works.")
                except Exception:
                    pass
            broken = ("NOT COMPARABLE: this backend cannot emit a parseable "
                      "tool call (no tool call was made in any chain), so the "
                      "chains scored zero by construction." + alt)
        deep.append({
            "label": lab,
            "color": lighten(col, 0.45) if "ollama" in f else col,
            "n_chains": a.get("n_chains"), "n_trials": a.get("n_trials"),
            "backend": "ollama" if "ollama" in f else "llama.cpp",
            "broken": broken,
            "agentic": {
                "pass_hat_k_reliability": a.get("pass_hat_k_reliability"),
                "per_trial_trace_rate": a.get("per_trial_trace_rate"),
                "final_answer_accuracy_no_tools_required":
                    a.get("final_answer_accuracy_no_tools_required"),
                "mean_tool_calls": a.get("mean_tool_calls"),
                "fabrication_rate": None,
                "n_chains": a.get("n_chains"),
                "neg_refusal_rate": a.get("neg_refusal_rate")},
            "agentic_ci": a.get("ci95_wilson"),
        })
    # screening agentic subsets, for depth contrast (chain counts shown)
    for m in models:
        if not any(r["label"].startswith(m["label"]) for r in deep):
            deep.append({
                "label": m["label"] + " (screen-subset)", "color": m["color"],
                "n_chains": m["agentic"].get("n_chains"),
                "n_trials": None,
                "backend": "llama.cpp",
                "broken": (m.get("broken") or {}).get("agentic"),
                "agentic": dict(m["agentic"]),
                "agentic_ci": m.get("agentic_ci"),
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
        # The row is a MODEL, not a file: the same label as in every other
        # section, so the reader does not have to learn a new naming scheme
        # for one chart. The instrument is marked in the tooltip/badge.
        base = os.path.basename(f)[:-5]
        lab = base
        for suf in ("-cs-txt", "-cs"):
            if lab.endswith(suf):
                lab = lab[:-len(suf)]
                break
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
        cs = [f for f in (glob.glob(os.path.join(RESULTS, "*-cs.json"))
                          + glob.glob(os.path.join(RESULTS, "*-cs-txt.json")))
              if famkey(os.path.basename(f)[:-5].replace("-cs-txt", "")
                        .replace("-cs", "")) == fam]
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
               "meta": {"title": title, "runs": len(models)}}
    return payload


def render(payload, out):
    html = PAGE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(html)
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPORTS, "screening.html"))
    ap.add_argument("--title", default="llmbench screening — model comparison")
    args = ap.parse_args()
    payload = build_payload(args.title)
    html = render(payload, args.out)
    print(f"wrote {args.out} ({len(html) // 1024} KB, "
          f"{len(payload['models'])} models)")


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
.nd{color:#8b96a3;font-size:11px;margin:2px 12px 8px;line-height:1.5}
.nd b{color:#b9c2cc}
.warn{background:#3a2a12;border:1px solid #7a5a1e;color:#f0c674;border-radius:9px;padding:0 7px;font-size:11px;cursor:help}
</style></head><body><div class="wrap">
<h1 id="t"></h1><div class="sub" id="chips"></div>
<div class="sub" style="margin:2px 0 10px"><b>How to read the marks:</b>
<span class="warn">⚠ incomplete</span> = the run behind that row is missing items (never a score) ·
<span class="warn">⚠ n/c</span> = the cell is real but <i>not comparable</i> (broken tool-calling backend, or a multiple-choice mechanism that was never qualified) ·
<span class="warn">⚠ txt</span> / lighter bar = multiple-choice cell measured by the <i>text-match</i> mechanism (generated answer matched to option, not single-token letter logprob) ·
<span class="warn">—</span> = not measured in this instrument (an empty slot, never a zero) ·
<span class="warn">0%</span> = a measured zero (a real score that would otherwise draw no bar) ·
<span class="warn">◐</span> = text-match cell in the heatmap ·
bars show accuracy; hover any bar for its exact 95% Wilson interval (also printed in the full data table).</div>
<h2 style="margin-top:20px">Coverage: who ran what</h2>
<p class="note" style="font-size:12px;color:#8b96a3">Record counts per instrument. Every section below compares only within one instrument — cross-section ranks are never computed.</p>
<div style="overflow-x:auto"><table id="cov"></table></div>
<div class="chart"><h2>Capability radar (headline metric per suite, higher is better)</h2>
<p class="note">One axis per suite. Lint = clean rate, NetDiag/DevOps = mean score, Agentic = pass^k reliability, rest = accuracy. One cell is a broken-backend artifact and is marked everywhere it appears: nemotron's agentic score comes from a llama.cpp build that cannot emit a parseable tool call, so its chains could not use tools (the same weights reach 0.895 through Ollama). It is drawn here as measured, but it is not a like-for-like comparison — see the ⚠ n/c marks below.</p><div id="c-radar"></div><div class="nd" id="nd-c-radar"></div></div>
<div class="chart"><h2>Custom suites — accuracy</h2>
<p class="note">Hover any bar for its exact 95% Wilson interval (never the normal approximation), n and chance baseline. Lighter bars marked ⚠ txt are the text-match instrument, not letter-logprob peers.</p><div id="c-custom" style="height:460px"></div><div class="nd" id="nd-c-custom"></div></div>
<div class="chart"><h2>Standard s30 suites — accuracy</h2>
<p class="note">Stratified 30-item official-split instances. Hover for the Wilson interval, n and chance baseline. MMLU-6, HellaSwag and WinoGrande are POSITION-DEBIASED: the upstream option order put 10-18 points of the answer in one letter, so the gold positions were permuted deterministically (same questions, same distractors) and those cells were re-measured. Every cell also publishes the score a constant picker would get, so the remaining confound is visible.</p><div id="c-std" style="height:440px"></div><div class="nd" id="nd-c-std"></div></div>
<div class="chart"><h2>Capability heatmap</h2>
<p class="note">Cells outlined in amber were measured with the OTHER multiple-choice mechanism (text match) - a different instrument, not a peer of the cells around it. Lighter = stronger (0 to 1). "\u2014" = the suite was not measured for that model <i>in this instrument</i>. A cell marked with an amber \u26a0 is real but not comparable (a broken tool-calling backend). k2-horizon's multiple-choice cells come from the TEXT-MATCH instrument - its llama.cpp build cannot score answer letters - so they are a different measurement and are never ranked against the letter-logprob cells of the other models. Hover any cell for its instrument, n and unparsed rate.</p><div id="c-heat" style="height:380px"></div><div class="nd" id="nd-c-heat"></div></div>
<div class="chart"><h2>Efficiency: capability vs decode speed</h2>
<p class="note">Bubble area = model file GB. Faster + higher is better; top-right wins. Speed here is the DEDICATED isolated pass: one model on an otherwise idle box, fresh server, two warm-up prompts discarded, then the median decode rate of 12 identical prompts - not a by-product of a long benchmark run. Server builds are named per model above: the qualifier instrument is entirely b11558, screening/CS-100/agentic are b11714; no comparison on this page spans two builds.</p><div id="c-eff" style="height:400px"></div><div class="nd" id="nd-c-eff"></div></div>
<div class="chart"><h2>English vs Persian (custom + Belebele-fa standard)</h2>
<p class="note">Gap = English accuracy − Persian accuracy, in percentage points.</p><div id="c-lang" style="height:380px"></div><div class="nd" id="nd-c-lang"></div></div>
<div class="chart"><h2>Multiple-choice vs chance baseline</h2>
<p class="note">Bars = accuracy; ✕ = chance (mean 1/k over items, not flat 25%). A bar at chance level knows nothing.</p><div id="c-chance" style="height:440px"></div><div class="nd" id="nd-c-chance"></div></div>
<div class="chart"><h2>MMLU-6 by subject (n=5 each)</h2>
<p class="note">Tiny-n: ±35pp intervals. Read rankings, not gaps.</p><div id="c-mmlu" style="height:380px"></div><div class="nd" id="nd-c-mmlu"></div></div>
<div class="chart"><h2>Agentic: reliability, trace, answer-only</h2>
<p class="note">pass^k requires tool use in order; answer-only ignores tools. Their gap = "knew but would not use tools". Cells use each model's deepest run (chain count shown); depths differ, so compare shapes, not just ranks.</p><div id="c-agent" style="height:360px"></div><div class="nd" id="nd-c-agent"></div></div>
<div class="chart"><h2>In-run decode rate (historical, under background load)</h2>
<p class="note">Kept for the record: these per-suite rates were taken as a by-product of long benchmark runs while a runaway desktop thumbnailer held half the CPU (REVIEW5), so they UNDERSTATE true speed. Any speed claim on this page uses the isolated pass instead.</p><div id="c-tps" style="height:340px"></div></div>
<div class="chart"><h2>Isolated throughput, every model</h2>
<p class="note">One model on an otherwise idle box, fresh server, two warm-up prompts discarded, then the median of 12 identical prompts. The spread (p25-p75, min-max) is in the tooltip; the foreign-CPU column is what the desktop was using at measurement time. These replace the in-run rates above, which were taken while a runaway thumbnailer held half the CPU.</p><div id="c-speed" style="height:300px"></div><div class="nd" id="nd-c-speed"></div></div>
<div class="chart"><h2>Text-match MC: the second mechanism (separate instrument)</h2>
<p class="note">A model that answers with the option's TEXT instead of the letter ("4" rather than "B") puts no probability mass on the letters, so the letter-logprob mechanism cannot score it - that model is unmeasured there, not wrong. This is the other mechanism: the generated answer is read and matched to the option. It is a DIFFERENT instrument: its categories are separate and are never ranked against the letter-logprob numbers above. Hover for the unparsed rate (answers that matched no option - counted as incorrect, never guessed).</p><div id="c-txt" style="height:380px"></div><div class="nd" id="nd-c-txt"></div></div>
<div class="chart"><h2>Qualifier: identical fast subsets, all models</h2>
<p class="note">Same items for everyone (deterministic subsets); ranks models for screening, never against full splits. All six qualifier runs share ONE server build (b11558, 2026-10-17) so the ranking is internally consistent; the screening instrument was measured on b11714. The two are never compared with each other. See <code>python3 qualify.py</code> for CUT/ADVANCE.</p><div id="c-qual" style="height:440px"></div><div class="nd" id="nd-c-qual"></div></div>
<div style="overflow-x:auto"><table id="tbl-q"></table></div>
<div class="chart"><h2>Agentic deep-dive: every model with 19-chain evidence</h2>
<p class="note">Deep files (19 chains) plus screening subsets (8 chains, chain count shown) for models without one. pass^k requires tool use; compare within equal chain counts.</p><div id="c-deep" style="height:380px"></div><div class="nd" id="nd-c-deep"></div></div>
<div class="chart"><h2>CS-100 decider: full official CS subject</h2>
<p class="note">100 items, ±10pp — the instrument that acquitted qwen's s30 gap (all pairs TIE). Hover for Wilson intervals.</p><div id="c-cs" style="height:360px"></div><div class="nd" id="nd-c-cs"></div></div>
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
function shortOf(c){
  if(D.short[c]) return D.short[c];
  const base = c.replace(/_txt$/,'');
  if(D.short[base]) return D.short[base];
  return c;
}
function tipOf(m, c, cell){
  cell = cell || cellOf(m, c);
  const sname = shortOf(c);
  if(!cell || cell.v==null)
    return `${m.label}<br/><b>${sname}</b>: — <span style="color:#8b96a3">(not measured in this instrument)</span>`;
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
/* Grouped-bar layout constants. The bar series and the "not measured"
   markers MUST use the same numbers, or the markers drift away from the
   slots they belong to. One source of truth for both. */
const CATGAP = 0.30, BARGAP = 0.20;
/* the note printed under a chart: which models have no data at all, and
   which cells came from the other multiple-choice mechanism */
function chartNote(shown, absent, get, cats){
  const bits=[];
  if(absent && absent.length)
    bits.push(`<b>no data in this chart:</b> ${absent.join(', ')} — never measured here, not a zero`);
  const txtModels=shown.filter(m=>{
    const cells = get ? cats.map(c=>get(m,c)) : Object.keys(m.cats||{}).map(c=>m.cats[c]);
    return cells.some(x=>x&&x.src==='text_match');
  }).map(m=>m.label);
  if(txtModels.length)
    bits.push(`<b>other instrument:</b> ${txtModels.join(', ')} — multiple-choice cells measured by text match (lighter bars), not by single-token letter logprob`);
  return bits.length? `<div class="nd">${bits.join(' · ')}</div>` : '';
}
/* shared grouped-bar builder.
   - a model with NO data in this chart is removed from the series and named
     in the note: a legend entry that never draws a bar reads as a bug;
   - a model with PARTIAL data keeps its slot, and the empty slots get an
     explicit "not measured" marker instead of a silent void;
   - a not-comparable cell keeps the model colour at reduced opacity and
     carries a ⚠ on the bar, so it can never be read as a peer score. */
function bars(id, cats, models, opts){
  opts = opts || {};
  const get = opts.get || ((m,c)=>cellOf(m,c));
  const rotate = opts.rotate==null?28:opts.rotate;
  const fs = opts.fs==null?11:opts.fs;
  const ch = mk(id);
  const rows = models.map(m=>({m, cells: cats.map(c=>get(m,c))}));
  const shown = rows.filter(r=>r.cells.some(x=>x && x.v!=null));
  const absent = rows.filter(r=>!r.cells.some(x=>x && x.v!=null)).map(r=>r.m.label);
  const M = shown.length;
  if(!M){
    ch.setOption({title:{show:true,left:'center',top:'middle',
      text:'no model has data in this instrument',
      textStyle:{color:'#8b96a3',fontSize:13}}});
    return;
  }
  const series = shown.map((r,mi)=>({
    name:r.m.label, type:'bar',
    itemStyle:{color:r.m.color, borderRadius:[3,3,0,0]},
    barGap:'20%', barCategoryGap:'30%',
    data:r.cells.map((x,ci)=>{
      if(!x || x.v==null) return '-';
      const cat = cats[ci];
      if(r.m.broken && r.m.broken[opts.catKey?opts.catKey(cat):cat])
        return {value:x.v, itemStyle:{color:r.m.color, opacity:0.55, borderRadius:[3,3,0,0]},
                label:{show:true, position:'top', formatter:'⚠', color:'#f0c674', fontSize:12}};
      if(x.src==='text_match')
        return {value:x.v, itemStyle:{color:r.m.color, opacity:0.62, borderRadius:[3,3,0,0]},
                label:(x.v===0)?{show:true,position:'top',formatter:'0%',color:'#8b96a3',fontSize:10}:undefined};
      // A real 0% draws NO bar, and an empty slot reads as "missing".
      // Print the zero so a measured zero can never be mistaken for a gap.
      if(x.v===0)
        return {value:0, label:{show:true, position:'top', formatter:'0%',
          color:'#8b96a3', fontSize:10}};
      return x.v;})
  }));
  /* explicit "not measured" markers: a hollow dashed stub with an em dash,
     drawn in the slot of the model that has no value there */
  const na=[];
  shown.forEach((r,mi)=>r.cells.forEach((x,ci)=>{
    if(!x || x.v==null) na.push([ci,mi]);}));
  if(na.length){
    series.push({type:'custom', name:'not measured', silent:true, z:6,
      renderItem:(p,api)=>{
        const ci=api.value(0), mi=api.value(1);
        const band=api.size([1,0])[0];
        const usable=band*(1-CATGAP);
        const w=usable/(M+BARGAP*(M-1));
        const start=api.coord([ci,0])[0]-usable/2;
        const cx=start+w*(mi*(1+BARGAP)+0.5);
        const y0=api.coord([0,0])[1];
        const h=Math.max(4, Math.abs(api.coord([0,0.05])[1]-y0));
        return {type:'group',children:[
          {type:'rect', shape:{x:cx-w/2+1, y:y0-h, width:Math.max(3,w-2), height:h},
           style:{fill:'rgba(120,132,148,0.12)', stroke:'#66768a', lineWidth:1, lineDash:[2,2]}},
          {type:'text', style:{x:cx, y:y0-h-2, text:'—', fill:'#8494a6',
           fontSize:11, textAlign:'center', textVerticalAlign:'bottom'}}]};
      }, data:na});
  }
  ch.setOption({legend:{textStyle:{color:'#cfd6de'},top:0,
      data: shown.map(r=>r.m.label).concat(opts.legendExtra||[])},
    tooltip:{trigger:'axis', axisPointer:{type:'shadow'},
      formatter:ps=>{
        if(!ps||!ps.length) return '';
        const c=cats[ps[0].dataIndex];
        let h=`<b>${shortOf(c)}</b>`;
        shown.forEach(r=>{ h+='<br/>'+tipOf(r.m,c,r.cells[ps[0].dataIndex]); });
        return h;}},
    grid:{bottom:opts.bottom==null?58:opts.bottom,top:34,containLabel:true},
    xAxis:{type:'category',data:cats.map(c=>opts.label?opts.label(c):shortOf(c)),
      axisLabel:{color:'#9aa4b2',rotate:rotate,fontSize:fs},
      axisLine:{lineStyle:{color:'#3a4756'}},axisTick:{show:false}},
    yAxis:{type:'value',min:0,max:1,
      axisLabel:{color:'#cfd6de',formatter:v=>Math.round(v*100)+'%'},
      splitLine:{show:false},axisLine:{show:false},axisTick:{show:false}},
    series: series.concat(opts.extraSeries||[])});
  const note=document.getElementById('nd-'+id);
  if(note) note.innerHTML=chartNote(shown.map(r=>r.m), absent, get, cats);
}
/* 1. radar */
(function(){
  const axes = ['code','rust','embedded_cpp','science','electronics','persian',
    'english','netdiag','devops','agentic','std_mmlu_s30','std_gsm8k_s30'];
  mk('c-radar').setOption({radar:{center:['50%','45%'], radius:'78%',
    indicator:axes.map(c=>({name:shortOf(c),max:1})),
    axisName:{color:'#9aa4b2',fontSize:11},splitArea:{areaStyle:{color:['#161d26','#141a22']}}},
    legend:{textStyle:{color:'#cfd6de'},bottom:0},
    // NOTE: for a radar series p.value is the WHOLE value array, not the
    // point's value - pct(p.value) printed "NaN%". The point's own value
    // is p.value[p.dimensionIndex].
    tooltip:{formatter:p=>{
      const v = Array.isArray(p.value) ? p.value[p.dimensionIndex] : p.value;
      const c = axes[p.dimensionIndex!=null?p.dimensionIndex:0];
      const m = D.models.find(x=>x.label===p.seriesName);
      if(v==null) return `${p.seriesName}<br/>${shortOf(c)}: — <span style="color:#8b96a3">(not measured in this instrument)</span>`;
      let t = `${p.seriesName}<br/><b>${shortOf(c)}: ${pct(v)}</b>`;
      const x = m ? (m.cats||{})[c] : null;
      if(x && x.lo!=null) t += `<br/>95% CI ${pct(x.lo)} – ${pct(x.hi)} · n=${x.n}`;
      else if(x && x.n!=null) t += `<br/>n=${x.n}`;
      if(x && x.src==='text_match')
        t += `<br/><span style="color:#f0c674">⚠ text-match instrument (not single-token letter logprob)</span>`;
      if(m && m.broken && m.broken[c])
        t += `<br/><span style="color:#f0c674">⚠ ${m.broken[c]}</span>`;
      return t;}},
    series:D.models.map(m=>({name:m.label,type:'radar',
      itemStyle:{color:m.color},lineStyle:{color:m.color},areaStyle:{color:m.color,opacity:0.12},
      // ONE instrument per chart: the letter-logprob cells only. Pulling in
      // the text-match cells here would silently rank two mechanisms
      // against each other - the one thing this report must never do.
      // A model without a letter-logprob score gets a GAP, not a zero.
      data:[{value:axes.map(c=>{const x=(m.cats||{})[c];return x&&x.v!=null?x.v:null;}),name:m.label}]}))});
  const nd=document.getElementById('nd-c-radar');
  if(nd) nd.innerHTML='One instrument only: the letter-logprob cells. A GAP means the model has no letter-logprob score for that suite (k2-horizon\u2019s backend cannot score answer letters, so its multiple-choice numbers live in the text-match section). A gap is never a zero.';
})();
/* the two headline bar charts */
bars('c-custom', D.custom, D.models);
bars('c-std', D.std, D.models);
/* 3. heatmap */
(function(){
  // The value dimension MUST be the third element: a fourth element made
  // ECharts colour every cell by that flag instead of by the score, which
  // painted the whole map in the minimum colour and highlighted exactly
  // one cell. The flags now travel in a side table.
  const NODATA='#1b222c';
  const cats=[...D.custom,...D.std], data=[], flags={}, srcs={}, miss={};
  D.models.forEach((m,mi)=>cats.forEach((c,ci)=>{
    const x=m.cats[c];
    const t=(m.txt_cats&&m.txt_cats[c])||null;
    const cell=(x&&x.v!=null)?x:((t&&t.v!=null)?t:null);
    const isText=!((x&&x.v!=null))&&!!(t&&t.v!=null);
    if(!cell || cell.v==null){
      // "not measured" is drawn as its own cell with its own fill and its
      // own label. A null value renders as an empty hole that looks like a
      // rendering failure, and it is NOT a zero.
      miss[mi+':'+ci]=true;
      data.push({value:[ci,mi,0], itemStyle:{color:NODATA, borderColor:'#11161d', borderWidth:1}});
      return;
    }
    const val = Math.round(cell.v*1000)/1000;
    const item = {value:[ci,mi,val]};
    if(isText){
      // A visibly different cell: the value was measured with the OTHER
      // multiple-choice mechanism, so it must not be read as a peer of the
      // letter-logprob cells around it.
      item.itemStyle = {borderColor:'#f0c674', borderWidth:2};
      srcs[mi+':'+ci]='text';
    } else if(m.broken && m.broken[c]) {
      item.itemStyle = {borderColor:'#f0c674', borderWidth:2};
      flags[mi+':'+ci]='brk';
    }
    data.push(item);
  }));
  mk('c-heat').setOption({tooltip:{formatter:p=>{
      const mi=p.value[1],ci=p.value[0],m=D.models[mi],c=cats[ci];
      const x=m.cats[c];
      const t=(m.txt_cats&&m.txt_cats[c])||null;
      const cell=(x&&x.v!=null)?x:((t&&t.v!=null)?t:null);
      if(!cell||cell.v==null) return `${m.label}<br/>${shortOf(c)}: — <span style="color:#8b96a3">(not measured in this instrument)</span>`;
      let s=`${m.label}<br/><b>${shortOf(c)}: ${pct(cell.v)}</b>`;
      if(srcs[mi+':'+ci]==='text')
        s+=`<br/><span style="color:#f0c674">text-match instrument: this model's backend cannot score answer letters, so its MC was measured by reading the generated answer. NOT comparable with the letter-logprob cells of the other models.</span>`;
      if(cell.range) s+=`<br/>range ${pct(cell.range[0])}-${pct(cell.range[1])} (an unparsed answer could have been right)`;
      if(cell.unparsed!=null) s+=`<br/>unparsed ${(cell.unparsed*100).toFixed(1)}%`;
      if(flags[mi+':'+ci]==='brk') s+=`<br/><span style="color:#f0c674">⚠ ${m.broken[c]}</span>`;
      s+=`<br/>n=${cell.n}`;
      return s;}},
    grid:{height:'76%',top:14,left:110,right:60}, xAxis:{type:'category',data:cats.map(c=>shortOf(c)),
      axisLabel:{color:'#9aa4b2',rotate:32,fontSize:10},axisTick:{show:false},
      splitArea:{show:false}},
    yAxis:{type:'category',data:D.models.map(m=>m.label),axisLabel:{color:'#cfd6de'}},
    visualMap:{min:0,max:1,calculable:true,orient:'vertical',right:4,top:'middle',
      itemWidth:12,itemHeight:110,text:['100%','0%'],
      textStyle:{color:'#8b96a3',fontSize:10},
      inRange:{color:['#2b3a4a','#3d6b8e','#5470C6','#91CC75','#FAC858']}},
    series:[{type:'heatmap',data,
      // The label colour follows the cell's own lightness: white on the
      // dark cells, near-black on the bright ones. One fixed colour made
      // the strongest cells the least readable.
      label:{show:true,fontSize:9,
        color:p=>{const v=p.value&&p.value[2];
          if(miss[p.value[1]+':'+p.value[0]]) return '#6b7a8d';
          return (v!=null && v>0.72)? '#0d1117':'#fff';},
        formatter:p=>{const k=p.value[1]+':'+p.value[0];
          if(miss[k]) return '—';
          const pc=Math.round(p.value[2]*100)+'%';
          if(flags[k]==='brk') return '⚠'+pc;
          if(srcs[k]==='text') return '◐'+pc;
          return pc;}},
      itemStyle:{borderColor:'#11161d',borderWidth:1},
      emphasis:{itemStyle:{borderColor:'#e6e8eb',borderWidth:2}},
      encode:{x:0,y:1,value:2}}]});
  const nd=document.getElementById('nd-c-heat');
  if(nd) nd.innerHTML = '<b>marks:</b> ◐ = measured with the text-match instrument · ⚠ = real but not comparable · — = not measured in this instrument (never a zero).';
})();
/* 4. efficiency: capability vs decode speed */
(function(){
  const pts=[];
  D.models.forEach(m=>{
    if(m.macro_custom==null || m.mean_tps==null || m.tps_source!=='isolated') return;
    pts.push({m, kind:'PRIMARY-CUSTOM', v:m.macro_custom, yoff:0});
    if(m.macro_std!=null) pts.push({m, kind:'PRIMARY-STD', v:m.macro_std, yoff:0});
  });
  if(!pts.length){mk('c-eff').setOption({title:{show:true,left:'center',top:'middle',
    text:'no isolated throughput yet', textStyle:{color:'#8b96a3',fontSize:13}}});return;}
  const ch=mk('c-eff');
  const mkSeries=(kind,color,symbol,sizeOf)=>({
    name:kind, type:'scatter', symbol:symbol,
    // ECharts hands a symbolSize callback the data ITEM, not always the
    // value array: with object items {value:[...]} the old d[2] read
    // undefined and every bubble got size NaN (drawn as nothing).
    symbolSize:d=>{const v=Array.isArray(d)?d:(d&&d.value)||[];return sizeOf(v[2]);},
    itemStyle:{color:color, opacity:0.92, borderColor:'#0f141b', borderWidth:1},
    label:{show:kind==='PRIMARY-CUSTOM', position:'right', distance:6,
      color:'#9aa4b2', fontSize:10,
      formatter:p=>p.value[3]},
    data:pts.filter(p=>p.kind===kind).map(p=>({
      value:[p.m.mean_tps, p.v, p.m.fileGB, p.m.label],
      itemStyle: (p.m[kind==='PRIMARY-CUSTOM'?'macro_custom_txt':'macro_std_txt'])
        ? {color:color, opacity:0.72, borderColor:'#f0c674', borderWidth:2, borderType:'solid'}
        : {color:color, opacity:0.92, borderColor:'#0f141b', borderWidth:1}}))});
  ch.setOption({
    legend:{textStyle:{color:'#cfd6de'},top:0},
    tooltip:{formatter:p=>{
      const v=(p.value||(p.data&&p.data.value)||[]);
      const m=D.models.find(x=>x.label===v[3]);
      const txt=m&&(m.macro_custom_txt||m.macro_std_txt);
      return `${v[3]}<br/><b>${p.seriesName}: ${pct(v[1])}</b>`+
        `${txt?' <span style="color:#f0c674">(⚠ MC via text-match)</span>':''}`+
        `<br/>${v[0]} tok/s (isolated pass) · ${v[2]} GB`;}},
    grid:{left:64,right:120,top:34,bottom:52},
    xAxis:{type:'value',name:'decode tok/s (isolated pass)',nameLocation:'middle',
      nameGap:30,nameTextStyle:{color:'#8b96a3',fontSize:11},
      min:v=>Math.floor(v.min-1), max:v=>Math.ceil(v.max+1),
      ...AX, splitLine:{show:false}, axisLine:{show:false}},
    yAxis:{type:'value',name:'macro-mean',nameLocation:'middle',nameGap:38,
      nameTextStyle:{color:'#8b96a3',fontSize:11},min:0,max:1,
      ...AX, splitLine:{show:false}, axisLine:{show:false},axisTick:{show:false},
      axisLabel:{color:'#cfd6de',formatter:v=>Math.round(v*100)+'%'}},
    series:[mkSeries('PRIMARY-CUSTOM','#5470C6','circle',gb=>14+Math.sqrt(gb)*8),
             mkSeries('PRIMARY-STD','#EE6666','diamond',gb=>11+Math.sqrt(gb)*6)]});
  const nd=document.getElementById('nd-c-eff');
  if(nd) nd.innerHTML='Bubble area = model file GB. The model name is printed next to the custom-macro bubble; the red diamond is the same model\'s standard macro.';
})();
/* 5. bilingual */
(function(){
  const cats=['english','persian','std_belebele_fa_s30'];
  bars('c-lang', cats, D.models, {rotate:0});
})();
/* 6. MC vs chance */
(function(){
  const cats=[...D.custom.filter(c=>['science','electronics','persian','english'].includes(c)),
              ...D.std.filter(c=>c!=='std_gsm8k_s30')];
  // The chance baseline is a per-category constant, so it is drawn on top
  // of the bars as a marker rather than as another model series.
  const chance=cats.map(c=>{
    const xs=D.models.map(m=>cellOf(m,c)).filter(x=>x&&x.chance!=null).map(x=>x.chance);
    return xs.length?Math.max(...xs):null;});
  bars('c-chance', cats, D.models, {rotate:26, fs:11,
    legendExtra:['chance baseline'],
    // A target ring, not a blob: a filled marker on top of a bar reads as
    // a data point. The legend names it; the note under the chart explains it.
    extraSeries:[{name:'chance baseline',type:'scatter',symbol:'circle',symbolSize:13,
      itemStyle:{color:'#11161d',borderColor:'#e6e8eb',borderWidth:2},
      data:chance}]});
  const nd=document.getElementById('nd-c-chance');
  if(nd) nd.innerHTML='✕ marks the chance baseline (mean 1/k over items, not a flat 25%). A bar at chance level knows nothing.';
})();
/* 7. MMLU subjects */
(function(){
  const subs=[...new Set(D.models.flatMap(m=>Object.keys(m.mmlu)))].sort();
  const get=(m,s)=> m.mmlu[s] ? {v:m.mmlu[s].acc, n:m.mmlu[s].n,
      src: m.mmlu_txt?'text_match':null} : null;
  bars('c-mmlu', subs, D.models, {rotate:22, fs:10, get:get,
    label:(c)=> (c.split('/')[1]||c).replace(/_/g,' ')});
  const nd=document.getElementById('nd-c-mmlu');
  if(nd) nd.innerHTML='n=5 per subject, so every cell carries roughly ±35pp. Lighter bars are the text-match instrument.';
})();
/* 8. agentic: reliability, trace, answer-only */
function agenticChart(id, models, opts){
  opts = opts || {};
  const rows=[['pass_hat_k_reliability','pass^k (tools required)'],
    ['per_trial_trace_rate','trace rate'],
    ['final_answer_accuracy_no_tools_required','answer-only']];
  const get=(m,k)=>{
    const v=m.agentic?m.agentic[k]:null;
    if(v==null) return null;
    const o={v:v, n:m.agentic.n_chains, ci:m.agentic_ci};
    // `broken` is a dict on the screening rows and a plain string on the
    // deep-dive rows: both mean the same thing.
    const br = (typeof m.broken === 'string') ? m.broken
             : (m.broken && m.broken.agentic);
    if(br) o.broken=br;
    return o;};
  const ch=mk(id);
  const mrows = models.map(m=>({m, cells: rows.map(r=>get(m,r[0]))}));
  const shown = mrows.filter(r=>r.cells.some(x=>x&&x.v!=null));
  const absent = mrows.filter(r=>!r.cells.some(x=>x&&x.v!=null)).map(r=>r.m.label);
  ch.setOption({legend:{textStyle:{color:'#cfd6de'},top:0,
      data: shown.map(r=>r.m.label)},
    tooltip:{trigger:'axis', axisPointer:{type:'shadow'},
      formatter:ps=>{
        if(!ps||!ps.length) return '';
        const c=rows[ps[0].dataIndex];
        let h=`<b>${c[1]}</b>`;
        shown.forEach(r=>{
          const x=r.cells[ps[0].dataIndex];
          h+='<br/>'+r.m.label+': '+(x&&x.v!=null
            ? `<b>${pct(x.v)}</b>`+(x.ci?` · 95% CI ${pct(x.ci[0])}–${pct(x.ci[1])}`:'')+
              ` · ${r.m.agentic.mean_tool_calls} calls/trial · ${r.m.agentic.n_chains} chains`+
              (x.broken?`<br/><span style="color:#f0c674">⚠ ${x.broken}</span>`:'')
            : '<span style="color:#8b96a3">— not measured</span>');
        });
        return h;}},
    grid:{bottom:44,top:34,containLabel:true},
    xAxis:{type:'category',data:rows.map(r=>r[1]),axisLabel:{color:'#cfd6de'},
      axisTick:{show:false},axisLine:{lineStyle:{color:'#3a4756'}}},
    yAxis:{type:'value',min:0,max:1,
      axisLabel:{color:'#cfd6de',formatter:v=>Math.round(v*100)+'%'},
      splitLine:{show:false},axisLine:{show:false},axisTick:{show:false}},
    series: shown.map(r=>({name:r.m.label,type:'bar',
      itemStyle:{color:r.m.color, borderRadius:[3,3,0,0]},
      barGap:'20%', barCategoryGap:'30%',
      data:r.cells.map(x=>{
        if(!x||x.v==null) return '-';
        if(x.broken) return {value:x.v,itemStyle:{color:r.m.color,opacity:0.55,borderRadius:[3,3,0,0]},
          label:{show:true,position:'top',formatter:'⚠',color:'#f0c674',fontSize:12}};
        return x.v;})}))});
  const nd=document.getElementById('nd-'+id);
  if(nd) nd.innerHTML = (absent.length?`<b>no data in this chart:</b> ${absent.join(', ')} — never measured here, not a zero · `:'')+
    'A ⚠ bar is real but NOT comparable: that backend could not emit a parseable tool call, so the chains could not use tools.';
}
agenticChart('c-agent', D.models);
/* 8b. isolated throughput table */
(function(){
  const rows=D.speeds||[];
  if(!rows.length){mk('c-speed').setOption({title:{show:true,left:'center',top:'middle',
    text:'no isolated measurements yet',textStyle:{color:'#8b96a3',fontSize:13}}});return;}
  const fam={};
  rows.forEach(r=>{ fam[r.label]=r.color||'#5470C6'; });
  mk('c-speed').setOption({
    tooltip:{trigger:'item',formatter:p=>{const r=rows[p.dataIndex];
      if(!r) return '';
      return `<b>${r.label}</b><br/>median ${r.tps} tok/s<br/>p25 ${r.p25} - p75 ${r.p75}<br/>min ${r.min} - max ${r.max}<br/>${r.n} prompts · foreign CPU ${r.foreign_cpu}%`;}},
    grid:{left:110,right:70,top:24,bottom:46,containLabel:false},
    xAxis:{type:'value',name:'decode tok/s (median)',nameLocation:'middle',nameGap:26,
      nameTextStyle:{color:'#8b96a3',fontSize:11},...AX,splitLine:{show:false},axisLine:{show:false}},
    yAxis:{type:'category',data:rows.map(r=>r.label),axisLabel:{color:'#cfd6de'},axisTick:{show:false}},
    series:[{type:'bar', itemStyle:{borderRadius:[0,3,3,0]},
      data:rows.map(r=>({value:r.tps, itemStyle:{color:fam[r.label]||'#5470C6'}})),
      label:{show:true,position:'right',color:'#e6e8eb',formatter:p=>p.value+' tok/s'}}]});
})();
/* 8c. text-match MC (a separate instrument) */
(function(){
  const rows=D.txt||[];
  if(!rows.length){mk('c-txt').setOption({title:{show:true,left:'center',top:'middle',
    text:'no text-match runs yet',textStyle:{color:'#8b96a3',fontSize:13}}});return;}
  const cats=D.txt_cats;
  bars('c-txt', cats, rows, {rotate:30, fs:10,
    get:(r,c)=> r.cats[c] ? r.cats[c] : null});
  const nd=document.getElementById('nd-c-txt');
  if(nd) nd.innerHTML='A DIFFERENT instrument from the letter-logprob charts above: the generated answer is read and matched to the option. Its numbers are never ranked against the letter-logprob ones.';
})();
/* 9. throughput */
(function(){
  const rows=[['code','Code tok/s'],['std_gsm8k','GSM8K tok/s']];
  mk('c-tps').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>{const r=rows[p.dataIndex];
      return `${p.seriesName}<br/>${r?r[1]:''}: <b>${p.value}</b>`;}},
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
  bars('c-qual', QC, D.q, {rotate:42, fs:10});
  let h='<tr><th>model</th><th>qual macro</th><th>code</th><th>agentic rel</th></tr>';
  D.q.forEach(m=>{
    const vals=QC.map(c=>m.cats[c]).filter(x=>x&&x.v!=null).map(x=>x.v);
    const macro=vals.length?pct(vals.reduce((a,b)=>a+b,0)/vals.length):'—';
    const code=m.cats.code, ag=m.cats.agentic;
    const ci=x=>x&&x.lo!=null?` [${pct(x.lo)},${pct(x.hi)}]`:'';
    const nb=Object.keys(m.broken||{});
    const badge = nb.length ? ` <span class="warn" title="${m.broken[nb[0]]}">⚠ n/c ×${nb.length}</span>` : '';
    h+=`<tr><td><b style="color:${m.color}">●</b> ${m.label}${m.gaps&&m.gaps.length?' <span class="warn" title="'+m.gaps.join('&#10;')+'">⚠</span>':''}${badge}</td><td><b>${macro}</b></td>`+
      `<td>${code&&code.v!=null?pct(code.v)+ci(code):'—'}</td><td>${ag&&ag.v!=null?pct(ag.v)+ci(ag):'—'}</td></tr>`;
  });
  document.getElementById('tbl-q').innerHTML=h;
})();
agenticChart('c-deep', D.deep);
/* CS-100: one metric, six models -> a horizontal bar chart reads better
   than six fat vertical bars under a single rotated label */
function csBars(id, rows){
  const data=rows.map(r=>({r, cell: cellOf(r,'std_mmlu_cs')})).filter(x=>x.cell&&x.cell.v!=null);
  if(!data.length){mk(id).setOption({title:{show:true,left:'center',top:'middle',
    text:'no CS-100 data',textStyle:{color:'#8b96a3',fontSize:13}}});return;}
  const ch=mk(id);
  ch.setOption({
    tooltip:{trigger:'item',formatter:p=>{
      const d=data[p.dataIndex];
      if(!d) return '';
      const c=d.cell;
      let s=`<b>${d.r.label}</b><br/><b>MMLU-CS (100): ${pct(c.v)}</b>`;
      if(c.src==='text_match')
        s+=`<br/><span style="color:#f0c674">⚠ text-match instrument (generated answer matched to option)</span>`+
           (c.range?`<br/>range ${pct(c.range[0])}–${pct(c.range[1])}`:'')+
           (c.unparsed!=null?`<br/>unparsed ${(c.unparsed*100).toFixed(1)}%`:'');
      if(c.lo!=null) s+=`<br/>95% CI ${pct(c.lo)} – ${pct(c.hi)} · n=${c.n}`;
      if(c.chance!=null) s+=`<br/>chance ${pct(c.chance)}`;
      return s;}},
    grid:{left:150,right:80,top:20,bottom:44,containLabel:false},
    xAxis:{type:'value',min:0,max:1,
      axisLabel:{color:'#cfd6de',formatter:v=>Math.round(v*100)+'%'},
      splitLine:{show:false},axisLine:{show:false},axisTick:{show:false}},
    yAxis:{type:'category',data:data.map(d=>d.r.label),
      axisLabel:{color:'#cfd6de'},axisTick:{show:false},
      inverse:true},
    series:[{type:'bar', itemStyle:{borderRadius:[0,3,3,0]},
      data:data.map(d=>({value:d.cell.v,
        itemStyle:{color:d.r.color, opacity: d.cell.src==='text_match'?0.62:1,
          borderColor: d.cell.src==='text_match'?'#f0c674':'transparent',
          borderWidth: d.cell.src==='text_match'?2:0}})),
      label:{show:true,position:'right',color:'#e6e8eb',
        formatter:p=>pct(p.value)+ (data[p.dataIndex].cell.src==='text_match'?' ◐':'')}}]});
  const nd=document.getElementById('nd-'+id);
  if(nd) nd.innerHTML='◐ = measured with the text-match instrument (a different mechanism, not comparable with the letter-logprob rows).';
}
/* cs100 section */
(function(){
  csBars('c-cs', D.cs);
  let h='<tr><th>model</th><th>CS-100</th><th>95% CI</th></tr>';
  D.cs.forEach(m=>{
    const x=cellOf(m,'std_mmlu_cs');
    if(!x||x.v==null) return;
    const badge=x.src==='text_match'?` <span class="warn" title="text-match instrument · unparsed ${(x.unparsed*100).toFixed(1)}% · range ${pct(x.range[0])}–${pct(x.range[1])}">⚠ txt</span>`:'';
    h+=`<tr><td><b style="color:${m.color}">●</b> ${m.label}${badge}</td><td><b>${pct(x.v)}</b></td><td>${pct(x.lo)}–${pct(x.hi)} · n=${x.n}</td></tr>`;
  });
  h+='<tr><td colspan="3" style="text-align:left;color:#8b96a3">k2-horizon is measured with the TEXT-MATCH mechanism (⚠ txt): its backend cannot score answer letters, so this cell is a different measurement and is not ranked against the letter-logprob rows.</td></tr>';
  h+='<tr><td colspan="3" style="text-align:left;color:#8b96a3">Paired McNemar on shared items, all three pairs: TIE (closest p=0.053). The s30 gap did not replicate.</td></tr>';
  document.getElementById('tbl-cs').innerHTML=h;
})();
window.addEventListener('resize',()=>{CHARTS.forEach(c=>c.resize());});
</script></body></html>
"""


if __name__ == "__main__":
    main()
