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
STD_CATS = ["std_arc_challenge_s30", "std_arc_easy_s30", "std_hellaswag_s30",
            "std_winogrande_s30", "std_truthfulqa_mc_s30", "std_belebele_fa_s30",
            "std_mmlu_s30", "std_gsm8k_s30", "std_mmlu_cs"]
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
            "n": n, "chance": s.get("chance_baseline")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPORTS, "screening.html"))
    ap.add_argument("--title", default="llmbench screening — model comparison")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(RESULTS, "*.json"))
                   if not f.endswith("-q.json") and not f.endswith("-cs.json")
                   and not f.endswith("-agentic.json")
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

    def build_row(f, label=None, agentic_pref=True):
        d = json.load(open(f))
        s = d.get("summary", {})
        ag_sum = s
        if agentic_pref:
            ag_file = os.path.join(RESULTS, os.path.basename(f)[:-5] + "-agentic.json")
            if os.path.exists(ag_file):
                try:
                    ag_sum = json.load(open(ag_file)).get("summary", s)
                except Exception:
                    pass
        cats = {c: cell(ag_sum if c == "agentic" else s, c)
                for c in CUSTOM_CATS + STD_CATS}
        vals_c = [cats[c]["v"] for c in CUSTOM_CATS
                  if cats[c] and isinstance(cats[c]["v"], (int, float))]
        vals_s = [cats[c]["v"] for c in STD_CATS
                  if cats[c] and isinstance(cats[c]["v"], (int, float))]
        dg = d.get("provenance", {}).get("model_digest", {}) or {}
        lab = label or os.path.basename(f)[:-5]
        return {
            "label": lab,
            "color": famcolor(lab),
            "fileGB": round((dg.get("size_bytes") or 0) / 1e9, 2),
            "digest": str(dg.get("digest", ""))[-12:],
            "macro_custom": round(sum(vals_c) / len(vals_c), 4) if vals_c else None,
            "macro_std": round(sum(vals_s) / len(vals_s), 4) if vals_s else None,
            "mean_tps": (d.get("backend_totals") or {}).get("mean_gen_tps"),
            "elapsed_h": round(d.get("elapsed_s", 0) / 3600, 2),
            "n_recs": len(d.get("records", [])),
            "cats": cats,
            "mmlu": (s.get("std_mmlu_s30") or {}).get("by_subject", {}),
            "agentic": {k: ag_sum.get("agentic", {}).get(k) for k in
                        ("pass_hat_k_reliability", "per_trial_trace_rate",
                         "final_answer_accuracy_no_tools_required",
                         "mean_tool_calls", "fabrication_rate", "n_chains")},
            "tps": {c: (s.get(c) or {}).get("gen_tps_mean")
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
        qualifier.append({
            "label": lab, "color": famcolor(lab),
            "cats": {c: cell(s, c) for c in qcats},
            "n_recs": len(d.get("records", [])),
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
        if any(row["cells"]):
            cov_cells.append(row)
    payload = {"models": models,
               "custom": CUSTOM_CATS, "std": STD_CATS, "short": SHORT,
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
</style></head><body><div class="wrap">
<h1 id="t"></h1><div class="sub" id="chips"></div>
<h2 style="margin-top:20px">Coverage: who ran what</h2>
<p class="note" style="font-size:12px;color:#8b96a3">Record counts per instrument. Every section below compares only within one instrument — cross-section ranks are never computed.</p>
<div style="overflow-x:auto"><table id="cov"></table></div>
<div class="chart"><h2>Capability radar (headline metric per suite, higher is better)</h2>
<p class="note">One axis per suite. Lint = clean rate, NetDiag/DevOps = mean score, Agentic = pass^k reliability, rest = accuracy.</p><div id="c-radar"></div></div>
<div class="chart"><h2>Custom suites — accuracy with 95% Wilson intervals</h2>
<p class="note">Whiskers are exact 95% Wilson intervals (never the normal approximation). Hover for n and chance baseline.</p><div id="c-custom" style="height:460px"></div></div>
<div class="chart"><h2>Standard s30 suites — accuracy with 95% Wilson intervals</h2>
<p class="note">Stratified 30-item official-split instances. Same-interval semantics as above.</p><div id="c-std" style="height:440px"></div></div>
<div class="chart"><h2>Capability heatmap</h2>
<p class="note">Darker = stronger. Missing = suite absent for that model.</p><div id="c-heat" style="height:380px"></div></div>
<div class="chart"><h2>Efficiency: capability vs decode speed</h2>
<p class="note">Bubble area = model file GB. Faster + higher is better; top-right wins.</p><div id="c-eff" style="height:400px"></div></div>
<div class="chart"><h2>English vs Persian (custom + Belebele-fa standard)</h2>
<p class="note">Gap = English accuracy − Persian accuracy, in percentage points.</p><div id="c-lang" style="height:380px"></div></div>
<div class="chart"><h2>Multiple-choice vs chance baseline</h2>
<p class="note">Bars = accuracy; ✕ = chance (mean 1/k over items, not flat 25%). A bar at chance level knows nothing.</p><div id="c-chance" style="height:440px"></div></div>
<div class="chart"><h2>MMLU-6 by subject (n=5 each)</h2>
<p class="note">Tiny-n: ±35pp intervals. Read rankings, not gaps.</p><div id="c-mmlu" style="height:380px"></div></div>
<div class="chart"><h2>Agentic: reliability, trace, answer-only</h2>
<p class="note">pass^k requires tool use in order; answer-only ignores tools. Their gap = "knew but would not use tools". Cells use each model's deepest run (chain count shown); depths differ, so compare shapes, not just ranks.</p><div id="c-agent" style="height:360px"></div></div>
<div class="chart"><h2>Throughput: mean decode tok/s</h2>
<p class="note">Server-side counters on identical prompts per model; wall-clock cross-checked. MC suites excluded (1 token each). Cross-era caution: runs began on different server builds — re-running all finalists on one build before publishing.</p><div id="c-tps" style="height:340px"></div></div>
<div class="chart"><h2>Qualifier: identical fast subsets, all models</h2>
<p class="note">Same items for everyone (deterministic subsets); ranks models for screening, never against full splits. See <code>python3 qualify.py</code> for CUT/ADVANCE.</p><div id="c-qual" style="height:440px"></div></div>
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
  D.models.map(m=>`<span class="chip"><b style="color:${m.color}">●</b> ${m.label} · ${m.fileGB} GB · <code>…${m.digest}</code> · ${m.elapsed_h} h</span>`).join('');
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
function tipOf(m, c, cell){
  if(!cell || cell.v==null) return `${m.label}<br/>${D.short[c]}: —`;
  let t = `${m.label}<br/><b>${D.short[c]}: ${cell.v}</b>`;
  if(cell.lo!=null) t += `<br/>95% CI [${cell.lo}, ${cell.hi}] · n=${cell.n}`;
  if(cell.chance!=null) t += `<br/>chance ${cell.chance}`;
  return t;
}
/* 1. radar */
(function(){
  const axes = ['code','rust','embedded_cpp','science','electronics','persian',
    'english','netdiag','devops','agentic','std_mmlu_s30','std_gsm8k_s30'];
  mk('c-radar').setOption({radar:{indicator:axes.map(c=>({name:D.short[c],max:1})),
    axisName:{color:'#9aa4b2',fontSize:11},splitArea:{areaStyle:{color:['#161d26','#141a22']}}},
    legend:{textStyle:{color:'#cfd6de'},bottom:0},
    tooltip:{}, series:D.models.map(m=>({name:m.label,type:'radar',
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
      data:cats.map(c=>{const x=m.cats[c];return x&&x.v!=null?x.v:'-';}),
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
    data.push([ci,mi,x&&x.v!=null?Math.round(x.v*1000)/1000:null]);
  }));
  mk('c-heat').setOption({tooltip:{formatter:p=>{
      const m=D.models[p.value[1]],c=cats[p.value[0]];return tipOf(m,c,m.cats[c]);}},
    grid:{height:'78%',top:10}, xAxis:{type:'category',data:cats.map(c=>D.short[c]),
      axisLabel:{color:'#9aa4b2',rotate:30,fontSize:10},splitArea:{show:true}},
    yAxis:{type:'category',data:D.models.map(m=>m.label),axisLabel:{color:'#cfd6de'}},
    visualMap:{min:0,max:1,calculable:true,orient:'vertical',right:0,top:'middle',
      textStyle:{color:'#9aa4b2'},inRange:{color:['#2b3a4a','#3d6b8e','#5470C6','#91CC75','#FAC858']}},
    series:[{type:'heatmap',data,label:{show:true,fontSize:9,color:'#fff',
      formatter:p=>p.value[2]==null?'—':p.value[2]}}]});
})();
/* 4. efficiency */
(function(){
  mk('c-eff').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{formatter:p=>`${p.seriesName} · ${p.data[3]}<br/>macro ${p.data[1]} · ${p.data[0]} tok/s · ${p.data[2]} GB`},
    xAxis:{name:'mean tok/s',...AX}, yAxis:{name:'macro-mean',max:1,...AX},
    series:['macro_custom','macro_std'].map((k,ki)=>({name:ki?'PRIMARY-STD':'PRIMARY-CUSTOM',
      type:'scatter',symbolSize:d=>18+Math.sqrt(d[2])*14,
      itemStyle:{color:ki?'#EE6666':'#5470C6',opacity:0.9},
      data:D.models.filter(m=>m[k]!=null&&m.mean_tps!=null)
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
    tooltip:{trigger:'item',formatter:p=>`${p.seriesName}<br/>${p.name}: <b>${p.value}</b> (n=5)`},
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
      return `${m.label}<br/>${rows[p.dataIndex][1]}: <b>${p.value}</b><br/>calls ${m.agentic.mean_tool_calls} · chains ${m.agentic.n_chains}`;}},
    xAxis:{type:'category',data:rows.map(r=>r[1]),axisLabel:{color:'#cfd6de'}},yAxis:{max:1,...AX},
    series:models.map(m=>({name:m.label,type:'bar',itemStyle:{color:m.color},
      data:rows.map(r=>m.agentic[r[0]]!=null?m.agentic[r[0]]:'-')}))});
}
agenticChart('c-agent', D.models);
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
  h+=`<tr><td>★ PRIMARY-CUSTOM</td>${macC.map(v=>`<td>${v!=null?v:'—'}</td>`).join('')}</tr>`;
  h+=`<tr><td>★ PRIMARY-STD</td>${macS.map(v=>`<td>${v!=null?v:'—'}</td>`).join('')}</tr>`;
  cats.forEach(c=>{
    h+=`<tr><td>${D.short[c]}</td>`+D.models.map(m=>{
      const x=m.cats[c];
      if(!x||x.v==null) return '<td>—</td>';
      let s=`${x.v}`;
      if(x.lo!=null) s+=`<br/><span style="color:#8b96a3">[${x.lo}, ${x.hi}] n=${x.n}</span>`;
      return `<td>${s}</td>`;}).join('')+'</tr>';
  });
  document.getElementById('tbl').innerHTML=h;
})();
/* coverage matrix */
(function(){
  let h='<tr><th>model</th><th>screening recs</th><th>qualifier recs</th><th>agentic-deep recs</th><th>CS-100 recs</th></tr>';
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
    h+=`<tr><td><b style="color:${m.color}">●</b> ${m.label}</td><td>${macro}</td>`+
      `<td>${code&&code.v!=null?code.v+ci(code):'—'}</td><td>${ag&&ag.v!=null?ag.v+ci(ag):'—'}</td></tr>`;
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
    h+=`<tr><td><b style="color:${m.color}">●</b> ${m.label}</td><td>${x.v}</td><td>[${x.lo}, ${x.hi}] n=${x.n}</td></tr>`;
  });
  h+='<tr><td colspan="3" style="text-align:left;color:#8b96a3">Paired McNemar on shared items, all three pairs: TIE (closest p=0.053). The s30 gap did not replicate.</td></tr>';
  document.getElementById('tbl-cs').innerHTML=h;
})();
window.addEventListener('resize',()=>{CHARTS.forEach(c=>c.resize());});
</script></body></html>
"""


if __name__ == "__main__":
    main()
