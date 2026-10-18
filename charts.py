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
                   if not f.endswith("-q.json"))
    if not files:
        raise SystemExit(f"no screening runs in {RESULTS}")
    models = []
    for i, f in enumerate(files):
        d = json.load(open(f))
        s = d.get("summary", {})
        cats = {c: cell(s, c) for c in CUSTOM_CATS + STD_CATS}
        vals_c = [cats[c]["v"] for c in CUSTOM_CATS
                  if cats[c] and isinstance(cats[c]["v"], (int, float))]
        vals_s = [cats[c]["v"] for c in STD_CATS
                  if cats[c] and isinstance(cats[c]["v"], (int, float))]
        dg = d.get("provenance", {}).get("model_digest", {}) or {}
        models.append({
            "label": os.path.basename(f)[:-5],
            "color": COLORS[i % len(COLORS)],
            "fileGB": round((dg.get("size_bytes") or 0) / 1e9, 2),
            "digest": str(dg.get("digest", ""))[-12:],
            "macro_custom": round(sum(vals_c) / len(vals_c), 4) if vals_c else None,
            "macro_std": round(sum(vals_s) / len(vals_s), 4) if vals_s else None,
            "mean_tps": (d.get("backend_totals") or {}).get("mean_gen_tps"),
            "elapsed_h": round(d.get("elapsed_s", 0) / 3600, 2),
            "cats": cats,
            "mmlu": (s.get("std_mmlu_s30") or {}).get("by_subject", {}),
            "agentic": {k: s.get("agentic", {}).get(k) for k in
                        ("pass_hat_k_reliability", "per_trial_trace_rate",
                         "final_answer_accuracy_no_tools_required",
                         "mean_tool_calls", "fabrication_rate", "n_chains")},
            "tps": {c: (s.get(c) or {}).get("gen_tps_mean")
                    for c in ("code", "std_gsm8k")},
        })
    payload = {"models": models,
               "custom": CUSTOM_CATS, "std": STD_CATS, "short": SHORT,
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
<p class="note">pass^k requires tool use in order; answer-only ignores tools. Their gap = "knew but would not use tools".</p><div id="c-agent" style="height:360px"></div></div>
<div class="chart"><h2>Throughput: mean decode tok/s</h2>
<p class="note">Server-side counters on identical prompts per model; wall-clock cross-checked. MC suites excluded (1 token each). Cross-era caution: runs began on different server builds — re-running all finalists on one build before publishing.</p><div id="c-tps" style="height:340px"></div></div>
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
function barsWithCI(id, cats, title){
  const ch = mk(id), M = D.models.length;
  const series = [];
  D.models.forEach((m,mi)=>{
    series.push({name:m.label,type:'bar',itemStyle:{color:m.color},
      data:cats.map(c=>{const x=m.cats[c];return x&&x.v!=null?x.v:'-';}),
      barGap:'18%', barCategoryGap:'32%'});
  });
  // whisker data: [catIndex, modelIndex, lo, hi]
  const wd = [];
  D.models.forEach((m,mi)=>cats.forEach((c,ci)=>{
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
      const m=D.models.find(x=>x.label===p.seriesName);
      return tipOf(m,cats[p.dataIndex],m.cats[cats[p.dataIndex]]);}},
    grid:{bottom:70,top:40}, xAxis:{type:'category',data:cats.map(c=>D.short[c]),
      axisLabel:{color:'#9aa4b2',rotate:28,fontSize:11},axisLine:{lineStyle:{color:'#3a4756'}}},
    yAxis:{type:'value',max:1,...AX}, series});
}
/* wire whisker tooltip via data */
barsWithCI('c-custom', D.custom); barsWithCI('c-std', D.std);
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
(function(){
  const rows=[['pass_hat_k_reliability','pass^k (tools required)'],
    ['per_trial_trace_rate','trace rate'],
    ['final_answer_accuracy_no_tools_required','answer-only']];
  mk('c-agent').setOption({legend:{textStyle:{color:'#cfd6de'}},
    tooltip:{trigger:'item',formatter:p=>{const m=D.models.find(x=>x.label===p.seriesName);
      return `${m.label}<br/>${rows[p.dataIndex][1]}: <b>${p.value}</b><br/>calls ${m.agentic.mean_tool_calls} · chains ${m.agentic.n_chains}`;}},
    xAxis:{type:'category',data:rows.map(r=>r[1]),axisLabel:{color:'#cfd6de'}},yAxis:{max:1,...AX},
    series:D.models.map(m=>({name:m.label,type:'bar',itemStyle:{color:m.color},
      data:rows.map(r=>m.agentic[r[0]]!=null?m.agentic[r[0]]:'-')}))});
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
window.addEventListener('resize',()=>{CHARTS.forEach(c=>c.resize());});
</script></body></html>
"""


if __name__ == "__main__":
    main()
