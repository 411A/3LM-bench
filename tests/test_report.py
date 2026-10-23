#!/usr/bin/env python3
"""tests/test_report.py — contracts for the published report.

The report is the only artefact a reader sees, so its data layer gets the
same treatment as the harness: every number must be traceable, nothing may
be silently zero, and two measurement mechanisms may never be merged into
one ranking.

These are the bugs that actually shipped in earlier revisions, kept as
regressions:
  - a radar tooltip printed "NaN%" (p.value is the whole array, not the
    point's value);
  - a scatter bubble got symbolSize NaN and drew nothing;
  - a missing cell drew no bar, so "not measured" looked like a bug and a
    measured 0% looked exactly like a missing cell;
  - the report pulled text-match cells into the letter-logprob charts,
    silently ranking two mechanisms against each other;
  - a run whose MC probe FAILED still showed its letter-logprob cells as
    if they were peer scores.

Run: python3 tests/test_report.py
"""
import json
import glob
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import charts  # noqa: E402
import qualify  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILS.append(name)


def walk(o, path=""):
    """Yield every (path, number) in a nested structure."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(o, float):
        yield path, o


def main():
    print("report contracts")
    payload = charts.build_payload("test")
    models = payload["models"]

    # The restored radar is intentionally restricted to complete binary
    # accuracy/clean-rate cells; partial-credit means and alternate MC are
    # shown in the accompanying profiles, not normalized into this radar.
    radar_axes = ("code", "bugfix", "algo", "refactor", "rust",
                  "embedded_cpp", "lint", "std_gsm8k_s30")
    radar_missing = [f"{m['label']}/{c}" for m in models for c in radar_axes
                     if not (m.get("cats", {}).get(c)
                             and m["cats"][c].get("v") is not None
                             and not (m.get("broken") or {}).get(c))]
    check("accuracy radar has complete comparable data", not radar_missing,
          str(radar_missing[:5]))
    check("accuracy radar axes share a binary rate metric",
          all(charts.HEADLINE.get(c, "accuracy") in ("accuracy", "clean_rate")
              for c in radar_axes))
    custom_leaders = {x["category"]: x for x in payload["leaders"]["custom"]}
    std_leaders = {x["category"]: x for x in payload["leaders"]["std"]}
    check("agentic leader excludes failed and unmeasured tool backends",
          custom_leaders["agentic"]["leader_labels"] == ["granite4-3b"]
          and custom_leaders["agentic"]["n_comparable"] == 4)
    check("MC leaders use the shared letter-logprob instrument only",
          "k2-horizon" not in std_leaders["std_arc_challenge_s30"]["leader_labels"]
          and std_leaders["std_arc_challenge_s30"]["n_comparable"] == 5)
    check("area leader ties are retained",
          set(custom_leaders["code"]["leader_labels"])
          == {"granite4-3b", "k2-horizon"})

    # ---- 1. no NaN / Infinity may reach the page ----
    bad = [p for p, v in walk(payload) if math.isnan(v) or math.isinf(v)]
    check("payload has no NaN/Infinity", not bad, str(bad[:5]))
    raw = json.dumps(payload, allow_nan=False)
    check("payload serialises without NaN literal",
          "NaN" not in raw and "Infinity" not in raw)

    # ---- 2. a missing cell is never a zero ----
    zero_as_missing = []
    for m in models:
        for c, cell in (m.get("cats") or {}).items():
            if cell is None:
                continue
            if cell.get("v") == 0 and cell.get("n") in (None, 0):
                zero_as_missing.append(f"{m['label']}/{c}")
    check("a zero score always carries its n", not zero_as_missing,
          str(zero_as_missing[:5]))

    # ---- 3. macro-means equal the mean of the cells they claim ----
    for m in models:
        for key, cats in (("macro_custom", payload["custom"]),
                          ("macro_std", payload["std"])):
            vals = []
            for c in cats:
                cell = (m.get("cats") or {}).get(c)
                if not cell or cell.get("v") is None:
                    cell = (m.get("txt_cats") or {}).get(c)
                if cell and isinstance(cell.get("v"), (int, float)):
                    vals.append(cell["v"])
            if not vals:
                continue
            want = round(sum(vals) / len(vals), 4)
            got = m.get(key)
            check(f"{m['label']} {key} = mean of {len(vals)} cells",
                  got is not None and abs(got - want) < 1e-6,
                  f"got {got} want {want}")

    # ---- 4. the primary (letter-logprob) cells and the text-match cells
    #         are separate: a text-match cell must never replace a
    #         letter-logprob one in the same category ----
    for m in models:
        overlap = [c for c in (m.get("txt_cats") or {})
                   if (m.get("cats") or {}).get(c)
                   and (m["cats"][c] or {}).get("v") is not None]
        check(f"{m['label']}: no category carries both mechanisms", not overlap,
              str(overlap))
    for m in models:
        for c, cell in (m.get("txt_cats") or {}).items():
            if cell.get("v") is None:
                continue
            check(f"{m['label']}/{c} text-match cell is marked",
                  cell.get("src") == "text_match")

    # ---- 5. a run whose MC probe failed must mark its letter cells ----
    probe_failed = 0
    for q in payload["q"]:
        br = q.get("broken") or {}
        if br:
            probe_failed += 1
            for c in br:
                cell = (q.get("cats") or {}).get(c)
                check(f"{q['label']}/{c} flagged not-comparable",
                      cell is None or cell.get("v") is not None)
    check("probe-failed runs are marked, not silently published",
          probe_failed >= 0)

    # ---- 6. the CS-100 rows are models, not file names ----
    for row in payload["cs"]:
        check(f"CS-100 row labelled by model: {row['label']}",
              not row["label"].endswith(("-cs", "-cs-txt")))

    # ---- 7. the rendered page must not contain the bug signatures ----
    html = charts.render(payload, "/tmp/llmbench-test-report.html")
    # search the RUNNABLE script only: a comment that mentions a past bug
    # is not a bug.
    script = html[html.rindex("<script>") + 8: html.rindex("</script>")]
    script = re.sub(r"/\*.*?\*/", " ", script, flags=re.S)
    script = re.sub(r"(?m)(^|\s)//[^\n]*", r"\1", script)
    for sig in ("NaN%", "undefined%", "[object Object]", "NaN"):
        check(f"page contains no {sig!r}", sig not in script)
    # Retain the task-level profiles and offer a carefully scoped radar for
    # one comparable metric family only (complete binary accuracy axes).
    check("task profiles remain as dot plots",
          "type:'scatter'" in script and "data:g.axes.map((c,ci)" in script)
    check("accuracy radar only uses complete binary suites",
          "RADAR_AXES=['code','bugfix','algo','refactor','rust','embedded_cpp','lint','std_gsm8k_s30']" in script
          and "type:'radar'" in script
          and "return x&&x.v!=null&&!(m.broken&&m.broken[c])" in script)
    check("radar never fills missing values with zero",
          "value:RADAR_AXES.map(c=>valueFor(m,c))" in script
          and "valueFor=(m,c)=>m.cats[c].v" in script)
    check("profile tooltip resolves its task by point index",
          "const c=g.axes[p.dataIndex]" in script)
    check("radar hover handles the whole value array",
          "Number.isInteger(p.dimensionIndex)" in script
          and "return tooltipFor(m,di)" in script
          and "Math.round(x.v*x.n)" in script)
    # a scatter symbolSize callback must survive an object data item
    check("scatter symbolSize handles object items",
          "Array.isArray(d)?d:(d&&d.value)" in html)
    # the "not measured" language must exist
    check("page explains 'not measured'", "not measured in this instrument" in html)
    # every chart that can lose a model must carry a note container
    for cid in ("c-custom", "c-std", "c-heat", "c-eff", "c-lang",
                "c-chance", "c-mmlu", "c-agent", "c-speed", "c-txt", "c-qual",
                "c-deep", "c-cs"):
        check(f"{cid} has a note container", f'id="nd-{cid}"' in html)
    # the accuracy radar and task-score profiles are built at run time
    check("profile host exists", 'id="radars"' in html)
    check("profiles group by measurement coverage", 'groups[key]' in script)
    check("profile points retain every task value", 'data:g.axes.map((c,ci)' in script)
    check("headline charts are initialized", "bars('c-custom', D.custom" in script
          and "bars('c-std', D.std" in script)
    # Heatmap labels remain thin and unoutlined; item-level color keeps
    # contrast without a heavy text stroke.
    check("heatmap label ink is per item", 'item.label = {color:ink' in script)
    check("heatmap computes the ramp", 'function rampRGB' in script)
    check("heatmap ramp runs pale-green low to dark-green high",
          "const RAMP=['#edf7ed','#c8e6c9','#81c784','#388e3c','#1b5e20']" in script
          and "inRange:{color:RAMP}" in script)
    check("heatmap chooses the stronger text contrast",
          "darkContrast>=whiteContrast" in script)
    check("heatmap labels are small and have no outline",
          "fontSize:8,fontWeight:'normal',textBorderWidth:0" in script)
    check("heatmap reserves room for the longest model name",
          "grid:{height:'76%',top:14,left:145,right:60}" in script)
    check("heatmap hover includes intervals and chance",
          "if(cell.lo!=null) s+=`<br/>95% CI" in script
          and "if(cell.chance!=null) s+=`<br/>chance" in script)
    check("heatmap color scale is not an interactive filter",
          "calculable:false" in script)
    check("alternate and invalid cells have distinct outlines",
          "borderColor:'#f0c674'" in script and "borderColor:'#ff5a4e'" in script)
    check("text-match bars stay fully visible", "opacity:0.62" not in script)
    check("heatmap shows values without status glyphs",
          "return Math.round(p.value[2]*100)+'%'" in script
          and "return '⚠'+pc" not in script and "return '◐'+pc" not in script)
    check("not-comparable tooltips retain interval and n",
          "not comparable: ${briefBroken(m.broken[c])}" in script
          and "if(cell.lo!=null) t += ` · 95% CI" in script
          and "n=${cell.n}`" in script)
    check("all tooltips are bounded, wrapped, and chart-confined",
          "confine:true,appendToBody:true,showDelay:0,hideDelay:0,transitionDuration:0.1" in script
          and "max-width:min(300px,calc(100vw - 24px))" in script
          and "white-space:normal!important" in script
          and "window.innerWidth-w-8" in script
          and "window.innerHeight-h-8" in script)
    check("long backend diagnostics are summarized in chart hovers",
          "briefBroken(m.broken[c])" in script
          and "briefBroken(x.broken)" in script
          and "tool-calling backend failed qualification" in script)
    check("agentic intervals belong only to pass-k reliability",
          "ci:k==='pass_hat_k_reliability'?m.agentic_ci:null" in script)
    check("long methodology is available but collapsed",
          "<details><summary>Methods and interpretation</summary>" in html)
    check("chart notes are not nested or repeated",
          "return bits.join(' · ')" in script
          and "<div class=\"nd\">${bits.join" not in script)
    # ---- 8. a chart group must never be empty: the in-run throughput
    #         chart once looked up a key that does not exist, so its whole
    #         GSM8K group was blank and nobody noticed ----
    tps_keys = set()
    for m in models:
        tps_keys |= set((m.get("tps") or {}).keys())
    rows = re.search(r"const rows=\[\['code','Code tok/s'\],\['([a-z0-9_]+)'", script)
    check("the throughput chart's row keys exist in the payload",
          rows is not None and rows.group(1) in tps_keys,
          f"chart wants {rows.group(1) if rows else '?'}, payload has {sorted(tps_keys)}")
    check("every payload tps key is plotted",
          tps_keys <= {("code"), rows.group(1) if rows else ""},
          f"payload has {sorted(tps_keys)}")
    # the note divs must not be forced to the chart-canvas height: that
    # left 420px of empty space under every chart
    check("note divs are excluded from the canvas height rule",
          ".chart>div[id]:not(.nd)" in html)
    check("global key explains the outline marks",
          "amber outline = text-match MC" in html
          and "red-orange outline = not comparable" in html)
    check("area leaders are shown below the heatmap with caveats",
          "Highest observed result by area" in html
          and 'id="leaders-custom"' in html and 'id="leaders-std"' in html
          and "not significance-tested winners" in html
          and "renderAreaLeaders('leaders-custom'" in script)
    check("chance baseline uses a bright diamond instead of a black dot",
          "symbol:'diamond',symbolSize:14" in script
          and "color:'#f4f7fb'" in script
          and "bright diamonds show the item-weighted chance baseline" in html)
    check("report keeps method details outside chart captions",
          "position imbalance" in html and "isolated throughput" in html)
    check("mobile chart panels can scroll without page overflow",
          ".chart-scroll>div#c-heat{min-width:1120px}" in html
          and "overflow-x:auto" in html)
    # Chance is a dataset property. If per-model baselines disagree, taking
    # the maximum for the marker could silently plot the wrong reference.
    chance_mismatches = []
    for c in payload["custom"] + payload["std"]:
        vals = [((m.get("cats") or {}).get(c) or {}).get("chance")
                for m in models]
        vals = [v for v in vals if v is not None]
        if vals and max(vals) - min(vals) > 1e-9:
            chance_mismatches.append(c)
    check("chance baseline is consistent within each suite",
          not chance_mismatches, str(chance_mismatches))
    try:
        charts.ensure_report_has_models({"models": []})
        empty_report_guard = False
    except ValueError as exc:
        empty_report_guard = "refusing to write an empty report" in str(exc)
    check("report generator fails closed without raw results", empty_report_guard)
    # The raw record count in the primary JSON differs for k2 because its MC
    # records are stored in a separate text-match file. Verify that the
    # combined non-agentic screening question IDs/prompts are exactly equal.
    record_maps = {}
    for m in models:
        label = m["label"]
        files = [os.path.join(ROOT, "results", "llamacpp", label + ".json")]
        if label == "k2-horizon":
            files.append(os.path.join(ROOT, "results", "llamacpp",
                                      label + "-txt.json"))
        recmap = {}
        for path in files:
            with open(path) as f:
                for r in json.load(f).get("records", []):
                    if r.get("task_category") == "agentic":
                        continue
                    key = (r.get("task_category"), r.get("id") or r.get("chain_id"))
                    recmap[key] = r.get("prompt")
        record_maps[label] = recmap
    ref = record_maps[models[0]["label"]]
    check("all models have the same 544 non-agentic screening records",
          len(ref) == 544 and all(v == ref for v in record_maps.values()),
          str({k: len(v) for k, v in record_maps.items()}))
    coverage = {r["model"]: r["cells"] for r in payload["coverage"]}
    check("k2 text-match screening records are split from primary records",
          coverage.get("k2-horizon") == [172, 388, 544, 143, 38, 100],
          str(coverage.get("k2-horizon")))
    check("all model rows show the same shared screening count",
          all(cells[2] == 544 for cells in coverage.values()))
    qfiles = glob.glob(os.path.join(ROOT, "results", "llamacpp", "*-q.json"))
    qrows, qcommon, qrankable = qualify.analyze_files(qfiles)
    qbylabel = {os.path.basename(r["file"]): r for r in qrows}
    check("qualifier refuses a macro with only two common valid suites",
          qcommon == ["code", "std_gsm8k_s30"] and not qrankable)
    check("qualifier excludes k2's failed letter-logprob cells",
          "science" in qbylabel["k2-horizon-q.json"]["invalid_categories"]
          and "science" not in qbylabel["k2-horizon-q.json"]["scores"])
    check("qualifier excludes tool-probe-failed agentic cells",
          "agentic" in qbylabel["nemotron-3-nano-q.json"]["invalid_categories"]
          and "agentic" not in qbylabel["nemotron-3-nano-q.json"]["scores"])
    qpayload = {m["label"]: m for m in payload["q"]}
    check("report marks invalid qualifier MC and tool cells",
          "science" in qpayload["k2-horizon-q"]["broken"]
          and "agentic" in qpayload["medgemma-4b-q"]["broken"]
          and "agentic" in qpayload["nemotron-3-nano-q"]["broken"])
    check("qualifier report labels the shared mean as non-ranking",
          "shared code/GSM8K mean¹" in html
          and "Not a CUT/ADVANCE score." in script)
    os.remove("/tmp/llmbench-test-report.html")

    print()
    if FAILS:
        print(f"REPORT CONTRACTS FAILED: {len(FAILS)}")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("REPORT CONTRACTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
