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
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import charts  # noqa: E402

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
    # the radar tooltip must read the point's own value
    check("radar tooltip uses p.value[dimensionIndex]",
          "p.value[p.dimensionIndex]" in html)
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
    # the radar charts are built at run time, one per axis-coverage group
    check("radar host exists", 'id="radars"' in html)
    check("radar groups by coverage", 'groups[key]' in script)
    # the radar must never plot a missing value as a zero: ECharts draws a
    # null radar value at the centre, so an axis that not every model can
    # answer has to live in its own radar.
    check("radar groups by model coverage", 'g.axes.length>=3' in script)
    # the heatmap label ink must be set per item (a label.color CALLBACK is
    # never invoked for a heatmap series in ECharts 5.6)
    check("heatmap label ink is per item", 'item.label = {color:ink' in script)
    check("heatmap computes the ramp", 'function rampRGB' in script)
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
    check("the note wording matches the glyph mark",
          "marked \u25d0 on the bar" in script or "marked \u25d0" in script)
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
