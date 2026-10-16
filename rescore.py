#!/usr/bin/env python3
"""
Re-aggregate existing result files with the current scoring code.

Results files hold the raw records - prompts, completions, extracted code,
compiler output, linter diagnostics - so scoring can be recomputed offline
without re-running a model. That separation is deliberate: scoring code gets
fixed, and the fix applies retroactively instead of costing hours of inference
again.

The category is taken from the RUNNERS table keyed on each record's `task`
field, not from the record's own `category` key, because an earlier version of
run.py let a task file decide its own category and that merged three categories
into one. Results written before that fix are corrected here.

Usage:
  python3 rescore.py                # rewrite every results/*.json in place
  python3 rescore.py --dry-run      # show what would change
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def recategorize(rec):
    task = rec.get("task")
    cat = R.RUNNERS[task][1] if task in R.RUNNERS else rec.get("category")
    if rec.get("category") != cat:
        rec["task_category"] = rec.get("category")
        rec["category"] = cat
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--files", default="")
    args = ap.parse_args()

    pats = args.files.split(",") if args.files else [os.path.join(RESULTS, "**", "*.json")]
    files = []
    for p in pats:
        files.extend(sorted(glob.glob(p, recursive=True)))
    files = [f for f in files if not f.endswith("speed.json")]

    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"  {os.path.basename(f):<28} unreadable: {e}")
            continue
        if "records" not in d:
            continue
        before = {c: s.get("accuracy", s.get("mean_score"))
                  for c, s in d.get("summary", {}).items()}
        recs = [recategorize(dict(r)) for r in d["records"]]
        after = R.aggregate(recs)
        d["records"] = recs
        d["summary"] = after
        # Backfill the partial flag on files written before it existed:
        # any ERROR stub means the run never completed cleanly.
        if any("ERROR" in str(r.get("id", "")) for r in recs):
            d["partial"] = True
        changed = {c: v for c, v in after.items() if before.get(c) != v.get("accuracy", v.get("mean_score"))}
        name = os.path.basename(f)
        if args.dry_run:
            print(f"  {name:<28} {len(after)} categories"
                  + (f"  CHANGED: {list(changed)}" if changed else "  (no change)"))
        else:
            json.dump(d, open(f, "w"), ensure_ascii=False, indent=1)
            print(f"  {name:<28} rescored -> {len(after)} categories"
                  + (f"  (changed: {', '.join(changed)})" if changed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())