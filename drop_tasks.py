#!/usr/bin/env python3
"""drop_tasks.py — invalidate the records of given tasks in a result file.

Why this exists: when an INSTANCE changes (the position-debiasing of the
MMLU-6 / HellaSwag / WinoGrande s30 sets), the item ids stay the same but
the items' content does not. A resume would then see those ids as "already
done" and keep the stale answers. Dropping the records of the affected
tasks makes the coverage gap explicit, and the next resume re-runs exactly
those items and nothing else.

Usage:
  python3 drop_tasks.py <file> <task> [<task> ...] [--dry-run]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = json.load(open(args.file))
    recs = d.get("records", [])
    before = len(recs)
    keep = [r for r in recs if r.get("task") not in args.tasks]
    dropped = before - len(keep)

    cfg = d.get("config") or {}
    try:
        exp = R.expected_coverage(
            cfg.get("tasks", []), cfg.get("caps_per_task") or {},
            {k: v for k, v in (cfg.get("reps_per_task") or {}).items()
             if k != "(all)"} or {"(all)": cfg.get("reps", 1)},
            cfg.get("subsample", 0), cfg.get("limit", 0))
        gaps = R.coverage_report(keep, exp)
    except Exception as e:  # noqa: BLE001
        print(f"  coverage recompute failed ({e}); keeping the stored gaps")
        gaps = d.get("coverage_gaps") or []

    print(f"  {os.path.basename(args.file)}: dropped {dropped} records "
          f"({', '.join(args.tasks)}) -> {len(keep)} remain; "
          f"gaps now: {gaps or 'none'}")
    if args.dry_run:
        return 0
    d["records"] = keep
    d["summary"] = R.aggregate(keep)
    d["coverage_gaps"] = gaps
    d["partial"] = bool(gaps)
    R.write_atomic(args.file, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
