#!/usr/bin/env python3
"""
Show exactly WHY items failed, without digging through JSON.

Usage:
  python3 failures.py results/nemotron-3-nano.json [task] [--show-code N]

Lists every non-passing record with its error class, the tail of stderr, and
(optionally) the first N lines the model actually wrote. Execution failures
(nonzero_exit/timeout/compile_error) print the assertion or compiler message;
scoring failures (MC wrong pick, devops misses, netdiag thirds, lint diags)
print the gold-vs-got that decided them.
"""
import argparse
import json


def card(r):
    cat = r.get("category", "?")
    rid = r.get("id") or r.get("chain_id")
    head = f"{r.get('task')}/{rid} rep={r.get('rep')} [{cat}]"
    lines = [head]
    if cat in ("code", "bugfix", "algo", "refactor", "rust", "embedded_cpp"):
        lines.append(f"  error : {r.get('exec_error')} stage={r.get('exec_stage')}")
        tail = (r.get("stderr_tail") or "").strip().splitlines()
        if tail:
            lines.append(f"  stderr: {tail[-1][:220]}")
        if cat == "rust" and r.get("clippy"):
            c = r["clippy"]
            lines.append(f"  clippy: n={c.get('n_diagnostics')} compiles={c.get('compiles')}")
    elif "correct" in r:
        lines.append(f"  gold={r.get('gold_index')} chose={r.get('chosen_index')} "
                     f"({r.get('chosen_letter')}) mass={r.get('letter_mass')}")
    elif cat == "devops":
        lines.append(f"  score={r.get('score')} missed={r.get('missed')} "
                     f"violations={r.get('violations')}")
    elif cat == "netdiag":
        lines.append(f"  score={r.get('score')} verdict={r.get('verdict_ok')} "
                     f"dev={r.get('device_ok')} cause={r.get('cause_ok')} "
                     f"{r.get('detail')}")
    elif cat == "lint":
        d = r.get("detail") or {}
        sub = d.get("ruff") or d.get("clippy") or {}
        lines.append(f"  n_diags={r.get('n_diagnostics')} gaming={r.get('noqa_gaming')} "
                     f"msgs={sub.get('messages', [])[:4]}")
    elif cat == "agentic":
        lines.append(f"  outcome={r.get('outcome_ok')} "
                     f"trace={r.get('trace_ok')}/{r.get('trace_total')} "
                     f"calls={r.get('n_tool_calls')} stop={r.get('stopped')}")
        for t in r.get("trace_detail", [])[:4]:
            lines.append(f"    - {t}")
        lines.append(f"  final : {(r.get('final_answer') or '')[:200]}")
    else:
        lines.append(f"  score={r.get('score')} passed={r.get('passed')}")
    return lines, r


def failed(r):
    cat = r.get("category")
    if cat in ("code", "bugfix", "algo", "refactor", "rust", "embedded_cpp"):
        return not r.get("passed")
    if "correct" in r:
        return not r.get("correct")
    if cat == "lint":
        return (r.get("n_diagnostics") or 0) > 0
    if cat == "agentic":
        return not (r.get("outcome_ok") and r.get("trace_ok") == r.get("trace_total")
                    and (r.get("n_tool_calls") or 0) > 0
                    or (r.get("expect_no_calls") and r.get("outcome_ok")
                        and (r.get("n_tool_calls") or 0) == 0))
    return (r.get("score") if isinstance(r.get("score"), (int, float)) else 1.0) < 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("task", nargs="?", default="")
    ap.add_argument("--show-code", type=int, default=0)
    args = ap.parse_args()
    d = json.load(open(args.results))
    recs = [r for r in d.get("records", [])
            if (not args.task or r.get("task") == args.task) and failed(r)]
    print(f"{len(recs)} failing records"
          + (f" in task {args.task}" if args.task else " total"))
    for r in recs:
        lines, _ = card(r)
        print("\n".join(lines))
        if args.show_code:
            code = (r.get("code_extracted") or "").splitlines()[:args.show_code]
            for ln in code:
                print(f"    | {ln[:160]}")
        print()


if __name__ == "__main__":
    main()
