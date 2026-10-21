#!/usr/bin/env python3
"""
llmbench: one comprehensive, all-in-one evaluation of a local LLM.

Run ONE model per invocation. That is a deliberate constraint, not a limitation:
running two models concurrently on 2 cores splits them between the same pipe and
halves each one's throughput, and any timing comparison between them would then
be measuring contention rather than model speed. Sequential runs are the only
way the speed numbers mean anything.

Every measurement is written to JSON with full provenance: the exact prompt, the
exact completion, extracted code, compiler/linter diagnostics verbatim, token
counts, server-reported and wall-clock timings, and the frozen inference config.
Results are re-scoreable offline without re-running the model.

Conditions are identical across models by construction: the config dict is
written into the result file and a --config-hash is recorded, so any drift
between runs is detectable rather than assumed away.

Usage:
  python3 run.py --backend ollama   --model nemotron-3-nano:4b
  python3 run.py --backend llamacpp --model k2-horizon --port 8091
  python3 run.py ... --tasks science,english --reps 3
  python3 run.py ... --limit 5      # fast smoke pass
"""
import argparse
import hashlib
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agentic                                            # noqa: E402
from backend import Backend, BackendError                 # noqa: E402
# imported as a module, not by name: run_rust is also the name of a task runner
# below, and a by-name import would be shadowed into infinite recursion.
import scorers as S                                       # noqa: E402
from stats import mc_bundle                                # noqa: E402

check_devops = S.check_devops
extract_code = S.extract_code
lint_python = S.lint_python
lint_rust = S.lint_rust
normalize_fa = S.normalize_fa
run_python = S.run_python
rustfmt_check = S.rustfmt_check
score_netdiag = S.score_netdiag
wilson = S.wilson

TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---------------------------------------------------------------- frozen config
# Anything that varies per model must NOT live here. These are held constant so
# differences in the score table are attributable to the weights, not the setup.
FROZEN = {
    "num_ctx": 4096,
    "num_gpu_layers": 0,
    "gen_temperature": 0.7,      # 0.0 makes MiniCPM5-2B loop forever and return ""
    "gen_top_p": 0.95,           # pinned here so it enters config_hash; the
                                 # backend default must match (asserted at startup)
    "mc_temperature": 0.0,       # MC is scored from a single token's distribution
    "gen_max_tokens": 512,
    "agent_max_tokens": 200,
    "seed_base": 20261012,
}

SYSTEM_CODER = ("You are a precise coding assistant. Output only the requested code "
                "inside a single fenced code block, with no explanation.")
SYSTEM_MC = "Answer with a single letter and nothing else."
# The second mechanism's system prompt: the option's text, not a letter.
SYSTEM_MC_TEXT = "Answer with the exact text of the correct option, and nothing else."


def T(name):
    return 0,0  # placeholder, replaced below


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tasks(name):
    # The _txt variants are the SAME items as their letter-logprob
    # counterparts, measured with the other mechanism. The alias keeps one
    # set of items on disk; the category suffix keeps the results apart.
    name = TASK_ALIAS.get(name, name)
    p = os.path.join(TASKS_DIR, name + ".json")
    if not os.path.exists(p):
        raise SystemExit(f"missing task file: {p}")
    return json.load(open(p))


# Generation is ~400 tokens per item at 3-8 tok/s, so the full 728-item code set
# is ~16 h per repetition. Subsampling uses a fixed seed and sorts the selection
# so the same items are chosen for every model - comparing models on different
# subsets would confound the model difference with the item difficulty difference.
import random as _random


def subsample(tasks, n, seed=20261012):
    if not n or n >= len(tasks):
        return tasks
    idx = sorted(_random.Random(seed).sample(range(len(tasks)), n))
    return [tasks[i] for i in idx]


def _err1(stderr, width=100):
    """The most informative failure line: for tracebacks it is the LAST line
    (the exception), for rustc/g++ it is the FIRST 'error' line (the last
    lines are span markers and footers). Never crashes, never empty."""
    lines = (stderr or "").strip().splitlines()
    if not lines:
        return ""
    for ln in lines:
        if "error" in ln.lower():
            return ": " + ln.strip()[:width]
    return ": " + lines[-1].strip()[:width]


def _mem_guard(b, args, where):
    """Per-item OOM guard for long generation loops (the between-tasks check
    never fires inside a 30-item GSM8K rep). Raises BackendError tagged
    OOM-GUARD so the rep handler skips its doomed retry and stubs at once:
    retrying against a fat server burns a full timeout for nothing."""
    budget = getattr(args, "server_mem_budget", 0) or 0
    if not budget:
        return
    rss = b.server_rss_mb() if hasattr(b, "server_rss_mb") else None
    if rss and rss > budget:
        raise BackendError(
            f"OOM-GUARD: server RSS {rss} MiB over budget {budget} at {where}")


def _item_error(t, rep, cat, e, extra=None):
    """Per-item last resort: one bad item must never cost its whole rep.

    A rep-level crash used to discard every completed item into a single
    ERROR stub (an empty model output once reached a log line that assumed
    an int, killing 5 good MBPP items). BackendError is NEVER routed here:
    it propagates to the rep handler, which owns retry/abort. The log line
    itself uses no format specs, so it cannot crash either.
    """
    import traceback as _tb
    tail = " | ".join(_tb.format_exc().strip().splitlines()[-2:])
    tid = t.get("id", "?") if isinstance(t, dict) else "?"
    log(f"    !! {tid} {type(e).__name__}: {str(e)[:160]}")
    # The #ERROR suffix is the harness-artifact convention: aggregate() and
    # compare() both exclude such records from scoring (missing, never zero),
    # while task+rep fields keep them attributable and resumable.
    rec = {"id": f"{tid}#ERROR", "rep": rep, "category": cat,
           "passed": False, "correct": False, "score": 0.0,
           "error": f"{type(e).__name__}: {e}"[:300],
           "trace": tail[:300],
           "tokens": {"prompt": 0, "eval": 0},
           "speed": {"gen_tps": None, "wall_s": None}}
    if cat == "net_diag":
        rec["is_anomaly"] = t.get("is_anomaly", True)
        rec.update(verdict_ok=False, device_ok=False, cause_ok=False,
                   verdict_unclear=False)
    if cat == "agentic":
        rec.update(chain_id=tid, outcome_ok=False,
                   trace_ok=0, trace_total=len(t.get("steps", []) or [1]),
                   n_tool_calls=0, stopped="exception", calls=[],
                   expect_no_calls=bool(t.get("expect_no_calls")))
    if extra:
        rec.update(extra)
    return rec


# ============================================================ task runners
def run_code(b, tasks, args, rep, progress=None):
    """Python generation + execution + lint."""
    out = _ProgressList(progress)
    n_tasks = len(tasks)
    for i, t in enumerate(tasks):
        try:
            _mem_guard(b, args, "run_code-item-%d" % (i + 1))
            r = b.generate([{"role": "system", "content": SYSTEM_CODER},
                            {"role": "user", "content": t["prompt"]}],
                           max_tokens=FROZEN["gen_max_tokens"],
                           temperature=FROZEN["gen_temperature"],
                           seed=FROZEN["seed_base"] + rep)
            raw = r["content"]
            code = extract_code(raw, "python")
            code, completion_style = S.reconcile_prefix(t["prompt"], code, "python")
            syn_ok, syn_err = S.python_syntax_ok(code)
            ex = {"passed": False, "error": "syntax_error", "stderr_tail": syn_err} \
                if not syn_ok else run_python(code, t.get("test", ""))
            lint = lint_python(code) if code.strip() else {"available": False, "empty": True,
                                                           "n_diagnostics": None, "clean": False,
                                                           "noqa_gaming": False, "lines": 0}
            out.append({
                "id": t["id"], "rep": rep, "category": "code",
                "prompt": t["prompt"], "completion_raw": raw, "code_extracted": code,
                "test": t.get("test", ""),
                "completion_style": completion_style,
                "syntax_ok": syn_ok, "syntax_error": syn_err,
                "passed": bool(ex["passed"]), "exec_error": ex.get("error"),
                "exec_stage": ex.get("stage"), "stderr_tail": ex.get("stderr_tail", "")[:400],
                "lint": lint,
                "tokens": {"prompt": r["prompt_tokens"], "eval": r["eval_tokens"]},
                "speed": {"gen_tps": r["gen_tps"], "prompt_tps": r["prompt_tps"],
                          "ttft_s": r["ttft_s"], "wall_s": r["wall_s"],
                          "load_s": r.get("load_s"), "prompt_cached_tokens": r.get("prompt_cached_tokens")},
            })
            style = "" if completion_style == "complete" else f" [{completion_style}]"
            # The failure REASON belongs on the log line: "fail nonzero_exit" alone
            # forces a JSON dig for every failure. First stderr line carries it
            # (AssertionError vs NameError vs timeout) in ~100 chars.
            err1 = (ex.get("stderr_tail") or "").strip().splitlines()
            err1 = (": " + err1[-1][:100]) if (err1 and not ex.get("passed")) else ""
            ndiag = lint.get("n_diagnostics")
            ndiag_s = f"{ndiag:>2}" if isinstance(ndiag, int) else " -"
            log(f"    [{i+1}/{n_tasks}] {t['id']:<32} {'PASS' if ex['passed'] else 'fail ' + str(ex.get('error')):<22} "
                f"lint={ndiag_s} {r['gen_tps']}tok/s{style}{err1}")
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            out.append(_item_error(t, rep, "code", e))
    return out


def run_rust(b, tasks, args, rep, progress=None):
    """Rust generation + compile + run + clippy + rustfmt."""
    out = _ProgressList(progress)
    n_tasks = len(tasks)
    for i, t in enumerate(tasks):
        try:
            _mem_guard(b, args, "run_rust-item-%d" % (i + 1))
            r = b.generate([{"role": "system", "content": SYSTEM_CODER},
                            {"role": "user", "content": t["prompt"]}],
                           max_tokens=FROZEN["gen_max_tokens"],
                           temperature=FROZEN["gen_temperature"],
                           seed=FROZEN["seed_base"] + rep)
            raw = r["content"]
            code = extract_code(raw, "rust")
            res = S.run_rust(code, t["tests"], t["entry_point"]) if code.strip() else \
                {"passed": False, "error": "empty_output", "stage": "extract"}
            clip = lint_rust(code) if code.strip() else {"available": False, "empty": True,
                                                            "n_diagnostics": None, "clean": False}
            fmt = rustfmt_check(code) if code.strip() else {"available": False}
            out.append({
                "id": t["id"], "rep": rep, "category": "rust",
                "prompt": t["prompt"], "completion_raw": raw, "code_extracted": code,
                "test": "\n".join(t["tests"]),
                "passed": bool(res.get("passed")), "exec_error": res.get("error"),
                "exec_stage": res.get("stage"),
                "stderr_tail": (res.get("stderr_tail") or "")[:500],
                "wrote_fn_main": res.get("wrote_main"), "used_test_attr": res.get("uses_test_attr"),
                "clippy": clip, "rustfmt": fmt,
                "tokens": {"prompt": r["prompt_tokens"], "eval": r["eval_tokens"]},
                "speed": {"gen_tps": r["gen_tps"], "ttft_s": r["ttft_s"], "wall_s": r["wall_s"],
                          "load_s": r.get("load_s"), "prompt_cached_tokens": r.get("prompt_cached_tokens")},
            })
            log(f"    [{i+1}/{n_tasks}] {t['id']:<32} {'PASS' if res.get('passed') else 'fail ' + str(res.get('error')):<22}{_err1(res.get("stderr_tail")) if not res.get("passed") else ""} "
                f"clippy={clip.get('n_diagnostics', 0)} {r['gen_tps']}tok/s")
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            out.append(_item_error(t, rep, "rust", e))
    return out


def run_mc(b, tasks, args, rep, progress=None):
    """Multiple choice scored from one forward pass via top-k logprobs.

    Returns accuracy plus a confidence distribution, because accuracy alone hides
    the difference between a model that knows the answer and one that guessed
    uniformly. Mean top-1 probability on the letter set is the tell.
    """
    out = _ProgressList(progress)
    n_tasks = len(tasks)
    for i, t in enumerate(tasks):
        try:
            _mc = b.mc_logprobs(t["question"], t["choices"], system=SYSTEM_MC)
            if len(_mc) == 5:           # older signature, no letter_mass
                idx, letter, dist, meta, mode = _mc
                letter_mass = max(dist.values()) if dist else 0.0
            else:
                idx, letter, dist, meta, mode, letter_mass = _mc
            ok = (idx == t["answer"])
            # `correct` is the argmax pick, full stop. It used to be
            # `ok and letter_mass >= 0.15`, which recorded a model that picked the
            # gold letter as INCORRECT whenever its total mass on the letter set
            # fell below 15%. That is a systematic under-report: it penalises exactly
            # the uncertain-but-right models the suite is trying to characterise,
            # and it was inert on confident models (a live sweep showed all masses
            # >= 0.9949), so it would have silently done nothing until the day it
            # mattered. Low mass is a real property of the response and is reported
            # as `informed`/`letter_mass`; it is not a scoring penalty.
            out.append({
                "id": t["id"], "rep": rep, "category": t.get("category", "mc"),
                "lang": t.get("lang"), "subject": t.get("subject"),
                "question": t["question"], "choices": t["choices"],
                "test": t["question"],
                "gold_index": t["answer"],
                "chosen_index": idx, "chosen_letter": letter,
                "correct": ok,
                "pick_correct": ok,
                # low-information: the model committed little total mass to any letter.
                # Reported, not penalised. A gate on this field, rather than on the
                # pick, is the correct use of it.
                "letter_mass": round(letter_mass, 4),
                "informed": bool(letter_mass >= 0.15),
                "confidence": round(letter_mass, 4),
                "distribution": {k: round(v, 6) for k, v in sorted(dist.items())},
                "scoring_mode": mode,
                "tokens": {"prompt": meta["prompt_tokens"], "eval": meta["eval_tokens"]},
                "speed": {"gen_tps": None, "ttft_s": meta["ttft_s"],
                          "wall_s": meta["wall_s"]},
            })
            log(f"    [{i+1}/{n_tasks}] {t['id']:<28} {'OK ' if ok else 'X  '} "
                f"picked={letter} gold={chr(65 + t['answer'])} "
                f"mass={letter_mass:.3f} ({meta['wall_s']}s)")
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            out.append(_item_error(t, rep, t.get("category", "mc"), e))
    return out


def run_netdiag(b, tasks, args, rep, progress=None):
    out = _ProgressList(progress)
    n_tasks = len(tasks)
    for i, t in enumerate(tasks):
        try:
            _mem_guard(b, args, "run_netdiag-item-%d" % (i + 1))
            msgs = [{"role": "system", "content":
                     "You are a network engineer. Given the topology, the configuration and "
                     "the reported symptom, diagnose the problem. If the evidence shows "
                     "everything is actually healthy, say so plainly. Be concise."},
                    {"role": "user", "content":
                        f"Topology: {t['topology']}\n\nConfiguration:\n{t['config']}\n\n"
                        f"Reported symptom: {t['symptom']}\n\nObservations:\n{t['observed']}\n\n"
                        f"Is there an actual fault? If yes, which device and what is the root "
                        f"cause? If no, say the network is healthy and no action is needed."}]
            r = b.generate(msgs, max_tokens=256, temperature=FROZEN["gen_temperature"],
                           seed=FROZEN["seed_base"] + rep)
            s = score_netdiag(r["content"], t)
            out.append({
                "id": t["id"], "rep": rep, "category": "net_diag",
                "prompt": msgs[-1]["content"], "completion_raw": r["content"],
                "is_anomaly": t["is_anomaly"], "root_cause": t["root_cause"],
                "score": s["score"], "verdict_ok": s["verdict_ok"],
                "device_ok": s["device_ok"], "cause_ok": s["cause_ok"], "detail": s["detail"],
                "tokens": {"prompt": r["prompt_tokens"], "eval": r["eval_tokens"]},
                "speed": {"gen_tps": r["gen_tps"], "wall_s": r["wall_s"],
                          "load_s": r.get("load_s"), "prompt_cached_tokens": r.get("prompt_cached_tokens")},
            })
            log(f"    [{i+1}/{n_tasks}] {t['id']:<28} score={s['score']:.2f} "
                f"{'(false premise)' if not t['is_anomaly'] else t['root_cause']:<14} {r['gen_tps']}tok/s")
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            out.append(_item_error(t, rep, "net_diag", e))
    return out


def run_devops(b, tasks, args, rep, progress=None):
    out = _ProgressList(progress)
    n_tasks = len(tasks)
    for i, t in enumerate(tasks):
        try:
            _mem_guard(b, args, "run_devops-item-%d" % (i + 1))
            r = b.generate([{"role": "system", "content": SYSTEM_CODER},
                            {"role": "user", "content": t["prompt"]}],
                           max_tokens=FROZEN["gen_max_tokens"],
                           temperature=FROZEN["gen_temperature"],
                           seed=FROZEN["seed_base"] + rep)
            c = check_devops(r["content"], t["checks"])
            out.append({
                "id": t["id"], "rep": rep, "category": "devops",
                "prompt": t["prompt"], "completion_raw": r["content"],
                "score": c["score"], "passed": c["passed"],
                "must_hit": f"{c['must_hit']}/{c['must_total']}", "missed": c["missed"],
                "violations": c["violations"], "empty": c["empty"],
                "tokens": {"prompt": r["prompt_tokens"], "eval": r["eval_tokens"]},
                "speed": {"gen_tps": r["gen_tps"], "wall_s": r["wall_s"],
                          "load_s": r.get("load_s"), "prompt_cached_tokens": r.get("prompt_cached_tokens")},
            })
            log(f"    [{i+1}/{n_tasks}] {t['id']:<32} {c['must_hit']}/{c['must_total']} "
                f"score={c['score']:.2f} miss={','.join(c['missed']) or '-'}")
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            out.append(_item_error(t, rep, "devops", e))
    return out


def run_lint(b, tasks, args, rep, progress=None):
    """The lint-as-benchmark suite. No unit tests: the artifact is the point."""
    out = _ProgressList(progress)
    n_tasks = len(tasks)
    for i, t in enumerate(tasks):
        try:
            _mem_guard(b, args, "run_lint-item-%d" % (i + 1))
            lang = t["lang"]
            sysmsg = SYSTEM_CODER
            r = b.generate([{"role": "system", "content": sysmsg},
                            {"role": "user", "content": t["prompt"]}],
                           max_tokens=FROZEN["gen_max_tokens"],
                           temperature=FROZEN["gen_temperature"],
                           seed=FROZEN["seed_base"] + rep)
            code = extract_code(r["content"], lang)
            if lang == "python":
                # An empty extraction is a total failure (refusal, prose-only reply),
                # not a style success. It used to record n_diagnostics=0/clean=True
                # and count toward the clean_rate numerator.
                lint = lint_python(code) if code.strip() else {"available": False, "empty": True,
                                                                "n_diagnostics": None, "clean": False,
                                                                "noqa_gaming": False}
                fmt = _ruff_format(code)
                detail = {"ruff": lint, "format": fmt}
                primary = lint.get("n_diagnostics")
            else:
                sets = ("correctness", "style", "complexity", "perf", "pedantic", "nursery")
                clip = lint_rust(code, sets=sets) if code.strip() else {"available": False, "empty": True,
                                                                       "n_diagnostics": None, "clean": False,
                                                                       "noqa_gaming": False}
                fmt = rustfmt_check(code) if code.strip() else {"available": False}
                detail = {"clippy": clip, "format": fmt}
                primary = clip.get("n_diagnostics")
            out.append({
                "id": t["id"], "rep": rep, "category": "lint", "group": t["group"],
                "lang": lang, "prompt": t["prompt"], "completion_raw": r["content"],
                "code_extracted": code, "n_diagnostics": primary,
                "empty": not code.strip(),
                "noqa_gaming": (lint.get("noqa_gaming") if lang == "python"
                                else clip.get("noqa_gaming")) if code.strip() else False,
                "detail": detail,
                "tokens": {"prompt": r["prompt_tokens"], "eval": r["eval_tokens"]},
                "speed": {"gen_tps": r["gen_tps"], "wall_s": r["wall_s"],
                          "load_s": r.get("load_s"), "prompt_cached_tokens": r.get("prompt_cached_tokens")},
            })
            log(f"    [{i+1}/{n_tasks}] {t['id']:<32} "
                f"diags={(f'{primary:>3}' if isinstance(primary, int) else '  -')} "
                f"gaming={out[-1]['noqa_gaming']} fmt={fmt.get('already_formatted', '?')} "
                f"{r['gen_tps']}tok/s")
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            out.append(_item_error(t, rep, "lint", e))
    return out


def _ruff_format(code):
    import subprocess
    from scorers import TMP, shutil_which
    b = shutil_which("ruff")
    if not b or not code.strip():
        return {"available": False}
    p = os.path.join(TMP, "fmtchk.py")
    open(p, "w").write(code)
    try:
        r = subprocess.run([b, "format", "--diff", "--isolated", p],
                           capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return {"available": False}
    return {"available": True, "already_formatted": r.returncode == 0,
            "diff_lines": len([x for x in r.stdout.splitlines()
                               if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))])}


def run_agentic(b, tasks, args, rep, progress=None):
    out = []
    # Honour the caller's selection: ignoring `tasks` meant --limit and
    # --per-task-cap silently had no effect on this suite.
    wanted = {t["id"] for t in tasks} if tasks else None
    chains = [c for c in agentic.CHAINS if wanted is None or c["id"] in wanted]
    n_tasks = len(chains)
    rss_budget = getattr(args, "server_mem_budget", 0) or 0
    rss_of = getattr(b, "server_rss_mb", lambda: None)
    for i, c in enumerate(chains):
        # Memory guard INSIDE the long rep: agentic holds dozens of trials in
        # one rep with no task boundary between them. A ballooning server
        # (measured: 6.8GB anon in 15 min, then OOM-killed) must stop the rep
        # early — completed chains are already checkpointed, resume fills gaps.
        if rss_budget:
            rss = rss_of() if callable(rss_of) else None
            if rss and rss > rss_budget:
                log(f"    !! server RSS {rss} MiB over budget {rss_budget} — "
                    f"stopping rep at chain {i + 1}/{n_tasks}, resume continues")
                break
        try:
            r = agentic.run_chain(b, c, max_tokens=FROZEN["agent_max_tokens"],
                                  temperature=FROZEN["gen_temperature"],
                                  seed=FROZEN["seed_base"] + rep)
            r["rep"] = rep
            r["category"] = "agentic"
            r["prompt"] = c["prompt"]
            if progress is not None:
                progress(r)  # persisted immediately; see main loop
            else:
                out.append(r)
            log(f"    [{i+1}/{n_tasks}] {c['id']:<28} trace={r['trace_ok']}/{r['trace_total']} "
                f"outcome={'OK' if r['outcome_ok'] else 'X'} "
                f"calls={r['n_tool_calls']} stop={r['stopped']}")
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            out.append(_item_error(c, rep, "agentic", e))
    return out


def run_cpp(b, tasks, args, rep, progress=None):
    """Embedded C++ that is actually compiled and executed.

    The mock Arduino/ESP32 header is prepended to whatever the model wrote, so
    the model is free to use digitalWrite/millis/INPUT_PULLUP and have it mean
    something. The task's assertions decide pass or fail - no reviewer, no LLM
    judge.
    """
    out = _ProgressList(progress)
    n_tasks = len(tasks)
    for i, t in enumerate(tasks):
        try:
            _mem_guard(b, args, "run_cpp-item-%d" % (i + 1))
            r = b.generate([{"role": "system", "content": SYSTEM_CODER},
                            {"role": "user", "content": t["prompt"]}],
                           max_tokens=FROZEN["gen_max_tokens"],
                           temperature=FROZEN["gen_temperature"],
                           seed=FROZEN["seed_base"] + rep)
            raw = r["content"]
            code = extract_code(raw, "cpp")
            if not code.strip():
                res = {"passed": False, "error": "empty_output", "stage": "extract",
                       "n_warnings": None}
            else:
                res = S.run_cpp(code, t["test"], t["header"])
            out.append({
                "id": t["id"], "rep": rep, "category": "embedded_cpp",
                "prompt": t["prompt"], "completion_raw": raw, "code_extracted": code,
                "passed": bool(res.get("passed")), "exec_error": res.get("error"),
                "exec_stage": res.get("stage"),
                "stderr_tail": (res.get("stderr_tail") or "")[:600],
                "n_warnings": res.get("n_warnings"),
                "test_source": t["test"],
                "tokens": {"prompt": r["prompt_tokens"], "eval": r["eval_tokens"]},
                "speed": {"gen_tps": r["gen_tps"], "ttft_s": r["ttft_s"], "wall_s": r["wall_s"],
                          "load_s": r.get("load_s"), "prompt_cached_tokens": r.get("prompt_cached_tokens")},
            })
            log(f"    [{i+1}/{n_tasks}] {t['id']:<32} {'PASS' if res.get('passed') else 'fail ' + str(res.get('error')):<22}{_err1(res.get("stderr_tail")) if not res.get("passed") else ""} "
                f"warn={res.get('n_warnings')} {r['gen_tps']}tok/s")
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            out.append(_item_error(t, rep, "embedded_cpp", e))
    return out


def run_gsm8k(b, tasks, args, rep, progress=None):
    """GSM8K grade-school math via generation + normalised exact match.

    Official protocol (Cobbe et al.): 8-shot CoT in the paper, but for tiny
    CPU models we use 0-shot with 'reason step by step, finish with #### N'
    and score the normalised final number. Prompt and shots are recorded
    per record so the deviation is auditable.
    """
    out = _ProgressList(progress)
    n_tasks = len(tasks)
    for i, t in enumerate(tasks):
        try:
            _mem_guard(b, args, "run_gsm8k-item-%d" % (i + 1))
            msgs = [{"role": "system", "content":
                     "You are a careful math assistant. Reason step by step, "
                     "then write the final numeric answer after ####."},
                    {"role": "user", "content":
                     f"{t['question']}\n\nReason step by step, then finish with "
                     f"the final number as #### <number>."}]
            r = b.generate(msgs, max_tokens=FROZEN["gen_max_tokens"],
                           temperature=FROZEN["gen_temperature"],
                           seed=FROZEN["seed_base"] + rep)
            s = S.score_gsm8k(r["content"], t["gold"])
            out.append({
                "id": t["id"], "rep": rep, "category": "gsm8k",
                "prompt": msgs[-1]["content"], "completion_raw": r["content"],
                "test": t["question"], "gold": t["gold"],
                "correct": bool(s["correct"]), "passed": bool(s["correct"]),
                "pred": s["pred"], "extracted": s["extracted"],
                "tokens": {"prompt": r["prompt_tokens"], "eval": r["eval_tokens"]},
                "speed": {"gen_tps": r["gen_tps"], "ttft_s": r["ttft_s"],
                          "wall_s": r["wall_s"],
                          "load_s": r.get("load_s"),
                          "prompt_cached_tokens": r.get("prompt_cached_tokens")},
            })
            log(f"    [{i+1}/{n_tasks}] {t['id']:<28} {'OK ' if s['correct'] else 'X  '} "
                f"pred={s['pred'][:16]:<16} gold={s['gold'][:16]:<16} {r['gen_tps']}tok/s")
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            out.append(_item_error(t, rep, "gsm8k", e))
    return out


class _ProgressList(list):
    """A list that reports every appended record to a callback.

    Durability: a 30-item rep used to live entirely in memory until the rep
    finished, so a suspend/OOM/kill mid-rep lost every completed item of that
    rep (observed twice: the laptop's idle-suspend made logind SIGTERM the
    worker). With this, every finished item is checkpointed to disk the
    moment it is scored - a kill can now only cost the item in flight.
    """

    def __init__(self, progress=None):
        super().__init__()
        self._progress = progress

    def append(self, item):
        super().append(item)
        self._emit(item)

    def extend(self, items):
        for it in items:
            self.append(it)

    def _emit(self, item):
        if not self._progress:
            return
        try:
            self._progress(item)
        except Exception as e:  # noqa: BLE001
            log(f"  !! progress callback failed: {type(e).__name__}: {e}")


CHECKPOINT_HOOK = {"fn": None}


def _install_signal_handlers():
    """Checkpoint-and-exit on SIGTERM/SIGINT.

    The laptop's power manager (xfce4-power-manager) suspends the box on
    idle; logind then SIGTERMs every user process (KillUserProcesses=yes by
    default). Unhandled, the worker died holding the results of the whole
    rep in memory and the item that had just been scored was lost.
    """

    def handler(signum, _frame):
        fn = CHECKPOINT_HOOK.get("fn")
        if fn:
            try:
                fn()
                log(f"  [checkpoint written on signal {signum}]")
            except Exception as e:  # noqa: BLE001
                log(f"  !! checkpoint on signal failed: {type(e).__name__}: {e}")
        os._exit(128 + signum)

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, handler)
        except (ValueError, OSError):
            pass


def run_mc_text(b, tasks, args, rep, progress=None):
    """Multiple choice scored from a GENERATED answer matched to the option.

    The second MC mechanism. The letter-logprob mechanism needs the model to
    put its probability mass on the option letters; some models reply with
    the option's TEXT instead ("4" instead of "B"), and then the mass never
    lands on the letters and the suite is unmeasurable. This runner reads the
    generated answer and matches it to an option (letter, exact text, or
    the text inside a sentence).

    It is a DIFFERENT instrument: the categories are suffixed with _txt and
    are never merged with the letter-logprob cells, so the two are never
    ranked against each other.
    """
    out = _ProgressList(progress)
    n_tasks = len(tasks)
    for i, t in enumerate(tasks):
        try:
            _mem_guard(b, args, "run_mc_text-item-%d" % (i + 1))
            # The text mechanism asks for the option's TEXT (a letter is
            # still accepted by the matcher). Asking for the letter is what
            # the primary mechanism does, and it is exactly what this model
            # ignores - the prompt has to meet the model where it answers.
            msgs = [{"role": "system", "content": SYSTEM_MC_TEXT},
                    {"role": "user", "content":
                     f"{t['question']}\n\n" +
                     "\n".join(f"{chr(ord('A') + j)}) {c}"
                               for j, c in enumerate(t["choices"])) +
                     "\n\nAnswer with the exact text of the correct option, "
                     "and nothing else.\nAnswer:"}]
            g = b.generate(msgs, max_tokens=48, temperature=0.0)
            idx, how = S.score_mc_text(g["content"], t["choices"])
            ok = (idx is not None and idx == t["answer"])
            out.append({
                "id": t["id"], "rep": rep, "category": t.get("category", "mc"),
                "lang": t.get("lang"), "subject": t.get("subject"),
                "question": t["question"], "choices": t["choices"],
                "gold_index": t["answer"],
                "chosen_index": idx, "chosen_letter":
                    (chr(ord("A") + idx) if idx is not None else None),
                "correct": ok, "pick_correct": ok,
                "mc_mechanism": "text_match",
                "match_kind": how,
                "completion_raw": g["content"],
                "unparsed": idx is None,
                "tokens": {"prompt": g["prompt_tokens"], "eval": g["eval_tokens"]},
                "speed": {"gen_tps": g["gen_tps"], "ttft_s": g["ttft_s"],
                          "wall_s": g["wall_s"]},
            })
            log(f"    [{i+1}/{n_tasks}] {t['id']:<28} {'OK ' if ok else 'X  '} "
                f"match={how:<18} {g['gen_tps']}tok/s")
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            out.append(_item_error(t, rep, "mc_text", e))
    return out


# Task name -> the task file it reads (the _txt variants reuse the same
# items; only the scoring mechanism differs).
TASK_ALIAS = {}

RUNNERS = {
    "code_humaneval": (run_code, "code"),
    "code_mbpp": (run_code, "code"),
    "bugfix": (run_code, "bugfix"),
    "algo": (run_code, "algo"),
    "refactor": (run_code, "refactor"),
    "code_rust": (run_rust, "rust"),
    "embedded_cpp": (run_cpp, "embedded_cpp"),
    "science": (run_mc, "science"),
    "electronics": (run_mc, "electronics"),
    "persian": (run_mc, "persian"),
    "english": (run_mc, "english"),
    "netdiag": (run_netdiag, "netdiag"),
    "devops": (run_devops, "devops"),
    "lint": (run_lint, "lint"),
    "agentic": (run_agentic, "agentic"),
    # Standard published benchmarks (tasks/std_*.json via standard_fetch.py).
    # MC suites reuse the single-token logprob runner; GSM8K uses generation.
    "std_arc_challenge": (run_mc, "std_arc_c"),
    "std_arc_easy": (run_mc, "std_arc_e"),
    "std_hellaswag": (run_mc, "std_hellaswag"),
    "std_winogrande": (run_mc, "std_winogrande"),
    "std_truthfulqa_mc": (run_mc, "std_truthfulqa"),
    "std_belebele_fa": (run_mc, "std_belebele_fa"),
    "std_mmlu": (run_mc, "std_mmlu"),
    "std_gsm8k": (run_gsm8k, "std_gsm8k"),
    # Stratified 30-item screening instances (tasks/std_*_s30.json via
    # standard_strata.py). Separate keys/categories so s30 numbers can never
    # mix with full-split numbers. Upstream files untouched.
    "std_arc_challenge_s30": (run_mc, "std_arc_challenge_s30"),
    "std_arc_easy_s30": (run_mc, "std_arc_easy_s30"),
    "std_hellaswag_s30": (run_mc, "std_hellaswag_s30"),
    "std_winogrande_s30": (run_mc, "std_winogrande_s30"),
    "std_truthfulqa_mc_s30": (run_mc, "std_truthfulqa_mc_s30"),
    "std_belebele_fa_s30": (run_mc, "std_belebele_fa_s30"),
    "std_mmlu_s30": (run_mc, "std_mmlu_s30"),
    "std_gsm8k_s30": (run_gsm8k, "std_gsm8k_s30"),
    # Full official high-school CS subject (100 items): deep-dive instrument
    # for CS gaps the s30 (n=5) can only suggest, never prove.
    "std_mmlu_cs": (run_mc, "std_mmlu_cs"),
}
ALL_TASKS = list(RUNNERS)
CUSTOM_TASKS = ["code_humaneval", "code_mbpp", "bugfix", "algo", "refactor",
                "code_rust", "embedded_cpp", "science", "electronics",
                "persian", "english", "netdiag", "devops", "lint", "agentic"]
STANDARD_TASKS = [t for t in ALL_TASKS if t.startswith("std_")]


# ============================================================ aggregation
MC_CATS = {"science", "electronics", "persian", "english",
           "std_arc_c", "std_arc_e", "std_hellaswag", "std_winogrande",
           "std_truthfulqa", "std_belebele_fa", "std_mmlu",
           "std_arc_challenge_s30", "std_arc_easy_s30", "std_hellaswag_s30",
           "std_winogrande_s30", "std_truthfulqa_mc_s30",
           "std_belebele_fa_s30", "std_mmlu_s30", "std_mmlu_cs"}
GSM8K_CATS = {"std_gsm8k", "std_gsm8k_s30"}
# The text-match MC families: the same items, the other scoring mechanism.
MC_CATS_TXT = {c + "_txt" for c in
               ("science", "electronics", "persian", "english",
                "std_arc_challenge_s30", "std_arc_easy_s30", "std_hellaswag_s30",
                "std_winogrande_s30", "std_truthfulqa_mc_s30",
                "std_belebele_fa_s30", "std_mmlu_s30", "std_mmlu_cs")}
CODE_CATS = {"code", "bugfix", "algo", "refactor", "rust"}


_CLOCK_OFFSET = [time.time() - time.monotonic()]


def note_suspend(where):
    """Detect a suspend/resume and say so out loud.

    The wall clock and the monotonic clock advance together while the
    machine is awake; a suspend advances only the wall clock. The gap is
    therefore exactly the time the box was asleep - and on 2026-10-20 it
    was 6h48m, invisible in the log until the wall-clock arithmetic was
    checked. A silent multi-hour stall in a benchmark is indistinguishable
    from a model that stopped answering, which is why this exists.
    """
    off = time.time() - time.monotonic()
    delta = off - _CLOCK_OFFSET[0]
    if delta > 30:
        log(f"  !! the machine was ASLEEP for {delta/60:.1f} min during {where} "
            f"(suspend/resume) - the item below was still scored on resume")
        _CLOCK_OFFSET[0] = off
    return delta


class PauseRun(Exception):
    """Raised when the machine must stop benchmarking for a while.

    The power policy belongs to the human, not to the harness. On a laptop
    the system suspends itself to preserve the battery, and a benchmark
    must NOT override that: it pauses cleanly at the last scored item and
    lets the operator plug the charger in. The run then continues with the
    same command - the resume is item-level, so nothing is lost.
    """


def battery_state():
    """(status, capacity_percent) of the first battery, or (None, None)."""
    for path in ("/sys/class/power_supply/BAT0",
                 "/sys/class/power_supply/BAT1",
                 "/sys/class/power_supply/Battery"):
        try:
            with open(os.path.join(path, "status")) as f:
                status = f.read().strip()
            with open(os.path.join(path, "capacity")) as f:
                cap = int(f.read().strip())
            return status, cap
        except (OSError, ValueError):
            continue
    return None, None


def check_power_policy(where, min_battery):
    """Pause the run if the machine is on a dying battery.

    Only ACTS when the battery is discharging and below the threshold: a
    charging machine is fine, a desktop with no battery is fine, and a
    healthy battery is fine. The system's own sleep/hibernate policy stays
    exactly as the operator configured it - this only avoids starting work
    that the power policy is about to interrupt.
    """
    status, cap = battery_state()
    if status is None or cap is None:
        return
    if status.lower() in ("discharging", "discharging-not-charging") \
            and cap < min_battery:
        raise PauseRun(
            f"battery at {cap}% and {status} (threshold {min_battery}%) - "
            f"pausing at {where}")
    if status.lower() in ("discharging", "discharging-not-charging"):
        # Once per task, not once per item: a per-item warning turns a
        # 350-item run into 350 identical lines.
        _warned = getattr(check_power_policy, "_warned", None)
        if _warned != cap:
            check_power_policy._warned = cap
            log(f"  (running on battery at {cap}% - the system may sleep to "
                f"save power; the run pauses cleanly at {min_battery}%)")


def pending_items(tasks, name, rep, done_ids):
    """Items of (task, rep) that still have to run.

    Item-level resume: an item is skipped only if a record for it is ALREADY
    in the file. A stub never matches (its id carries the #ERROR suffix), so a
    failed item always re-runs - and a completed item is never re-run, which
    is what makes a power cut cost one item instead of the whole run.
    """
    return [t for t in tasks if (name, rep, t.get("id")) not in done_ids]


def write_atomic(path, payload):
    """Write JSON so that a power cut can never corrupt the file.

    A plain open(path, "w") truncates first: a power cut during the write
    leaves a half-written file, and the next resume would find unparseable
    JSON and start the whole run from zero - the exact loss the checkpoint
    exists to prevent. Writing to a sibling temp file and renaming it is
    atomic on POSIX: the file on disk is always either the previous
    complete checkpoint or the new complete one.
    """
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        # A failed write must not leave debris next to the real file, and
        # must not touch the previous checkpoint.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    # fsync the directory so the rename itself survives a power cut.
    try:
        dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except (OSError, AttributeError):
        pass


def _is_stub(r):
    return ("ERROR" in str(r.get("id", ""))
            or r.get("stopped") in ("error", "exception"))


def split_superseded(records):
    """Separate stubs that a later successful attempt replaced.

    A stub means "this attempt failed". A rep-level stub (id "<task>#ERROR")
    marks a whole failed round; a per-item stub (id "<item>#ERROR") marks one
    failed item. When the same (task, rep) later produced valid records, the
    stub is historical evidence of the failed round, not a failure of the
    final result - counting it would print "14 errors" on a cell that is
    complete. Live stubs (items that never succeeded) stay in the records
    and are still reported as errors.
    """
    valid_by_key = {}
    for r in records:
        if _is_stub(r):
            continue
        valid_by_key.setdefault((r.get("task"), r.get("rep")), set()).add(
            r.get("id") or r.get("chain_id"))
    live, superseded = [], []
    for r in records:
        if not _is_stub(r):
            live.append(r)
            continue
        key = (r.get("task"), r.get("rep"))
        raw = str(r.get("id") or r.get("chain_id") or "")
        item = raw[:-6] if raw.endswith("#ERROR") else raw
        if item == r.get("task"):
            covered = bool(valid_by_key.get(key))          # rep-level stub
        else:
            covered = item in valid_by_key.get(key, ())     # per-item stub
        (superseded if covered else live).append(r)
    return live, superseded


def coverage_report(records, expected):
    """Which (task, rep) pairs lack full valid-record coverage.

    Coverage, not stub-presence, decides whether a result file is partial:
    a stub that a later attempt replaced with a real record must not keep
    the file "partial" forever, and a task that never ran must always count
    as a gap. Stubs are never coverage.
    """
    valid_ids = {}
    for r in records:
        if _is_stub(r):
            continue
        # Agentic records carry `chain_id`; every other family carries `id`.
        valid_ids.setdefault((r.get("task"), r.get("rep")), set()).add(
            r.get("id") or r.get("chain_id"))
    gaps = []
    for (tname, trep), n_exp in sorted(expected.items()):
        got_n = len(valid_ids.get((tname, trep), ()))
        if got_n < n_exp:
            gaps.append(f"{tname}/rep{trep}: {got_n}/{n_exp}")
    return gaps


# The second MC mechanism, as separate categories. A model that answers with
# the option text instead of the letter is not broken - it is measured with
# the mechanism that can read it, and its numbers live in different
# categories so nothing is ever ranked across mechanisms.
for _c in ("science", "electronics", "persian", "english",
           "std_arc_challenge_s30", "std_arc_easy_s30", "std_hellaswag_s30",
           "std_winogrande_s30", "std_truthfulqa_mc_s30",
           "std_belebele_fa_s30", "std_mmlu_s30", "std_mmlu_cs"):
    _n = _c + "_txt"
    RUNNERS[_n] = (run_mc_text, _n)
    TASK_ALIAS[_n] = _c


def aggregate(records):
    """Per-category scores with Wilson intervals, plus pass@k and pass^k."""
    by = {}
    for r in records:
        by.setdefault(r["category"], []).append(r)
    summary = {}
    for cat, rs in by.items():
        # ERROR stubs (harness/backend failures) are NOT model scores: they are
        # excluded from every denominator and counted separately. Scoring them
        # as failures corrupted real leaderboards (a dead server once zeroed a
        # model's code and agentic columns). Missing = missing, never zero.
        err_ids = {r.get("id") for r in rs
                   if "ERROR" in str(r.get("id", ""))
                   or r.get("stopped") in ("error", "exception")}
        n_errors = sum(1 for r in rs if r.get("id") in err_ids)
        rs = [r for r in rs if r.get("id") not in err_ids]
        n = len(rs)
        if n == 0:
            summary[cat] = {"n": 0, "n_items": 0, "n_samples": 0,
                            "accuracy": None,
                            "error": "no valid records: all are harness errors",
                            "n_errors": n_errors,
                            "error_ids": sorted(err_ids)}
            continue
        if cat in CODE_CATS or cat == "embedded_cpp":
            k = sum(1 for r in rs if r.get("passed"))
            n_samples = len({r["id"] for r in rs})
            n_correct = sum(1 for i in {r["id"] for r in rs} if any(x["id"] == i and x.get("passed") for x in rs))
            lo, hi = wilson(k, n)
            entry = {
                "n_items": n_samples, "n_samples": n,
                "passed": k, "accuracy": round(k / n, 4),
                "ci95_wilson": [round(lo, 4), round(hi, 4)],
                "pass_at_1": round(k / n, 4),
                "parse_fail_rate": round(sum(1 for r in rs if not r.get("syntax_ok", True)) / n, 4)
                                   if cat not in ("rust", "embedded_cpp") else
                                   round(sum(1 for r in rs if r.get("exec_error") == "empty_output") / n, 4),
            }
            if n_samples > 1:
                entry["pass_at_1_strict"] = round(n_correct / n_samples, 4)
            if cat == "rust":
                entry["wrote_fn_main_rate"] = round(
                    sum(1 for r in rs if r.get("wrote_fn_main")) / n, 4)
                cl = [r.get("clippy", {}).get("n_diagnostics") for r in rs
                      if isinstance(r.get("clippy", {}).get("n_diagnostics"), int)]
                if cl:
                    entry["clippy_mean_diags"] = round(sum(cl) / len(cl), 2)
                    entry["clippy_clean_rate"] = round(sum(1 for c in cl if c == 0) / len(cl), 4)
                fo = [r["rustfmt"].get("already_formatted") for r in rs
                      if r.get("rustfmt", {}).get("available")]
                if fo:
                    entry["rustfmt_already_formatted_rate"] = round(sum(1 for x in fo if x) / len(fo), 4)
            elif cat == "embedded_cpp":
                wn = [r.get("n_warnings") for r in rs if isinstance(r.get("n_warnings"), int)]
                if wn:
                    entry["gcc_warning_mean"] = round(sum(wn) / len(wn), 2)
                    entry["warning_free_rate"] = round(sum(1 for w in wn if w == 0) / len(wn), 4)
                entry["compile_error_rate"] = round(
                    sum(1 for r in rs if r.get("exec_error") == "compile_error") / n, 4)
                entry["assert_fail_rate"] = round(
                    sum(1 for r in rs if r.get("exec_error") == "assertion_failed") / n, 4)
            else:
                ld = [r["lint"].get("n_diagnostics") for r in rs
                      if isinstance(r.get("lint", {}).get("n_diagnostics"), int)]
                if ld:
                    entry["ruff_mean_diags"] = round(sum(ld) / len(ld), 2)
                    entry["ruff_clean_rate"] = round(sum(1 for d in ld if d == 0) / len(ld), 4)
                    entry["ruff_lint_density"] = round(
                        sum(r["lint"].get("lint_density", 0) for r in rs
                            if r.get("lint", {}).get("available")) / max(1, sum(1 for r in rs if r.get("lint", {}).get("available"))), 4)
                entry["ruff_noqa_gaming_rate"] = round(
                    sum(1 for r in rs if r.get("lint", {}).get("noqa_gaming")) / n, 4)
        elif cat in MC_CATS or cat in MC_CATS_TXT:
            # An accuracy without a null model is not interpretable: at n=14 a
            # 43% score sits inside the 95% interval of pure guessing. Every MC
            # figure therefore ships with the chance baseline, the smartest
            # zero-knowledge strategy's score, a one-sided exact binomial
            # p-value, and the accuracy REQUIRED at this n to exclude chance -
            # so no MC number can be read naked. See stats.py.
            entry = dict(mc_bundle(rs, None))
            entry["mc_mechanism"] = ("text_match" if cat in MC_CATS_TXT
                                     else "letter_logprob")
            # Pooled reps inflate n: Wilson/p-value assume independent items.
            # Report both so a 35x2 pooled n=70 is never read as 70 items.
            entry["n_items"] = len({r.get("id") for r in rs})
            entry["n_samples"] = n
            entry["note"] = ("n pools reps; CIs/p-values computed on n_samples "
                             "and are optimistic vs n_items")
            if cat in MC_CATS_TXT:
                # Letter-mass confidence is a letter-logprob concept; the text
                # mechanism reports its own failure mode instead.
                entry["unparsed_rate"] = round(
                    sum(1 for r in rs if r.get("unparsed")) / n, 4)
                entry["match_kinds"] = {
                    k: sum(1 for r in rs if r.get("match_kind") == k)
                    for k in ("letter", "text", "text_in_sentence")}
            else:
                entry["mean_confidence"] = round(
                    sum(r.get("confidence", 0) for r in rs) / n, 4)
                entry["low_information_rate"] = round(
                    sum(1 for r in rs if not r.get("informed", True)) / n, 4)
            entry["tokens_per_item"] = round(
                sum(r["tokens"]["eval"] for r in rs) / n, 2)
            # Degenerate-answerer control: if a model answered one letter every
            # time, what would it score? Reported per model so a high score
            # from a letter-monotonic model is visible as such.
            picks = [r.get("chosen_letter") for r in rs if r.get("chosen_letter")]
            if picks:
                from collections import Counter as _C
                top_letter, top_cnt = _C(picks).most_common(1)[0]
                gold_letter_hits = sum(
                    1 for r in rs if r.get("chosen_letter") == r.get("chosen_letter")
                    and r.get("correct") and r.get("chosen_letter") == top_letter)
                entry["most_common_pick"] = top_letter
                entry["most_common_pick_rate"] = round(top_cnt / len(picks), 4)
                entry["accuracy_from_single_letter"] = round(gold_letter_hits / n, 4)
            subs = {}
            for r in rs:
                s = subs.setdefault(r.get("subject") or "all", [0, 0])
                s[0] += 1
                s[1] += 1 if r.get("correct") else 0
            entry["by_subject"] = {k2: {"n": v[0], "acc": round(v[1] / v[0], 4)}
                                   for k2, v in sorted(subs.items())}
        elif cat in GSM8K_CATS:
            k = sum(1 for r in rs if r.get("correct"))
            lo, hi = wilson(k, n)
            n_items = len({r.get("id") for r in rs})
            entry = {"n": n, "n_items": n_items, "n_samples": n,
                     "correct": k, "accuracy": round(k / n, 4),
                     "ci95_wilson": [round(lo, 4), round(hi, 4)],
                     "empty_pred_rate": round(
                         sum(1 for r in rs if not (r.get("pred") or "")) / n, 4),
                     "protocol": "0-shot #### extraction (paper uses 8-shot CoT)"}
        elif cat == "netdiag":
            mean = sum(r["score"] for r in rs) / n
            lo, hi = wilson(sum(1 for r in rs if r["score"] == 1.0), n)
            fa = [r for r in rs if not r["is_anomaly"]]
            an = [r for r in rs if r["is_anomaly"]]
            entry = {"n": n, "mean_score": round(mean, 4),
                     "ci95_wilson_perfect": [round(lo, 4), round(hi, 4)],
                     "false_premise_n": len(fa),
                     "false_premise_correct": sum(1 for r in fa if r["score"] == 1.0),
                     "false_alarm_rate": round(
                         sum(1 for r in fa if not r["verdict_ok"]) / max(1, len(fa)), 4),
                     "real_fault_detected_rate": round(
                         sum(1 for r in an if r["verdict_ok"]) / max(1, len(an)), 4)}
        elif cat == "devops":
            # .get(...,0.0) so an error stub record without a score is counted as
            # zero rather than raising KeyError after all inference is done.
            m = sum(r.get("score", 0.0) for r in rs) / n
            k = sum(1 for r in rs if r.get("passed"))
            lo, hi = wilson(k, n)
            entry = {"n": n, "mean_score": round(m, 4),
                     "fully_correct": k, "exact_rate": round(k / n, 4),
                     "ci95_wilson": [round(lo, 4), round(hi, 4)],
                     "empty_output_rate": round(sum(1 for r in rs if r.get("empty")) / n, 4)}
        elif cat == "lint":
            nd = [r["n_diagnostics"] for r in rs if isinstance(r.get("n_diagnostics"), int)]
            lo, hi = wilson(sum(1 for r in rs if r.get("n_diagnostics") == 0), n)
            entry = {"n": n,
                     "mean_diagnostics": round(sum(nd) / max(1, len(nd)), 3),
                     "median_diagnostics": sorted(nd)[len(nd) // 2] if nd else None,
                     "clean_rate": round(sum(1 for r in rs if r.get("n_diagnostics") == 0) / n, 4),
                     "ci95_wilson_clean": [round(lo, 4), round(hi, 4)],
                     "noqa_gaming_rate": round(sum(1 for r in rs if r.get("noqa_gaming")) / n, 4),
                     "empty_output_rate": round(sum(1 for r in rs if r.get("empty")) / n, 4)}
            for lang, key in (("python", "python_ruff"), ("rust", "rust_clippy")):
                # .get() not []: the BackendError stub record has no 'lang' key,
                # and indexing it raised KeyError inside aggregate() after the
                # inference had already been paid for.
                sub = [r for r in rs if r.get("lang") == lang]
                if sub:
                    # .get() may be None for empty outputs; those are excluded
                    # from the mean (they are still penalised in clean_rate via
                    # the full-n denominator below, since None != 0).
                    sub_nd = [r["n_diagnostics"] for r in sub
                              if isinstance(r.get("n_diagnostics"), int)]
                    entry[f"{lang}_mean_diagnostics"] = round(
                        sum(sub_nd) / max(1, len(sub_nd)), 3)
                    entry[f"{lang}_clean_rate"] = round(
                        sum(1 for r in sub if r.get("n_diagnostics") == 0) / len(sub), 4)
                    fo = [r["detail"].get("format", {}).get("already_formatted") for r in sub
                          if r["detail"].get("format", {}).get("available")]
                    if fo:
                        entry[f"{lang}_already_formatted_rate"] = round(
                            sum(1 for x in fo if x) / len(fo), 4)
            for s in ("pedantic", "nursery", "correctness", "style", "complexity", "perf"):
                vals = [r["detail"]["clippy"]["by_set"].get(s, {}).get("n") for r in rs
                        if r.get("lang") == "rust" and isinstance(
                            (r.get("detail") or {}).get("clippy", {}).get("by_set", {})
                            .get(s, {}).get("n"), int)
                        and r["detail"]["clippy"]["by_set"][s]["n"] >= 0]
                if vals:
                    entry[f"clippy_{s}_mean"] = round(sum(vals) / len(vals), 3)
        elif cat == "agentic":
            n_items = len({r["chain_id"] for r in rs})
            # The headline counts a chain as solved ONLY if the model used the
            # tools and completed every expected step AND got the answer right.
            #
            # It used to be `outcome_ok` alone, which is scored purely from the
            # final answer TEXT. A model that never calls a tool and recites a
            # memorised answer scored pass_hat_k_reliability = 1.0 with
            # trace 0/16 and mean_tool_calls 0.0 - so the suite's headline
            # measured knowledge, not the multi-call tool use it exists to
            # measure. Verified against a mock backend, not inferred.
            def _solved(r):
                # No-call (irrelevance) chains invert the tool requirement: the
                # correct behaviour is a grounded refusal with ZERO calls, so
                # the positive `n_tool_calls > 0` gate would make them unsolvable
                # by construction. Solved = refusal verdict AND nothing called.
                if r.get("expect_no_calls"):
                    return bool(r.get("outcome_ok")
                                and (r.get("n_tool_calls") or 0) == 0)
                return bool(r.get("outcome_ok")
                            and r.get("trace_ok") == r.get("trace_total")
                            and (r.get("n_tool_calls") or 0) > 0)
            k_items = sum(1 for i in {r["chain_id"] for r in rs}
                          if all(_solved(x) for x in rs if x["chain_id"] == i))
            traces = [r["trace_ok"] / max(1, r["trace_total"]) for r in rs]
            # pass^k: fraction of chains where EVERY repeat succeeded
            pk = k_items / n_items if n_items else 0
            entry = {
                "n_chains": n_items, "n_trials": n,
                # headline: requires tool use
                "per_trial_outcome_rate": round(sum(1 for r in rs if _solved(r)) / n, 4),
                "per_trial_trace_rate": round(sum(traces) / n, 4),
                "pass_hat_k_reliability": round(pk, 4),
                # secondary: final-answer text alone, kept because it separates
                # "did not know" from "knew but did not use the tools", which is
                # a different failure and worth seeing.
                "final_answer_accuracy_no_tools_required": round(
                    sum(1 for r in rs if r.get("outcome_ok")) / n, 4),
                "pass_hat_1": round(sum(1 for r in rs if _solved(r)) / n, 4),
                "reps_per_chain": n / n_items if n_items else 0,
                "halt_without_first_tool_call_rate": round(
                    sum(1 for r in rs if r["stopped"] == "no_tool_call") / n, 4),
                "mean_tool_calls": round(sum(r["n_tool_calls"] for r in rs) / n, 2),
                # Refusal (no-call) split: reported separately so a model that
                # calls on everything cannot hide behind the positive chains,
                # and a model that refuses everything cannot hide behind these.
                "neg_n_chains": sum(1 for i in {r["chain_id"] for r in rs}
                                    if any(x.get("expect_no_calls") for x in rs
                                           if x["chain_id"] == i)),
                "neg_refusal_rate": round(
                    sum(1 for r in rs if r.get("expect_no_calls") and _solved(r)) /
                    max(1, sum(1 for r in rs if r.get("expect_no_calls"))), 4),
                "fabrication_rate": round(
                    sum(1 for r in rs if r["stopped"] == "final_answer" and r["trace_ok"] == 0) / n, 4),
                "mean_gen_tps": round(
                    sum(c["gen_tps"] or 0 for r in rs for c in r["calls"]) /
                    max(1, sum(len(r["calls"]) for r in rs)), 2),
            }
            lo, hi = wilson(k_items, n_items)
            entry["ci95_wilson"] = [round(lo, 4), round(hi, 4)]
        else:
            continue
        tps = [r["speed"]["gen_tps"] for r in rs if r.get("speed", {}).get("gen_tps")]
        # For single-token MC, decode rate is meaningless: 1 token in a
        # microsecond of eval_duration reads as 1e6 tok/s. The wall-clock time
        # per item is the honest number there, since it is dominated by prompt
        # processing, which is what actually costs something on this hardware.
        if tps and cat not in MC_CATS and cat not in MC_CATS_TXT:
            entry["gen_tps_mean"] = round(sum(tps) / len(tps), 2)
            entry["gen_tps_median"] = sorted(tps)[len(tps) // 2]
        walls = [r["speed"]["wall_s"] for r in rs if r.get("speed", {}).get("wall_s")]
        if walls:
            entry["wall_s_per_item"] = round(sum(walls) / len(walls), 2)
        if cat in MC_CATS and tps:
            entry["gen_tps_mean_NA"] = "single-token scoring; use wall_s_per_item"
        ev = [r["tokens"]["eval"] for r in rs if r.get("tokens", {}).get("eval")]
        if ev:
            entry["eval_tokens_mean"] = round(sum(ev) / len(ev), 1)
        entry["n_errors"] = n_errors
        if n_errors:
            entry["error_ids"] = sorted(err_ids)
        summary[cat] = entry
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "llamacpp"], required=False,
                    help="not needed with --resume (taken from the file)")
    ap.add_argument("--model", required=False,
                    help="not needed with --resume (taken from the file)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model-path", default=None,
                    help="local weight file for digesting (llamacpp/GGUF): "
                         "its sha256 is recorded in provenance")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="max items per task (smoke)")
    ap.add_argument("--subsample", type=int, default=0,
                    help="deterministically sample N items per generation-heavy task")
    ap.add_argument("--per-task-cap", default="",
                    help="per-task overrides, e.g. code_mbpp=60,agentic=8")
    ap.add_argument("--reps-for", default="",
                    help="per-task rep overrides, e.g. code_mbpp=1,agentic=3")
    ap.add_argument("--agentic-reps", type=int, default=0,
                    help="reps for the agentic suite only (it is cheap and its "
                         "variance is the whole point, so it wants more reps)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--offline", action="store_true",
                    help="refuse network-dependent paths: standard task files "
                         "must already exist (see standard_fetch.py --check)")
    ap.add_argument("--timeout", type=int, default=900,                    help="per-request seconds (default 900; qualifier uses 300: "
                         "a hung generation must not cost 30 min twice)")
    ap.add_argument("--min-battery", type=int, default=30,
                    help="pause the run when the machine is on battery below "
                         "this percentage (default 30). The system power "
                         "policy is never overridden: the run pauses cleanly "
                         "and resumes item-level after the charger is in.")
    ap.add_argument("--probe-only", action="store_true",
                    help="run backend self-qualification and exit (no tasks)")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip backend self-qualification (you accept unmeasured "
                         "pairing risk; results are marked unprobed)")
    ap.add_argument("--server-mem-budget", type=int, default=5500, metavar="MIB",
                    help="abort the run (keeping checkpoints) when our llama-server "
                         "exceeds this RSS in MiB. 0 disables. RSS includes mmap'd "
                         "weights, so size it for weights + headroom: 5500 fits a "
                         "2.7GB model on a 7.6GB box with margin before OOM. "
                         "Deliberately NOT part of config_hash: it changes abort "
                         "behavior, never scores.")
    ap.add_argument("--resume", default=None, metavar="RESULTS_JSON",
                    help="resume this file: rerun only missing/stubbed "
                         "(task, rep) pairs using the file's own config. "
                         "Connection (--backend/--model/--port) must match; "
                         "task flags are ignored")
    args = ap.parse_args()

    resumed_cfg = None
    if not args.resume and (not args.backend or not args.model):
        raise SystemExit("need --backend + --model (or --resume a results file)")
    if args.resume:
        if not os.path.exists(args.resume):
            raise SystemExit(f"nothing to resume: {args.resume} not found")
        prev0 = json.load(open(args.resume))
        resumed_cfg = prev0.get("config") or {}
        for k in ("backend", "model", "port"):
            cli = {"backend": args.backend, "model": args.model,
                   "port": args.port}[k]
            if cli is not None and cli != resumed_cfg.get(k):
                raise SystemExit(
                    f"--resume refuses: --{k} {cli} != file's {resumed_cfg.get(k)} "
                    f"(resuming across setups poisons comparability)")
        args.backend = resumed_cfg["backend"]
        args.model = resumed_cfg["model"]
        args.port = resumed_cfg.get("port")
        args.tasks = ",".join(resumed_cfg.get("tasks", []))
        args.reps = resumed_cfg.get("reps", args.reps)
        args.limit = resumed_cfg.get("limit", 0)
        args.subsample = resumed_cfg.get("subsample", 0)
        args.offline = resumed_cfg.get("offline", False)
        args.timeout = resumed_cfg.get("timeout", 900)
        args.per_task_cap = ",".join(
            f"{k}={v}" for k, v in (resumed_cfg.get("caps_per_task") or {}).items())
        rep_for_resumed = {k: v for k, v in
                           (resumed_cfg.get("reps_per_task") or {}).items()
                           if k in RUNNERS}
        args.agentic_reps = 0
        args.out = args.resume
        log(f"resume mode: {len(prev0.get('records', []))} records present, "
            f"only gaps will run")

    names = ALL_TASKS if args.tasks == "all" else args.tasks.split(",")
    bad = [n for n in names if n not in RUNNERS]
    if bad:
        raise SystemExit(f"unknown task(s): {bad}\navailable: {ALL_TASKS}")

    caps = {}
    for kv in filter(None, args.per_task_cap.split(",")):
        k, _, v = kv.partition("=")
        if k.strip() in RUNNERS and v.strip().isdigit():
            caps[k.strip()] = int(v)

    rep_for = {}
    for kv in filter(None, args.reps_for.split(",")):
        k, _, v = kv.partition("=")
        if k.strip() in RUNNERS and v.strip().isdigit():
            rep_for[k.strip()] = int(v)
    if args.agentic_reps and "agentic" in names:
        rep_for["agentic"] = args.agentic_reps
    if resumed_cfg:
        rep_for.update(rep_for_resumed)

    port = args.port or (11434 if args.backend == "ollama" else 8090)
    b = Backend(args.backend, args.model, port=port, api_key=args.api_key,
                timeout=args.timeout,
                num_ctx=FROZEN["num_ctx"], n_gpu_layers=FROZEN["num_gpu_layers"],
                model_path=args.model_path)
    if not b.alive():
        raise SystemExit(f"no server answering on {b.host}:{b.port}")
    import inspect as _inspect
    _tp = _inspect.signature(Backend.generate).parameters["top_p"].default
    assert _tp == FROZEN["gen_top_p"], \
        f"backend top_p default {_tp} != FROZEN {FROZEN['gen_top_p']}: sampling differs from config_hash"

    provenance = {"model_digest": b.model_digest(), "versions": b.server_versions()}
    try:
        provenance["cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                             if l.startswith("model name")][0]
        provenance["kernel"] = __import__("platform").release()
    except Exception:  # noqa: BLE001
        pass

    # Backend self-qualification: prove each I/O mechanism works BEFORE hours
    # of inference. Broken MC scoring or tool parsing would otherwise produce
    # zeros that look exactly like model failures (seen live, twice).
    import probes as _P  # noqa: E402
    skipped = []
    if args.probe_only or not args.no_probe:
        log("probing backend mechanisms (generation, logprob, tools)...")
        probes = _P.qualify(b)
        for k, (ok, detail) in probes.items():
            log(f"  probe {k:<11} {'OK  ' if ok else 'BROKEN'} "
                f"{json.dumps(detail)[:140]}")
        provenance["backend_probes"] = {
            k: {"ok": ok, **detail} for k, (ok, detail) in probes.items()}
        # server version surfaces on first generate (system_fingerprint)
        provenance["versions"] = b.server_versions()
    else:
        probes = {}
        provenance["backend_probes"] = {"skipped": True}
    if args.probe_only:
        log("probe-only: exiting before tasks")
        return
    if resumed_cfg:
        if probes:
            log("resume mode: probes are advisory only, task list unchanged")
    elif not args.no_probe:
        gen_ok = probes.get("generation", (False,))[0]
        if not gen_ok:
            raise SystemExit("probe FAILED: backend cannot generate — aborting, "
                             "no scores to collect")
        skipped = []
        if not probes.get("mc_logprob", (False,))[0]:
            # Only the letter-logprob families are unmeasurable. The _txt
            # families read a GENERATED answer, so they need the generation
            # probe (which passed) - dropping them here would throw away a
            # perfectly measurable suite.
            dropped = [n for n in names
                       if RUNNERS[n][1] in MC_CATS and not n.endswith("_txt")]
            if dropped:
                skipped += [(n, "mc_logprob probe failed: letters carry no mass "
                               "(this model answers with the option text, not "
                               "the letter - measure it with the _txt families)") for n in dropped]
                names = [n for n in names if n not in dropped]
        if not probes.get("tools", (False,))[0]:
            dropped = [n for n in names if RUNNERS[n][1] == "agentic"]
            if dropped:
                skipped += [(n, "tools probe failed: no parsed tool call") for n in dropped]
                names = [n for n in names if n not in dropped]
        for n, reason in skipped:
            log(f"  SKIP {n}: {reason} (missing, never zero)")
        if not names:
            raise SystemExit("probe FAILED every requested family — nothing "
                             "measurable, aborting")
    else:
        # --no-probe: pairing risk unmeasured (logged below via provenance flag)
        log("WARNING: --no-probe, pairing risk unmeasured")
    log(f"digest   : {provenance['model_digest']}")
    log(f"versions : {provenance['versions']}")

    label = args.label or args.model.replace("/", "_").replace(":", "_")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.out or os.path.join(RESULTS_DIR, f"{label}.json")

    if resumed_cfg:
        # Adopt the file's config verbatim: recomputing would inject newer
        # keys (e.g. timeout) and the hash could never match an older file,
        # silently restarting instead of resuming.
        cfg = dict(resumed_cfg)
    else:
        cfg = dict(FROZEN)
        cfg["backend"] = args.backend
        cfg["port"] = port
        cfg["model"] = args.model
        cfg["reps"] = args.reps
        cfg["reps_per_task"] = rep_for or {"(all)": args.reps}
        cfg["caps_per_task"] = caps
        cfg["tasks"] = names
        cfg["limit"] = args.limit
        cfg["subsample"] = args.subsample
        cfg["offline"] = bool(args.offline)
        cfg["timeout"] = args.timeout
    if args.offline:
        missing = [n for n in names
                   if n.startswith("std_") and not os.path.exists(
                       os.path.join(TASKS_DIR, n + ".json"))]
        if missing:
            raise SystemExit(
                f"offline and missing standard task files: {missing}\n"
                f"run standard_fetch.py --fetch ... online first, or "
                f"standard_fetch.py --check to audit the cache")
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]

    log("=== llmbench run ===")
    log(f"model    : {args.model}  via {args.backend}:{port}")
    log(f"tasks    : {', '.join(names)}")
    log(f"reps     : {args.reps}   limit: {args.limit or 'none'}")
    log(f"config   : {cfg_hash}")
    log(f"out      : {out_path}")

    # record peak memory so a swap-thrashing run is distinguishable from a slow one
    def meminfo():
        d = {}
        for ln in open("/proc/meminfo"):
            k, _, v = ln.partition(":")
            d[k] = int(v.split()[0])
        return d

    mi = meminfo()
    log(f"mem      : {mi['MemTotal']//1024} MiB total, {mi['MemAvailable']//1024} MiB available, "
        f"swap free {mi.get('SwapFree',0)//1024} MiB")

    # t_start is a MONOTONIC base (for elapsed); t_wall is the wall clock.
    # They were conflated: "started_unix" held monotonic seconds, so every
    # result file claimed to start in 1970 and no run could be dated.
    records, t_start, t_wall = [], time.monotonic(), time.time()
    min_battery = getattr(args, "min_battery", 30)
    server_dead = False
    consec_err = 0
    # Coverage ledger: expected (task, rep) -> item count, and the set of
    # record objects already streamed to disk by the per-item progress hook.
    expected = {}
    appended = set()
    _install_signal_handlers()
    # Resume: a dead run restarts mid-model instead of from zero. Completed
    # (task, rep) pairs are read back from the existing output file - but ONLY
    # if its config_hash matches, so a config change never silently inherits
    # another config's records. Resume is ITEM-level: completed item ids are
    # skipped individually, so a kill mid-rep loses nothing already recorded
    # (a BackendError stub never marks its items done - its id differs).
    done_ids = set()
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            if prev.get("config_hash") == cfg_hash:
                records = list(prev.get("records", [])) \
                    + list(prev.get("superseded_stubs", []))
                for r in records:
                    done_ids.add((r.get("task"), r.get("rep"),
                                  r.get("id") or r.get("chain_id")))
                log(f"resume: {len(records)} records from matching config {cfg_hash}")
            else:
                log(f"config changed ({prev.get('config_hash')} -> {cfg_hash}); "
                    f"starting fresh, old file overwritten at first checkpoint")
        except Exception as e:  # noqa: BLE001
            log(f"resume: unreadable {out_path} ({type(e).__name__}); starting fresh")
            records = []
    for ti, name in enumerate(names):
        if server_dead:
            log(f"  skipping {name}: server is dead, run aborted here")
            break
        fn, cat = RUNNERS[name]
        # The agentic chains live in agentic.py, not a task file: they carry a
        # live tool environment, so they cannot be serialised as static prompts.
        if name == "agentic":
            tasks = [{"id": c["id"]} for c in agentic.CHAINS]
        else:
            tasks = load_tasks(name)
        total_available = len(tasks)
        tasks = subsample(tasks, caps.get(name, 0) or args.subsample)
        if args.limit:
            tasks = tasks[:args.limit]
        reps_here = rep_for.get(name, args.reps)
        for _r in range(reps_here):
            expected[(name, _r)] = len(tasks)
        log(f"--- [{ti+1}/{len(names)}] {name} ({len(tasks)}/{total_available} items x {reps_here} reps) ---")

        def checkpoint():
            live, superseded = split_superseded(records)
            agg = aggregate(live)
            # The checkpoint MUST carry the partial flag: a checkpoint is what
            # a killed run leaves behind, and the resume loop decides whether
            # to keep going by reading exactly this file.
            gaps = coverage_report(live, expected)
            payload = {"schema": "llmbench/1", "config": cfg, "config_hash": cfg_hash,
                       "started_unix": t_wall,
                       "started_monotonic": t_start,
                       "elapsed_s": round(time.monotonic() - t_start, 1),
                       "provenance": provenance,
                       "summary": agg, "records": live,
                       "superseded_stubs": superseded,
                       "coverage_gaps": gaps,
                       "partial": bool(gaps) or bool(server_dead),
                       "backend_totals": b.summary()}
            write_atomic(out_path, payload)
            log(f"  [checkpoint written: {len(records)} records]")

        # A suspend/OOM kill must still leave the file with everything scored
        # so far, so the signal handler writes the same checkpoint.
        CHECKPOINT_HOOK["fn"] = lambda: checkpoint()

        t0 = time.monotonic()
        for rep in range(reps_here):
            pending = pending_items(tasks, name, rep, done_ids)
            if not pending:
                log(f"  rep {rep + 1}/{reps_here} already complete; skipping")
                continue
            # Completed items are KEPT (not dropped): resume is item-level, so
            # a kill mid-rep loses nothing already recorded. Stubs never match
            # pending (their ids differ), so failed items always re-run.
            log(f"  rep {rep + 1}/{reps_here} "
                f"({len(pending)}/{len(tasks)} pending)")
            # Every task reports item-by-item: each finished item is written
            # to disk immediately, so a suspend/OOM/kill can only cost the
            # item in flight. A 19-chain agentic rep and a 30-item math rep
            # are both far too much evidence to hold in memory.
            def prog(r, _cat=cat, _name=name, _rep=rep):
                note_suspend(f"{name}/rep{rep} item {r.get('id')}")
                r["task_category"] = r.get("category")
                r["category"] = _cat
                r["task"] = _name
                r["rep"] = _rep
                r.setdefault("id", r.get("chain_id"))
                records.append(r)
                appended.add(id(r))
                if "ERROR" not in str(r.get("id", "")):
                    done_ids.add((_name, _rep, r.get("id")))
                checkpoint()
                # AFTER the checkpoint: a pause must never cost the item.
                try:
                    check_power_policy(f"{_name} item {r.get('id')}", min_battery)
                except PauseRun as e:
                    log(f"PAUSED: {e}")
                    log("  the run stops here; plug the charger in and re-run "
                        "the same command to continue from the last item")
                    os._exit(75)
            attempt = 0
            while True:
                try:
                    got = fn(b, pending, args, rep, progress=prog)
                    break
                except BackendError as e:
                    # One retry after evicting the runner and cooling down: last
                    # night a runner OOM crash at rep level wiped 50 items into
                    # a single stub record, twice. A retry costs minutes; the
                    # stub costs the whole rep's statistical value.
                    # BUT: if the server is dead (OOM-killed, connection
                    # refused), retrying burns another full timeout and every
                    # later task stubs identically. Probe liveness first.
                    attempt += 1
                    log(f"  !! backend error: {e}")
                    if "OOM-GUARD" in str(e):
                        log("  !! over-budget server: retry would burn a full "
                            "timeout against the same fat server — stubbing now")
                        attempt = 99
                    if attempt > 1:
                        got = [{"id": f"{name}#ERROR", "rep": rep, "category": cat,
                                "error": str(e)[:300], "passed": False, "correct": False,
                                "score": 0.0, "n_diagnostics": None, "noqa_gaming": False,
                                "chain_id": "ERR", "outcome_ok": False, "trace_ok": 0,
                                "trace_total": 1, "n_tool_calls": 0, "stopped": "error",
                                "calls": [], "tokens": {"eval": 0}, "speed": {"gen_tps": None}}]
                        if not b.alive():
                            log("  !! server is dead — aborting remaining tasks, "
                                "keeping checkpoints (resume after restarting it)")
                            server_dead = True
                        consec_err += 1
                        if consec_err >= 3 and not server_dead:
                            log("  !! 3 consecutive backend failures — server is "
                                "wedged, aborting remaining tasks")
                            server_dead = True
                        break
                    if not b.alive():
                        log("  !! server is dead — skipping retry, recording stub")
                        got = [{"id": f"{name}#ERROR", "rep": rep, "category": cat,
                                "error": str(e)[:300], "passed": False, "correct": False,
                                "score": 0.0, "n_diagnostics": None, "noqa_gaming": False,
                                "chain_id": "ERR", "outcome_ok": False, "trace_ok": 0,
                                "trace_total": 1, "n_tool_calls": 0, "stopped": "error",
                                "calls": [], "tokens": {"eval": 0}, "speed": {"gen_tps": None}}]
                        log("  !! server is dead — aborting remaining tasks, "
                            "keeping checkpoints (resume after restarting it)")
                        server_dead = True
                        break
                    log("  evicting runner, 30s cooldown, then retrying rep once")
                    b.evict()
                    time.sleep(30)
                except Exception as e:  # noqa: BLE001
                    # Any exception other than BackendError used to propagate and kill
                    # the process. Checkpoints are per-rep, so an abort discards at
                    # most one rep - but a scorer bug is still a bug: record it and
                    # move on. This is not hypothetical: a NameError in a scorer
                    # fallback path (backend's letter-less logprob branch) once
                    # discarded hours of inference.
                    # Every item is scored independently, so one bad item must never be
                    # able to end the run.
                    import traceback as _tb
                    log(f"  !! UNEXPECTED {type(e).__name__} in {name} rep {rep + 1}: {e}")
                    log("  " + " | ".join(_tb.format_exc().strip().splitlines()[-3:]))
                    got = [{"id": f"{name}#ERROR", "rep": rep, "category": cat,
                            "error": f"{type(e).__name__}: {e}"[:300], "passed": False,
                            "correct": False, "score": 0.0, "n_diagnostics": None,
                            "noqa_gaming": False, "chain_id": "ERR", "outcome_ok": False,
                            "trace_ok": 0, "trace_total": 1, "n_tool_calls": 0,
                            "stopped": "exception", "calls": [],
                            "tokens": {"eval": 0}, "speed": {"gen_tps": None}}]
                    break
            for r in got:
                if id(r) in appended:
                    continue  # already streamed to disk by the per-item hook
                # The RUNNERS table is authoritative for the category. Letting a
                # record keep its own `category` silently merged bugfix/algo/
                # refactor into "code" and left science_mc/fa_mc unrecognised,
                # which is invisible in the output but wrong in every summary.
                r["task_category"] = r.get("category")
                r["category"] = cat
                r["task"] = name
                records.append(r)
            if not any("ERROR" in str(r.get("id", "")) for r in got):
                consec_err = 0  # clean rep resets the circuit breaker
            el = time.monotonic() - t0
            log(f"  rep {rep + 1} done in {el/60:.1f} min  (elapsed {el/60:.1f} min)")
            checkpoint()           # per-rep: a dead run loses one rep, not the task
        # Bound runner memory growth between tasks (measured 2.8GB -> 6.8GB
        # across one sweep, ending in an OOM kill). Reload is ~10s of mmap.
        log(f"  evicting runner between tasks ({b.evict()})")
        rss = b.server_rss_mb()
        if rss:
            log(f"  server RSS: {rss} MiB")
            if 0 < args.server_mem_budget < rss:
                log(f"  !! server RSS {rss} MiB over budget "
                    f"{args.server_mem_budget} MiB — aborting remaining tasks, "
                    f"restart the server and resume")
                server_dead = True
                break
        time.sleep(15)

    live_records, superseded_stubs = split_superseded(records)
    coverage_gaps = coverage_report(live_records, expected)
    if coverage_gaps:
        log(f"  coverage gaps (file is PARTIAL): {'; '.join(coverage_gaps)}")

    mi = meminfo()
    result = {
        "schema": "llmbench/1",
        "config": cfg, "config_hash": cfg_hash,
        "started_unix": t_wall, "started_monotonic": t_start,
        "elapsed_s": round(time.monotonic() - t_start, 1),
        "provenance": provenance,
        "host": {"mem_total_mib": mi["MemTotal"] // 1024,
                 "mem_available_end_mib": mi["MemAvailable"] // 1024,
                 "swap_free_end_mib": mi.get("SwapFree", 0) // 1024},
        "backend_totals": b.summary(),
        "summary": aggregate(live_records),
        "records": live_records,
        # Stubs from attempts that a later run replaced: kept as evidence of
        # the failed rounds, never counted against the final result.
        "superseded_stubs": superseded_stubs,
        # partial = the file does not cover the requested work. Decided by
        # COVERAGE (valid records per task/rep vs items expected), not by the
        # mere presence of an old stub: a stub that a later attempt replaced
        # with a real record must not keep the file "partial" forever.
        "partial": bool(server_dead) or bool(coverage_gaps),
        "coverage_gaps": coverage_gaps,
        "server_dead": bool(server_dead),
        # Skipped-by-probe suites: explicitly unmeasured, never zero.
        "skipped": [{"task": n, "reason": reason} for n, reason in skipped],
    }
    if result["partial"]:
        log("WARNING: PARTIAL run — stubbed/unrun tasks present; "
            "resume after fixing the backend, do not quote these scores")
    write_atomic(out_path, result)
    log(f"=== done in {(time.monotonic()-t_start)/60:.1f} min -> {out_path}")
    for cat, s in result["summary"].items():
        log(f"    {cat:<16} {json.dumps({k: v for k, v in s.items() if k != 'by_subject'})[:190]}")


if __name__ == "__main__":
    main()