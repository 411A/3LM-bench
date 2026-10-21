#!/usr/bin/env python3
"""Unit gate: pure-function contracts of scorers/stats/aggregate. No model,
no network, no toolchain. Fails loudly (non-zero exit) on any violation.

Covers: wilson, pass_at_k/pass_hat_k, extract_code, extract_math_answer +
score_gsm8k, normalize_fa, netdiag_verdict/score_netdiag, check_devops
(keyword echo MUST fail, genuine artifact MUST pass), mc_bundle chance math,
stub exclusion in aggregate.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scorers as S
from stats import mc_bundle, binom_sf
import run as R

fails = []


def check(label, cond):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}")
    if not cond:
        fails.append(label)


# stats
lo, hi = S.wilson(8, 32)
check("wilson(8,32) inside [0,1] and ordered", 0 <= lo < 0.25 < hi <= 1)
check("wilson(0,0) is (0,1)", S.wilson(0, 0) == (0.0, 1.0))
check("pass_at_k(4,1,1)==0.25", abs(S.pass_at_k(4, 1, 1) - 0.25) < 1e-9)
check("pass_hat_k all-fail is 0 (not inverted)",
      S.pass_hat_k(4, 0, 2) == 0.0)
check("binom_sf certain event is 1", binom_sf(0, 10, 0.25) == 1.0)

# code extraction
check("tagged block preferred",
      "return a" in S.extract_code("x\n```python\ndef f():\n    return a\n```", "python"))
check("prose is not code", S.extract_code("I cannot help with that.") == "")
check("last block wins",
      S.extract_code("```python\ndef f():\n    return 1\n```\n```python\ndef f():\n    return 2\n```",
                     "python").strip().endswith("return 2"))

# GSM8K math
check("#### extraction", S.extract_math_answer("blah #### 18") == "18")
check("last-number fallback", S.extract_math_answer("so 9 apples") == "9")
check("gsm8k 18.0==18", S.score_gsm8k("#### 18.0", "18")["correct"])
check("gsm8k wrong is wrong", not S.score_gsm8k("#### 19", "18")["correct"])
check("gsm8k empty is wrong", not S.score_gsm8k("no numbers here", "18")["correct"])

# Persian
check("yeh/kaf/digit normalization",
      S.normalize_fa("يك ۱۲۳") == S.normalize_fa("یک 123"))
check("ZWNJ collapse", " " in S.normalize_fa("کتاب‌ها"))

# netdiag
check("healthy verdict", S.netdiag_verdict("The network is healthy, no action needed.") is False)
check("fault verdict", S.netdiag_verdict("DNS resolution fails on every client.") is True)
check("unclear verdict", S.netdiag_verdict("I looked at the ticket.") is None)
s = S.score_netdiag("The network is healthy, no action needed.",
                    {"is_anomaly": False, "faulty_device": None, "root_cause": "none"})
check("healthy item scores 1.0", s["score"] == 1.0)
s = S.score_netdiag("Yes broken, trust me.",
                    {"is_anomaly": False, "faulty_device": None, "root_cause": "none"})
check("false alarm scores 0.0", s["score"] == 0.0)

# devops
echo_checks = {"must_contain": ["FROM", "EXPOSE"], "must_not_contain": ["PRIVILEGED"]}
check("keyword echo rejected", S.check_devops("FROM EXPOSE", echo_checks)["score"] == 0.0)
art = ("FROM python:3.12\nEXPOSE 8000\nUSER appuser\n"
       "HEALTHCHECK CMD true\nWORKDIR /app\nCOPY . .\n")
dt = {"must_contain": ["FROM", "EXPOSE 8000", "USER", "HEALTHCHECK"],
      "must_not_contain": ["USER root"]}
check("genuine artifact passes", S.check_devops(art, dt)["score"] == 1.0)
check("banned string zeroes", S.check_devops(
    art.replace("USER appuser", "USER root"), dt)["score"] == 0.0)

# mc_bundle chance with mixed choice counts
recs = [{"correct": True, "choices": ["a", "b", "c"], "gold_index": 0},
        {"correct": False, "choices": ["a", "b"], "gold_index": 1}]
b = mc_bundle(recs, None)
check("chance is mean 1/k (not flat 25%)",
      abs(b["chance_baseline"] - (1 / 3 + 1 / 2) / 2) < 1e-4)
check("bundle carries significance fields",
      all(k in b for k in ("p_value_vs_chance", "min_acc_to_reject_chance",
                           "best_constant_baseline")))

# stub exclusion
stub = [{"id": "t#ERROR", "rep": 0, "category": "code", "task": "t",
         "passed": False, "correct": False, "score": 0.0, "stopped": "error",
         "tokens": {"eval": 0}, "speed": {}},
        {"id": "t1", "rep": 0, "category": "code", "task": "t",
         "passed": True, "tokens": {"eval": 0}, "speed": {},
         "lint": {"n_diagnostics": 0, "noqa_gaming": False}}]
a = R.aggregate(stub)
check("stub excluded from accuracy", a["code"]["accuracy"] == 1.0)
check("stub counted in n_errors", a["code"]["n_errors"] == 1)

# per-item bulkhead: one exploding item must not cost its rep; the server
# going away must still propagate to the rep handler (retry/abort owner)
from backend import BackendError
from backend import Backend
check("rss watchdog None for ollama",
      Backend("ollama", "x").server_rss_mb() is None)
check("rss watchdog None when server absent",
      Backend("llamacpp", "x", port=59998).server_rss_mb() is None)


class _Boom:
    def generate(self, *a, **k):
        raise ValueError("kaboom")
    def mc_logprobs(self, *a, **k):
        raise ValueError("kaboom")


class _Dead:
    def generate(self, *a, **k):
        raise BackendError("gone")
    def mc_logprobs(self, *a, **k):
        raise BackendError("gone")


_tasks = [{"id": "t1", "prompt": "def f():\n    return 1\n", "test": "assert True"},
          {"id": "t2", "prompt": "def g():\n    return 2\n", "test": "assert True"}]
_got = R.run_code(_Boom(), _tasks, None, 0)
check("item errors recorded per item, rep survives",
      len(_got) == 2 and all("#ERROR" in r["id"] for r in _got))
try:
    R.run_code(_Dead(), _tasks, None, 0)
    check("BackendError propagates to rep handler", False)
except BackendError:
    check("BackendError propagates to rep handler", True)


# backend self-qualification probes (mock backends: mechanism, not models)
import probes as _PB


class _HealthyBackend:
    def generate(self, messages, **kw):
        if kw.get("tools"):
            return {"content": "", "tool_calls": [
                {"function": {"name": "echo_word",
                              "arguments": '{"word": "hello"}'}}]}
        return {"content": "OK", "tool_calls": []}

    def mc_logprobs(self, q, choices, system=None, prefill=""):
        return (0, "A", {"A": 0.97, "B": 0.03}, {}, "logprob", 0.97)


class _BrokenBackend:
    def generate(self, messages, **kw):
        return {"content": "let me think about that", "tool_calls": []}

    def mc_logprobs(self, q, choices, system=None, prefill=""):
        return (-1, "?", {"A": 0.02}, {}, "generated")


class _ExplodingBackend:
    def generate(self, *a, **k):
        raise RuntimeError("down")

    def mc_logprobs(self, *a, **k):
        raise RuntimeError("down")


_ok, _d = _PB.probe_mc(_HealthyBackend())
check("probe_mc passes healthy backend", _ok and _d["mean_mass"] >= 0.9)
_ok, _d = _PB.probe_mc(_BrokenBackend())
check("probe_mc fails zero-mass backend", not _ok)
_ok, _d = _PB.probe_tools(_HealthyBackend())
check("probe_tools passes parsed calls", _ok and _d.get("name") == "echo_word")
_ok, _d = _PB.probe_tools(_BrokenBackend())
check("probe_tools fails narration-only backend", not _ok)
_ok, _ = _PB.probe_generation(_HealthyBackend())
check("probe_generation passes non-empty", _ok)
_ok, _ = _PB.probe_generation(_BrokenBackend())
check("probe_generation passes any text (even narration)", _ok)
_q = _PB.qualify(_ExplodingBackend())
check("qualify never raises, all broken",
      set(_q) == {"generation", "mc_logprob", "tools"}
      and not any(ok for ok, _ in _q.values()))

# ---- durability contracts (per-item checkpointing + signal safety) ----
# A laptop suspend makes logind SIGTERM the worker; the results of a whole
# rep used to die with it. These contracts keep the fix from regressing.
_got = []
_pl = R._ProgressList(_got.append)
_pl.append({"id": "a"})
_pl.extend([{"id": "b"}, {"id": "c"}])
check("progress list streams every appended record",
      [r["id"] for r in _got] == ["a", "b", "c"])
check("progress list keeps list semantics",
      len(_pl) == 3 and _pl[0]["id"] == "a")

_recs = [
    {"task": "t", "rep": 0, "id": "x", "correct": True},
    {"task": "t", "rep": 0, "id": "y", "correct": False},
    {"task": "t", "rep": 0, "id": "t#ERROR", "correct": False},
]
check("coverage counts valid records only",
      R.coverage_report(_recs, {("t", 0): 3}) == ["t/rep0: 2/3"])
check("coverage is clean when everything is backed",
      R.coverage_report(_recs[:2], {("t", 0): 2}) == [])
check("a stub that was later replaced does not keep the file partial",
      R.coverage_report(_recs, {("t", 0): 2}) == [])
check("a task that never ran is always a gap",
      R.coverage_report([], {("t", 0): 30}) == ["t/rep0: 0/30"])
check("reps are counted separately",
      R.coverage_report([{"task": "t", "rep": 0, "id": "x"}],
                        {("t", 0): 1, ("t", 1): 1}) == ["t/rep1: 0/1"])

# SIGTERM must checkpoint and exit 143 (the suspend killer).
_sig = (
    "import os, signal, sys, tempfile\n"
    "sys.path.insert(0, %r)\n"
    "import run as R\n"
    "p = os.path.join(tempfile.mkdtemp(), 'ckpt.json')\n"
    "R.CHECKPOINT_HOOK['fn'] = lambda: open(p, 'w').write('saved')\n"
    "R._install_signal_handlers()\n"
    "os.kill(os.getpid(), signal.SIGTERM)\n"
    "print('NOT REACHED')\n"
) % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import subprocess
_proc = subprocess.run([sys.executable, "-c", _sig], capture_output=True,
                       text=True, timeout=120)
check("SIGTERM exits 143 (clean, not a crash)", _proc.returncode == 143)
check("SIGTERM path is reachable (handler installed)", "NOT REACHED" not in _proc.stdout)

# ---- superseded stubs must not be reported as errors of a complete cell --
_live, _sup = R.split_superseded([
    {"task": "t", "rep": 0, "category": "t", "id": "a", "correct": True},
    {"task": "t", "rep": 0, "category": "t", "id": "b", "correct": False},
    {"task": "t", "rep": 0, "category": "t", "id": "t#ERROR", "stopped": "error"},
])
check("a replaced rep-level stub is superseded, not an error",
      len(_live) == 2 and len(_sup) == 1)
_live2, _sup2 = R.split_superseded([
    {"task": "t", "rep": 0, "category": "t", "id": "a", "correct": True},
    {"task": "t", "rep": 0, "category": "t", "id": "b", "correct": True},
    {"task": "t", "rep": 0, "category": "t", "id": "b#ERROR", "stopped": "error"},
])
check("a per-item stub whose item later succeeded is superseded",
      len(_live2) == 2 and len(_sup2) == 1)
_live3, _sup3 = R.split_superseded([
    {"task": "t", "rep": 0, "category": "t", "id": "a", "correct": True},
    {"task": "t", "rep": 0, "category": "t", "id": "b#ERROR", "stopped": "error"},
    {"task": "u", "rep": 0, "category": "u", "id": "u#ERROR", "stopped": "error"},
])
check("a stub for an item that never succeeded stays live",
      len(_live3) == 3 and len(_sup3) == 0)
check("a complete cell reports no errors after separation",
      R.aggregate(_live).get("t", {}).get("n_errors", 0) == 0)
check("coverage counts agentic records by chain_id (no id field)",
      R.coverage_report([{"task": "agentic", "rep": 0, "chain_id": "c1"},
                         {"task": "agentic", "rep": 0, "chain_id": "c2"}],
                        {("agentic", 0): 2}) == [])
check("a missing agentic chain is still a gap",
      R.coverage_report([{"task": "agentic", "rep": 0, "chain_id": "c1"}],
                        {("agentic", 0): 2}) == ["agentic/rep0: 1/2"])

# ---- power-cut primitives: atomic writes and item-level resume --------
import tempfile as _tf
_d = _tf.mkdtemp()
_p = os.path.join(_d, "ckpt.json")
R.write_atomic(_p, {"records": [1, 2, 3]})
check("atomic write produces a readable file",
      json.load(open(_p))["records"] == [1, 2, 3])
try:
    R.write_atomic(_p, {"records": "not-json-ok", "x": object()})
except Exception:
    pass
check("a failed write leaves the previous checkpoint intact",
      json.load(open(_p))["records"] == [1, 2, 3])
check("a failed write leaves no temp file",
      not os.path.exists(_p + ".tmp"))
_t = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
check("pending_items skips only completed items",
      [x["id"] for x in R.pending_items(_t, "t", 0, {("t", 0, "b")})] == ["a", "c"])
check("a stub never counts as done (failed items re-run)",
      [x["id"] for x in R.pending_items(_t, "t", 0, {("t", 0, "b#ERROR")})]
      == ["a", "b", "c"])
check("a different rep is a different item",
      [x["id"] for x in R.pending_items(_t, "t", 1, {("t", 0, "a")})] == ["a", "b", "c"])

# ---- the power policy belongs to the human, not to the harness ---------
_st, _cap = R.battery_state()
check("battery_state reads the machine (or reports none)",
      _st is None or isinstance(_st, str))
_orig = R.battery_state
try:
    R.battery_state = lambda: ("Discharging", 12)
    try:
        R.check_power_policy("test", 30)
        check("a draining battery below the threshold pauses the run", False)
    except R.PauseRun:
        check("a draining battery below the threshold pauses the run", True)
    R.battery_state = lambda: ("Charging", 12)
    try:
        R.check_power_policy("test", 30)
        check("a charging battery never pauses the run", True)
    except R.PauseRun:
        check("a charging battery never pauses the run", False)
    R.battery_state = lambda: ("Discharging", 80)
    try:
        R.check_power_policy("test", 30)
        check("a healthy battery on battery power never pauses the run", True)
    except R.PauseRun:
        check("a healthy battery on battery power never pauses the run", False)
    R.battery_state = lambda: (None, None)
    try:
        R.check_power_policy("test", 30)
        check("a desktop with no battery never pauses the run", True)
    except R.PauseRun:
        check("a desktop with no battery never pauses the run", False)
finally:
    R.battery_state = _orig

print()
if fails:
    print(f"{len(fails)} UNIT FAILURES")
    raise SystemExit(1)
print("UNIT OK: all contracts hold.")
