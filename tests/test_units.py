#!/usr/bin/env python3
"""Unit gate: pure-function contracts of scorers/stats/aggregate. No model,
no network, no toolchain. Fails loudly (non-zero exit) on any violation.

Covers: wilson, pass_at_k/pass_hat_k, extract_code, extract_math_answer +
score_gsm8k, normalize_fa, netdiag_verdict/score_netdiag, check_devops
(keyword echo MUST fail, genuine artifact MUST pass), mc_bundle chance math,
stub exclusion in aggregate.
"""
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

print()
if fails:
    print(f"{len(fails)} UNIT FAILURES")
    raise SystemExit(1)
print("UNIT OK: all contracts hold.")
