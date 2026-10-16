#!/usr/bin/env python3
"""
Negative controls: prove the RUNNER rejects wrong answers.

verify_tasks.py proved that correct solutions pass. That is not enough, and the
HumanEval defect is the proof: the runner scored a deliberately wrong stub as a
pass on 164/164 items because it executed a test body that defined a function and
never called it. The verifier appended the missing call and agreed with the
runner's verdicts, so it could not catch the difference between the two code
paths.

This file tests the path the benchmark actually takes - the same call run.py
makes, with the same task JSON - against answers that are known to be wrong. If
any wrong answer scores a pass, the suite is not measuring what it claims.

This is a permanent fixture, not a one-off check. Every new task type needs a
negative control here before it goes in the suite.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scorers as S  # noqa: E402

T = os.path.join(HERE, "tasks")
fails = []


def check(name, passed, should_pass, detail=""):
    ok = (passed == should_pass)
    if not ok:
        fails.append((name, passed, should_pass, detail))
    flag = "OK " if ok else "BUG"
    print(f"  {flag} {name:<44} scored={str(passed):<6} should={str(should_pass)}")
    return ok


def load(n):
    return json.load(open(os.path.join(T, n + ".json")))


print("=== HumanEval: wrong stub must FAIL (the D1 probe) ===")
he = load("code_humaneval")
n_pass = 0
for t in he:
    r = S.run_python("def %s(*a, **k): return None" % t["entry_point"], t["test"])
    n_pass += bool(r["passed"])
check(f"HumanEval wrong stub over {len(he)} items", n_pass == 0, True,
      f"{n_pass} wrong stubs scored PASS")

# A stub that returns a wrong value of the right shape.
n_pass2 = 0
for t in he:
    r = S.run_python("def %s(*a, **k): return 0" % t["entry_point"], t["test"])
    n_pass2 += bool(r["passed"])
check("HumanEval wrong-value stub", n_pass2 == 0, True, f"{n_pass2} scored PASS")

print()
print("=== HumanEval: canonical must still PASS ===")
n_ok = sum(1 for t in he
           if S.run_python(t["prompt"] + t["canonical_solution"], t["test"],
                           timeout=15).get("passed"))
check(f"HumanEval canonical over {len(he)} items", n_ok == len(he), True, f"{n_ok}/{len(he)}")

print()
print("=== MBPP: wrong stub must FAIL ===")
mb = load("code_mbpp")
n_pass = 0
for t in mb[:60]:
    r = S.run_python("def similar_elements(): return 0", t["test"])
    n_pass += bool(r["passed"])
check("MBPP wrong stub (60 items)", n_pass == 0, True, f"{n_pass} scored PASS")

print()
print("=== bugfix/algo/refactor: correct reference PASSES, broken FAILS ===")
import verify_tasks as V
for suite in ("bugfix", "algo", "refactor"):
    rows = load(suite)
    bad = 0
    for t in rows:
        ref = V.CODE_REF.get(t["id"])
        if not ref:
            continue
        r = S.run_python(ref, t["test"], timeout=10)
        ok = r.get("passed")
        # and the identity version must fail
        r2 = S.run_python("def identity(*a, **k): return None", t["test"], timeout=10)
        if not ok or r2.get("passed"):
            bad += 1
    n = sum(1 for t in rows if t["id"] in V.CODE_REF)
    check(f"{suite}: reference passes, broken fails", bad == 0, True, f"{bad} items wrong")

print()
print("=== Rust: wrong impl must FAIL ===")
rs = load("code_rust")
bad = []
for t in rs:
    ref = V.RUST_REF.get(t["id"])
    if not ref:
        continue
    # Mutation must actually change the source, otherwise the control tests
    # nothing. A digit-swap is a no-op for functions with no digits in them, which
    # is what made this control silently vacuous the first time it was written.
    mutations = [
        ref.replace(".rev()", ""),          # reverse_string / is_palindrome
        ref.replace("return true", "return false"),
        ref.replace("return false", "return true"),
        ref.replace("a % b", "a + b"),       # gcd
        ref.replace("s += x", "s -= x"),
        ref.replace("> 0", "< 0"),
        ref.replace("== 0", "== 1"),
        ref.replace("{ return 0; }", "{ return 1; }"),
    ]
    mutations = [m for m in mutations if m != ref]
    mutations = mutations + [ref + "\nfn extra_unused() { let _z: i32 = 1; }"]  # no-op sanity
    r = S.run_rust(ref, t["tests"], t["entry_point"])
    fatal = None
    for m in mutations:
        if m == ref:
            continue
        r2 = S.run_rust(m, t["tests"], t["entry_point"])
        if r2.get("passed") is True and r2.get("exec_stage") == "exec":
            # a mutation that still passes the assertions is not covered by the
            # test suite for that item, which is worth reporting on its own
            fatal = m
            break
    if not r.get("passed"):
        bad.append((t["id"], "reference FAILED", r.get("error")))
    elif fatal is not None:
        print(f"    note {t['id']}: a mutation survives the assertions "
              f"(weak test, not a wrong score)")
print("  (references all pass):", all(not b[1].startswith("reference") for b in bad) if bad else True)
bad2 = [b for b in bad if b[1].startswith("reference")]
check("Rust: every reference passes", len(bad2) == 0, True, str(bad2[:2]))

print()
print("=== MC: gold-distribution bias must be gone ===")
from collections import Counter
for suite in ("science", "electronics", "persian", "english"):
    rows = load(suite)
    n = len(rows)
    hist = [Counter(r["answer"] for r in rows).get(i, 0) for i in range(4)]
    best = max(hist) / n
    # a constant-letter answerer must not exceed 50%, and no letter may be absent
    check(f"{suite}: best-constant answer accuracy <= 0.50", best <= 0.50, True,
          f"best={best:.0%} hist={hist}")
    zero = sum(1 for h in hist if h == 0)
    check(f"{suite}: every letter used at least once", zero == 0, True, f"hist={hist}")

print()
print("=== agentic: polarity must be correct ===")
import agentic as A
vok = True
for txt, exp in [("The refund is not allowed.", False),
                 ("The refund IS allowed.", True),
                 ("Cannot be refunded.", False),
                 ("I need more information.", None)]:
    if A.verdict_yes_no(txt) != exp:
        vok = False
check("verdict_yes_no on 4 polarity cases", vok, True)
ref = next(c for c in A.CHAINS if c["id"] == "chain/refund-eligibility")
check("refund chain expects NOT-allowed", ref.get("expect_yes") is False, True)

print()
print("=== agentic: out-of-stock polarity inverts (verdict_kind=stock) ===")
sok = True
for txt, exp in [("It is out of stock.", True),
                 ("The SKU is not in stock.", True),
                 ("No units are in stock at WH-EAST.", True),
                 ("It has 0 units in stock.", True),
                 ("Yes, 12 units are in stock.", False)]:
    if A.stock_verdict(txt) != exp:
        sok = False
check("stock_verdict on 5 phrasings", sok, True)

print()
print("=" * 74)
print("ADVERSARIAL CONTROLS - added after the second audit")
print("=" * 74)

print()
print("=== agentic: the headline must NOT be maxable with zero tool calls ===")
GOLD = {
    "chain/refund-eligibility": "No. The refund is not allowed for order ORD-1003 because it is processing.",
    "chain/shipping-cost": "The shipping estimate for APAC is a cost of $25.00 with 7 transit days.",
    "chain/restock": "There are 12 units of SKU-77 in stock at WH-EAST.",
    "chain/discounted-total": "The discounted total is $80.55.",
    "chain/out-of-stock-detect": "Yes, SKU-05 is out of stock at WH-EAST, 0 units.",
    "chain/eu-shipping": "Shipping to EU costs $12.00 and takes 4 days.",
    "chain/cancel-then-check": "Yes, cancellation is allowed for a processing order.",
    "chain/tier-aware": "Customer CUST-1 is gold tier.",
    "neg/policy-cancel-delivered": "No, cancellation is not allowed for delivered order ORD-1001.",
    "neg/policy-refund-cancelled": "No, a refund is not allowed for cancelled order ORD-1004.",
}
# The no-tool cheater control covers POSITIVE chains only: on irrelevance
# chains zero calls is the CORRECT behaviour, so a memorised refusal would
# score there by design. That is not gaming, it is the test working - the
# gaming surface for no-call chains is the always-CALL cheater, controlled
# separately below.
POS_CHAINS = [c for c in A.CHAINS if not c.get("expect_no_calls")]
NEG_CHAINS = [c for c in A.CHAINS if c.get("expect_no_calls")]
assert set(GOLD) == {c["id"] for c in POS_CHAINS}, \
    f"GOLD must cover exactly the positive chains: {sorted(set(GOLD) ^ {c['id'] for c in POS_CHAINS})}"
import types                                                    # noqa: E402
import run as R                                                 # noqa: E402
recs = []
for c in POS_CHAINS:
    g = GOLD[c["id"]]

    def gen(msgs, _g=g, **kw):        # recites a memorised answer; never calls a tool
        return {"content": _g, "tool_calls": [], "prompt_tokens": 0,
                "eval_tokens": 0, "ttft_s": 0.0, "wall_s": 0.0,
                "gen_tps": None, "first_token_top": []}
    b = types.SimpleNamespace(generate=gen, alive=lambda: True)
    r = A.run_chain(b, c, max_tokens=200, temperature=0.0, seed=1)
    r["category"] = "agentic"
    r["tokens"] = {"eval": 0}
    r["speed"] = {"gen_tps": None}
    recs.append(r)
agg = R.aggregate(recs)["agentic"]
check("no-tool cheater: headline pass_hat_k == 0.0",
      agg["pass_hat_k_reliability"] == 0.0, True,
      f"got {agg['pass_hat_k_reliability']}")
check("no-tool cheater: headline outcome rate == 0.0",
      agg["per_trial_outcome_rate"] == 0.0, True,
      f"got {agg['per_trial_outcome_rate']}")
check("no-tool cheater: final-answer-only accuracy still 1.0 (reported separately)",
      agg["final_answer_accuracy_no_tools_required"] == 1.0, True)

print()
print("=== agentic: order and dependency are enforced ===")
c = next(x for x in A.CHAINS if x["id"] == "chain/refund-eligibility")


def _mock(seq):
    def gen(msgs, **kw):
        turn = len([m for m in msgs if m.get("role") == "assistant"])
        for t, n, a in seq:
            if t == turn:
                return {"content": "", "tool_calls": [{"function": {"name": n,
                                "arguments": a}}], "prompt_tokens": 0,
                        "eval_tokens": 0, "ttft_s": 0.0, "wall_s": 0.0,
                        "gen_tps": None, "first_token_top": []}
        return {"content": "The refund is not allowed.", "tool_calls": [],
                "prompt_tokens": 0, "eval_tokens": 0, "ttft_s": 0.0,
                "wall_s": 0.0, "gen_tps": None, "first_token_top": []}
    return types.SimpleNamespace(generate=gen, alive=lambda: True)


GA = {"order_id": "ORD-1003"}
GP = {"action": "refund", "order_status": "processing"}
ok_order = A.run_chain(_mock([(0, "get_order", GA), (1, "check_policy", GP)]),
                       c, max_tokens=200, temperature=0.0, seed=1)
ok_rev = A.run_chain(_mock([(0, "check_policy", GP), (1, "get_order", GA)]),
                     c, max_tokens=200, temperature=0.0, seed=1)
ok_par = A.run_chain(_mock([(0, "get_order", GA), (0, "check_policy", GP)]),
                     c, max_tokens=200, temperature=0.0, seed=1)
check("correct order scores full trace", ok_order["trace_ok"] == 2, True)
check("REVERSED order does NOT score full trace", ok_rev["trace_ok"] < 2, True)
check("same-turn (guessed args) does NOT score full trace", ok_par["trace_ok"] < 2, True)

print()
print("=== agentic: irrelevance chains (no-call negatives) ===")
REFUSAL = ("I cannot do that - none of my tools can look up that information. "
           "I can help with orders, inventory, customers, or shipping estimates.")


def _text_only(txt):
    def gen(msgs, **kw):
        return {"content": txt, "tool_calls": [], "prompt_tokens": 0,
                "eval_tokens": 0, "ttft_s": 0.0, "wall_s": 0.0,
                "gen_tps": None, "first_token_top": []}
    return types.SimpleNamespace(generate=gen, alive=lambda: True)


def _caller(tool, args, final):
    calls = [{"content": "", "tool_calls": [{"function": {"name": tool,
             "arguments": args}}], "prompt_tokens": 0, "eval_tokens": 0,
             "ttft_s": 0.0, "wall_s": 0.0, "gen_tps": None, "first_token_top": []},
             {"content": final, "tool_calls": [], "prompt_tokens": 0,
              "eval_tokens": 0, "ttft_s": 0.0, "wall_s": 0.0,
              "gen_tps": None, "first_token_top": []}]
    state = {"i": 0}

    def gen(msgs, **kw):
        r = calls[min(state["i"], 1)]
        state["i"] += 1
        return r
    return types.SimpleNamespace(generate=gen, alive=lambda: True)


neg_ids = sorted(c["id"] for c in NEG_CHAINS)
check("5 irrelevance chains exist",
      neg_ids == ["neg/irrelevance-external-tracking", "neg/irrelevance-hours",
                  "neg/irrelevance-opinion", "neg/irrelevance-password-reset",
                  "neg/irrelevance-sku-attribute"], True, f"got {neg_ids}")
honest = [A.run_chain(_text_only(REFUSAL), c, max_tokens=200, temperature=0.0, seed=1)
          for c in NEG_CHAINS]
check("honest refusal solves all 5 irrelevance chains",
      all(r["chain_ok"] for r in honest), True,
      f"got {[r['chain_ok'] for r in honest]}")
callers = [A.run_chain(_caller("get_order", {"order_id": "ORD-1001"}, REFUSAL), c,
                       max_tokens=200, temperature=0.0, seed=1) for c in NEG_CHAINS]
check("always-call cheater scores 0 on all 5 irrelevance chains",
      all(not r["chain_ok"] for r in callers), True,
      f"got {[r['chain_ok'] for r in callers]}")
fab = [A.run_chain(_text_only("I have no hours tool; we are open 9-5 on Sundays."),
                   next(c for c in NEG_CHAINS if c["id"] == "neg/irrelevance-hours")),
       A.run_chain(_text_only("No, refund not allowed."),
                   next(c for c in NEG_CHAINS if c["id"] == "neg/irrelevance-password-reset"))]
check("refusal + fabricated fact fails (no disclaimer-free pass)",
      all(not r["chain_ok"] for r in fab), True)
neg_recs = []
for c, r in zip(NEG_CHAINS, honest):
    r["category"] = "agentic"
    neg_recs.append(r)
neg_agg = R.aggregate(neg_recs)["agentic"]
check("neg split reports neg_n_chains == 5", neg_agg.get("neg_n_chains") == 5, True,
      f"got {neg_agg.get('neg_n_chains')}")
check("neg split refusal rate == 1.0 on honest refusals",
      neg_agg.get("neg_refusal_rate") == 1.0, True,
      f"got {neg_agg.get('neg_refusal_rate')}")

print()
print("=== agentic: string-typed tool arguments (llamacpp/OpenAI shape) ===")


def gen_str(msgs, **kw):
    turn = len([m for m in msgs if m.get("role") == "assistant"])
    if turn == 0:
        return {"content": "", "tool_calls": [{"function": {"name": "get_order",
                "arguments": '{"order_id": "ORD-1003"}'}}], "prompt_tokens": 0,
                "eval_tokens": 0, "ttft_s": 0.0, "wall_s": 0.0,
                "gen_tps": None, "first_token_top": []}
    return {"content": "The refund is not allowed.", "tool_calls": [],
            "prompt_tokens": 0, "eval_tokens": 0, "ttft_s": 0.0,
            "wall_s": 0.0, "gen_tps": None, "first_token_top": []}


try:
    rs = A.run_chain(types.SimpleNamespace(generate=gen_str, alive=lambda: True),
                     c, max_tokens=200, temperature=0.0, seed=1)
    check("string arguments do not crash run_chain", True, True)
except Exception as e:                                              # noqa: BLE001
    check(f"string arguments do not crash run_chain ({type(e).__name__}: {e})", False, True)

print()
print("=== backend: a letter-less top-k must not crash (D7's own trigger path) ===")
import backend as BK                                              # noqa: E402
bk = BK.Backend.__new__(BK.Backend)
bk.host, bk.port, bk.api_key = "127.0.0.1", 11434, None
bk.generate = lambda msgs, **kw: {
    "content": "The", "tool_calls": [], "prompt_tokens": 1, "eval_tokens": 1,
    "ttft_s": 0.0, "wall_s": 0.0, "gen_tps": None,
    "first_token_top": [{"token": "The", "logprob": -0.5},
                        {"token": "answer", "logprob": -1.2},
                        {"token": "is", "logprob": -1.8}]}
try:
    bk.mc_logprobs("What is 2+2?", ["3", "4", "5", "6"])
    check("letter-less top-k returns a fallback verdict", True, True)
except Exception as e:                                              # noqa: BLE001
    check(f"letter-less top-k ({type(e).__name__}: {e})", False, True)

print()
print("=== backend: duplicate letter tokens ACCUMULATE, they do not clobber ===")
bk2 = BK.Backend.__new__(BK.Backend)
bk2.host, bk2.port, bk2.api_key = "127.0.0.1", 11434, None
bk2.generate = lambda msgs, **kw: {
    "content": "A", "tool_calls": [], "prompt_tokens": 1, "eval_tokens": 1,
    "ttft_s": 0.0, "wall_s": 0.0, "gen_tps": None,
    "first_token_top": [{"token": "A", "logprob": -1.609438},
                        {"token": "B", "logprob": -0.916291},
                        {"token": "A.", "logprob": -1.049822}]}
import math as _m                                                 # noqa: E402
w_a = _m.exp(-1.609438) + _m.exp(-1.049822)      # "A" + "A."
w_b = _m.exp(-0.916291)
idx, letter, dist, meta, mode, mass = bk2.mc_logprobs("q", ["a", "b", "c", "d"])
check("accumulated A out-weighs B (pick must be A)", letter == "A", True,
      f"got {letter}")

print()
print("=== prose around correct code must still execute ===")
FN = ("def has_close_elements(numbers, threshold):\n"
      "    for i, a in enumerate(numbers):\n"
      "        for j, b in enumerate(numbers):\n"
      "            if i != j and abs(a-b) < threshold:\n"
      "                return True\n"
      "    return False")
he0 = load("code_humaneval")[0]
prose_cases = {
    "non-skip preamble": "Let me write this:\n" + FN,
    "prose on the 2nd line": "Here is the solution.\nis the implementation:\n" + FN,
    "trailing prose": FN + "\n\nThis checks whether two numbers are close.",
    "prose both sides": "Let me write this:\n" + FN + "\nThat is the answer.",
}
pok = True
for label, txt in prose_cases.items():
    got = S.extract_code(txt, "python")
    if got.strip() != FN.strip():
        print(f"      {label}: extracted {got[:60]!r}")
        pok = False
check("prose-wrapped code extracted exactly, 4 phrasings", pok, True)
check("prose-wrapped code EXECUTES",
      S.run_python(S.extract_code("Let me write this:\n" + FN, "python"),
                   he0["test"]).get("passed"), True)

print()
print("=== a fresh build must not reintroduce the position bias ===")
import subprocess                                                 # noqa: E402
import tempfile                                                    # noqa: E402
with tempfile.TemporaryDirectory() as td:
    # copy the builders and the tasks dir, then rebuild from scratch in isolation
    import shutil
    for f in ("build_tasks2.py", "build_electronics.py", "debias_mc.py"):
        shutil.copy(os.path.join(HERE, f), td)
    os.makedirs(os.path.join(td, "tasks"), exist_ok=True)
    for m in ("build_tasks2.py", "build_electronics.py"):
        subprocess.run([sys.executable, os.path.join(td, m)],
                       cwd=td, capture_output=True, timeout=300)
    worst = 0.0
    for name in ("science", "electronics", "persian", "english"):
        p = os.path.join(td, "tasks", name + ".json")
        if not os.path.exists(p):
            check(f"fresh build produced {name}", False, True)
            continue
        rr = json.load(open(p))
        hh = [sum(1 for r in rr if r["answer"] == i) for i in range(4)]
        b = max(hh) / len(rr)
        worst = max(worst, b)
        print(f"      {name:<13} hist={hh} best-constant={b:.1%}")
check("fresh build stays balanced (<=30% on any letter)", worst <= 0.30, True,
      f"worst={worst:.1%}")

print()
print("=" * 74)
if fails:
    print(f"{len(fails)} FAILURES:")
    for n_, p, s, d in fails:
        print(f"   {n_}: scored={p} should={s} {d}")
else:
    print("ALL NEGATIVE + ADVERSARIAL CONTROLS PASS")
sys.exit(1 if fails else 0)
