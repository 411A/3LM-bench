#!/usr/bin/env python3
"""
Agentic multi-call tool use.

Design follows tau-bench / AgentBench rather than single-shot tool calling,
because that is the failure mode these models actually have: qwen3.5 emits a
correct first tool call and then fabricates the tool's return value on turn 2
instead of waiting for it. A harness that only checks the first call reports
that model as working. This one checks the whole chain.

Every task is a chain of dependent steps. Step k+1 needs a value only the tool
at step k can produce, so the chain cannot be short-circuited: if the model
hallucinates step 1's result, step 2 is scored wrong automatically.

Scoring follows tau-bench's pass^k, not pass@k. pass@k ("did any of k trials
work") rewards lucky runs; pass^k ("did all k trials work") is the reliability a
user actually experiences from an agent. Both are reported because they disagree
sharply on these models.
"""
import json
import re

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_order",
        "description": "Look up an order by its ID. Returns status and total amount.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string", "description": "Order ID like ORD-1001"}},
            "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "check_inventory",
        "description": "Check how many units of a SKU are in stock at a warehouse.",
        "parameters": {"type": "object", "properties": {
            "sku": {"type": "string"}, "warehouse": {"type": "string"}},
            "required": ["sku", "warehouse"]}}},
    {"type": "function", "function": {
        "name": "lookup_customer",
        "description": "Get a customer's tier (standard/silver/gold) and region.",
        "parameters": {"type": "object", "properties": {
            "customer_id": {"type": "string"}},
            "required": ["customer_id"]}}},
    {"type": "function", "function": {
        "name": "get_shipping_estimate",
        "description": "Get the shipping cost and transit days for a destination.",
        "parameters": {"type": "object", "properties": {
            "region": {"type": "string"}},
            "required": ["region"]}}},
    {"type": "function", "function": {
        "name": "apply_discount",
        "description": "Compute the discounted total for an order given a percent.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "percent": {"type": "number"}},
            "required": ["amount", "percent"]}}},
    {"type": "function", "function": {
        "name": "check_policy",
        "description": "Return whether a refund/cancel action is allowed by policy.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string"}, "order_status": {"type": "string"}},
            "required": ["action", "order_status"]}}},
]

WORLD = {
    "orders": {
        "ORD-1001": {"status": "delivered", "total": 250.00, "sku": "SKU-77", "qty": 2},
        "ORD-1002": {"status": "shipped", "total": 89.50, "sku": "SKU-12", "qty": 1},
        "ORD-1003": {"status": "processing", "total": 430.00, "sku": "SKU-40", "qty": 3},
        "ORD-1004": {"status": "cancelled", "total": 15.00, "sku": "SKU-05", "qty": 1},
    },
    "inventory": {
        ("SKU-77", "WH-EAST"): 12, ("SKU-77", "WH-WEST"): 0,
        ("SKU-12", "WH-EAST"): 4, ("SKU-12", "WH-WEST"): 25,
        ("SKU-40", "WH-EAST"): 7, ("SKU-40", "WH-WEST"): 3,
        ("SKU-05", "WH-EAST"): 0, ("SKU-05", "WH-WEST"): 100,
    },
    "customers": {
        "CUST-1": {"tier": "gold", "region": "EU"},
        "CUST-2": {"tier": "standard", "region": "NA"},
        "CUST-3": {"tier": "silver", "region": "APAC"},
    },
    "shipping": {
        "EU": {"cost": 12.0, "days": 4}, "NA": {"cost": 8.0, "days": 2},
        "APAC": {"cost": 25.0, "days": 9},
    },
    "policy": {
        ("refund", "delivered"): True, ("refund", "shipped"): True,
        ("refund", "processing"): False, ("refund", "cancelled"): False,
        ("cancel", "processing"): True, ("cancel", "delivered"): False,
    },
}


def run_tool(name, args):
    """Execute a tool. Never raises - a bad call returns an error string, which
    is exactly the feedback a real agent gets."""
    a = args or {}
    try:
        if name == "get_order":
            o = WORLD["orders"].get(str(a.get("order_id", "")).upper())
            return json.dumps(o) if o else json.dumps({"error": "order not found"})
        if name == "check_inventory":
            k = (str(a.get("sku", "")), str(a.get("warehouse", "")))
            v = WORLD["inventory"].get(k)
            return json.dumps({"sku": k[0], "warehouse": k[1],
                               "units_in_stock": v if v is not None
                               else "no such warehouse/sku"})
        if name == "lookup_customer":
            c = WORLD["customers"].get(str(a.get("customer_id", "")))
            return json.dumps(c) if c else json.dumps({"error": "customer not found"})
        if name == "get_shipping_estimate":
            s = WORLD["shipping"].get(str(a.get("region", "")).upper())
            return json.dumps(s) if s else json.dumps({"error": "unknown region"})
        if name == "apply_discount":
            amt, pct = float(a.get("amount", 0)), float(a.get("percent", 0))
            return json.dumps({"amount": amt, "percent": pct,
                               "discounted_total": round(amt * (1 - pct / 100.0), 2)})
        if name == "check_policy":
            k = (str(a.get("action", "")).lower(), str(a.get("order_status", "")).lower())
            v = WORLD["policy"].get(k)
            return json.dumps({"action": k[0], "order_status": k[1],
                               "allowed": bool(v) if v is not None else "unknown status"})
        return json.dumps({"error": f"no such tool: {name}"})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# Each chain declares its expected outcome EXPLICITLY:
#   expect_yes=True/False  -> a yes/no verdict question, scored by verdict_yes_no()
#   answer_regex=...       -> a value question, scored by a literal match
# Never mix the two, and never derive the expectation from whether a
# negative-sounding pattern matched - that is the bug that made the refund
# chain backwards.
CHAINS = [
    {"id": "chain/refund-eligibility",
     "prompt": ("Customer CUST-1 wants a refund for order ORD-1003. Check the order, "
                "check the refund policy for that order status, and then tell me "
                "whether the refund is allowed. Use the tools - do not guess."),
     "steps": [("get_order", {"order_id": "ORD-1003"}),
               ("check_policy", {"action": "refund", "order_status": "processing"})],
     "expect_yes": False,          # policy says refund on 'processing' is NOT allowed
     "max_turns": 6},
    {"id": "chain/shipping-cost",
     "prompt": ("For customer CUST-3, look up their region and the shipping estimate "
                "for that region. Report the shipping cost and transit days. Use the tools."),
     "steps": [("lookup_customer", {"customer_id": "CUST-3"}),
               ("get_shipping_estimate", {"region": "APAC"})],
     "answer_regex": r"25(\.0)?\b",
     "max_turns": 6},
    {"id": "chain/restock",
     "prompt": ("Order ORD-1001 includes 2 units of its SKU. Check inventory for that "
                "SKU at warehouse WH-EAST and tell me how many units are in stock. Use the tools."),
     "steps": [("get_order", {"order_id": "ORD-1001"}),
               ("check_inventory", {"sku": "SKU-77", "warehouse": "WH-EAST"})],
     "answer_regex": r"\b12\b",
     "max_turns": 6},
    {"id": "chain/discounted-total",
     "prompt": ("Order ORD-1002 costs a known amount. Look it up, apply a 10 percent "
                "discount, and give me the discounted total. Use the tools."),
     "steps": [("get_order", {"order_id": "ORD-1002"}),
               ("apply_discount", {"amount": 89.50, "percent": 10})],
     "answer_regex": r"80\.55",
     "max_turns": 6},
    {"id": "chain/out-of-stock-detect",
     "prompt": ("A customer ordered SKU-05 from warehouse WH-EAST. Check inventory and "
                "tell me whether it is out of stock. Use the tools."),
     "steps": [("check_inventory", {"sku": "SKU-05", "warehouse": "WH-EAST"})],
     "expect_yes": True,           # it IS out of stock (units_in_stock == 0)
     "verdict_kind": "stock",      # polarity inverts vs the generic classifier
     "max_turns": 5},
    {"id": "chain/eu-shipping",
     "prompt": ("What would shipping to region EU cost and how many days would it take? "
                "Use the shipping tool."),
     "steps": [("get_shipping_estimate", {"region": "EU"})],
     "answer_regex": r"12(\.0)?\b",
     "max_turns": 5},
    {"id": "chain/cancel-then-check",
     "prompt": ("A customer wants to cancel order ORD-1003. Check its status, then check "
                "the cancellation policy for that status. Use the tools."),
     "steps": [("get_order", {"order_id": "ORD-1003"}),
               ("check_policy", {"action": "cancel", "order_status": "processing"})],
     "expect_yes": True,           # cancel on 'processing' IS allowed
     "max_turns": 6},
    {"id": "chain/tier-aware",
     "prompt": ("Look up customer CUST-1 and tell me their tier and region. Use the tool, "
                "do not answer from memory."),
     "steps": [("lookup_customer", {"customer_id": "CUST-1"})],
     "answer_regex": r"gold",
     "max_turns": 5},
    # --- policy refusals: the tools COULD serve the request, policy forbids it.
    # Same dependency structure as the positive chains (step-2 args observable
    # only from step-1 results), polarity flipped. Fit the current harness:
    # normal trace match + expect_yes=False.
    {"id": "neg/policy-cancel-delivered",
     "prompt": ("Customer wants to cancel order ORD-1001. Check its status, then check "
                "the cancellation policy for that status, and tell me whether "
                "cancellation is allowed. Use the tools - do not guess."),
     "steps": [("get_order", {"order_id": "ORD-1001"}),
               ("check_policy", {"action": "cancel", "order_status": "delivered"})],
     "expect_yes": False,          # policy says cancel on 'delivered' is NOT allowed
     "max_turns": 6},
    {"id": "neg/policy-refund-cancelled",
     "prompt": ("Customer CUST-2 wants a refund for order ORD-1004. Check the order, "
                "check the refund policy for that order status, and tell me whether "
                "the refund is allowed. Use the tools - do not guess."),
     "steps": [("get_order", {"order_id": "ORD-1004"}),
               ("check_policy", {"action": "refund", "order_status": "cancelled"})],
     "expect_yes": False,          # policy says refund on 'cancelled' is NOT allowed
     "max_turns": 6},
    # --- irrelevance: NO tool can answer; correct = zero calls + grounded refusal.
    # Each prompt names a real entity (SKU-77, CUST-1, TRK-88192-shaped IDs) as a
    # relevance trap: a keyword-driven model calls a look-alike tool and converts
    # the result into an answer the tools cannot support. Scored by
    # verdict_kind="refusal" (disclaimer AND no fabricated payload) with
    # expect_no_calls=True. Requires the SYSTEM relevance sentence above;
    # without it these measure instruction conflict, not relevance detection.
    {"id": "neg/irrelevance-opinion",
     "prompt": ("Is SKU-77 a good birthday gift for a teenager? Give me your personal "
                "recommendation on whether they will like it. If no tool can answer "
                "that, say so - do not guess."),
     "steps": [], "verdict_kind": "refusal", "expect_no_calls": True,
     "max_turns": 5},
    {"id": "neg/irrelevance-hours",
     "prompt": ("What are your store opening hours on Sundays, and are you open on "
                "public holidays? If no tool provides hours, say so instead of "
                "calling a tool."),
     "steps": [], "verdict_kind": "refusal", "expect_no_calls": True,
     "max_turns": 5},
    {"id": "neg/irrelevance-password-reset",
     "prompt": ("Please reset the password for customer CUST-1 right now, and confirm "
                "once it is done. If you cannot do that with your tools, say so."),
     "steps": [], "verdict_kind": "refusal", "expect_no_calls": True,
     "max_turns": 5},
    {"id": "neg/irrelevance-sku-attribute",
     "prompt": ("What color and weight is SKU-99, and is it compatible with SKU-12? "
                "Do not guess - check only what your tools actually return."),
     "steps": [], "verdict_kind": "refusal", "expect_no_calls": True,
     "max_turns": 5},
    {"id": "neg/irrelevance-external-tracking",
     "prompt": ("Where is my DHL package TRK-88192 right now? Track it live and give "
                "me the current city. If no tool can track external parcels, say so."),
     "steps": [], "verdict_kind": "refusal", "expect_no_calls": True,
     "max_turns": 5},
]

SYSTEM = ("You are a support agent with access to tools. You MUST call the provided "
          "tools to get real data - never invent or assume a tool result. After the "
          "tool returns, answer the user concisely. "
          "If none of the tools can fulfill the request, say so and do not call "
          "any tool.")


_NEG = r"(?:not|no|cannot|couldn't|won't|isn't|aren't|never|unavailable|denied|disallowed)"
# Affirmative stems. No trailing \b: "allowed"/"permitted"/"available" are
# inflections of the stem, and a trailing boundary rejects every one of them.
_POS = r"(?:allow|permit|possib|\byes\b|\bcan\b|availab|valid|eligib|stock|ok)"

_NEG_ONLY_RE = re.compile(r"\b" + _NEG + r"\b", re.I)
_NEG_RE = re.compile(r"\b" + _NEG + r"\b[^.!?]{0,32}?\s*" + _POS, re.I)
_POS_RE = re.compile(r"\b" + _POS, re.I)


def verdict_yes_no(text):
    """Classify a final answer as affirmative, negative, or indeterminate.

    The scheme this replaces was inverted on the refund chain: the gold is
    "not allowed", the old regex matched the substring "allow" inside
    "not allowed", and `answer_expect` was False - so the correct answer scored
    0 while the wrong one scored 1. Verified directly against agentic.CHAINS.

    Negation has to be resolved before the keyword, because matching a bare
    keyword inverts polarity for exactly the cases that matter. A negation cue
    within ~32 characters ahead of an affirmative keyword makes it negative;
    a bare affirmative is positive; nothing found is indeterminate.
    """
    if not text:
        return None
    t = re.sub(r"\s+", " ", text)
    if _NEG_RE.search(t):
        return False
    if _POS_RE.search(t):
        return True
    # A bare negation with no affirmative keyword anywhere ("Refund denied",
    # "It cannot be refunded") is still a negative answer.
    if _NEG_ONLY_RE.search(t):
        return False
    return None


def stock_verdict(text):
    """Classify an answer to "is it OUT OF STOCK?".

    The generic verdict_yes_no cannot be used here. "The SKU is not in stock"
    pairs the negation with the affirmative stem "stock" and returns False - but
    the answer to "is it out of stock" is YES. Out-of-stock is the NEGATIVE state
    of in-stock, so polarity inverts relative to the stem, and two of the five
    most natural phrasings of the CORRECT answer scored 0.

    Scored on the stock state itself rather than on English negation, and on the
    unit count where one is given.
    """
    if not text:
        return None
    t = re.sub(r"\s+", " ", text).lower()
    if re.search(r"out of stock|no units|zero units|0 units|not in stock|"
                 r"no [a-z ]{0,12}in stock|unavailable|out of inventory|"
                 r"cannot be (?:ordered|shipped)", t):
        return True
    m = re.search(r"\b(\d+)\s*units?\b", t)
    if m:
        return int(m.group(1)) == 0
    if re.search(r"in stock|available|can be ordered", t):
        return False
    return None


def refusal_verdict(text):
    """Did the model groundly refuse (no applicable tool) WITHOUT fabricating?

    Strict bool, never None: a no-call chain's outcome must be decidable.
    Pass requires BOTH:
      (a) a disclaimer tying the refusal to tool capability, and
      (b) NO fabricated payload (amounts, counts, tiers, times, specs).

    verdict_yes_no cannot score these: a generic "I cannot help" reads False,
    so expect_yes=False would credit any negative-sounding text - including a
    wrong-policy answer or a refusal followed by an invented fact.
    """
    if not text:
        return False
    t = re.sub(r"\s+", " ", text).lower()
    disclaimer = re.search(
        r"(no|none)(\s+of)?.{0,40}(tool|function).{0,40}(can|apply|provide|perform)|"
        r"cannot .{0,24}(look up|check|do that|reset|track|perform|provide)|"
        r"do not have .{0,32}(tool|access)|don'?t have access to|"
        r"not something (i can|i can do|we can) (do|check|look up)|"
        r"outside what i can|no tool", t)
    if not disclaimer:
        return False
    fabricated = re.search(
        r"\$\s*\d|\b\d+\s*(units|days|percent)|\b(gold|silver)\b|"
        r"standard\s+(tier|member|customer)|\b(open|closed)\b.{0,12}"
        r"\b(\d|morning|evening|am|pm)\b|9\s*(am|pm)|"
        r"\b(red|blue|green|black|\d+\s*(kg|g|cm|mm))\b", t)
    return not bool(fabricated)


def _norm_args(d):
    # `arguments` is a JSON STRING in the OpenAI chat-completions spec, which is
    # what llama.cpp's /v1/chat/completions route returns; Ollama returns a dict.
    # Calling .items() on the raw string raised AttributeError and aborted the
    # run on the FIRST tool call any model made on the llamacpp backend - the
    # headline tool-use measurement, dead on the backend most of these GGUFs need.
    # _safe_json degrades an unparseable string to {"_raw": ...} rather than
    # raising, so a malformed call costs that step, not the run.
    if not isinstance(d, dict):
        d = _safe_json(d)
    return {str(k).lower(): str(v).lower() for k, v in (d or {}).items()}


def _call_name(tc):
    if isinstance(tc, dict):
        if "function" in tc and isinstance(tc["function"], dict):
            return tc["function"].get("name"), tc["function"].get("arguments")
        return tc.get("name"), tc.get("arguments") or tc.get("parameters")
    return None, None


def _safe_json(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:  # noqa: BLE001
            return {"_raw": x}
    return {}


def run_chain(b, chain, max_tokens=200, temperature=0.7, seed=None):
    """Drive one chain to completion. Returns a full trace."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": chain["prompt"]}]
    trace, n_tool_calls = [], 0
    final, stopped = "", "max_turns"

    for turn in range(chain["max_turns"]):
        r = b.generate(msgs, max_tokens=max_tokens, temperature=temperature,
                       tools=TOOL_SPECS, seed=seed)
        final = r["content"] or ""
        tcs = r.get("tool_calls") or []
        trace.append({
            "turn": turn, "content": final,
            "tool_calls": [_call_name(t) for t in tcs], "raw_tool_calls": tcs,
            "gen_tps": r["gen_tps"], "eval_tokens": r["eval_tokens"],
            "wall_s": r["wall_s"], "prompt_tokens": r["prompt_tokens"],
        })
        if not tcs:
            stopped = "no_tool_call" if turn == 0 else "final_answer"
            break
        n_tool_calls += len(tcs)
        for i, tc in enumerate(tcs):
            name, a = _call_name(tc)
            out = run_tool(name, _safe_json(a))
            n_tool_calls += 0
            msgs.append({"role": "assistant", "content": final if i == 0 else "",
                         "tool_calls": [{"id": f"call_{turn}_{i}", "type": "function",
                                         "function": {"name": name, "arguments": a}}]})
            msgs.append({"role": "tool", "content": out, "name": name or "unknown"})
    else:
        stopped = "max_turns"

    # Ordered trace match. The previous matcher asked "was each expected call ever
    # made" with no order and no turn information, so a model that fired the calls
    # in reverse scored the same as one that followed the dependency chain, and a
    # model that issued every tool in ONE turn - guessing the arguments it could
    # only have learned from a tool result - also scored full credit.
    #
    # The dependency here is real: step k+1's arguments are values only the tool
    # at step k can reveal (chain/restock does not name the SKU in the prompt; it
    # is inside get_order's result). So step k+1 must happen in a STRICTLY LATER
    # TURN than step k. A same-turn pair means the argument was guessed.
    called = [(n, _norm_args(a), t["turn"])
              for t in trace for n, a in t["tool_calls"]]
    trace_ok, detail = 0, []
    cursor = 0          # index into `called`; enforces relative order
    prev_turn = -1
    for want_name, want_args in chain["steps"]:
        hit = False
        for i in range(cursor, len(called)):
            got_name, got_args, got_turn = called[i]
            if got_name != want_name:
                continue
            # Boundary-aware containment, not raw substring. A raw `v in arg`
            # test credits "xxORD-1003xx" for "ORD-1003" - an argument that no
            # tool result contained and no honest model emits, but which still
            # banks full trace credit. Word boundaries keep the intended
            # tolerance ("order ORD-1003 please" still hits) while rejecting
            # values embedded inside junk. Values without word characters fall
            # back to substring (boundaries cannot exist for them).
            def _arg_hit(v, a):
                v, a = str(v).lower(), str(a)
                if re.search(r"\w", v[:1]) and re.search(r"\w", v[-1:]):
                    return bool(re.search(r"\b" + re.escape(v) + r"\b", a))
                return v in a
            if not all(_arg_hit(v, got_args.get(str(k).lower(), ""))
                       for k, v in want_args.items()):
                continue
            if got_turn <= prev_turn:
                # Same turn as its dependency: the argument was guessed, not read
                # out of a tool result. Record it and do not advance.
                if not any(d.startswith(want_name + ":same_turn") for d in detail):
                    detail.append(
                        f"{want_name}:called in the same turn as the step it "
                        f"depends on (argument guessed, not observed)")
                continue
            hit = True
            cursor = i + 1
            prev_turn = got_turn
            break
        if hit:
            trace_ok += 1
        elif not any(d.startswith(want_name + ":") for d in detail):
            detail.append(f"{want_name}: never called with expected args")

    m = None
    if "answer_regex" in chain:
        m = re.search(chain["answer_regex"], final, re.I)
        outcome_ok = bool(m)
    elif chain.get("verdict_kind") == "refusal":
        outcome_ok = refusal_verdict(final)
    else:
        v = stock_verdict(final) if chain.get("verdict_kind") == "stock" \
            else verdict_yes_no(final)
        outcome_ok = (v == chain.get("expect_yes"))
    # A chain counts as solved only if the tools were actually used, in order,
    # and the final answer is right. outcome_ok alone is scored from answer TEXT
    # and is trivially maxed by reciting a memorised answer.
    #
    # No-call (irrelevance) chains invert the tool requirement: the correct
    # behaviour is ZERO calls, so `n_tool_calls > 0` would make them unsolvable
    # by construction. Solved = grounded refusal AND nothing called.
    if chain.get("expect_no_calls"):
        chain_ok = bool(outcome_ok and n_tool_calls == 0)
    else:
        chain_ok = bool(outcome_ok and trace_ok == len(chain["steps"])
                        and n_tool_calls > 0)
    return {
        "chain_id": chain["id"], "stopped": stopped, "n_tool_calls": n_tool_calls,
        "chain_ok": chain_ok, "expect_no_calls": bool(chain.get("expect_no_calls")),
        "final_answer": final, "answer_matched": bool(m) if m is not None else (verdict_yes_no(final)),
        "verdict": verdict_yes_no(final), "expected_verdict": chain.get("expect_yes"),
        "outcome_ok": outcome_ok,
        "trace_ok": trace_ok, "trace_total": len(chain["steps"]),
        "trace_detail": detail, "called_tools": [n for n, _a, _t in called],
        "calls": trace,
    }