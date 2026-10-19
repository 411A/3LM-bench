#!/usr/bin/env python3
"""Backend self-qualification: prove each I/O mechanism works before a run.

A benchmark score is meaningless when the harness path itself is broken
(proven twice: qwen4b letter-choice degenerate on llama-server, nemotron blob
making zero tool calls on llama-server while both work via Ollama). These
probes test MECHANISM, never capability:

  mc_logprob  do answer letters carry probability mass? (mass, not correctness:
              a wrong-but-confident model is measurable; a zero-mass model is not)
  tools       does a forced tool call come back parsed? (parsed, not wise:
              any parsed call proves the renderer+parser round-trip works)
  generation  does a short completion come back non-empty?

Verdicts gate task families: broken MC skips all run_mc suites, broken tools
skips agentic, broken generation aborts. Skipped = explicitly missing with a
reason, never zero. On --resume, probes are advisory-only (the file's task
list must not change or its hash breaks).

Thresholds are set from measured evidence (healthy pairings commit mass
>= 0.95; broken ones <= 0.5): MC_MASS_FLOOR = 0.5 over 3 probes.
"""
import sys

MC_ITEMS = [
    ("What is 2+2?", ["3", "4", "5"], 1),
    ("Which is a prime number?", ["4", "6", "7", "9"], 2),
    ("What colour is the sky on a clear day?", ["green", "blue", "red"], 1),
]
MC_MASS_FLOOR = 0.5

PROBE_TOOLS = [{"type": "function", "function": {
    "name": "echo_word",
    "description": "Echo back one word. Call this with word='hello'.",
    "parameters": {"type": "object",
                   "properties": {"word": {"type": "string"}},
                   "required": ["word"]}}}]


def _mass_of(mc_result):
    # mc_logprobs returns 5- or 6-tuples across backend versions
    if len(mc_result) == 5:
        _, _, _, _, _ = mc_result
        return None
    return mc_result[5]


def probe_mc(b):
    """Returns (ok, detail). ok requires mean letter mass >= floor."""
    masses, picks = [], []
    for q, choices, _gold in MC_ITEMS:
        try:
            r = b.mc_logprobs(q, choices,
                              system="Answer with a single letter and nothing else.")
        except Exception as e:  # noqa: BLE001
            return False, {"error": f"{type(e).__name__}: {e}"}
        m = _mass_of(r)
        masses.append(m if isinstance(m, (int, float)) else 0.0)
        picks.append(r[1])
    mean_mass = sum(masses) / len(masses)
    return mean_mass >= MC_MASS_FLOOR, {
        "mean_mass": round(mean_mass, 4), "masses": [round(m, 4) for m in masses],
        "floor": MC_MASS_FLOOR,
        "note": "mechanism only: mass proves letters are scored, "
                "correctness is the model's business"}


def probe_tools(b, tries=3):
    """Returns (ok, detail). ok requires >= 1 parsed tool call in `tries`."""
    last = None
    for _ in range(tries):
        try:
            r = b.generate(
                [{"role": "system", "content":
                  "You have a tool. You MUST call echo_word with word='hello'. "
                  "Do not answer in text."},
                 {"role": "user", "content": "Call echo_word now."}],
                max_tokens=64, temperature=0.0, tools=PROBE_TOOLS)
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            continue
        tcs = r.get("tool_calls") or []
        if tcs:
            fn = tcs[0].get("function", tcs[0]) if isinstance(tcs[0], dict) else {}
            return True, {"parsed": True,
                          "name": (fn.get("name") if isinstance(fn, dict) else None),
                          "note": "a parsed call proves renderer+parser work; "
                                  "wisdom is measured by the agentic suite"}
        last = (r.get("content") or "")[:120]
    return False, {"parsed": False, "last": last,
                   "note": "no parsed call in %d tries: tool suites would "
                           "score zero for harness reasons" % tries}


def probe_generation(b):
    """Returns (ok, detail). ok requires non-empty content."""
    try:
        r = b.generate([{"role": "user", "content": "Reply with exactly: OK"}],
                       max_tokens=8, temperature=0.0)
    except Exception as e:  # noqa: BLE001
        return False, {"error": f"{type(e).__name__}: {e}"}
    ok = bool((r.get("content") or "").strip())
    return ok, {"non_empty": ok}


def qualify(b):
    """Run all three probes. Returns {name: (ok, detail)} — never raises."""
    out = {}
    for name, fn in (("generation", probe_generation),
                     ("mc_logprob", probe_mc),
                     ("tools", probe_tools)):
        try:
            out[name] = fn(b)
        except Exception as e:  # noqa: BLE001
            out[name] = (False, {"error": f"probe crashed: {e}"})
    return out


if __name__ == "__main__":
    sys.path.insert(0, __import__("os").path.dirname(
        __import__("os").path.abspath(__file__)))
    from backend import Backend
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", default="nemotron-3-nano:4b")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    b = Backend(args.backend, args.model,
                port=args.port or (11434 if args.backend == "ollama" else 8090))
    import json
    print(json.dumps({k: {"ok": ok, **d} for k, (ok, d) in qualify(b).items()},
                     indent=1))
