# Benchmark correctness review

Everything below was verified by executing code, not by reading it. Each claim
has a reproducing input.

**Status: no benchmarking data is usable yet.** All previous result files were
deleted, including `results/_salvage-nemotron-codeonly.json` — it carried zero
information for the reason described in D1.

---

## Critical — a wrong answer scored as a correct answer

### D1. HumanEval never executed a single assertion

`run.py` passed the stored `test` field to `run_python`. HumanEval's test field
*defines* `check(candidate)` and never *calls* it, so the runner executed a
function definition and exited 0.

```
run_python("def has_close_elements(numbers, threshold): return False", test)
    -> passed=True          # never ran
run_python(same, test + "\ncheck(has_close_elements)")
    -> passed=False
```

All 164 items, 40 of which were in the salvage file, passed on any parseable
stub. **My earlier claim that nemotron's 98.75% was genuine was wrong**, and I
should not have stated it. I verified the completion looked correct and ruled out
three explanations; I never checked that the test actually ran. That is the exact
error the suite was supposed to catch.

`verify_tasks.py` could not catch this: it appended the missing `check()` call and
therefore validated a *different code path* from the runner's. Now `negcontrols.py`
tests the runner itself.

Fixed by baking `check(entry_point)` into the stored test field.

### D2. Answer-position bias made a knowledge-free model score 69%

Gold-index distribution before the fix:

| file | n | A | B | C | D | best constant |
|---|---|---|---|---|---|---|
| electronics | 58 | 12 | **40** | 6 | **0** | **69%** |
| english | 15 | 1 | **10** | 4 | **0** | **67%** |
| science | 35 | 12 | 16 | 5 | 2 | 46% |

A model that unconditionally answers "B" would have scored 69% on electronics and
67% on english. D was absent entirely from two files, so "never answer D" was a
free 100% rule on that option. Nothing in the harness detected it.

Fixed by `debias_mc.py`, a seeded per-item shuffle with the gold index remapped.
Gold *text* is verified preserved — only position moves. Max index is now 26–43%,
within sampling noise of uniform, and no letter is unused.

### D3. `sys.exit(0)` and `os._exit(0)` scored a PASS

`run_python` scored `returncode == 0`. Both calls terminate the subprocess with
status 0 *before* the assertions, so the test never ran and the item passed.

Fixed by appending a sentinel that only prints once the whole test body has
completed. A pass now requires both exit 0 **and** the sentinel. Same fix applied
to the Rust and C++ harnesses.

### D4. Agentic polarity was inverted — the correct answer scored 0

`chain/refund-eligibility` golds on "not allowed". The regex matched the
substring `allow` inside `not allowed`, and `answer_expect` was `False`.

```
"The refund is not allowed."  -> outcome_ok=False     # correct answer, scored 0
"The refund IS allowed."      -> outcome_ok=True       # wrong answer, scored 1
```

`chain/cancel-then-check` had the same flaw in the other direction: `"Cancellation
is not allowed."` matched `allow` and scored correct.

Fixed by replacing regex polarity with `verdict_yes_no()`, a negation-aware
classifier, and an explicit `expect_yes: True/False` per chain. 13/13 polarity
cases verified.

---

## Major

### D5. `extract_code` picked the first block, and prose counted as code

- Two code blocks, second correct → the **wrong** one was scored.
- `"I cannot help with that."` was returned as code and handed to the compiler.
- Nested triple-backticks in a docstring truncated the block to `"""use `.

Fixed: prefer tagged blocks, then the **last** untagged block, validate blocks
with a code heuristic, and return `""` for prose.

### D6. Duplicate letter tokens clobbered instead of accumulating

`dist[L] = lp` (assignment). A top-k over token *strings* routinely contains `A`,
`A.`, `**A` and `A)` as separate entries, so whichever was serialised last won.

```
raw: A=0.20  B=0.40  A.=0.35   -> true mass A=0.55, B=0.40
harness picks B; reverse the order and it still picks B
```

The answer depended on JSON key order. Fixed by accumulating.

### D7. `max()` on an empty dict aborted the entire run

A single item with all-zero logprobs raised `ValueError` inside `mc_logprobs`.
`run.py` caught only `BackendError`, so it propagated out of `run_mc` and killed
the process — losing the current task and every task after it, since checkpoints
only fire between tasks. Reproduced with llama.cpp-style pruned logits at `-1e9`.

### D8. Confidence was structurally incapable of being low

`confidence` renormalised *inside* the letter set, so it was bounded below by
`1/len(letters)`. A model with 0.04% belief in any answer reported confidence
0.25–1.00. Now the **absolute** letter mass is returned and `run_mc` gates on it,
so a model with no real preference is recorded as low-information rather than
silently scored at 50%.

### D9. `pass_hat_k` computed `1 - pass@k`

`C(n-c,k)/C(n,k)` is hypergeometric *and* inverted — total failure reported 1.0.
Correct tau-bench estimator is `(c/n)^k`. It was dead code, but a wrong inverted
formula in a shared module is a trap regardless.

### D10. `aggregate()` raised KeyError after the inference was paid for

Three categories (`lint`, `devops`, `rust`-detail) indexed `r["lang"]` /
`r["score"]` on records without those keys. Fixing to `.get()`.

### D11. `"(A"` was rejected

`strip("*:)")` left the leading paren, so `Answer (A` — the form the prompt
itself primes for, since options render as `A) choice` — was dropped and the item
forced down the fallback path.

---

## Determinism

Regex, ruff, rustc, g++ and clippy are deterministic: fixed input, fixed fixed
toolchain version, always the same output. The real risk was **sampling**, and it
was measured rather than assumed:

```
case        temp   seed  unique  length  verdict
short        0.7  20261012   1/3        2  DETERMINISTIC
code         0.7  20261012   1/3       45  DETERMINISTIC
long         0.7  20261012   1/3      858  DETERMINISTIC
long         0.0  20261012   1/3      894  DETERMINISTIC

3 distinct seeds at temp=0.7 -> 3 unique outputs (seed honoured)
MC path (3 runs) -> 1 unique verdict (no sampling stage at all)
```

Both conditions hold: the seed reaches the sampler (different seeds → different
outputs, so the run is not accidentally reproducible), and a fixed seed gives
byte-identical output across short, code and 858-token generations.

`determinism.py` re-measures this for every model and backend.

---

## Fixtures now in place

| file | what it proves |
|---|---|
| `verify_tasks.py` | correct solutions **pass** (all 806 tasks) |
| `negcontrols.py` | the **runner** rejects wrong answers — the check D1 hid from |
| `verify_cpp.py` | every embedded task distinguishes right from wrong |
| `determinism.py` | fixed seed is reproducible and the seed is actually honoured |
| `debias_mc.py` | gold-position distribution is near-uniform |

Any new task type needs a negative control in `negcontrols.py` before it enters
the suite. That is the rule D1 violated.
