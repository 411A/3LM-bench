# Second adversarial audit — findings, fixes, and verification

Source: independent audit by another OpenCode agent, 19 findings (4 CRITICAL,
10 MAJOR, 5 MINOR), executed in an isolated scratch copy. Every finding below
was **re-verified by running its trigger path** before any fix was applied, and
**re-verified after**. Three auditor findings did not survive my own replication
attempt and are marked as such rather than being quietly counted as fixes.

Task count after this pass: **854** (was 806). Persian 14 → 40, English 15 → 45.

---

## CRITICAL

### C1 — wrong gold, `fa/fa-math-1` (Persian)

5 books cost 80,000 → 16,000 each → 3 books = **48,000**. The gold was
`۲۴۰ هزار` = 240,000 = `80000 × 3` — the "forgot to divide by 5" error — and the
correct answer was **not among the choices at all**. A live 4B model picked the
wrong gold at 0.997 confidence, so the item rewarded the exact mistake small
models make and penalised correct reasoning.

Gold corrected to `۴۸ هزار`; distractors are now the characteristic errors (×3
without dividing, ×2, divisor-as-multiplier). `fa-math-2` and `fa-math-3` were
hand-checked and were already correct.

### C2 — the agentic headline was maxable with zero tool calls

`pass_hat_k_reliability` was computed from `outcome_ok`, which is scored purely
from the **final answer text**. A mock backend that never called a tool and
recited memorised answers scored:

```
per_trial_outcome_rate = 8/8 = 1.000
pass_hat_k_reliability = 1.000      <-- the headline
per_trial_trace_rate   = 0.000
mean_tool_calls        = 0.0
```

The headline now requires `outcome_ok AND trace_ok == trace_total AND
n_tool_calls > 0`. The same mock now scores **0.0** on the headline while
`final_answer_accuracy_no_tools_required` still reports 1.0, so "did not know"
stays distinguishable from "knew but would not use the tools".

### C3 — DevOps: a bare keyword echo scored 1.0 on 10/10

`check_devops` scored substring presence, so `'FROM EXPOSE 8000 USER
HEALTHCHECK'` — no file at all — passed everything, including a task requiring a
pinned action version and one requiring a secret NOT be printed.

**First fix over-corrected and I reverted it.** Requiring each requirement to
anchor a line scored genuine artifacts at 0.5–0.88, because YAML list items
(`- uses:`), inline `FROM … AS build`, and values under a key never anchor a
line. What actually separates an artifact from a keyword salad is size and
spread, so that is what is enforced: ≥3 non-empty lines, and requirements spread
across ≥2 distinct lines.

Result: echo **0/10**, two-line echo **0/10**, genuine artifacts **10/10**.
`verify_devops.py` holds all three sides as a permanent fixture, because this
scorer has now failed in both directions at different times.

### C4 — `backend.py` used `re` without importing it

The `mc_logprobs` fallback — added to fix D7 — called `re.search` in a module
with no `import re`. Any MC item whose top-20 tokens contained no parseable
letter killed the whole run, and `run.py` caught only `BackendError`. A full run
could die at 80% with the current task file unwritten.

Fixed, plus a structural fix: `run.py` now catches **any** exception per task and
records a stub, because checkpoints only fire between task files and no single
item should be able to end a 40-hour run.

---

## MAJOR

| # | Defect | Resolution |
|---|---|---|
| 5 | `el/ard/spi-divider` gold was SPSR; SPR1:SPR0 live in **SPCR** | Gold → SPCR; stem now names the bits. A live model had answered SPCR at 0.968 and been scored wrong |
| 6 | `sci/cs-5` asked for stable **and** in-place **and** O(n log n) — no such option; merge sort is not in-place | "in-place" replaced with the explicit O(n) auxiliary space trade-off |
| 7 | MC gate `correct = ok and mass >= 0.15` recorded confident-but-below-threshold gold picks as **wrong** | `correct` is the argmax pick. Low mass reported as `informed`/`letter_mass`, never scored. Inert on confident models (all masses ≥0.9949), so it would have done nothing until the day it mattered |
| 8 | netdiag verdict was `NOT(healthy keywords)` — plain-English correct answers scored 0.0 on **36%** of the suite, and a correct real-fault answer containing "working correctly" scored 0.0 | Two-sided classifier with **positive evidence required on both sides**; a scoped clearance no longer cancels a fault word; an unevidenced verdict gets 1/3 instead of zeroing the item |
| 9 | Prose before/after correct code reached the compiler → correct answers scored as syntax errors | For Python the code span is chosen by **`ast.parse`**, not a regex. A preamble phrased in any way the regex didn't anticipate can no longer cost the answer. 8/8 prose cases extract exactly; prose-wrapped code executes |
| 10 | The de-bias was a manual script **called by nothing** — a rebuild regenerated the 69%-on-B suite | `rebalance()` is now called by the builders, so the de-biased file *is* the build output. Also switched from random shuffle to **least-used round-robin** (shuffle is only approximately balanced), and `debias_mc.py` now resolves `tasks/` relative to `__file__` instead of hardcoding `~/llmbench` — previously running it from a copy rewrote the real benchmark |
| 11 | Trace matcher ignored order and turn, so reversed and same-turn (guessed-args) traces scored 2/2 | Ordered match; step *k+1* must occur in a **strictly later turn** than step *k* |
| 12 | "The SKU is not in stock" scored **wrong** on the out-of-stock chain | `stock_verdict()` scores the stock state, not English negation. Polarity genuinely inverts: out-of-stock is the negative state of in-stock |
| 13 | `_norm_args` crashed on string-typed tool arguments — the OpenAI/llama.cpp format, i.e. **the whole llamacpp path** | Normalised through `_safe_json`. Reproduced deterministically with the OpenAI response shape |
| 14 | `fa/fa-mc-7` leaked its gold into the question, and 3 of 4 choices collapsed to one string under the suite's own `normalize_fa` | Unrepairable in place — `normalize_fa` treats Arabic-Indic and Persian digits as equivalent *because they are*, so a digit-script question contradicts the scorer. Replaced with a vocabulary item. **My replacement then leaked the gold again**, and `verify_golds.py` caught it; the stem now gives the English gloss |

### MINOR

- `el/sens/ldr-divider`: an unstated divider topology had two defensible answers. The wiring is now stated.
- 4 netdiag golds: `net/1` blamed the wrong device (a `/30` on 10.0.1.5 does not contain 10.0.1.1); `net/0`'s gold cause had no supporting evidence (added); `net/9`'s "duplicate" wasn't one (10.0.0.5 ≠ 10.0.0.1, corrected); `net/10`'s symptom contradicted its own observation.
- Two compose tasks banned `version:` — the standard header of every correct compose file. The ban was itself a defect.
- `dev_ok`/`cause_ok` reported `None` instead of `False` on a miss, because `a or re.search(...)` yields `None`.
- README claimed confidence was "renormalised" (stale — it is absolute), and claimed the agentic chain "cannot be short-circuited".

---

## 9 further gold defects, found by independent re-audit of all 122 MC golds

Every item was re-judged individually, with arithmetic recomputed.

| id | defect | fix |
|---|---|---|
| `sci/arc-1` | Transduction and Transformation are **both** bacterial DNA transfer; "immediately" discriminates nothing | stem now names the vector (bacteriophage) |
| `sci/cs-1` | "average … in the worst case" is self-contradictory | one case only |
| `sci/cs-8` | Segregated fit **also** suffers external fragmentation — a second correct option | replaced with fixed-partition paging |
| `el/ard/pwm-timer0` | analogWrite uses Timer2 on pins 3/11 — unstated pin gave two correct answers | stem now names pin 9 |
| `el/sens/oversampling` | choice 1 restated the gold | replaced with a false claim |
| `fa/fa-nli-1` | golded contradiction; "some" → "all" is **neutral**, not a contradiction | gold → neutral, and **all three NLI items now state their label semantics** |
| `fa/fa-nli-3` | inconsistent with nli-1 under any single reading | resolved by the above |
| `en/en-mc-7` | "A compiler is to software as a:" is ungrammatical and has no valid answer | replaced with a well-formed item |
| `fa/fa-mc-7` | gold quoted verbatim in the stem | see MAJOR 14 |

The NLI items first defined their labels *using the choice strings*, which put the
gold into every stem. `verify_golds.py` caught it immediately; the options are now
numeric and the stem describes the three cases without quoting any option.

---

## The question that actually matters: is any score better than guessing?

A multiple-choice percentage on its own is not interpretable. Every MC figure
now ships with the chance baseline, the best constant-letter answerer's score,
the lift, a Wilson interval, an exact one-sided binomial p-value, and the
accuracy **required at that n** before guessing can be excluded.

```
                  n   chance  best-constant  must reach to prove non-random
science           35   25.0%     25.7%        40%
electronics       58   25.0%     25.9%        36%
persian           40   25.6%     27.5%        40%     (was n=14 -> 57%)
english           45   25.0%     26.7%        38%     (was n=15 -> 53%)
```

This is why Persian and English were expanded. At n=14 a model had to score 57%
before the data could exclude chance at all, and the two items separating 43%
from 57% were the entire margin — the suite could not distinguish a good model
from a coin flip. The binomial tail is validated exact to 1e-12 against
brute-force rational arithmetic.

---

## Where my reasoning was flawed

The auditor's central point was that each of my fixes had been validated against
the *specific* input that exposed the defect and then declared done, three times
in a row:

1. **D7's fix was never run on its own trigger path.** The empty-distribution
   fallback was added; nobody fed `mc_logprobs` a letter-less top-k. The fix
   crashed with `NameError` on exactly the input it was written for.
2. **The agentic fix was verified on polarity, never on the metric's meaning.**
   Four polarity cases passed while the headline was maxable with no tool calls.
3. **The de-bias fix was verified on the stored files, never on the build.** The
   histograms were correct; the build that regenerates them was not.

And I reproduced the pattern myself this round: the first DevOps fix over-
corrected and penalised genuine artifacts. Only running real Dockerfiles caught
it — reading the new code would not have.

The rule now enforced rather than remembered: **every fix is verified by running
its trigger path and its near-miss, and any task type enters the suite only with
a control in `negcontrols.py`.**

## Control coverage after this pass

| file | proves |
|---|---|
| `verify_tasks.py` | correct answers **pass** (854 tasks) |
| `negcontrols.py` | wrong answers **fail**; no-tool cheater cannot max the headline; reversed and same-turn traces lose credit; string tool args don't crash; letter-less top-k doesn't crash; duplicate letters accumulate; prose-wrapped code extracts and runs; **a fresh build stays balanced** |
| `verify_golds.py` | arithmetic golds recomputed independently; no gold leaked into its own question; structural integrity; gold-position balance |
| `verify_devops.py` | keyword echo rejected, genuine artifacts accepted, banned strings penalised |
| `verify_cpp.py` | embedded C++ distinguishes right from wrong |
| `determinism.py` | fixed seed reproducible **and** honoured |

All six pass. Three of them (`verify_golds`, `verify_devops`, and the adversarial
block in `negcontrols`) exist only because of this audit, and each caught a real
defect — including two I introduced while fixing others.