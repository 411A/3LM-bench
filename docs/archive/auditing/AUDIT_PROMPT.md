# Adversarial audit prompt — llmbench

> Archived prompt: its counts and protocol description are historical. Check
> current project behavior in `../../../README.md` and `../../METHODOLOGY.md`.

Paste this entire file as the prompt to any model you want to use as an auditor.
It is written to be self-contained: the auditor needs no other context.

---

You are auditing a benchmark suite called `llmbench` that measures the coding,
reasoning, language and tool-use ability of small local LLMs (2–4B parameters) on
weak hardware. Your job is to find defects **that would make the benchmark
report a wrong conclusion**. You are being adversarial on purpose.

The author has already found and fixed 11 defects, listed at the bottom under
ALREADY FOUND. Do not re-report those. Find what remains.

**The bar for a finding:** you must be able to state a concrete input and the
wrong verdict it produces. "This code is untidy" is not a finding. "A model that
answers B always, scores 69% on electronics" is a finding.

## Where things are

Read `README.md` first for design intent, then the code.
Everything is Python 3.14; no dependencies outside the standard library.

```
run.py             task runners + aggregation; the entry point
backend.py         unified client over Ollama and llama.cpp; MC logprob scoring
scorers.py         code extraction, sandboxed execution, lint, netdiag/devops scoring
agentic.py         tool specs, a simulated world, and 8 dependent tool-call chains
build_tasks.py     HumanEval + MBPP extraction; bugfix/algo/refactor/rust sets
build_tasks2.py    science, Persian, English, netdiag, DevOps
build_electronics.py   electronics MC + embedded C++, mock Arduino header
build_lint.py      ruff and clippy task sets
verify_tasks.py    positive controls (correct answers must pass)
negcontrols.py     negative controls (wrong answers must fail)
verify_cpp.py      embedded C++: right answer passes, plausible wrong answer fails
determinism.py     reproducibility under a fixed seed
tasks/*.json       custom tasks + tasks/std_*.json (standard, via standard_fetch.py)
```

## How scoring works

**Code tasks** — model output is extracted from markdown, compiled/run in a
subprocess against unit tests with a wall-clock timeout and an address-space cap.
A pass requires exit 0 **and** a sentinel printed after the test body.

**Multiple choice** — the prompt ends exactly where the answer letter goes; one
token is generated and the top-k logprob distribution over letters is read. This
is deliberate: at 3–8 tok/s, scoring 122 MC items by generation costs hours, and
by logprob costs minutes.

**Tool calling** — 8 chains of dependent steps; `pass^k = (c/n)^k` is the
headline metric, with trace scoring per step.

## What to attack, in priority order

### 1. Question and answer correctness (highest value)

The suites are mostly hand-authored. Any wrong gold answer, ambiguous question,
or leading prompt is a real defect because it silently corrupts the comparison.

- Is every gold answer **actually correct**? Check `science`, `electronics`,
  `persian`, `english`, and the embedded C++ assertions. Compute anything
  numerical by hand. The author found a wrong expected CRC this way.
- Are questions **ambiguous** — two defensible readings, or an unstated
  convention that the model would have to guess? (Example: HumanEval-style tasks
  must say whether the model writes the whole function or just the body.)
- Is the **gold answer leaked** in the question text? Count how often the gold
  string appears elsewhere in the prompt.
- Is any question **leading** — does the phrasing make one option obviously
  correct without knowledge?
- Do the **distractors** make sense, or is one obviously wrong/generated, letting
  a model eliminate rather than know?
- Persian: check ZWNJ, Arabic vs Persian letters, digit scripts, and whether a
  *correct* Persian answer could be marked wrong by normalisation.
- Are any physics/chemistry/biology/electronics facts actually wrong as stated?

### 2. Logical defects in scoring

- Can a **wrong answer score correct**? Construct one for each scorer.
- Can a **right answer score wrong**? Construct one. Argue which direction is
  more frequent, because that determines whether a model is under- or
  over-reported.
- Is any metric **labelled incorrectly** — named "strict" but computed leniently,
  or inverted? Re-derive the formulas.
- Is `confidence`/`letter_mass` used for anything, or decorative?
- Does aggregation assume fields that some record lacks?

### 3. Gaming

Could a model score well **without the ability being measured**? Look for:
always answering the same letter; echoing required keywords (`check_devops`
substring matching); suppressing a linter instead of satisfying it; a prompt
that rewards saying "healthy" regardless of evidence.

### 4. Extraction and parsing

`extract_code` and `_strip_preamble` are heuristic. Find completions that are
mangled, or where prose reaches the compiler. Construct inputs, run them.

### 5. Agentic chains

Read `agentic.py`'s `CHAINS` and `run_chain`. Are the expected outcomes correct?
Is `verdict_yes_no` sound, or does it misread negation? Could a chain pass without
the model actually using the tools?

### 6. Determinism

`determinism.py` claims fixed-seed reproducibility and that seeds are honoured.
Falsify it if you can. Note that the LLM servers are on this machine only if
running — if not, reason about the code paths instead of calling them.

## Method

**Execute the code.** The author's own audit caught a defect where the test field
defined a function and never called it — zero assertions ran, and 164 items
passed on any parseable stub. Reading the code suggested it was fine. Only
running it found it.

Write and run small Python scripts against `scorers.py`, `agentic.py` and
`tasks/*.json`. Use `python3 -c` or heredocs. g++, rustc, ruff, clippy-driver,
rustfmt are installed.

For every claim, give the **exact command and its output**.

## Output format

For each finding:

```
### [CRITICAL|MAJOR|MINOR] Short title

Claim: one sentence.
Evidence: exact command + output proving it.
Impact: what the benchmark reports wrongly, and in which direction.
Fix: concrete suggestion.
```

- **CRITICAL** — a wrong answer scores correct, or a right answer scores wrong,
  in a way that plausibly occurs during a real run.
- **MAJOR** — plausible trigger, or corrupts a metric's meaning.
- **MINOR** — cosmetic, or needs unusual input.

Then a short section listing what you checked and found **correct**, so the
author knows what is already covered. Prefer "I could not break this, here is
what I tried" over silence.

Do not speculate. If you cannot reproduce it, say so explicitly and mark it
UNVERIFIED.

## ALREADY FOUND — do not re-report these

1. HumanEval test field defined `check()` but never called it → 164 items passed
   on any parseable stub. Fixed (call baked into the stored test).
2. Answer-position bias: 69% of golds at index B in electronics, 0% at D. Fixed
   by seeded shuffle in `debias_mc.py`.
3. `sys.exit(0)` / `os._exit(0)` scored a PASS. Fixed with a post-test sentinel.
4. Agentic polarity inverted: correct "not allowed" scored 0, wrong "allowed"
   scored 1. Fixed with `verdict_yes_no()` and explicit `expect_yes`.
5. `extract_code` took the first block and passed prose to the compiler; nested
   backticks truncated code. Fixed.
6. Duplicate letter tokens clobbered instead of accumulating. Fixed.
7. `max()` on an empty distribution aborted the whole run. Fixed.
8. Confidence renormalised within the letter set, so it could never go below
   1/len(letters). Fixed (absolute letter mass, gated at 0.15).
9. `pass_hat_k` computed `C(n-c,k)/C(n,k)` = `1 - pass@k`, inverting the metric.
   Fixed to `(c/n)^k`.
10. `aggregate()` raised `KeyError` in 3 categories on records lacking fields.
    Fixed with `.get()`.
11. `"(A"` rejected because `strip("*:)")` left the leading paren. Fixed.

## One thing the author got wrong and wants to know about

The author told the user that a model's 98.75% HumanEval score was "genuine",
ruling out three explanations, and never checked that the test actually executed.
If you find a claim in `../../../README.md` or
`../reviews/REVIEW.md` that the code does not support,
say so plainly. I want to know where my reasoning was flawed, not reassured.
