# Local LLM Code Benchmark — Research Specification
Machine: i5-6300U / 2 cores / 7.6 GB RAM, llama.cpp or Ollama @ 3-8 tok/s. No Docker, no node.

## 0. ENVIRONMENT CORRECTIONS (verified on this box)

| Brief said | Reality | Evidence |
|---|---|---|
| clippy NOT installed | **clippy 0.1.97 IS installed** | `cargo clippy --version`; `rustup component list --installed` shows `clippy-x86_64-unknown-linux-gnu` |
| clippy requires cargo | **clippy-driver runs standalone** | `clippy-driver f.rs --edition 2021 --emit=metadata -o OUT --error-format=json -W clippy::pedantic` works with zero Cargo.toml |

This changes the plan: you do NOT need cargo project scaffolding for Rust linting. Use `clippy-driver` directly.

Also verified: `ruff check - --isolated --select ... --output-format json` reads from stdin (no temp files).
`ruff 0.17.0` ships 971 rules, 139 of them preview-only (14.3%).

---

## 1. CODE GENERATION

### 1.1 HumanEval — STANDARD, use as-is
Source: Chen et al. 2021, arXiv:2107.03374. Data: `openai/human-eval`, `data/HumanEval.jsonl.gz`, 164 problems.
Fields: `task_id`, `prompt`, `entry_point`, `canonical_solution`, `test`.

Verified prompt shape (HumanEval/0): imports, `def <entry_point>(<typed args>) -> <ret>:`, then a docstring
containing the spec and `>>>` doctests. The model completes the **body only**.

Official harness (`human_eval/execution.py`) builds:
```
check_program = problem["prompt"] + completion + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"
```
Runs in a `multiprocessing.Process`, under `signal.setitimer` timeout, with `reliability_guard()` that
blots out `os.kill`, `shutil.rmtree`, etc. Only `result[0] == "passed"` counts.

PITFALL: `reliability_guard` is explicitly documented as NOT a security sandbox. On an untrusted local model
this is acceptable, but never point it at a directory with anything you care about.

Scoring: pass@k. The canonical estimator (Chen et al. §3, and MultiPL-E `pass_k.py`):
```
pass@k = 1 - C(n-c, k) / C(n, k)
```
Numerically stable form actually used in `nuprl/MultiPL-E/pass_k.py`:
```python
def estimator(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```
Reported as the MEAN over problems. FINAL score = mean of per-problem pass@k.

HARD CONSTRAINT (stated in openai/human-eval README): **you cannot estimate pass@k when n < k.** With 1-3
samples per task on this hardware, only pass@1 and pass@2 (at n>=2) are defined. Do not report pass@k > n.

### 1.2 MBPP — STANDARD, but use the sanitized split
Source: Austin et al. 2021, arXiv:2108.07732. Full MBPP = 974 problems (task_id 1-974), verified locally.
Sanitized = 427 problems, task_id 11-510 (human-verified, unambiguous, self-contained). Verified counts:
426 rows / ids 2-809 from the MultiPL-E copy of `sanitized-mbpp.json` — treat 427 as the canonical figure and
reconcile the 426 discrepancy if you depend on it.

Canonical prompt template (Austin et al. 2021, §"Text prompt"):
```
You are an expert Python programmer, and here is your task: {text}
Your code should pass these tests:
{test_list joined by newline}
[BEGIN]
{code}
[DONE]
```
Evaluation is 3-shot (task_ids 2,3,4 as the shots), and scoring runs `test_list` asserts against the
generated `code`. MBPP scores are typically 3-shot; HumanEval is 0-shot. Do not mix.

### 1.3 EvalPlus / HumanEval+ / MBPP+ — RECOMMENDED upgrade, cheap
Liu et al. 2023, arXiv:2305.01210 / NeurIPS 2023. HumanEval+ has ~80x the test inputs of HumanEval;
MBPP+ ~35x. They also publish **HumanEval+-mini**, ~47x smaller, which retains similar discriminating power.

This matters a lot here: original HumanEval tests are weak enough that ~20% of solutions that pass them
still fail on the harder inputs. A local 7B model will look better than it is on base HumanEval.

Cost check: EvalPlus is pure Python `assert`s, no network, no Docker. Practical on this box.
Pick HumanEval+-mini if you need to stay under ~10 min/model at 5 tok/s.

### 1.4 LiveCodeBench — NOT practical here
Jain et al. 2024, arXiv:2403.07974. Key property: problems are **time-windowed by release date**
(release_v1..v5) and you may only score problems released *after* the model's cutoff. That requires
knowing the cutoff, which open-weight models don't publish cleanly, and the problems are competitive-
programming scale (stdio + stdin parsing). A 3-8 tok/s model will not finish these in budget.
Use it only as a contamination *argument* in your writeup, not as a scored suite.

### 1.5 CodeContests — NOT practical here
13,610 Codeforces problems, IO-based (stdin/stdout diff). Same throughput problem as above. Cite as the
standard for competitive programming; do not run.

### 1.6 BigCodeBench — NOT practical here
Zhuo et al. ICLR 2025, arXiv:2406.15877. 1,140 tasks across 139 libraries / 7 domains, 723 function calls.
Requires the library APIs to be installed to execute the tests. No network => no. Cite only.

### 1.7 Aider Polyglot — WORTH IT, this is your best "algorithmic thinking" suite
225 hard Exercism exercises across C++, Go, Java, JavaScript, Python, **Rust**.
Prompt convention: the model is shown the existing source file and the test suite, and must emit
**SEARCH/REPLACE blocks** (aider's "diff" edit format), not whole-file rewrites.

Scoring: **pass rate** — fraction of exercises whose test suite passes after applying the model's edit.
Per-language breakdown is published (so you can report Python vs Rust separately for free).

This is the single best fit for your constraints: per-language, no network, deterministic tests,
moderate prompt sizes, and it covers Rust.

PITFALL: SEARCH/REPLACE formatting failures are scored as task failures, not as harness errors. Decide
and document whether format failures count. I recommend: report them separately, because for a 3B model
at low temperature the format failure rate may dominate the correctness signal.

### 1.8 Practical scoring stack for this machine
1. 0-shot HumanEval+ / MBPP+ via a pure-Python `exec` harness modelled on `execution.py` (no multiprocessing
   needed if you `subprocess` a fresh `python3 -c`; that is your isolation boundary on Linux with no Docker).
2. Mean pass@1 over n samples, plus the unbiased pass@k for k <= n.
3. Always log per-task n and c so the estimator is auditable.

---

## 2. BUG FIXING

### 2.1 The standard, and why you can't run it
- **SWE-bench** (Jimenez et al. 2024). Verified: the official harness page states the evaluation uses
  Docker for reproducible environments; the FAQ states Docker is required. Verified / Lite / Multimodal
  splits exist. Each instance needs the repo at a base commit + the project's own test command.
  **Not runnable here.** No Docker, and the repos are large.
- **Defects4J** (Just et al. 2014): 835+ Java bugs, pairs of buggy/fixed, needs a trigger test. Java,
  heavyweight build. Not runnable.
- **BugsInPy** (Widyasari et al., ICSE-SEIP 2020): 493 real Python bugs in 17 projects. Verified from the
  project README: **"Use Docker ... build the Docker image"**. Requires a per-project venv pinned to old
  dependency versions. **Not runnable without Docker.**
- **CodeRepair** (Chen et al., ICSE 2022): the Defects4J-derived code-repair pair set. Also Java.
- **CodeXGLUE / Bugs2Fix** (Lu et al. 2021, arXiv:2102.04664): ~20k buggy/fixed Java function pairs.
  Java; and it is a *pair-construction* dataset, not an executable pass/fail suite.

CONCLUSION: every standard bug-fixing benchmark is repo-scale + environment-pinned. There is no standard
that runs offline on a bare Python/Rust toolchain. Say this explicitly in your writeup; it is a finding.

### 2.2 The defensible local proxy: single-file bug-fix with embedded tests
No published standard exists for this. Construct it, and hold it to these rules:

**Task format** (my specification, not a citation):
```
# TASK <id>
# The function below has a bug. One of the assertions below fails.
# Fix the function. Do not change the assertions. Do not add new imports
# beyond <allowed stdlib list>.

<code under test, self-contained, no I/O, no network, no clock, no randomness>

# Failing assertion (verified to fail on the buggy version):
<single assert that raises AssertionError on the buggy code>

# Additional assertions that must keep passing (PASS_TO_PASS):
<2-5 asserts>
```

**Construction protocol** (this is what makes it a benchmark and not a toy):
1. Write the function CORRECT first.
2. Introduce one mutation (off-by-one, wrong operator, missing edge case, wrong variable).
3. Machine-verify: `FAIL_TO_PASS` assert fails on the buggy version; all `PASS_TO_PASS` asserts pass on
   the *fixed* version.
4. Ship only (buggy_code, FAIL_TO_PASS, PASS_TO_PASS). Never ship the fix.
5. Randomise assertion order so the model can't hard-code the first one.

**Scoring** — this is straight SWE-bench's methodology transplanted to a single file:
```
resolved   = FAIL_TO_PASS passes AND all PASS_TO_PASS still pass
score      = mean(resolved)   # report this as the headline bug-fix rate
```
Track two failure modes separately:
- `test_regression`: fixed the bug but broke a PASS_TO_PASS -> the model overfit to the visible assert.
- `partial_fix`: better to also report the per-assert pass rate, since with n<=3 samples the binary
  outcome is noisy. This mirrors RefactorBench's subtask scoring (see §3).

**Why this is defensible and not invented:** FAIL_TO_PASS / PASS_TO_PASS is SWE-bench's own scoring
construction; you are only changing the granularity from repo to file. State that lineage in the writeup.

**Pitfalls:**
- The model will find the visible assert and special-case it. That is why PASS_TO_PASS exists, and why you
  must hold out asserts the model never sees.
- Don't include anything needing network, filesystem, time, or randomness. You have no container to
  contain a misbehaving `os.system`.
- Run each candidate in a fresh `python3` subprocess with a wall-clock timeout. 2 seconds is plenty for
  these sizes and keeps the suite bounded.

### 2.3 Optional: harness-assisted repair loop
If you have budget, add a second arm where the failing assertion output is fed back once. Report it as a
separate column. RefactorBench's finding (below) applies here too: agents that lint-gate every edit break
on multi-step tasks, because intermediate broken states are part of the legitimate solution.

---

## 3. REFACTORING

### 3.1 Is there a standard? Yes, and it's RefactorBench — but it is repo-scale.
**RefactorBench: Evaluating Stateful Reasoning in Language Agents Through Code.**
Gautam, Garg, Jang, Sundaresan, Moghaddam (UC Berkeley + Microsoft), arXiv:2503.07832, 10 Mar 2025.
Repo: `github.com/microsoft/RefactorBench`. Read the full paper; the scoring method is the reusable part.

Verified design facts from the paper:
- 100 handcrafted tasks, **multi-file by construction** (2-31 files edited, mean 4.3), across 9 popular
  Python repos. Repo sizes: mean 2,328 files / 304K lines.
- Three mutually exclusive instruction sets: **lazy** (mean 16 words), **base** (20.6 words),
  **descriptive** (68.8 words).
- **Scoring: AST-based unit tests, not exact match.** For every core edit, developers write a unit test that
  parses the target file's AST and asserts the refactor has the correct *broad structure*, explicitly to
  remove dependence on exact-line matching. 2-27 subtests per task, mean 6.51, mean 131 lines of test code.
- **A task counts as solved only if ALL its subtask tests pass.** They also report subtask pass rate as a
  finer-grained signal (that is what they use for their state-representation ablation).
- Reference solutions are withheld from release to prevent overfitting. Only the test files ship.
- Baselines: SWE-agent + GPT-4 -> 12% lazy / 22% base / 27% descriptive. Human developer, 5 min limit: 87%.

Also relevant newer work (cited by RefactorBench; I did not read these in full, treat as pointers):
SWE-Refactor (arXiv:2602.03712, 1,099 Java refactorings from 18 repos), SmellBench (arXiv:2606.05574),
SWE-Bench ProMax (multilingual, 170 instances). All repo-scale, all infeasible here.

### 3.2 The local proxy — borrow RefactorBench's METHOD, shrink the SCALE
No standard exists at single-file scale. Here's the defensible construction, and it is directly derived
from RefactorBench's scoring approach:

**Task format:**
```
# TASK <id>
# Refactor the function below so that <stated goal>.
# Behaviour must not change: the given tests must still pass.
# Additional structural requirements are checked automatically.

<before>
# TESTS THAT MUST STILL PASS
<tests>
```
Derive the goal vocabulary from **Fowler, Refactoring (2nd ed. 2018)** — that is RefactorBench's own
reference point and citing it keeps you anchored to an established taxonomy, not an invented one.
Good single-function subset that is genuinely checkable:
- Extract Function / Inline Function
- Replace Conditional with Polymorphism (needs a new type -> use AST test)
- Replace Magic Literal with Named Constant
- Replace Loop with Pipeline
- Introduce Parameter Object
- Move Function / Move Method
- Replace Nested Conditional with Guard Clause
- Rename Method (functionality-preserving; the killer task because it has no exact-match ambiguity)

**Scoring — two gates, mirroring RefactorBench's "all subtests must pass" rule:**
```
gate_1 behaviour_preserved : all provided tests still pass        (hard requirement)
gate_2 structural          : all AST assertions pass              (the RefactorBench method)
resolved = gate_1 AND gate_2
subtask_rate = fraction of gate_2 assertions that pass     (report alongside, as RefactorBench does)
```
Report **subtask_rate** as the primary number when n is small: it has far lower variance than the binary
`resolved` and shows partial progress, exactly as RefactorBench found.

For the AST assertions, the RefactorBench approach is: parse the edited file with `ast`, then assert on
structure. E.g. for "Replace Loop with Pipeline":
- assert exactly one `ast.ListComp` exists in the function body
- assert no `ast.For` remains in the function body
- assert `ast.For` count decreased vs the `before` source
This is dependency-free (stdlib `ast`), no network, no Docker. Perfect for this box.

### 3.3 Complexity measurement
Standard tooling, all offline-capable:
- **radon** (`rubik/radon`): McCabe cyclomatic complexity, raw metrics, Halstead, **Maintainability Index**.
  Python-native, AST-based. The canonical Python choice. Cite it by name; don't hand-roll CC.
  Note: radon is a pip package. If you cannot pip-install, you have to vendor it or skip CC entirely.
- **lizard**: language-agnostic CC (supports Python AND Rust), used by several research studies.
  Relevant because it works on your Rust side too, where radon does not.
- Maintainability Index (Oman & Hagemeister, 1992) is the standard normalisation layered on CC.

### 3.4 Honest framing for your writeup
"No standard single-file refactoring benchmark exists. RefactorBench is the standard and requires
repository-scale contexts; we adopt its scoring method (AST-based subtask tests, all-must-pass,
subtask-rate reporting) at single-file scale and anchor the task taxonomy to Fowler 2018."

---

## 4. RUST AND PYTHON SPECIFICALLY

### 4.1 MultiPL-E is the answer, and I verified the exact format
Cassano et al., "MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation",
IEEE TSE 2023 (arXiv:2208.08227). Repo `nuprl/MultiPL-E`; data on HF `nuprl/MultiPL-E`.

Verified dataset sizes (via HF datasets-server):
- `humaneval-rs`: **156** problems
- `mbpp-rs`: **354** problems
Columns: `name`, `language`, `prompt`, `doctests`, `original`, `prompt_terminology`, `tests`, `stop_tokens`
`prompt_terminology` is `"reworded"` — the docstrings were reworded to remove Python-specific phrasing.

**Exact Rust prompt** (verified, HumanEval_0):
```rust
/// Check if in given vector of numbers, are any two numbers closer to each other than
/// given threshold.
/// >>> has_close_elements(vec![1.0, 2.0, 3.0], 0.5)
/// false
/// >>> has_close_elements(vec![1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
/// true
fn has_close_elements(numbers: Vec<f64>, threshold: f64) -> bool {
```
Note: docstring -> `/// ` comments; `typing.List[T]` -> `Vec<T>`; `int`->`isize`, `float`->`f64`,
`str`->`String`, `Dict`->`HashMap`, `Optional`->`Option`. Doctest examples are REWRITTEN into valid Rust
(`[1.0,2.0,3.0]` -> `vec![1.0,2.0,3.0]`, `True` -> `true`).

**Exact Rust tests** (appended verbatim):
```rust
}

fn main() {
    let candidate = has_close_elements;
    assert_eq!(candidate(vec![1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3), true);
    ...
}
```

**ANSWERS TO YOUR SPECIFIC QUESTIONS:**
- **Does the model need `fn main()`?** NO. The harness appends `fn main()` itself. The model writes only the
  function body. Adding one is a compile error (duplicate main) -> scored as failure. Say this in the prompt.
- **Does the model need `#[test]`?** NO. Tests are plain `assert_eq!` inside a harness-generated `fn main()`.
  Do not ask for `#[test]`; it is not used and won't be compiled here.
- **Stop token:** `["\n}"]` — verified as the `stop_tokens` field on every Rust row. Generation MUST stop at
  the closing brace of the function body. Enforce it in your sampler.

### 4.2 The canonical Rust evaluator — verified by running it
From `nuprl/MultiPL-E/evaluation/src/eval_rust.py`:
```python
build = subprocess.run(["rustc", path, "-o", basename], capture_output=True, timeout=15)
if build.returncode != 0: status = "SyntaxError"
else:
    output = subprocess.run([basename], capture_output=True, timeout=5)
    status = "OK" if output.returncode == 0 else "Exception"
```
Status vocabulary: `OK` | `SyntaxError` | `Exception` | `Timeout`.

I implemented this locally and verified end-to-end: correct solution -> OK; `true` stub -> Exception
(assert fires); `return 42` -> SyntaxError; empty completion -> SyntaxError. Whole 4-case sweep: 0.9 s wall.

NOTE: there is NO `reliability_guard` equivalent on the Rust side. `rustc` is safe, but the *compiled binary*
runs your model's code natively with no sandbox. Only ever run binaries from a task whose code you can
inspect, or accept that a malicious/broken model can touch the filesystem. Add `timeout` and run under a
scratch cwd.

### 4.3 Honest caveat on reconstructing the Rust set yourself
I attempted to regenerate HumanEval-Rust by running MultiPL-E's own translator
(`dataset_builder/humaneval_to_rs.py`) over `HumanEval.jsonl.gz`. Only **29/164** problems were
translatable — the rest fail on `implicitly untyped argument` / `unknown name list`. MultiPL-E reached 156
because they used hand-annotated type-annotated versions plus reworded prompts.

So: **download the MultiPL-E release, do not roll your own translation**, unless you are willing to
hand-annotate. This is a concrete, citable methodological point you should state.

### 4.4 Codeforces / Rust-specific algorithm benchmarks
I could NOT verify a canonical "rust-eval" or "CrateEval"/"RustBench" as a community-standard suite. My
searches surfaced nothing meeting that bar. **FLAG AS UNVERIFIED / LIKELY NON-EXISTENT.**

Closest legitimate thing to cite: **Aider Polyglot** covers Rust among its 6 languages, and MultiPL-E
covers Rust properly. Those are your citable Rust sources. For a Rust-specific algorithmics suite, the
defensible move is to use `mbpp-rs` (354 problems) + `humaneval-rs` (156) and be explicit that you are
using MultiPL-E, not a bespoke Rust benchmark.

---

## 5. LINTER-DRIVEN EVALUATION — THE NOVELTY

### 5.1 Prior art: essentially none as a *scoring method*
I searched specifically and found no published benchmark that scores "model writes code, we lint it, count
diagnostics" as a primary metric. What exists:
- **Linters as in-agent feedback**: SWE-agent's ACI runs a linter on each edit and rejects edits that
  introduce syntax errors. This is a *mechanism*, not a *metric* — the linter is never scored.
- **RefactorBench (arXiv:2503.07832) is your best prior art and it is an ARGUMENT AGAINST naive lint-gating.**
  Verified quotes from the paper: 78.4% of trajectories error in a code editing step; intermediate broken
  states are frequently *necessary*; they conclude "the practice of automatically enforcing strict linting
  rules and rejecting edits based on errors proves to be an impractical approach to scaling real world
  agents". They also measured that one common linting-error edit consumed ~1,466 tokens (o200k_base
  tokenizer) of agent context.
- **Complexity-metric-as-quality-gate** appears in applied papers (cyclomatic complexity, Halstead effort,
  Maintainability Index, SLOC compared human vs AI code) but is used as a descriptive statistic, not as a
  discriminative benchmark with a pass/fail contract.
- **LintCFG / automated linter configuration** papers (DSL-based compilation of coding standards into linter
  configs) exist but are about configuring linters, not about scoring models.

So your novelty claim is defensible IF you (a) cite the above as the boundary you are working against,
and (b) do not claim lint-as-a-hard-gate, which has published evidence against it.

### 5.2 Verified ruff facts (ruff 0.17.0, on this box)
- 971 rules total. Category counts: pedantic 310, suspicious 173, correctness 130, complexity 121,
  security 78, style 69, formatting 68, performance 7, restriction 15.
- 139 rules are preview-only (14.3%).
- Default select is `E4, E7, E9, F`. This is much narrower than "E,F,W".
- `--select ALL` is a SPECIAL CASE: it auto-disables mutually incompatible pydocstyle rules. Verified
  output: `warning: D203 and D211 are incompatible. Ignoring D203.` and same for D212/D213.
  So `ALL` is NOT a stable set across ruff versions.
- Selector precedence (documented): `ALL < category < linter group < rule prefix < rule`.
- Categories are ordered "in order of decreasing severity":
  correctness > suspicious > complexity > performance > style > security > formatting > pedantic > restriction.
  Only correctness, suspicious, complexity, performance, style are on by default.
- Exit codes: 0 clean, 1 violations found, 2 abnormal termination. (`--exit-zero` suppresses 1.)
- `--output-format json` emits: `cell, code, end_location, filename, fix, location, message, name,
  noqa_row, severity, url`. `fix` contains `applicability` (safe/unsafe) — useful, see below.

### 5.3 Verified clippy facts
- 838 lints total (from rust-lang.github.io/rust-clippy master index).
- Lint groups: `correctness, suspicious, style, complexity, perf, pedantic, nursery, restriction, cargo`.
- **PITFALL (verified by running):** `-W clippy::restriction` is wrong. Clippy emits
  `blanket_clippy_restriction_lints`: "`clippy::restriction` is not meant to be enabled as a group".
  You must enable restriction lints one at a time.
- `clippy::cargo` group members are `cargo-common-metadata, multiple-crate-versions,
  negative-feature-names, redundant-feature-names, wildcard-dependencies` — several need a real Cargo.toml.
  Without cargo context these produce noise or nothing. **Exclude the `cargo` group**; it is not a
  statement about the model's code.
- Use `clippy-driver`, not `cargo clippy`, for single-file eval (see §0).

### 5.4 The defensible scoring function
Raw diagnostic count is the wrong headline. It is dominated by D1xx (missing docstrings: 3 rules × N
functions) and line-length (E501), neither of which is a capability signal, and it scales with output
length. I built and ran a category-weighted scorer; on the same file it gave raw=9 / weighted=22, showing
the weighting is doing real work.

**Recommended metric — report four numbers, not one:**

```
(1) weighted_severity       = sum over diagnostics of W[category(rule)]
(2) correctness_violations  = count of category=="correctness" diagnostics
(3) lint_density            = weighted_severity / max(1, loc)      # length-normalised
(4) clean_rate              = fraction of samples with ZERO correctness+suspicious diagnostics
```

with weights derived from ruff's OWN documented severity ordering, not invented:
```
W = {correctness: 4, security: 3, suspicious: 3, performance: 2, complexity: 2,
     style: 1, formatting: 1, pedantic: 1, restriction: 1}
```
This is defensible because the ordering is the linter vendor's, not yours. Say exactly that in the writeup.

**Rule selection — do NOT use `ALL`.** Justification, all verifiable:
- `ALL` silently drops D203/D211 and D212/D213, so the rule set is version-dependent.
- 310 of 971 rules are `pedantic` and 15 are `restriction`; the ruff docs explicitly say pedantic and
  restriction are "too opinionated" and "not meant to be enabled as a whole" (paraphrasing restriction's
  own clippy analogue). Scoring on them measures disagreement with the vendor's house style, not skill.
- `CPY001` (missing copyright notice) fired on my test file under `ALL`. That is not a capability signal.
- `preview` rules change without semver. **Pin `select` explicitly and record `ruff --version` in results.**

Recommended explicit set, which is also the set you named:
```
select = ["E","F","W","I","N","UP","B","SIM","C4","PT","PTH","PERF","RUF"]
preview = false
```
Note: with this set, `D1xx` docstring rules and `CPY*` are excluded, which removes the dominant length artefact.

**Additional axes worth reporting, because they're cheap and each answers a different question:**
- **fixability**: split diagnostics by `fix.applicability`. An autofixable diagnostic (unused import,
  `== true`) is a much weaker failure than a non-fixable one. Report
  `unfixable_weighted` separately.
- **suppression gaming**: models sometimes emit `# noqa` to silence lint. Run with `RUF100`
  (`--extend-select RUF100`) to catch *unused* suppressions, and count model-emitted `# noqa` /
  `# type: ignore` / `#[allow(...)]` occurrences. A model that suppresses is not more clean, it is
  suppressing. Without this check your metric is trivially gameable and a reviewer will say so.
- **cross-tool agreement**: rustc's own lints (`unused_variables`, `dead_code`) are free (rustc runs
  anyway for the code-gen arm) and independent of clippy. Report them as a separate signal.

### 5.5 Design requirement that makes this a real benchmark
Score lint **jointly with correctness on the same sample**. A model that writes correct-but-ugly code and
a model that writes ugly-and-wrong code must be distinguishable, and "does it lint" alone can't tell you.
Report the 2x2:
```
                 lint_clean   lint_dirty
correct            A            B
incorrect          C            D
```
Then `usefulness = A/(A+C)` is the headline ("correct AND clean"), and `A/D` vs `B/D` tells you whether
lint-cleanliness predicts correctness at all for that model. That contingency question is what makes this
more than a style score.

Do NOT let the model iterate against the linter (RefactorBench's evidence says it backfires and burns
tokens). **Single-shot lint. One generation, then lint. No repair loop.**

---

## 6. SCORING METHODOLOGY

### 6.1 pass@k vs mean@k
- `pass@k` = P(at least one of k samples passes). Unbiased estimator per Chen et al. 2021 (formula in §1.1).
  Meaningful only when n >= k, and it is optimistic.
- `mean@k` = k * (c/n) — expected fraction of correct samples. Unbiased regardless of n.
- **For 1-3 samples, report pass@1 and mean@n. Do not report pass@k for k > n.**
- At 3-8 tok/s on 2 cores, the honest budget is n=1..3. State n in every table cell. Never average across
  different n without saying so.

### 6.2 Variance: this is your dominant threat
Best citable evidence: **"Do Repetitions Matter? strengthening Reliability in LLM Evaluations"**,
arXiv:2509.24086 (28 Sep 2025). Verified findings:
- **10/12 slices (83%) had at least one pairwise rank inversion** for single stochastic runs vs the 3-run
  majority. Single-run leaderboards are brittle.
- Averaging 1 -> 3 runs shrank SE only ~5% (0.0288 -> 0.0273), but **two runs removed ~83% of single-run
  rank inversions**. The benefit is rank stability, not narrower CIs.
- Their recommendation: **>= 2 repetitions** under stochastic decoding.
- They note Epoch AI runs each model 8-16 times; smaller teams can't afford that.
- Intra-class correlation (ICC) across their slices: mean 0.68; they treat <0.60 poor, 0.60-0.75 moderate,
  0.75-0.90 good. Report your ICC.
- Their best practice list: "treat evaluation as an experiment, report uncertainty, use >=2 repetitions".

CRITICAL PITFALL, cited in that paper: **Bowyer, K. et al., "Position: Don't Use the CLT in LLM Evals With
Fewer Than a Few Hundred Datapoints", arXiv:2503.01747.** With ~150-500 tasks you do NOT get Gaussian
intervals for free. Use **Wilson score intervals** for binomial proportions (that paper's recommended
alternative), not normal-approximation CIs.

Concrete recommendations for your setup:
- **n = 3** per task if the model is small enough that 1/3 samples succeed (low n is affordable when the
  success rate is low). If the model solves most tasks, drop to n=2 and spend the budget on more tasks.
  The cost-optimal allocation is more TASKS, not more SAMPLES, at 3-8 tok/s.
- Report mean and Wilson 95% CI for every headline number.
- Report per-task pass *rates* (c/n), not just the binary, so you can show the spread.

### 6.3 Normalising across tasks of different difficulty
Per-task pass rates are not comparable across tasks of different difficulty. Two defensible standard routes:

(a) **Mean over tasks** (what everyone does) — simple, but a task everyone fails contributes 0 to everyone
    and a task everyone solves contributes 1 to everyone, so most of the signal comes from mid-difficulty
    tasks. State your task difficulty distribution.

(b) **Item Response Theory.** Cited in the same paper: Schilling-Wilhelmi et al., "Lifting the Benchmark
    Iceberg with Item Response Theory" (ICLR 2025). Fits a 1PL/2PL model to get per-task difficulty and
    per-model ability, giving a latent ability estimate that IS comparable across task sets. This is the
    methodologically strongest normalisation and it is citable. Overkill for 150 tasks, defensible if you
    want to claim sophistication.

Recommendation: do (a) as the headline, and (b) as a robustness check if time allows. If you stratify
tasks into difficulty buckets (easy/medium/hard by reference-solution agreement across models), report the
buckets separately — that is cheap, transparent, and sidesteps the normalisation problem honestly.

### 6.4 Pairwise comparison (Elo / Bradley-Terry) — what it's for
Standard references: Chatbot Arena uses Bradley-Terry for statistical inference on model strength
(arena.ai/blog/extended-arena; statistical framework paper arXiv:2412.18407). Elo is the online
approximation; BT is the batch MLE. BT gives more stable ratings than Elo because it doesn't over-weight
recent matches.

**Recommendation for you: do NOT use Elo/BT for the headline.** They need human pairwise preferences.
You have no preference oracle; you'd be comparing pass/fail against pass/fail, which is what a mean already
tells you. Elo/BT is the right tool only if you collect *human* pairwise judgments of code quality.

**Use it for one thing**: if you do collect human pairwise preferences on the lint+correctness outputs,
BT is the correct aggregator and gives you uncertainty on the ratings. Then report the BT ratings with
confidence intervals.

### 6.5 Variance reduction: paired / common-random-numbers designs
These are standard experimental-design practice (Fisher 1935; Box/Hunter/Hunter 2005 — cited in
arXiv:2509.24086) and are the highest-leverage thing you can do here:
- **Fixed task set, fixed prompts, fixed decoding params, fixed seeds.** All models see the identical
  tasks. This is the CRN design and it's free.
- **Same seed sequence per task across models**, so a stochastic difference between models isn't a seed
  artefact.
- **Greedy decoding (temperature=0) as the primary arm** for reproducibility, and treat temperature>0
  as a separate stochasticity study. Note from arXiv:2509.24086 that higher temperature greatly increased
  flip rates (GPT-4 family ~35% of problems flipped at T=0.7). If you evaluate only at T>0 with n=1 you are
  measuring noise.
- **Paired comparison**: compare models per-task, then test the paired differences (McNemar for binary
  pass/fail, Wilcoxon signed-rank across tasks). This removes task-difficulty variance, which is the
  dominant term. Much tighter than comparing two independent means.
- **Bolster**: Miller (2024), "Adding Error Bars to Evals" (arXiv:2411.00640), bootstrapping over tasks.
  With no IID CLT assumption this is the safe way to get intervals at your sample size.

### 6.6 The reporting template
For every number in your results:
```
n_samples_per_task   : <n>
n_tasks              : <N>
decoder              : <model, quantisation, llama.cpp/Ollama version, seed>
prompt_hash          : <sha256 of exact prompt template>
ruff_version         : 0.17.0   (and the exact --select list)
rustc/clippy_version : 1.97.0 / 0.1.97
estimate             : <value>
CI                   : <Wilson 95%>
```

---

## 7. DURABILITY PROTOCOL (2026-10-20, after REVIEW4)

A benchmark that loses data to the machine it runs on is not a benchmark. The
protocol below is what makes "no missed data" a property of the harness
rather than a hope. Full incident write-up: `REVIEW4.md`.

### 7.1 Three invariants
1. **Nothing scored is ever lost.** Every finished item is written to disk in
   the second it is scored (`_ProgressList` → per-item checkpoint). A kill
   can cost the item *in flight*, never the rep.
2. **Nothing broken is ever presented as a score.** A backend failure
   produces a stub, which is excluded from every denominator and reported as
   missing. Missing is never zero.
3. **Nothing incomplete is ever presented as complete.** A result file is
   `partial` when coverage is short: for every (task, rep) pair the file
   must hold as many valid records as there are items to run. A stub that a
   later attempt replaced no longer keeps the file partial; a task that never
   ran is always a gap.

### 7.2 The failure modes this defends against
| Threat | Mechanism | Defence |
|--------|-----------|---------|
| Laptop idle-suspend | logind SIGTERMs user processes | `systemd-inhibit --what=sleep:idle` held for the whole run; SIGTERM handler checkpoints and exits 143 |
| Server OOM | kernel OOM-killer takes `llama-server` | per-item RSS guard (clean abort) + fresh server + item-level resume |
| Long rep killed | results in RAM until rep end | per-item checkpoint |
| Stubs counted as progress | resume loop sees "records" | progress counted from valid records only |
| Checkpoint written without `partial` | resume loop thinks it is done | checkpoint carries `partial` + `coverage_gaps` |

### 7.3 What a power cut costs (measured, not assumed)

`crashtest.py` (gate `crash`) simulates a power cut - a process that dies
with no cleanup, no flush and no final write - in the middle of a run, and
then resumes it. The measured result:

| Question | Measured answer |
|----------|-----------------|
| Is the checkpoint file valid JSON afterwards? | Yes - writes are atomic (temp file + `fsync` + `os.replace`), so the file on disk is always the previous complete checkpoint or the new complete one, never a half-written one |
| Are the items scored before the cut still there? | Yes - every item is checkpointed the moment it is scored |
| How many items are lost? | **One** - the one in flight |
| Does the resume re-run completed items? | No - resume is item-level: an item is skipped only if a record for it exists; a *stub* (a failed attempt) never counts as done, so failed items always re-run |
| Are there duplicates afterwards? | No - every item is present exactly once |

To continue after a power cut:

```bash
bash close_coverage.sh          # gap-based: re-checks every (model, instrument)
                                # cell and runs only what is still missing
bash run_llama.sh <model> screening --resume --until-complete 6   # one cell
```

The driver is idempotent: it asks "which (model, instrument) cells are
missing or incomplete?" and works only on those, so restarting it after a
reboot continues exactly where the queue stopped.

`results/` is not in git (generated data, by policy), so the last line of
defence is `snapshot.sh`: after every completed instrument the evidence
(`results/`, `reports/`, `tasks/`, manifests) is archived to
`~/llmbench-backups/llmbench-<timestamp>.tar.gz`, keeping the newest 20.

### 7.4 The power policy belongs to the human, not to the harness

On a laptop the system suspends (or suspends-then-hibernates) itself to
preserve the battery. A benchmark tool must **never override that**: on
2026-10-20 UPower's critical action fired at 8% battery and the machine was
asleep for 6h48m in the middle of a run.

What the harness does instead:

* **It never touches the system's sleep configuration.** No masked units, no
  edited `logind.conf`, no edited `UPower.conf`, no disabled power manager.
  (An earlier attempt to mask `sleep.target` was reverted the moment it was
  made - the operator's power policy is not the harness's to change.)
* **It yields to the policy.** The run checks the battery after every
  checkpointed item. If the machine is *discharging* and below
  `--min-battery` (default 30%), the run pauses: it finishes the item in
  flight, writes the checkpoint, and exits with code 75. The queue stops and
  says so. Nothing is lost - the resume is item-level.
* **It survives the interruption anyway.** The per-item checkpoint, the
  atomic write and the item-level resume mean a suspend, a power cut or a
  crash costs at most the item in flight.
* **It tells you.** `note_suspend()` compares the wall clock with the
  monotonic clock after every item; the gap is exactly the time the machine
  was asleep, and the run log now says so in minutes instead of leaving a
  silent multi-hour stall that looks like a model that stopped answering.

A laptop that is about to sleep is a *hardware* fact, and the answer is a
charger, not a disabled policy.

### 7.5 One definition per statistic

The project used to carry TWO Wilson implementations - `stats.wilson`
(exact quantile, z = 1.959963985) and `scorers.wilson` (rounded, z = 1.96).
They disagreed in the 4th decimal of the published interval, which was
enough to flip a rounded digit: a re-aggregated cell could differ from the
stored one. `scorers.py` now re-exports `stats.wilson` - one definition, one
source of truth - and `rescore.py` re-aggregates every finished file so the
corpus never mixes statistical conventions. The audit gate re-derives all
185 published numbers from the raw records and fails on any disagreement.

### 7.6 Two MC mechanisms, two instruments

The primary multiple-choice mechanism reads the model's probability mass on
the option LETTERS from a single forward pass. It needs the model to
answer in the requested format.

Some models do not: k2-horizon replies with the option's TEXT ("4" instead
of "B"), so its probability mass never lands on the letters - the letter
mechanism cannot score it at all. The harness does not paper over that: the
probe detects it, the family is skipped, and the cell stays MISSING (never
zero, never a guess).

A second mechanism exists for exactly that case: the model is asked for the
option's text, the generated answer is read, and it is matched to the option
(letter, exact text, the text inside a sentence, or a distinctive prefix of
a truncated option). **Every matching rule requires the match to be unique**
- when two options could claim the answer, the answer is `none`, counted as
incorrect and reported as `unparsed_rate`. The results live in categories
suffixed `_txt`, in their own section of the report, and are **never ranked
against the letter-logprob numbers**: two mechanisms are two instruments.

The two mechanisms answer different questions. The letter mechanism measures
"can this model use the answer format the benchmark asks for"; the text
mechanism measures "can this model answer the question". A model that scores
well under one and is unmeasurable under the other is a real finding, not a
harness failure - and it is reported as exactly that.

### 7.7 Rules of engagement on this box
- One model resident at a time; one run at a time (enforced by a lock).
- Never reuse a server whose previous run was partial — it is holding the
  memory that caused the abort. Restart for a fresh window.
- `--until-complete N` grinds a task through as many server lifetimes as it
  needs, stopping on completion **or** on a round that banks nothing
  (BLOCKED ≠ slow).
- Resume adopts the previous file's config verbatim, so a resumed cell is
  bit-identical in configuration to the cell it completes.

---

## WHAT I COULD NOT VERIFY

1. **"rust-eval", "CrateEval", "RustBench"** — no such canonical suite found. Treat as non-existent or
   at least non-standard. Use MultiPL-E + Aider Polyglot for Rust and say so.
2. **MBPP sanitized exact size** — canonical claim is 427 (ids 11-510). The MultiPL-E copy I downloaded has
   426 rows with ids 2-809. Reconcile before citing a precise number.
3. **SWE-Refactor / SmellBench / SWE-Bench ProMax** — found by citation only, not read.
4. **CodeContests, LiveCodeBench, BigCodeBench scoring internals** — read abstracts/leaderboards, not the
   full evaluation code. My recommendation that they're infeasible at 3-8 tok/s rests on problem size, which
   is documented, but I did not run them.
5. **No published lint-as-benchmark metric exists.** Absence of evidence from web search, not proof. State
   your novelty claim as "no standard found", not "none exists".
6. **radon / lizard availability** — I did NOT install or test these. They are standard tools, but on this
   box `pip install` may not work. Verify before committing to them in §3.

## REFERENCES
- Chen et al. 2021, HumanEval / pass@k. arXiv:2107.03374. github.com/openai/human-eval
- Liu et al. 2023, EvalPlus / HumanEval+ / MBPP+. arXiv:2305.01210. github.com/evalplus/evalplus
- Austin et al. 2021, MBPP. arXiv:2108.07732
- Cassano et al. 2023, MultiPL-E. IEEE TSE. arXiv:2208.08227. github.com/nuprl/MultiPL-E, HF nuprl/MultiPL-E
- Jimenez et al. 2024, SWE-bench. github.com/swe-bench/SWE-bench, swebench.com/SWE-bench/reference/harness
- Gautam et al. 2025, RefactorBench. arXiv:2503.07832. github.com/microsoft/RefactorBench
- Widyasari et al. 2020, BugsInPy. ICSE-SEIP. github.com/soarsmu/Bugsinpy (requires Docker)
- Just et al. 2014, Defects4J. github.com/rjust/defects4j
- Lu et al. 2021, CodeXGLUE / Bugs2Fix. arXiv:2102.04664
- Zhuo et al. 2025, BigCodeBench. ICLR. arXiv:2406.15877
- Jain et al. 2024, LiveCodeBench. arXiv:2403.07974
- Aider polyglot. aider.chat/docs/leaderboards/
- Astral, Ruff docs. docs.astral.sh/ruff/linter/ ; Ruff 0.17.0 (971 rules, category counts verified locally)
- Rust Clippy lint index. rust-lang.github.io/rust-clippy/master/index.html
- Alvarado Gonzalez et al. 2025, Do Repetitions Matter? arXiv:2509.24086
- Bowyer et al. 2025, Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints. arXiv:2503.01747
- Miller 2024, Adding Error Bars to Evals. arXiv:2411.00640
- Schilling-Wilhelmi et al. 2025, Lifting the Benchmark Iceberg with Item Response Theory. ICLR 2025
- Ameli et al., A Statistical Framework for Ranking LLM-Based Chatbots. arXiv:2412.18407
- Arena AI, Statistical Extensions of the Bradley-Terry and Elo Models. arena.ai/blog/extended-arena
- Oman & Hagemeister 1992, Maintenance Index (via radon docs)
- Fowler 2018, Refactoring, 2nd ed. (task taxonomy reference, per RefactorBench)