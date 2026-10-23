# 3LM Bench (Lean LLM Bench)

Reproducible, local-first evaluation of small LLMs (2–4B, CPU, 3–8 tok/s)
across **standard published benchmarks** plus hand-built suites for Persian,
electronics/embedded, DevOps, network diagnosis, lint quality, and tool use.

- Standard suites use official splits, fetched pinned from upstream
  (ARC, HellaSwag, WinoGrande, GSM8K, TruthfulQA-MC1, Belebele-fa, MMLU-6).
- Custom suites are labelled `custom` everywhere and never presented as
  official leaderboard scores.
- Every MC figure ships with chance baseline, best-constant baseline, Wilson
  interval, exact binomial p-value, and `min_acc_to_reject_chance`.
- Reports carry separate PRIMARY-STD and PRIMARY-CUSTOM macro-means plus
  paired McNemar/Holm comparison. No single "intelligence score".
- Stdlib-only Python, CPU-only, offline-capable, sandboxed code execution.

## Project development

The project as a whole—including its code, benchmark design, documentation,
and report—was built from human prompts with assistance from multiple AI
agents, then reviewed by a human. This is a development-process disclosure;
it is not a claim of external peer review.

The committed `reports/screening.html` is a static snapshot of the benchmark
results described below. Raw model outputs, run logs, downloaded standard
datasets, and generated standard task files are deliberately local-only and
ignored by Git; regenerating the snapshot requires restoring those local result
files. See [Repository layout](#repository-layout) and
[Contributing](CONTRIBUTING.md).

## Quick start

```bash
# Run these commands from the cloned repository root.
python3 --version  # Python 3.10+

# Optional: isolate the environment (there are no required pip packages).
python3 -m venv .venv
source .venv/bin/activate

# Fast, model-free sanity checks.
python3 -m py_compile ./*.py tests/*.py
python3 tests/test_units.py
python3 run.py --help

# Build custom tasks from the bundled HumanEval/MBPP source files.
python3 build_tasks.py && python3 build_tasks2.py \
  && python3 build_lint.py && python3 build_electronics.py

# fetch pinned standard benchmarks (one-time, ~19k items)
python3 standard_fetch.py --fetch-all
python3 verify_standard.py && python3 verify_tasks.py && python3 validate.py

# smoke: pipeline check only, never reported as scores
python3 run.py --backend ollama --model nemotron-3-nano:4b --limit 5 --reps 1 \
  --tasks science,english,std_arc_challenge,std_gsm8k

# screening (default for comparison): deterministic caps
python3 run.py --backend ollama --model nemotron-3-nano:4b --reps 2 \
  --per-task-cap code_humaneval=40,code_mbpp=50,std_hellaswag=200,std_gsm8k=60 \
  --agentic-reps 3

# full standard splits: Tier-3 overnight opt-in (see run_all.sh)
bash run_all.sh <label>     # one model, then stop for review
bash run_all.sh --all       # whole queue, strictly sequential

python3 report.py            # cross-model comparison table
python3 compare.py results/A.json results/B.json --holm   # only sanctioned A>B claim
python3 charts.py              # offline HTML report -> reports/screening.html
python3 rescore.py           # re-aggregate after changing scoring code
```

To run inference, start a local Ollama service or a compatible llama.cpp
server first. No hosted model API is used. The execution-based task families
also need their optional local toolchains (`ruff`, Rust, and/or `g++`); see
[`requirements.txt`](requirements.txt). For a complete local gate run, use
`bash gates.sh` after fetching standard data and restoring the raw result files
needed by the audit/report contracts. GitHub CI runs the model-free checks and
does not download datasets or run inference.

`run_all.sh` drives every model sequentially, unloading between each one and
waiting for a memory floor. `standard_fetch.py --check` verifies the data
cache without network.

## Why one model at a time

Two models resident at once on 2 cores split the same pipe. This is not
hypothetical: earlier work in this project measured parallel downloads starving
each other (6 KiB/s versus 447 KB/s from the same link). The same effect applies
to two inference servers, and it would corrupt every throughput number in the
table — which is one of the things this suite exists to measure. Sequential runs
are the only way the speed column means anything.

## Suites

### Standard (official splits, externally comparable)

| suite | source | n | scored by |
|---|---|---|---|
| `std_arc_challenge` | ARC-Challenge test (Clark et al. 2018) | 1170 | single-token logprobs |
| `std_arc_easy` | ARC-Easy test | 2350 | single-token logprobs |
| `std_hellaswag` | HellaSwag validation (test labels hidden) | 10042 | single-token logprobs |
| `std_winogrande` | WinoGrande-XL validation | 1267 | single-token logprobs |
| `std_truthfulqa` | TruthfulQA MC1, choices shuffled seed 20261012 (upstream fixes gold at 0) | 817 | single-token logprobs |
| `std_belebele_fa` | Belebele `pes_Arab` | 900 | single-token logprobs |
| `std_mmlu` | MMLU 6-subject core (elem/high math, physics, bio, CS, world hist) | 1446 | single-token logprobs, by-subject |
| `std_gsm8k` | GSM8K test, 0-shot `####` prompt (paper uses 8-shot CoT) | 1319 | generation + normalised exact match |

Stratified s30 screening instances (`tasks/std_*_s30.json`, built by
`standard_strata.py`, verified by `verify_standard.py`) — the default for
screening and qualifiers. Upstream files are never modified; the 30s are
separate instances with guaranteed subcategory coverage:

| suite | strata (≥3 each, enforced) |
|---|---|
| ARC-C / ARC-E s30 | exam family: Mercury ×10, state exams ×10 (≥5 distinct exam sources), national/intl ×10 |
| HellaSwag s30 | indomain ×15 (15 distinct activities), zeroshot ×15 (all 14 WikiHow activities) |
| WinoGrande s30 | surface-length terciles ×10 (upstream ships no topic labels — stated, not invented) |
| TruthfulQA s30 | choice-count bands 2–4 / 5–7 / 8+ ×10 (MC rows ship no topic fields — stated) |
| Belebele-fa s30 | question-number 1 ×15 + 2 ×15 across 30 distinct passages |
| MMLU-6 s30 | 6 subjects ×5 — physics, bio, CS, history and both maths all present |
| GSM8K s30 | solution steps 0–2 / 3–4 / 5+ ×10, counted from official `<<..>>` annotations |

Files are round-robin ordered, so any `--limit` prefix stays balanced, and the
builder *fails* (non-zero exit) if any stratum cannot supply its items — a
sampler that silently drops a subcategory is treated as broken. At n=30 the
Wilson interval is ~±18pp: enough to separate 70% from 40% (the qualifier
job), never enough to claim small wins.

See `benchmarks/catalog.json` for licenses, papers, verification dates, and
protocol deviations. Screening uses deterministic caps; full splits are
Tier-3. Custom MC sets complement — never replace — these.

### Custom (hand-made, labelled as such)

| category | what it measures | scored by |
|---|---|---|
| `code` | HumanEval (164) + MBPP-sanitized (426), Python | unit tests executed in a sandboxed subprocess |
| `bugfix` | single-file bug repair | embedded unit tests (SWE-bench lineage, dropped to one file) |
| `algo` | algorithmic problem solving | unit tests |
| `refactor` | behaviour-preserving restructure | tests must still pass |
| `rust` | Rust function synthesis | `rustc` + execution (MultiPL-E convention) |
| `embedded_cpp` | embedded C++ / Arduino / ESP32 logic | `g++` + execution against a mock Arduino API |
| `science` | ARC-C / MMLU-STEM style (custom screening) | single-token logprobs |
| `electronics` | Arduino, Raspberry Pi, ESP32, I/O buses, power, sensors | single-token logprobs |
| `persian` | Persian language, NLI and arithmetic | single-token logprobs + normalisation |
| `english` | grammar, syntax, logic, computing literacy | single-token logprobs |
| `netdiag` | network troubleshooting from config + symptom | false-premise-aware partial credit |
| `devops` | Dockerfile / compose / k8s / CI artifacts | gated static checks |
| `lint` | idiomatic code quality | `ruff` and `clippy` diagnostic counts |
| `agentic` | multi-call tool use with dependent steps | pass^k + trace scoring |

## Release state (2026-10-22)

**Six models; 26 complete local result files in the 2026-10-22 snapshot.**
222 published numbers re-derive from their raw records with zero mismatches
(`python3 audit.py`), and the local pre-production gates pass (`bash gates.sh`).

| instrument | coverage | notes |
|---|---|---|
| Screening (14 custom + 8 standard s30 suites) | 6/6 models | 544 identical non-agentic record IDs/prompts per model; k2 MC is stored in its separate text-match instrument |
| Qualifier | 6/6 files | non-agentic subsets match, but the current data have only 2 common qualified categories; no valid macro rank/CUT/ADVANCE |
| CS-100 decider (full MMLU-CS subject) | 5/6 | k2 via the text instrument |
| Agentic-19 (19 dependent tool chains) | 5/6 | medgemma has no tool support in any available runtime |
| Isolated throughput (dedicated quiet-box pass) | 6/6 | one model, fresh server, median of 12 identical prompts |
| Text-match MC (second mechanism) | k2 | see "Two multiple-choice mechanisms" below |

`reports/screening.html` is a single-file report snapshot: it embeds ECharts
and its license notices, so it can be opened without the `reports/assets`
directory. Raw results are not included, so a fresh clone can view the report
but cannot independently regenerate its data payload until local results are
restored. No chart ranks scores across different instruments.

The report is covered by its own gate (`python3 tests/test_report.py`, run
by `bash gates.sh`): the published payload must contain no NaN, a missing
cell is never a zero, the two multiple-choice mechanisms never share a
category, and a run whose MC probe failed must carry a not-comparable mark
on its letter-logprob cells.

### What the numbers do and do not support

- **n = 30 per standard screening cell.** The Wilson interval is ±13–17 pp.
  Most pairwise differences between models are *not* significant; read the
  interval, not the gap. The qualifier was designed to separate 70% from 40%,
  but its current aggregate is not rankable across the recorded mechanisms.
- **This is a profile, not a leaderboard.** The macro-means are exploratory
  (17 cells × 6 models is many comparisons). The only sanctioned A>B claim
  in the project is `compare.py`'s paired McNemar + Holm.
- **k2-horizon's multiple-choice numbers live in a different instrument.**
  Its llama.cpp build answers with the option's *text* instead of the letter,
  so the letter-logprob mechanism cannot score it at all. The text-match
  mechanism reads the generated answer; its cells carry a published
  `unparsed_rate` and an accuracy *range*, and are never ranked against the
  other five models' letter-logprob numbers.
- **Record-count caveat.** The coverage table splits k2's 388 screening MC
  records into the text-match column; its primary file therefore has fewer
  records even though the combined non-agentic screening items match. The
  displayed 544 common records exclude the separately reported agentic runs.
- **Qualifier caveat.** The saved qualifier files include k2 MC scores from a
  failed letter-logprob probe, tool-probe-failed agentic cells, and 3 vs 5
  agentic chains. `qualify.py` now excludes invalid/different-item cells and
  reports NOT RANKABLE when fewer than three common categories remain. No
  rerun is needed for the already-complete screening report.
- **medgemma-4b's agentic cell is unmeasured**, not zero: the model has no
  tool-calling support in Ollama (`does not support tools`) or llama.cpp.
- **Position confound.** MMLU-6, HellaSwag and WinoGrande are
  position-debiased (upstream put 10–18 points of the answer in one letter);
  their cells were re-measured. ARC-C, ARC-E and Belebele keep the upstream
  order and still carry a 0.05 confound - every MC cell publishes the score
  a constant picker would get, so the remainder is visible.
- **Two server builds.** The qualifier ran on llama-server b11558, the rest
  on b11714. No comparison on the page spans two builds, and the build is
  named per model.
- **Speed is the AC/charging number.** On battery the same box runs ~4×
  slower (the CPU drops to its powersave P-state), so the speed column is
  not a battery figure.
- **Upstream labels are upstream labels.** MMLU's own answer keys carry
  roughly 6% label noise, and the MMLU-6 cells hold 5 items per subject.

## Publishing and licenses

The project code is MIT-licensed. Bundled HumanEval/MBPP inputs and the ECharts
chart library have separate upstream terms and notices; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Downloaded standard dataset
caches, raw results, logs, and model weights are excluded from Git. The
committed HTML is a results snapshot, not a substitute for those raw inputs.
GitHub Actions runs compilation, shell syntax, model-free unit contracts, and
CLI help checks; it does not run models or fetch datasets.

### Durability: what survives a bad day

A benchmark that loses data to the machine it runs on is not a benchmark.
Every one of these is implemented and covered by a gate or a unit test:

| threat | behaviour | contract |
|---|---|---|
| power cut / OOM / kill | the item in flight is lost, nothing else | `crashtest.py` (gate `crash`) |
| laptop suspends mid-run | the run checkpoints and exits 143; the queue resumes | `note_suspend()` logs the sleep |
| battery dies | the run pauses cleanly at the last scored item and waits for the charger | `--min-battery` (default 30%) |
| server OOM | per-item RSS guard, clean abort, fresh server, item-level resume | `--until-complete` |
| checkpoint write interrupted | atomic write (temp + fsync + rename): the file is never half-written | `write_atomic()` |
| a task never reached | the coverage map is built *before* the task loop, so a missing suite shows as a gap | `coverage_report()` |
| the harness itself | 10 gate groups, 222-number audit, 60+ unit contracts | `bash gates.sh` |

### Two multiple-choice mechanisms

The primary mechanism reads the model's probability mass on the option
**letters** from a single forward pass. When a model ignores that format
and replies with the option's **text**, the primary mechanism cannot score
it - so there is a second mechanism that reads the generated answer and
matches it to the option (letter, exact text, the text inside a sentence,
or a distinctive prefix of a truncated option). Every matching rule requires
a **unique** match; an ambiguous answer is `none`, counted as incorrect and
reported as `unparsed_rate`. The two mechanisms are two instruments, in
separate categories, never ranked against each other.



**Multiple choice costs one token.** Ollama and llama.cpp both return the
model's top-k distribution at each position. The MC prompt ends exactly where the
answer letter goes, so asking for a single token and reading its top-k turns a
~40-token generation into 1 token. At 3-8 tok/s that is the difference between
the knowledge categories running in minutes and running for hours. Standard
suites reuse the same runner, so ARC/MMLU/HellaSwag stay affordable.

**Standard and custom are reported separately.** `report.py` renders
PRIMARY-STD (macro-mean over the 8 standard suites) and PRIMARY-CUSTOM
(macro-mean over the 14 custom suites). The two never mix: custom items use
hand-written prompts, standard items use official splits and protocols.

**An MC accuracy is never reported alone.** A bare percentage on a multiple-choice
set is not interpretable, because with n<100 it is usually not distinguishable
from guessing. Every MC figure therefore ships with the chance baseline (the mean
of 1/len(choices) over items, which is not 25% where items have three options),
the best constant-letter answerer's score, the lift over chance, a Wilson
interval, a one-sided exact binomial p-value, and — the number that actually
decides whether the score means anything — **`min_acc_to_reject_chance`: the
accuracy a model must reach at this n before the data can exclude guessing at
all**. At the original n=14 that was 57%; the Persian and English suites have
since been expanded to n=40 and n=45, which lowers the bar to 40% and 38%.
See `stats.py`. Pooled reps inflate n: records carry both `n_items` and
`n_samples` so a 35x2 pooled n=70 is never read as 70 items.

Confidence is the **absolute** probability mass on the letter set, not the
renormalised value within it. Renormalising inside the letters makes confidence
structurally incapable of dropping below 1/len(choices), so a model committing
5% of its mass to any answer reported confidence 1.00. Low mass is reported as
`informed` and `letter_mass`; it is deliberately **not** a scoring penalty,
because gating the score on it penalises exactly the uncertain-but-right models
the suite exists to characterise.

**Wilson intervals, not the normal approximation.** Per arXiv:2503.01747, the CLT is not valid below a
few hundred datapoints; the naive interval routinely extends outside [0,1] or
collapses to zero width. Every score is reported with its interval because a
bare number at n=20 claims precision the data does not support.

**TruthfulQA choices are shuffled.** Upstream MC1 fixes the gold at position 0
on all 817 items (verified). The official metric compares likelihoods
order-invariantly, but a letter-pick runner would let constant-"A" score 100%.
Choices are therefore shuffled with seed 20261012; the shuffle is recorded per
item and in `data/manifest.json`. ARC/HellaSwag/MMLU/Belebele keep upstream
order for literature comparability.

**GSM8K is 0-shot, and says so.** The paper uses 8-shot CoT; we prompt 0-shot
with `#### <number>` and score the normalised final number. The deviation is
recorded on every record (`protocol` field) and in the catalog. Do not compare
our GSM8K against 8-shot leaderboard numbers.

**pass^k, not pass@k, for agentic.** pass@k asks "did any trial work" and
rewards lucky runs. pass^k asks "did every trial work", which is the reliability
a user actually experiences.

**A chain counts as solved only if the tools were actually used.** pass^k is
computed from `chain_ok`, which requires the final answer to be correct AND every
expected tool call to have been made, in order, in a strictly later turn than the
step it depends on. Scoring pass^k from the final-answer text alone — which is
what this did originally — is gameable to 1.0 by a model that never calls a tool
and recites a memorised answer; that was verified against a mock backend, not
inferred. `final_answer_accuracy_no_tools_required` is reported alongside so the
difference between "did not know" and "knew but would not use the tools" stays
visible.

**Chains, not single calls.** Each agentic task is a sequence of dependent steps
where step *k+1* needs a value only the tool at step *k* can produce. Hallucinating
step 1 makes step 2 wrong automatically. Two claims about this were previously
overstated and are now enforced rather than asserted: a reversed-order trace and a
same-turn trace (both tools fired at once, arguments guessed) both score partial
trace credit, verified by control.

**False premises in network diagnosis.** 4 of 11 netdiag tickets (36%) report a
fault on a healthy network, and the correct answer is "nothing is broken"
(FaulT-Bench's majority-counterexample design). Without them you measure
compliance rather than diagnosis, and any model that always says "yes, broken"
scores well.

**A pinned ruff rule set, not `--select ALL`.** `--select ALL` fires `CPY001`
(copyright) and the whole pydocstyle `D1xx` family — house style, not defects.
Including them makes every model look equally bad and flattens the metric. The
pinned set excludes both.

**Lint is reported as four numbers, not one.** A raw diagnostic count rewards
noise: a file missing every docstring would out-score one with a genuine
correctness bug. Reported separately are the raw count, a severity-weighted
count, correctness-only violations, and lint density per line. `noqa` and
`allow(clippy::...)` usage is tracked separately as gaming — a model that
silences the linter has not satisfied it.

**Persian is normalised before comparison.** Arabic yeh vs Persian yeh, Arabic-
Indic vs Persian digits, ZWNJ and diacritics are corpus artefacts, not model
errors. Scoring raw Persian with exact match fails perfectly correct answers.

**clippy runs standalone.** `clippy-driver file.rs --edition 2021 --emit=metadata
--error-format=json -W clippy::pedantic` needs no Cargo.toml and no cargo
project, which is what makes lint-as-benchmark feasible at all here.

**Every embedded C++ task is verified twice.** `verify_cpp.py` compiles a correct
reference implementation against each task's assertions (it must pass), then
compiles a deliberately plausible-but-wrong implementation (it must fail). An
assertion that cannot distinguish right from wrong is not a test. This caught
five genuine harness bugs during construction — a wrong expected CRC, a
contradictory clamp case, a member/function name collision, an ill-specified
state machine, and four assertions too weak to catch the obvious mistake.
Warning counts before 2026-10-17 are inflated by ~31 header-stub warnings and
must be disregarded; the header span is now pragma-silenced so warnings measure
model code only.

## What this benchmark does not measure

- **Flash/RAM fit or core-version API availability** for real AVR targets. There
  is no cross-compiler here, so `analogWrite` on an old core is not caught. The
  embedded tasks score logic, which is what actually breaks in practice.
- **Persian generation quality.** Only multiple choice is scored; free-form
  Persian needs either a reference-based metric or a judge, and a 4B judge is
  worse than none.
- **Anything requiring network or Docker.** SWE-bench, FDE-Bench's execution
  layers and the containerised network benchmarks all need either. Their
  *formats* and *scoring rubrics* are copied; their datasets are not.
- **Sustained agentic reliability under long horizons.** 15 chains x 6 turns is
  enough to separate "calls tools" from "does not", not enough to characterise
  drift over hundreds of steps.
- **Medicine, chemistry, multimodal engineering, hardware-in-the-loop IoT.**
  Deliberately out of scope: ChemBench needs RDKit, EEE-Bench needs images,
  IoT-SkillsBench HIL needs real boards, and a 4B MedQA score measures recall,
  not clinical safety. See `benchmarks/catalog.json` research_only entries.

## Backend qualification (trust the pairing, not the logo)

Weights are not the product: the same GGUF can score 5/6 through one backend
and 0/12 through another (measured: qwen4b letter-choice; nemotron tool calls
at 0/56). Every run therefore probes its three I/O mechanisms first
(`probes.py`, ~2 min) and records the verdicts in provenance:

- `generation`: is anything emitted at all?
- `mc_logprob`: do answer letters carry mass (floor 0.5)? Mass proves the
  letters are scored; correctness stays the model's business.
- `tools`: does a forced call come back parsed? Parsing proves the
  renderer+parser round-trip; wisdom is the agentic suite's job.

A broken mechanism **skips its task families explicitly** (recorded under
`skipped`, missing-never-zero) instead of printing zeros. `python3 run.py ...
--probe-only` checks a pairing without running anything; `--no-probe`
disables it and marks results unprobed. On `--resume`, probes are advisory:
the file's task list never changes.

## Offline use
```bash
python3 standard_fetch.py --fetch-all   # online, once; fills data/raw + tasks/std_*.json
python3 standard_fetch.py --check       # offline verification of the cache
python3 run.py ... --offline            # refuses standard tasks missing from the local cache
```

Downloaded standard datasets and generated standard task files are never
committed (see `.gitignore`). The small HumanEval/MBPP source files are bundled
under their own licenses; see `THIRD_PARTY_NOTICES.md`. Do not redistribute
any benchmark material without following its upstream terms.

## Repository layout

```
benchmarks/          suite catalog, source metadata, and license notes
data/                manifest plus small licensed source inputs; raw downloads are ignored
docs/                methodology, contributor docs, and archived audit history
reports/             committed standalone HTML snapshot
tasks/               authored task definitions; standard generated tasks are ignored
tests/               model-free unit tests and local-data report contracts
*.py                 flat CLI entrypoints and shared modules (see below)
*.sh                 local queue, durability, and verification workflows
```

The Python files stay at the repository root intentionally: command-line
scripts import one another as top-level modules, and the shell workflows invoke
them by path. `docs/REPOSITORY_LAYOUT.md` maps the main commands and data flow;
historical `REVIEW*.md` investigations live under `docs/archive/reviews/` and
are archival evidence, not current protocol instructions.

Useful entrypoints: `run.py` runs tasks; `run_llama.sh`/`run_all.sh` manage
local queues; `standard_fetch.py` acquires standard data; `verify_*.py` check
inputs; `report.py`/`charts.py` render results; `compare.py` performs paired
tests; `audit.py` re-derives published values; and `gates.sh` runs the local
pre-production checks. See [Contributing](CONTRIBUTING.md) before changing
scorers, task files, or protocols.
