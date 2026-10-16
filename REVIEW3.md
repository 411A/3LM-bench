# REVIEW3 — methodology alignment with 2026 best practice (2026-10-15)

Three parallel research agents surveyed (1) execution-based code eval, (2) general
LLM eval practice, (3) small-model local benchmarking, all against primary
sources fetched live (papers, Eleuther harness, llama.cpp, Ollama docs, BFCL,
τ-bench, EvalPlus, MultiPL-E). `websearch` was down; anything the agents could
not re-verify is marked UNVERIFIED in their reports and is NOT acted on here.

Verdict per theme: ALIGNED (already correct), FIXED (gap closed this round,
verified by execution), or DEFERRED (blocked or needs a user decision).

## 1. Estimation & statistics — ALIGNED, then FIXED upward

- Chen unbiased pass@k: our code suites report accuracy over n=1–2 reps, which
  IS pass@1 (no extrapolation). No `(c/n)^k`-as-pass@k anywhere in code paths.
- `pass_hat_k_reliability` is τ-bench `pass^k` (empirical all-trials-succeed
  fraction), the correct object for agents — not the biased extrapolator the
  code-eval literature warns about. Naming matches τ-bench; docstring says so.
- Wilson intervals (never CLT at n<100), chance + best-constant baselines,
  exact binomial p-values, `min_acc_to_reject_chance`: already shipped (stats.py).
- FIXED: no paired test existed, so no A>B claim was expressible at all.
  Added `compare.py` (exact McNemar on shared (task,id,rep) pairs, Holm option).
  Verified on synthetic A>B/TIE files.
- FIXED: no pre-registered primary → 112-cell multiple-comparisons roulette.
  `report.py` now renders a macro-mean PRIMARY and labels every cell exploratory.

## 2. Determinism & sampling — ALIGNED with documented deviation

- Fixed seed per rep, fixed temp (0.7), fixed top_p (0.95), fixed num_ctx,
  single-flight, one model resident at a time. MC via single-token logprobs
  (no sampling stage).
- DEVIATION (deliberate): temp 0.7, not 0.0 — temp 0 makes MiniCPM5-2B loop
  forever and return "". Same temp for all models, so the comparison is fair;
  absolute scores are temp-conditional and reported as such.
- FIXED: top_p lived as a hidden backend default, invisible to config_hash.
  Now pinned in FROZEN and asserted at startup.
- Reps (A:2 / B:1) are below the literature's 3–5×. Kept: at 3–8 tok/s a third
  rep costs ~3h per model for CIs that stay ±13–18pp wide regardless. The
  paired design + Wilson + McNemar is the correct compensation; per-cell claims
  stay exploratory by rule.

## 3. Sandboxing — ALIGNED, deviation documented

- Subprocess + tempdir + wall-clock timeout + RLIMIT_AS/CPU/NPROC + preexec_fn.
  No Docker (user constraint); no network namespace available.
- Limits (py 8s/512MB; rustc 25s, run 5s/1GB) are looser than HumanEval's 3s
  but appropriate for a 2-core box where the TOOLCHAIN is the slow part.
  Documented here instead of changed: tightening to 3s would measure this
  laptop's scheduler, not the models.
- `reliability_guard`-style in-process exec is NOT used; all untrusted code
  runs in a child process. No network access is granted to test harnesses by
  construction (no server, no creds in env).

## 4. Throughput — FIXED

- Was: single blended `gen_tps` per record. Research (llama-bench protocol,
  Ollama timing fields) requires pp/tg/load split: first-request
  `load_duration` (measured 9.2s cold today) and KV-cache hits contaminate
  blended rates.
- FIXED: backend now records `load_s` + `prompt_cached_tokens` per call;
  every record's `speed` carries them. Report still shows means; load/cold
  analysis is available offline per record.

## 5. Isolation between models — FIXED

- Was: `unload` fired keep_alive:0 but never verified; with ollama's 5-min
  keep-alive a "sequential" queue can hold TWO models resident on 7.6 GB.
- FIXED: `run_all.sh` polls `/api/ps` until zero models are resident before
  each model (ollama backend). Verified logic; live check shows 1 resident
  only because of today's probe calls.

## 6. Provenance / eval-card — FIXED

- Was: config_hash + mem footers only. No model digest, no tool versions
  (auditor's gap #7: results not reproducible without them).
- FIXED: every results file now carries `provenance` (registry digest or GGUF
  sha256, ollama/rustc/ruff/g++/python versions, CPU, kernel). Verified live:
  digest `6cc467f0…`, ollama 0.40.2, ruff 0.17.0, rustc 1.97.0.

## 7. One-by-one workflow — FIXED (user directive)

- `run_all.sh` with no args ran the whole 40h queue. Now refuses and prints
  the model table; `bash run_all.sh <label>` runs exactly one model;
  `bash run_all.sh --all` is the explicit full queue. Rationale in header.

## 8. DEFERRED (blocked or needs user decision)

1. **HumanEval+/MBPP+ vendoring**: INVESTIGATED, then downgraded. Mirror agent
   retrieved byte-verified files (md5 match, 164 + 378 tasks). Direct comparison
   shows our HumanEval tests ARE the EvalPlus base tests (only the baked
   `check()` call differs, which is our runner contract) — swapping gains
   literally zero. The 80x case-multiplication lives in EvalPlus's
   input-mutation runner (needs their package + contract injection), a harness
   port, not a data swap. DEFERRED as engineering, not data.
2. **MBPP missing-import false negatives (13/426): FIXED.** Tests calling
   `math.isclose` without importing math failed any correct solution whose
   code didn't itself import math. Prepended `import math` (EvalPlus precedent:
   their assertions carry it). Verified: reconciled reference solutions now
   pass 377/377 mapped items (were 372/377).
3. **Code-item solvability: PROVEN.** Reconciled canonical solutions pass
   164/164 HumanEval and 377/377 mapped MBPP through the real sandbox. The 49
   MBPP ids absent from Plus are self-consistent (parse, entry names present
   in prompts, imports fixed) — kept, not dropped; no proven-broken item found.
4. **Private holdout + contamination Δ**: needs 20–30% never-released items
   per category — i.e. newly authored tasks. Not done.
3. **Persian pack** (Khayyam/Belebele-fas/PerCoR samples): same network block,
   plus licensing/weight questions. Current 40 hand-made items stand.
4. **Quant curves** (Q8 vs Q4_K_M same base): recommended before the sweep;
   costs ~2× pilot runs. Decision needed: which 2–3 pilots.
5. **FAIL_TO_PASS + PASS_TO_PASS tags, added_date/cutoff fields** on hand-made
   suites: cheap, not yet done. Do before queue if contamination slicing is
   wanted.
6. **Full BFCL / τ-bench with LLM user simulator**: correctly NOT adopted —
   too heavy and too noisy for this box. Our scripted-tools + AST/trace +
   state design is the literature's cheap-robust ladder.

## What changed this round (files)

- `backend.py`: `load_s`/`prompt_cached_tokens` per call; `model_digest()`;
  `server_versions()`.
- `run.py`: FROZEN `gen_top_p` + startup assert; `provenance` in checkpoints
  and final results; `load_s`/`cached` in all 6 speed dicts; empty-output and
  error-stub lint semantics (see session notes).
- `run_all.sh`: one-by-one default; `/api/ps` empty pre-flight.
- `report.py`: PRIMARY macro-mean + exploratory labels.
- `compare.py` (new): exact McNemar + Holm.
- `scorers.py`/`agentic.py`: audit-3 fixes (separate notes).

All 5 control suites pass after every change; live smoke (nemotron,
lint/devops/netdiag) ran end-to-end on the edited harness.
