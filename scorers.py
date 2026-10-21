#!/usr/bin/env python3
"""
Scorers for the llmbench suite.

Nothing here talks to a model. Each function takes raw model output plus task
metadata and returns a dict of metrics. Keeping scoring separate from inference
means a bad generation can be re-scored later without re-running the model.

Methodology notes (see docs/METHODOLOGY.md):

* Code is executed in a *subprocess* with a wall-clock timeout and an address
  space cap, never with exec() in the harness process. A model that writes
  `while True: pass` must not take the benchmark down with it.

* Rust tasks follow MultiPL-E's convention: the harness supplies `fn main()` and
  the assert_eq! tests. A model that writes its own `fn main()` gets a duplicate
  main compile error, which is a real failure of the instruction, not a harness
  bug - so we detect it explicitly and report it as `wrote_main`.

* Lint scoring reports four numbers rather than one, because a raw diagnostic
  count rewards noise: a file missing every docstring would score "worse" than
  one with a genuine correctness bug. We report raw, severity-weighted,
  correctness-only, plus noqa usage (which is the obvious way to game the
  metric).

* Persian text is normalised before any string comparison. Scoring raw Persian
  with exact match fails on Arabic yeh vs Persian yeh, Arabic-Indic vs Persian
  digits, and ZWNJ - none of which are model errors.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata

# --------------------------------------------------------------------- sandbox
# The sandbox lives on DISK, never in /tmp. On this box /tmp is a 3.8 GB
# tmpfs - i.e. RAM - and the code paths below leave a compiled binary per
# item (4.3 MB each). Nothing ever deleted them: after a week of runs there
# were 629 leftover directories holding 3.2 GB of RAM that the kernel had
# to swap out to accommodate. That is where the "high swap on an idle box"
# and a large part of the memory pressure came from.
#
# LLMBENCH_TMP overrides the location; ~/.llmbench-tmp is the default.
TMP_ROOT = os.environ.get("LLMBENCH_TMP") or os.path.join(
    os.path.expanduser("~"), ".llmbench-tmp")
try:
    os.makedirs(TMP_ROOT, exist_ok=True)
except OSError:
    TMP_ROOT = tempfile.gettempdir()
TMP = tempfile.mkdtemp(prefix="llmbench-", dir=TMP_ROOT)


def _sweep_stale(root, max_age_s=6 * 3600):
    """Remove sandbox directories left behind by crashed runs."""
    import shutil
    import time
    now = time.time()
    for name in os.listdir(root):
        if not name.startswith("llmbench-"):
            continue
        path = os.path.join(root, name)
        try:
            if now - os.path.getmtime(path) > max_age_s:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def _cleanup(*paths):
    """Delete a compiled artifact / source file after it has been run."""
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


try:
    _sweep_stale(TMP_ROOT)
except OSError:
    pass
RUFF_BIN = os.path.expanduser("~/.local/bin/ruff")

# Appended after the test body. Without it, model code that exits early (or a
# test that silently no-ops) is indistinguishable from a passing test.
_SENTINEL = "#__LLMBENCH_TESTS_COMPLETED__"


# --------------------------------------------------------------------- stats
# The Wilson interval has ONE definition in this project: stats.wilson, with
# the exact two-sided 95% normal quantile (z = 1.959963985). This module used
# to carry its own copy with the rounded z = 1.96 - the two disagreed in the
# 4th decimal of the published interval, which is enough to flip a rounded
# digit and to make a re-aggregated cell differ from the stored one.
def score_mc_text(answer, choices):
    """Match a free-text answer to a multiple-choice option.

    Some models ignore "answer with a single letter" and reply with the
    option's TEXT instead ("4" instead of "B"). The letter-logprob
    mechanism cannot score those (the probability mass never lands on the
    letters), so this is the second mechanism: read the answer, match it to
    an option. It is a DIFFERENT instrument and is reported as one - the
    categories are suffixed, never merged.

    Every rule requires the match to be UNIQUE. When two options could
    claim the answer, the answer is "none" - an unparsed answer is counted
    as incorrect, never guessed onto the nearest option.

    Returns (index or None, how), where how is one of "letter" | "text" |
    "text_in_sentence" | "text_prefix" | "none".
    """
    import re as _re
    a = (answer or "").strip()
    if not a:
        return None, "none"

    def norm(s):
        # Unicode-aware: the Persian suite answers in Persian, and an
        # ASCII-only normaliser turned every Persian answer into an empty
        # string (so every Persian item was "unparsed").
        return _re.sub(r"[^\w\s]+", " ", s.lower(), flags=_re.UNICODE).strip()

    def _letter_only(txt):
        m = _re.match(r"^\**\s*\(?([A-Z])\)?\s*\**[\s\.\):,-]*$", txt.upper())
        if not m:
            return None
        i = ord(m.group(1)) - ord("A")
        return i if 0 <= i < len(choices) else None

    # 0. the model echoed the question and the option list and then gave the
    #    answer at the END ("... A) x\nB) y\n\nAnswer: D"). The LAST
    #    "answer:" marker is the model's actual answer; the echo before it
    #    is noise.
    marks = [m.start() for m in _re.finditer(r"answer\s*:", a, _re.I)]
    if marks:
        tail = a[marks[-1] + len(_re.match(r"answer\s*:", a[marks[-1]:], _re.I).group(0)):]
        tail = tail.strip()
        li = _letter_only(tail)
        if li is not None:
            return li, "letter"
        for i, c in enumerate(choices):
            if norm(c) and norm(c) == norm(tail):
                return i, "text"
        a = tail or a

    # 1. a bare letter (A/B/C...), optionally wrapped in punctuation
    li = _letter_only(a)
    if li is not None:
        return li, "letter"
    # 1b. the letter with its option text ("A) O(n^2)")
    m = _re.match(r"^\**\s*\(?([A-Z])\)?\s*[\.\):,-]\s*(.*)$", a)
    if m:
        i = ord(m.group(1)) - ord("A")
        rest = m.group(2).strip()
        if 0 <= i < len(choices) and rest:
            if norm(rest) == norm(choices[i]):
                return i, "letter"
            if norm(choices[i]) and norm(rest).startswith(norm(choices[i])):
                return i, "letter"

    na = norm(a)
    normed = [norm(c) for c in choices]

    # 2. exactly the option text
    exact = [i for i, nc in enumerate(normed) if nc and na == nc]
    if len(exact) == 1:
        return exact[0], "text"

    # 3. a distinctive PREFIX of the option appears in the answer: the model
    #    started to write the option and the generation was cut off
    #    ("... approximately 3 x 10^8 m" for the option "3 x 10^8 m/s").
    pref_hits = []
    for i, nc in enumerate(normed):
        words = nc.split()
        for take in (3, 2):
            if len(words) < take:
                continue
            pref = " ".join(words[:take])
            if len(pref) < 6:
                continue
            if pref in na:
                pref_hits.append(i)
                break
    if len(set(pref_hits)) == 1:
        return pref_hits[0], "text_prefix"

    # 4. the answer starts with the option text
    starts = [i for i, nc in enumerate(normed) if nc and na.startswith(nc)]
    if len(starts) == 1:
        return starts[0], "text"

    # 5. the option text inside a sentence ("The answer is 4.")
    inside = [i for i, nc in enumerate(normed)
              if nc and _re.search(rf"(?:^|\s){_re.escape(nc)}(?:$|\s|[.,!?])", na)]
    if len(inside) == 1:
        return inside[0], "text_in_sentence"
    return None, "none"


from stats import wilson  # noqa: F401,E402  (re-exported: one definition)


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator (Chen et al. 2021), the HumanEval convention.

    n = number of samples, c = number correct, k = k in pass@k.
    """
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def pass_hat_k(n, c, k):
    """tau-bench / AgentBench pass^k: probability that ALL k trials succeed.

    pass@k asks "did it work at least once", which is the wrong question for an
    agent - a tool-calling model that nails the first call and fabricates the
    result scores 1.0 under pass@k and 0.0 under pass^k. The correct estimator is
    (c/n)^k for k draws with replacement from the empirical per-trial rate.

    The previous implementation was C(n-c,k)/C(n,k), which is hypergeometric AND
    inverted: that is exactly 1 - pass@k, so a model that failed everything
    reported pass^k = 1.0. It was dead code (run.py computes this inline), but a
    wrong inverted formula in a shared module is a trap regardless.
    """
    if n <= 0:
        return 0.0
    if k < 0:
        raise ValueError("k must be non-negative")
    return (c / n) ** k


# ------------------------------------------------------------ code extraction
_FENCE = re.compile(r"^```[ \t]*([a-zA-Z0-9_+#.-]*)[ \t]*\n(.*?)(?:^```|\Z)",
                    re.S | re.M)


def extract_code(text, lang=None):
    """Pull runnable code out of a model reply.

    Selection rules, in order of preference:
      1. a fenced block tagged with the requested language
      2. the LAST untagged block - models that write a wrong answer and then
         correct themselves put the fix last, and scoring the first block turned
         a correct reply into a failure
      3. the only/last block regardless of tag
      4. the whole reply

    An unterminated fence is still captured, so a model that forgets the closing
    ``` does not lose its answer.

    Nested fences (a fence marker inside a docstring) close the block early under
    the naive pattern; that truncation produced a two-word fragment and a
    guaranteed syntax error. Blocks are therefore validated for plausibility and
    the next candidate is preferred when a block is obviously cut off.
    """
    if not text:
        return ""
    blocks = [(m.group(1).lower(), m.group(2)) for m in _FENCE.finditer(text)]

    if lang:
        tagged = [b for tag, b in blocks if tag == lang.lower()]
        if tagged:
            cand = _best_candidate(tagged, lang)
            if cand:
                return cand
    untagged = [b for tag, b in blocks if tag in ("", "python", "py")]
    if lang == "python":
        untagged = [b for tag, b in blocks if tag in ("", "python", "py")]
    if untagged:
        cand = _best_candidate(list(reversed(untagged)), lang)
        if cand:
            return cand
    if blocks:
        cand = _best_candidate([b for _, b in reversed(blocks)], lang)
        if cand:
            return cand
    # No fence at all. Try the raw reply, but only if it looks like code rather
    # than prose - otherwise "I cannot help with that" would be submitted to the
    # compiler and produce a meaningless syntax error.
    if _looks_like_code(text):
        return _strip_preamble(text, lang)
    return ""


def _best_candidate(cands, lang=None):
    """Return the last candidate that is not an obviously truncated fragment.

    `fenced=True` marks content the model itself delimited with a code fence, in
    which case the interior is trusted verbatim (except for Python, which is
    parser-validated). See _strip_preamble for why re-interpreting a fence is
    destructive.
    """
    for c in reversed(cands):
        c = _strip_preamble(c, lang, fenced=True)
        if c and _looks_like_code(c):
            return c
    for c in reversed(cands):
        c = _strip_preamble(c, lang, fenced=True)
        if c:
            return c
    return ""


_CODE_HINT = re.compile(
    r"(?m)^\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+|class\s+\w+"
    r"|#\[|#\[derive|fn\s+\w+|use\s+\w+|let\s+\w+|const\s+\w*|var\s+\w+"
    r"|pub\s+fn|impl\b|trait\b|template\s*<|<\?php|\{\s*$|@\w+)")


def _looks_like_code(s):
    if not s or not s.strip():
        return False
    return bool(_CODE_HINT.search(s))


def _strip_preamble(s, lang=None, fenced=False):
    """Drop prose before and after the code, keeping only the code span.

    A FENCED block is returned essentially intact. That is the whole point of a
    fence: the model delimited the code itself, so re-interpreting the interior
    is not just unnecessary, it is destructive. An earlier version ran the same
    trailing "cut back to the last code-looking line" heuristic on fenced
    non-Python content and silently deleted most of a correct nginx config:
    `server {`, `listen 80;` and `proxy_pass ...;` do not match the code-line
    heuristic, so a fenced, correct artifact scored 0.0 while the byte-identical
    unfenced one scored 1.0. Models fence their output constantly, so this
    penalised correct answers in the exact form they are written.

    Python still goes through the parser (see _python_span), because a fenced
    Python block can genuinely contain prose around the function.

    For unfenced non-Python text only LEADING prose is removed. Cutting the tail
    on a line-pattern guess is unsafe when the language's syntax is not
    line-oriented (nginx, YAML, systemd units, GitHub Actions).
    """
    if not s:
        return ""
    if fenced and lang != "python":
        return s.strip("\n")
    lines = s.split("\n")

    if lang == "python":
        out = _python_span(lines)
        if out is not None:
            return out

    start = None
    for i, ln in enumerate(lines):
        t = ln.strip()
        if not t:
            continue
        if _looks_like_code_line(t):
            start = i
            break
    if start is None:
        return ""
    body = lines[start:]
    if lang == "python":
        # python already went through the parser above; unreachable in practice
        end = len(body)
        for j in range(len(body) - 1, -1, -1):
            t = body[j].strip()
            if not t:
                continue
            if _looks_like_code_line(t):
                end = j + 1
                break
        return "\n".join(body[:end]).rstrip()
    # Non-python, unfenced: keep the tail. Only the head is guessable.
    return "\n".join(body).strip("\n")


_CODE_LINE = re.compile(
    r"^(?:from\s|import\s|def\s|class\s|fn\s|use\s|pub\s|let\s|const\s|var\s"
    r"|#|//|/\*|\#\[|@\w+|<|\{|\}|\)|\]|"
    r"[A-Za-z_]\w*\s*[:=(]|return\b|if\b|for\b|while\b|try\b|elif\b|else\b"
    r"|with\b|print\(|assert\b)")


def _looks_like_code_line(t):
    return bool(_CODE_LINE.match(t))


def _python_span(lines):
    """Longest span of consecutive lines that parses as Python.

    Tries every start/end pair from the outside in, preferring the longest span
    that parses, so a leading docstring comment or a trailing blank block does
    not cost the answer. Returns None if nothing parses.
    """
    import ast
    n = len(lines)
    # candidate starts: every line that is not obviously prose
    starts = [i for i, ln in enumerate(lines)
              if ln.strip() and _looks_like_code_line(ln.strip())]
    if not starts:
        return None
    first = starts[0]
    best = None
    for end in range(n, first, -1):
        cand = "\n".join(lines[first:end]).strip()
        if not cand:
            continue
        try:
            ast.parse(cand)
        except SyntaxError:
            continue
        except ValueError:
            continue
        best = cand
        break
    if best is None:
        # nothing parses from the first code-looking line; try later starts
        for s in starts[1:]:
            cand = "\n".join(lines[s:]).strip()
            if not cand:
                continue
            try:
                ast.parse(cand)
                return cand
            except (SyntaxError, ValueError):
                continue
    return best


# ------------------------------------------------------------ python execution
def run_python(code, test, timeout=8.0, mem_mb=512):
    """Execute `code` then `test` in an isolated subprocess.

    Returns dict(passed, error, stage, stderr_tail).

    A pass requires BOTH a zero exit status AND the sentinel, which is only
    printed once the whole test body has run. `sys.exit(0)` and `os._exit(0)`
    from model code terminate the process before the sentinel, so a model cannot
    score a pass by exiting early.
    """
    src = code + "\n\n" + test + f'\nprint("{_SENTINEL}")\n'
    path = os.path.join(TMP, f"t{abs(hash(src)) % 10**10}.py")
    with open(path, "w") as f:
        f.write(src)

    def limit():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 2, int(timeout) + 3))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))

    try:
        p = subprocess.run([sys.executable, "-I", "-S", path], capture_output=True,
                           text=True, timeout=timeout, preexec_fn=limit,
                           env={"PATH": "/usr/bin:/bin", "HOME": TMP})
    except subprocess.TimeoutExpired:
        _cleanup(path)
        return {"passed": False, "error": "timeout", "stage": "exec",
                "stderr_tail": f"exceeded {timeout}s"}
    except Exception as e:  # noqa: BLE001
        _cleanup(path)
        return {"passed": False, "error": f"harness:{type(e).__name__}",
                "stage": "exec", "stderr_tail": str(e)[:300]}
    finally:
        _cleanup(path)

    out = (p.stderr or "")[-600:]
    # A model that calls sys.exit(0) / os._exit(0) would otherwise terminate the
    # subprocess with status 0 BEFORE the assertions run, and the original
    # `returncode == 0` test would score that a PASS. Verified: both
    # `import sys; sys.exit(0)` and `import os; os._exit(0)` returned
    # passed=True with the test never executed. The test now emits a sentinel on
    # completion; reaching the end of the test is required for a pass.
    ok = p.returncode == 0 and (p.stdout or "").strip().endswith(_SENTINEL)
    return {"passed": ok,
            "error": "" if ok else ("exit_before_test" if p.returncode == 0 else "nonzero_exit"),
            "stage": "exec", "returncode": p.returncode,
            "harness_completed": (p.stdout or "").strip().endswith(_SENTINEL),
            "stderr_tail": out}


def python_syntax_ok(code):
    try:
        compile(code, "<gen>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"{type(e).__name__}: {e}"


def reconcile_prefix(prompt, code, lang):
    """Handle a completion that continues the prompt instead of restating it.

    HumanEval's prompt already contains the `def` line and the docstring, and
    the canonical solution is only the body. Models split roughly evenly: some
    restate the whole function, some continue it and emit an indented fragment.
    Concatenating a fragment on its own raises IndentationError and scores a
    correct answer as a syntax error.

    So: if the code stands alone it is used as-is. If it does not compile, and
    the prompt looks like a code prefix for this language, try prompt + code.
    The original completion is preserved in the record either way, so the
    `completion_style` field is a real measurement of how often models do this
    rather than a silent normalisation.

    Returns (code_to_run, style) where style is complete | prefix_concatenated
    | unparseable.
    """
    ok, _ = python_syntax_ok(code)
    if ok:
        return code, "complete"
    if lang == "python" and re.search(r"(?m)^\s*def\s+\w+\s*\(.*\)[^:]*:\s*(#.*)?$", prompt):
        joined = prompt + ("\n" if not prompt.endswith("\n") else "") + code
        ok2, _ = python_syntax_ok(joined)
        if ok2:
            return joined, "prefix_concatenated"
    return code, "unparseable"


# ---------------------------------------------------------------- rust harness
def run_rust(fn_code, tests, fn_name, timeout=25.0):
    """Compile and run a Rust function with MultiPL-E's harness convention.

    The model supplies only the function. We append `fn main()` and the
    assert_eq! checks, so a stray `fn main()` in the output is a duplicate
    definition - reported rather than silently stripped.
    """
    wrote_main = bool(re.search(r"fn\s+main\s*\(", fn_code))
    uses_test_attr = "#[test]" in fn_code
    # Defensive: a non-string test entry would otherwise be formatted with
    # Python's repr and produce a compile error that looks like a model failure.
    tests = [str(t) for t in (tests or []) if t]
    if not tests:
        return {"passed": False, "error": "no_tests", "stage": "harness",
                "stderr_tail": "task supplied no test statements"}
    for t in tests:
        if not t.lstrip().startswith(("assert", "let ")):
            return {"passed": False, "error": "malformed_test", "stage": "harness",
                    "stderr_tail": f"test is not a Rust statement: {t[:120]}"}
    body = "\n".join(f"    {t}" for t in tests)
    harness = (f"{fn_code}\n\nfn main() {{\n{body}\n"
               f'    println!("{_SENTINEL}");\n}}\n')
    path = os.path.join(TMP, f"t{abs(hash(fn_code)) % 10**10}.rs")
    binp = path[:-3]
    with open(path, "w") as f:
        f.write(harness)

    c = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings",
                        "-o", binp, path], capture_output=True, text=True, timeout=timeout)
    if c.returncode != 0:
        _cleanup(path, binp)
        err = _error_first(c.stderr)
        reason = "compile_error"
        if wrote_main and "main" in (c.stderr or ""):
            reason = "wrote_fn_main"
        return {"passed": False, "error": reason, "stage": "compile",
                "stderr_tail": err, "wrote_main": wrote_main,
                "uses_test_attr": uses_test_attr}

    try:
        r = subprocess.run([binp], capture_output=True, text=True, timeout=5,
                           preexec_fn=lambda: __import__("resource").setrlimit(
                               __import__("resource").RLIMIT_AS, (1024 * 1024 * 1024,) * 2))
    except subprocess.TimeoutExpired:
        _cleanup(path, binp)
        return {"passed": False, "error": "timeout", "stage": "exec",
                "stderr_tail": "binary timed out", "wrote_main": wrote_main,
                "uses_test_attr": uses_test_attr}
    finally:
        # The compiled binary is 4-9 MB. Leaving one per item in the
        # sandbox is what filled 3.2 GB of RAM on this box.
        _cleanup(path, binp)
    return {"passed": r.returncode == 0 and _SENTINEL in (r.stdout or ""),
            "error": ("" if r.returncode == 0 and _SENTINEL in (r.stdout or "")
                      else ("early_exit" if r.returncode == 0 else "assertion_failed")),
            "stage": "exec", "stdout": (r.stdout or "")[-200:],
            "stderr_tail": (r.stderr or "")[-400:],
            "wrote_main": wrote_main, "uses_test_attr": uses_test_attr}


# ------------------------------------------------------------------ lint: ruff
# Pinned rather than --select ALL on purpose. ALL fires CPY001 (copyright) and
# the whole pydocstyle D1xx family, which are house-style preferences, not
# defects; including them makes every model look equally bad and flattens the
# metric. These are the rules that indicate an actual bug or real non-idiom.
RUFF_RULES = ("E4,E7,E9,F,B,SIM,C4,RET,PIE,PT,PTH,PERF,RUF,UP,ARG,S,BLE,"
              "PLR,PLW,A,C90,DTZ,T20,EM,ISC,ICN,G,LOG,Q,RSE,SLF,TID,TRY")

SEVERITY_WEIGHT = {"F": 5.0, "E9": 4.0, "B": 3.0, "S": 3.0, "A": 3.0,
                   "BLE": 3.0, "PLR": 2.0, "PLW": 2.0, "E7": 2.0, "E4": 1.5,
                   "ARG": 1.0, "SIM": 1.0, "RET": 1.0, "C4": 1.0, "PIE": 1.0,
                   "UP": 1.0, "PERF": 1.0, "RSE": 1.0, "T20": 1.0}


def _count_suppressions(code):
    """Count genuine lint-suppression directives in Python source.

    Returns (count, kinds). `kinds` distinguishes a line-level `# noqa` from a
    file-level `# ruff: noqa` / `# flake8: noqa`, which blanks the ENTIRE file -
    far more serious than silencing one line, and the exact form that a plain
    `#\\s*noqa` regex cannot see.

    Comments are read through `tokenize` rather than by scanning text, so the
    word "noqa" inside a docstring or a data string is not counted as
    suppression. If the source does not tokenise (it may be mid-edit, which is
    the point of this suite), the text scan is used as a fallback and every hit
    is reported as `line` so the flag is never silently lost.
    """
    import io
    import tokenize
    line_re = re.compile(r"#\s*noqa\b", re.I)
    file_re = re.compile(r"#\s*(?:ruff|flake8)\s*:\s*noqa\b", re.I)
    kinds = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type != tokenize.COMMENT:
                continue
            if file_re.search(tok.string):
                kinds.append("file")
            elif line_re.search(tok.string):
                kinds.append("line")
    except (tokenize.TokenError, IndentationError, SyntaxError):
        for ln in code.split("\n"):
            if file_re.search(ln):
                kinds.append("file")
            elif line_re.search(ln):
                kinds.append("line")
    return len(kinds), kinds


def lint_python(code, filename="gen.py"):
    """Run ruff with the pinned rule set; return diagnostics + four metrics."""
    if not RUFF_BIN or not os.path.exists(RUFF_BIN):
        p = shutil_which("ruff")
        if not p:
            return {"available": False}
        globals()["RUFF_BIN"] = p
    path = os.path.join(TMP, filename)
    with open(path, "w") as f:
        f.write(code)
    try:
        p = subprocess.run([RUFF_BIN, "check", "--no-cache", "--isolated",
                            "--select", RUFF_RULES, "--output-format", "json", path],
                           capture_output=True, text=True, timeout=30)
        diags = json.loads(p.stdout or "[]")
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)[:200]}

    codes = [d["code"] for d in diags]
    # ruff reports noqa-suppressed findings as absent entirely, so suppression
    # must be detected from the source.
    #
    # The previous detector was `len(re.findall(r"#\s*noqa", code))`. That is a
    # blind spot, not a rounding error: ruff's FILE-LEVEL directive is written
    # `# ruff: noqa`, in which `#` is followed by " ruff: ", so the pattern cannot
    # match. A model could prepend that one line to completely unmodified starter
    # code and record clean=True with noqa_gaming=False - byte-identical report
    # fields to a model that actually rewrote the code. It also missed `#NoQA`
    # (the directive is case-insensitive) and flagged the word "noqa" appearing
    # inside a docstring or data string, which is not suppression at all.
    #
    # Fixed by tokenising: only genuine COMMENT tokens count, never string
    # content, and both the line-level `# noqa` and the file-level
    # `# ruff: noqa` / `# flake8: noqa` forms are recognised.
    noqa_count, noqa_kinds = _count_suppressions(code)
    weighted = sum(SEVERITY_WEIGHT.get(re.match(r"[A-Z]+", c).group(0) if re.match(r"[A-Z]+", c) else "", 0.5)
                   for c in codes)
    correctness = [c for c in codes if c.startswith(("F", "E9", "B0", "B6", "B7"))]
    lines = max(1, len([l for l in code.split("\n") if l.strip()]))
    return {
        "available": True,
        "n_diagnostics": len(codes),
        "codes": sorted(set(codes)),
        "by_prefix": _prefix_hist(codes),
        "weighted_severity": round(weighted, 2),
        "n_correctness": len(correctness),
        "lint_density": round(len(codes) / lines, 3),
        "lines": lines,
        "noqa_count": noqa_count,
        "noqa_kinds": sorted(set(noqa_kinds)),
        # A file-level directive is categorically worse than a line-level one:
        # it silences the whole file rather than one finding. Reported
        # separately so a blanket suppression is visible as such and cannot hide
        # inside an aggregate `noqa_gaming_rate`.
        "noqa_gaming": noqa_count > 0,
        "noqa_file_level": "file" in noqa_kinds,
        "clean": len(codes) == 0,
        "messages": [f"{d['code']} {d['message'][:80]}" for d in diags[:12]],
    }


def _prefix_hist(codes):
    h = {}
    for c in codes:
        m = re.match(r"[A-Z]+", c)
        k = m.group(0) if m else "?"
        h[k] = h.get(k, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def shutil_which(name):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


# ------------------------------------------------------------- lint: clippy
# clippy-driver runs standalone -- no Cargo.toml, no cargo project, no network.
# That is what makes a lint-as-benchmark feasible at all on this box.
CLIPPY_SETS = {
    "correctness": ["clippy::correctness"],
    "pedantic": ["clippy::pedantic"],
    "nursery": ["clippy::nursery"],
    "style": ["clippy::style"],
    "complexity": ["clippy::complexity"],
    "perf": ["clippy::perf"],
}


def rustfmt_check(code):
    """Did the model produce already-formatted Rust? Cheap, and rustfmt is local.

    Implemented with `--check` and the exit status. The previous version ran
    `rustfmt --emit stdout <path>` and compared stdout to the source - but in file
    mode rustfmt PREPENDS a "filename:" header (and a blank line) to its stdout,
    so `formatted.strip() == orig.strip()` could never be true and the metric was
    a constant 0.0 for every model on every run. `rust_already_formatted_rate`
    therefore reported "no model can format Rust" for the whole suite. It also
    ran rustfmt twice and discarded the first result.
    """
    b = shutil_which("rustfmt")
    if not b:
        return {"available": False}
    path = os.path.join(TMP, "fmt.rs")
    with open(path, "w") as f:
        f.write(code)
    try:
        r = subprocess.run([b, "--edition", "2021", "--check", path],
                           capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)[:200]}
    # rc 0 = already formatted. rc 1 = would reformat. Other codes = rustfmt
    # could not parse it, which is a model error but not a formatting verdict.
    if r.returncode == 0:
        return {"available": True, "already_formatted": True, "parse_ok": True}
    if r.returncode == 1:
        diff = [l for l in (r.stdout or "").splitlines()
                if l.startswith(("+", "-")) and l[:3] not in ("+++", "---")]
        return {"available": True, "already_formatted": False, "parse_ok": True,
                "diff_lines": len(diff)}
    return {"available": True, "already_formatted": False, "parse_ok": False,
            "error": (r.stderr or "")[:200]}


# rustc emits a trailing human summary as a JSON diagnostic too:
#   {"message":"1 warning emitted", "code":null, "level":"warning"}
# Counting it inflated every set by 1 (46% of a small answer's total) and, worse,
# made it indistinguishable from a real finding.
_SUMMARY_MSG = re.compile(r"^\d+\s+(?:warning|error)s?\s+emitted", re.I)
# dead_code fires on any function the harness's own appended `fn main` never
# calls, so every answer carried it regardless of quality. Every prompt in this
# suite forbids writing `fn main`, so the harness adds it and manufactures the
# warning. dead_code is a rustc lint, not a clippy lint, and is not what the
# suite measures.
_STRUCTURAL_RUST_CODES = {"dead_code"}
_RUST_ALLOW = re.compile(r"#!?\[\s*allow\s*\(\s*clippy", re.I)
_RUST_BLANKET_ALLOW = re.compile(r"#!?\[\s*allow\s*\(", re.I)


def lint_rust(code, sets=("correctness", "pedantic")):
    driver = shutil_which("clippy-driver")
    if not driver:
        return {"available": False}
    path = os.path.join(TMP, "gen.rs")
    with open(path, "w") as f:
        f.write(code if "fn main" in code else code + "\nfn main() {}\n")
    out = {}
    any_timeout = False
    any_error = False
    for s in sets:
        meta = CLIPPY_SETS[s]
        cmd = [driver, path, "--edition", "2021", "--emit=metadata",
               "-o", os.path.join(TMP, "g.rmeta"), "--error-format=json"]
        for m in meta:
            cmd += ["-W", m]
        # dead_code is manufactured by the harness's own appended fn main.
        cmd += ["-A", "dead_code"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        except subprocess.TimeoutExpired:
            out[s] = {"n": -1, "error": "timeout"}
            any_timeout = True
            continue
        parsed = {"n": 0, "codes": [], "messages": []}
        for line in p.stderr.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = (d.get("message") or "").strip()
            if _SUMMARY_MSG.match(msg):
                continue                    # rustc's "N warnings emitted" footer
            lvl = d.get("level")
            # NOT named `code`: that shadows the function parameter, which then
            # arrives as None at the noqa scan below.
            dcode = (d.get("code") or {}).get("code")
            if lvl == "error" or (dcode and re.match(r"^E\d{4}$", dcode)):
                # a compile error is not a lint finding; it means the model
                # wrote code that does not build. Counted separately so
                # "does not compile" is never confused with "is idiomatic".
                parsed["errors"] = parsed.get("errors", 0) + 1
                any_error = True
                parsed["messages"].append(f"COMPILE_ERROR {msg[:70]}")
                continue
            if not dcode or dcode in _STRUCTURAL_RUST_CODES:
                continue
            parsed["n"] += 1
            parsed["codes"].append(dcode)
            parsed["messages"].append(f"{dcode} {msg[:70]}")
        out[s] = parsed

    if any_timeout:
        # A timeout is a harness failure, not a model success. The previous code
        # summed `if v["n"] > 0`, and the timeout branch sets n = -1, so every
        # set timing out gave total == 0 and clean == True - a perfect lint score
        # awarded for a rustc hang.
        return {"available": False, "error": "timeout", "by_set": out,
                "compiles": False, "clean": False, "n_diagnostics": None}
    total = sum(v["n"] for v in out.values())
    return {
        "available": True,
        "n_diagnostics": total,
        "clean": total == 0 and not any_error,
        "compiles": not any_error,
        "compile_error_count": sum(v.get("errors", 0) for v in out.values()),
        "by_set": out,
        # `allow (clippy::pedantic)` with a space, or capitalised, evaded the old
        # `allow\(clippy` pattern entirely.
        "noqa_count": len(_RUST_ALLOW.findall(code)),
        "noqa_kinds": sorted({("clippy_allow" if m else "blanket_allow")
                              for m in _RUST_ALLOW.findall(code)}) or
                            (["blanket_allow"] if _RUST_BLANKET_ALLOW.search(code) else []),
        "noqa_gaming": bool(_RUST_ALLOW.search(code) or _RUST_BLANKET_ALLOW.search(code)),
    }


# ------------------------------------------------------------------ embedded C++
# Compiled and executed for real. There is no AVR cross-compiler here, so we
# cannot check flash/RAM fit or core-version API availability; what we check is
# the logic, which is what actually breaks. Warnings are captured separately and
# scored as a quality signal: -Wall -Wextra on a short function is a reasonable
# proxy for sloppy embedded C++, and unlike clippy it needs no extra toolchain.
def _error_first(stderr, max_chars=900, max_err_lines=10, tail_lines=4):
    """Keep the error lines, not the warning noise.

    Model code under -Wall -Wextra can emit dozens of warnings that push the
    actual error outside any fixed-size tail window (the log then shows a bare
    span marker). Error lines are kept preferentially, plus a few trailing
    context lines, within the same size budget.
    """
    lines = (stderr or "").splitlines()
    errs = [l for l in lines if "error" in l.lower()][:max_err_lines]
    keep = errs + [l for l in lines[-tail_lines:] if l not in set(errs)]
    out = "\n".join(keep)
    return out[-max_chars:] if len(out) > max_chars else out
def run_cpp(code, test, header, timeout=40.0):
    # The mock header's unused API stubs emit ~31 -Wunused-function warnings
    # by themselves (measured), which buried real errors outside the stderr
    # window AND made warning_free_rate measure the harness, not the model.
    # The header span is wrapped in push/ignored/pop so model code stays
    # fully warned while header noise is silenced.
    _HDR_QUIET = ('#pragma GCC diagnostic push\n'
                  '#pragma GCC diagnostic ignored "-Wunused-function"\n')
    _HDR_POP = '#pragma GCC diagnostic pop\n'
    src = (_HDR_QUIET + header + "\n" + _HDR_POP + code
           + "\n\nint main() {\n" + test + f'\n    printf("{_SENTINEL}\\n");\n    return 0;\n}}\n')
    path = os.path.join(TMP, f"e{abs(hash(src)) % 10**11}.cpp")
    exe = path[:-4]
    with open(path, "w") as f:
        f.write(src)

    def limit():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 5,) * 2)

    try:
        c = subprocess.run(["g++", "-std=c++17", "-O1", "-Wall", "-Wextra",
                            "-Wno-unused-parameter", "-o", exe, path],
                           capture_output=True, text=True, timeout=timeout,
                           preexec_fn=limit)
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "compile_timeout", "stage": "compile",
                "stderr_tail": "g++ timed out", "n_warnings": None}
    except FileNotFoundError:
        return {"available": False, "error": "g++ not found"}

    stderr = c.stderr or ""
    n_warn = len([l for l in stderr.splitlines()
                  if l.strip().startswith(("warning:", "note:")) and l.strip().endswith("]")])
    n_warn = len(re.findall(r"(?m)^\S+:\d+:\d+: warning:", stderr))
    if c.returncode != 0:
        _cleanup(path)
        return {"passed": False, "error": "compile_error", "stage": "compile",
                "stderr_tail": _error_first(stderr), "n_warnings": n_warn}
    try:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=10,
                           preexec_fn=limit)
    except subprocess.TimeoutExpired:
        _cleanup(path, exe)
        return {"passed": False, "error": "timeout", "stage": "exec",
                "stderr_tail": "binary timed out (likely an infinite loop)", "n_warnings": n_warn}
    finally:
        # The compiled binary is ~4 MB. Leaving one per item in the sandbox
        # is what filled 3.2 GB of RAM on this box.
        _cleanup(path, exe)
    return {"passed": r.returncode == 0 and _SENTINEL in (r.stdout or ""),
            "error": ("" if r.returncode == 0 and _SENTINEL in (r.stdout or "")
                      else ("early_exit" if r.returncode == 0 else "assertion_failed")),
            "stage": "exec", "stdout_tail": (r.stdout or "")[-200:],
            "stderr_tail": (r.stderr or "")[-400:], "n_warnings": n_warn}


# ------------------------------------------------------- persian normalisation
_AR2FA = {"\u064A": "\u06CC",  # ARABIC YEH -> FARSI YEH
          "\u0649": "\u06CC",  # ALEF MAKSURA -> FARSI YEH
          "\u0643": "\u06A9",  # ARABIC KAF -> KEHEH
          "\u06AA": "\u06A9",
          "\u0623": "\u0627", "\u0622": "\u0627", "\u0625": "\u0627",  # ALEF variants
          "\u06D4": "\u0640", "\u0640": "",                      # yeh barree, tatweel
          }
_FA2AR_DIGITS = {chr(0x06F0 + i): str(i) for i in range(10)}
_AR2AR_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)}
_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\uFBB2-\uFBC1]")
_ARABIC_COMMA = {",": "،", ";": "؛", "?": "؟", "%": "٪"}


def normalize_fa(s, keep_zwnj=False):
    """Canonicalise Persian text for comparison.

    Without this, a model that correctly writes "تهران" scores zero against a
    key stored with Arabic yeh, and every Persian number is judged wrong because
    the model used Persian digits and the key used ASCII. Those are corpus
    artefacts, not capability gaps.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(str.maketrans(_AR2FA))
    s = s.translate(str.maketrans(_FA2AR_DIGITS))
    s = s.translate(str.maketrans(_AR2AR_DIGITS))
    s = _DIACRITICS.sub("", s)
    if not keep_zwnj:
        s = s.replace("\u200c", " ")  # ZWNJ -> space, then collapse
        s = re.sub(r"\s+", " ", s)
    # Arabic punctuation -> Persian, and vice versa, so either side matches
    s = s.translate(str.maketrans(_ARABIC_COMMA))
    s = s.translate(str.maketrans({v: k for k, v in _ARABIC_COMMA.items()}))
    return s.strip()


def token_f1(pred, gold, normalize=True):
    """Token-level F1 - the standard SCIR/BERTScore-style measure for Persian.

    Exact match is too brittle for free-form Persian; character-level F1 (as in
    the SCIR shared task) tolerates inflectional variation without giving credit
    for unrelated answers.
    """
    if normalize:
        pred, gold = normalize_fa(pred), normalize_fa(gold)
    p, g = pred.split(), gold.split()
    if not p or not g:
        return 1.0 if pred == gold else 0.0
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0) > 0:
            common[t] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------- devops static checks
def check_devops(text, checks):
    """Structural partial credit on a 0/.25/.5/.75/1 lattice, FDE-Bench style.

    Each must_contain is worth 1/N. No LLM judge: an LLM judge scoring a 2B
    model's Dockerfile would measure the judge's leniency, not the artifact.

    WHY THIS IS NOT PLAIN SUBSTRING MATCHING

    It used to be `c in text.lower()`. A model that emitted nothing but the
    required keywords - 'FROM EXPOSE 8000 USER HEALTHCHECK', no file at all -
    scored 1.0 on 10 of 10 tasks, including one requiring a pinned action version
    and one requiring that a secret NOT be printed, neither of which the echo
    satisfies. That measured prompt paraphrase, not artifact construction.

    The first fix anchored each requirement to the start of a line. That was
    wrong and was reverted after it was measured: it rejects legitimate file
    idioms - YAML list items ('- uses: actions/checkout@v4'), multi-stage
    Dockerfile keywords inside a FROM line ('FROM golang AS build'), and values
    under a key ('image: nginx') - so real artifacts scored 0.5-0.88 while only
    the echo was rejected. Over-correction is its own defect.

    What actually separates an artifact from a keyword salad is SIZE and SPREAD,
    not token position, so that is what is enforced:

      * an artifact floor - fewer than 3 non-empty lines is not a file, whatever
        tokens it contains. This alone rejects the one-line echo, and it is the
        property the echo was actually missing (it had no artifact);
      * spread - the requirements must be satisfied across at least 2 distinct
        lines, so they cannot all be crammed together.

    Token presence itself stays substring matching, because that is the right
    test for "this file expresses this requirement" and does not punish idiomatic
    formatting. The prompts were separately de-leaked so the requirement strings
    are no longer handed to the model verbatim.

    `must_not_contain` is also substring, which is correct for it: "must not
    appear anywhere" is genuinely a substring property.
    """
    got = extract_code(text)
    if not got:
        got = text or ""
    raw = got.lower()
    # Collapse ALL whitespace (including newlines) before substring matching:
    # the ban test was a raw match on un-normalised text, so a single extra
    # space or a tab evaded it: 'USER  root' (two spaces) and 'USER\troot'
    # both passed the non-root ban, and '${{secrets.TOKEN}}' - valid GitHub
    # Actions syntax - passed the do-not-print-the-secret ban. Those are the
    # two highest-value safety properties in the suite. Structure (line floor,
    # comments, spread) is measured separately on `lines` below, so collapsing
    # here cannot merge lines.
    g_flat = re.sub(r"\s+", " ", raw)
    need = [re.sub(r"\s+", " ", c.lower()).strip() for c in checks.get("must_contain", [])]
    ban = [re.sub(r"\s+", " ", c.lower()).strip() for c in checks.get("must_not_contain", [])]

    lines = [re.sub(r"[ \t\x0b\x0c\r]+", " ", ln).strip()
             for ln in raw.splitlines() if ln.strip()]

    # A file in which EVERY line is a comment is not an artifact, however many
    # lines it has. The previous floor was `len(lines) >= 3`, and a comment line
    # counts as a non-empty line, so a 100%-comment keyword salad scored 1.0 on
    # all 10 devops tasks - producing no file at all. Comments also have to
    # stay a minority: an artifact that is mostly commentary is a model
    # narrating its answer, not writing one.
    def _is_comment(ln):
        return ln.startswith("#") or ln.startswith("//") or ln.startswith("--") \
            or ln.startswith("/*") or ln.startswith(";")
    code_lines = [ln for ln in lines if not _is_comment(ln)]
    if not code_lines:
        return {"score": 0.0, "passed": False,
                "must_hit": 0, "must_total": len(need),
                "missed": list(need), "violations": [c for c in ban if c in g_flat],
                "empty": not got.strip(), "lines": len(lines),
                "reason": "every line is a comment: no artifact was produced"}
    comment_ratio = (len(lines) - len(code_lines)) / len(lines)
    if len(lines) < 3 or comment_ratio > 0.5:
        return {"score": 0.0, "passed": False,
                "must_hit": 0, "must_total": len(need),
                "missed": list(need), "violations": [c for c in ban if c in g_flat],
                "empty": not got.strip(), "lines": len(lines),
                "reason": (f"{len(lines)} non-empty line(s), "
                           f"{comment_ratio:.0%} comments"
                           + ("; needs >= 3 lines" if len(lines) < 3 else ""))}

    hits = [c for c in need if c in g_flat]
    # The requirement keywords themselves do not constitute an artifact. A 3- or
    # 4-line echo ("FROM\nexpose 8000\nUSER") clears the line floor and the
    # spread check because each keyword sits on its own line, yet no file was
    # produced. Require at least one substantive line that is NOT itself one of
    # the required tokens: a real Dockerfile has WORKDIR/COPY/RUN/CMD, a compose
    # file has image/ports/environment, a unit file has a [Section] body. That
    # is the difference between constructing an artifact and echoing a spec.
    need_set = {c.strip() for c in need}
    substantive = [ln for ln in code_lines
                   if ln.strip().lower() not in need_set
                   and not any(ln.strip().lower() == c for c in need)]
    if not substantive:
        return {"score": 0.0, "passed": False,
                "must_hit": len(hits), "must_total": len(need),
                "missed": [c for c in need if c not in g_flat],
                "violations": [c for c in ban if c in g_flat],
                "empty": not got.strip(), "lines": len(lines),
                "reason": "artifact contains only the required keywords and no "
                          "structure of its own"}
    hit_lines = {i for i, ln in enumerate(lines)
                 if any(c in ln for c in hits)}
    spread_ok = len(need) < 2 or len(hit_lines) >= 2

    score = len(hits) / len(need) if need else 0.0
    if not spread_ok:
        score *= 0.5           # all requirements crammed onto one line
    if ban:
        violations = [c for c in ban if c in g_flat]
        if violations:
            # A banned string is a hard requirement, not one more line on the
            # lattice. It previously cost only len(violations)/len(need), so an
            # answer that printed the secret entirely could still score 0.667 -
            # the worst possible outcome for the property being tested.
            score = 0.0
    else:
        violations = []
    score = min(1.0, score)
    return {
        "score": round(score, 3),
        "passed": score == 1.0,
        "must_hit": len(hits), "must_total": len(need),
        "missed": [c for c in need if c not in g_flat],
        "violations": violations,
        "spread_ok": spread_ok, "lines": len(lines),
        "empty": not got.strip(),
    }


# ------------------------------------------------------------ net diagnosis
# Two-sided verdict classification.
#
# The previous test was `says_broken = NOT (matches healthy-keywords)`. That makes
# every correct answer phrased without the magic words a FALSE ALARM, and it makes
# a correct real-fault diagnosis that happens to contain "working correctly" a
# MISSED FAULT. Measured on the suite as authored: "the ticket is wrong, the
# network is fine", "everything works fine", "all checks pass, nothing to do" and
# "connectivity is normal" all scored 0.0 on healthy items - 4 of 11, i.e. 36% of
# the suite - while "router-1 is working correctly for pings, but DNS resolution
# fails on every client" scored 0.0 on a real fault. The defect cut in BOTH
# directions.
#
# Now BOTH sides must be positively evidenced. A fault word wins over a scoped
# clearance, because these items always describe a real fault when there is one.
# And when neither side is evidenced the verdict is genuinely unclear, which gets
# a third of the item's credit rather than being scored 0 - zeroing an item for
# an ambiguous answer measured the rubric, not the model.
_FAULT_EV = re.compile(
    r"\b(?:fail(?:s|ed|ing|ure)?|unreachable|time(?:d)?\s*out|no response|"
    r"not respond(?:ing)?|cannot resolve|can'?t resolve|won'?t resolve|"
    r"servfail|nxdomain|misconfig(?:uration)?|address conflict|duplicate|"
    r"refused|reset(?:s|ting)?|broken|crash(?:es|ed|ing)?|degraded|"
    r"packet loss|dropp?(?:ed|ing|s)?)\b", re.I)
# Terms that name a diagnosed problem in this domain. Bare "fault" is safe here
# only because _NO_FAULT is tested first, so "no fault detected" is not read as a
# diagnosis.
_FAULT_EV2 = re.compile(
    r"\bfaults?\b|\bidle\b|vanish(?:ed|es|ing)?|adjacen\w+|not established|"
    r"\blost\b|\bgone\b|\boffline\b", re.I)
_HEALTHY_EV = re.compile(
    r"no (?:actual )?(?:fault|anomaly|problem|issue|error)|"
    r"nothing(?:'s| is) (?:actually )?(?:wrong|broken|to do)|"
    r"nothing to do|no action(?: (?:is )?(?:needed|required))?|"
    r"all checks pass|checks pass|everything works|"
    r"(?:network|connectivity|everything) (?:is |are )?(?:fine|healthy|normal|ok)|"
    r"working (?:fine|correctly|as expected)|"
    r"healthy|false premise|gather(?:ing)? more evidence|collect more data|"
    r"ticket is (?:wrong|incorrect)|no evidence of a fault|"
    r"(?:cannot|can't|unable to|could not|couldn't) reproduce|"
    r"names? resolv\w* (?:fine|ok|success\w*)|"
    r"(?:dns\s+)?resolution is (?:fine|healthy|working|ok(?:ay)?|normal|correct)|"
    r"reach\w* the internet|routing is (?:fine|normal|correct)", re.I)


_NO_FAULT = re.compile(
    r"\bno (?:actual |apparent )?(?:fault|anomaly|problem|issue|error)s?\b"
    r"|\bnot a fault\b|\bno evidence of (?:a )?fault\b"
    # A model that quotes the ticket in order to refute it ("the ticket's claim
    # that DNS is broken is incorrect") contains a fault word from the ticket
    # text. Without this the quotation is read as the model's own diagnosis and
    # a correct refutation scores as a false alarm.
    r"|\b(?:ticket|report|claim|allegation|premise)\b.{0,40}\b(?:is|are)\s+"
    r"(?:wrong|incorrect|false|not\s+supported)\b", re.I)


def netdiag_verdict(text):
    """Return True (a fault is present), False (healthy), or None (unclear).

    Order is load-bearing. An explicit denial of a fault is checked FIRST,
    because "no fault detected" contains the word "fault" and would otherwise be
    read as a diagnosis. Then a fault word anywhere wins, because these items
    describe a real fault when they have one and a scoped clearance ("router-1 is
    working correctly for pings, but DNS resolution fails") must not cancel it.
    """
    t = (text or "")
    if not t.strip():
        return None
    if _NO_FAULT.search(t):
        return False
    if _FAULT_EV.search(t) or _FAULT_EV2.search(t):
        return True
    if _HEALTHY_EV.search(t):
        return False
    return None


_NEG_CLAUSE = re.compile(
    r"\b(?:not|isn't|is not|aren't|no|never|rather than|instead of|"
    r"nothing to do with|unrelated to|rule out|ruled out)\b[^.]{0,24}$", re.I)
_ASSERT = re.compile(
    r"\b(?:is|are|was|caused by|root cause|cause[sd]?|because|"
    r"due to|fault(?:y)?(?: is)?|blame|culprit|the problem is|points? to|"
    r"responsible)\b", re.I)


def _asserted_token(text, token):
    """Is `token` ASSERTED as the cause, rather than merely mentioned - or denied?

    A bare substring test cannot tell "the root cause is a dns misconfig" from
    "DNS is fine, the problem is the cable" or from a verbatim echo of the config
    the model was shown. All three contain the gold token and all three scored
    full credit for that third. This requires the token to sit in an asserting
    clause and not to be denied on EITHER side of it.

    Negation usually FOLLOWS a short gold token ("dns is not the cause",
    "the vlan is fine"), so checking only the preceding clause - as the first
    version of this function did - let every denial through. Both windows are
    now examined; that first version was verified by reading, not by running a
    denial through it, which is exactly how it shipped broken.
    """
    if not token:
        return False
    post_neg = re.compile(
        r"\b(?:isn'?t|aren'?t|wasn'?t|is\s+not|are\s+not|was\s+not)\b|"
        r"\b(?:is|are|was)\s+(?:fine|healthy|ok(?:ay)?|normal|working|correct|good)\b|"
        r"\b(?:ruled?\s+out|rather than|instead of|nothing to do with|unrelated to)\b",
        re.I)
    for m in re.finditer(re.escape(token.lower()), text.lower()):
        start = max(0, m.start() - 60)
        clause = text[start:m.start()]
        after = text[m.end():m.end() + 40]
        if _NEG_CLAUSE.search(clause):
            continue                      # "DNS is not ... <token>"
        if post_neg.search(after):
            continue                      # "<token> is not the cause" / "is fine"
        if not _ASSERT.search(clause):
            # no assertion cue in this window; accept only a short, clean
            # clause (e.g. "faulty device: r2")
            if len(clause.strip()) > 24:
                continue
        return True
    return False


def score_netdiag(text, task):
    """FaulT-Bench-style scoring with a false-premise class.

    An item has three independent things to get right:
      1. anomaly verdict  (is anything actually broken?)
      2. device           (if so, where?)
      3. root cause class (if so, why?)
    Score is the fraction correct, so a model that always says "yes, broken"
    scores ~0.33 on this set rather than 1.0. An unclear verdict earns the
    verdict third and nothing more, rather than zeroing the whole item.
    """
    t = (text or "").lower()
    verdict = netdiag_verdict(text)
    if verdict is None:
        return {"score": round(1.0 / 3, 3), "verdict_ok": False,
                "verdict_unclear": True, "device_ok": False, "cause_ok": False,
                "detail": "verdict unclear: neither fault nor healthy evidence"}
    says_broken = verdict
    if task["is_anomaly"] and not says_broken:
        return {"score": 0.0, "verdict_ok": False, "device_ok": False,
                "cause_ok": False, "verdict_unclear": False,
                "detail": "missed a real fault"}
    if not task["is_anomaly"] and says_broken:
        return {"score": 0.0, "verdict_ok": False, "device_ok": False,
                "cause_ok": False, "verdict_unclear": False,
                "detail": "false alarm on a healthy network"}
    verdict_ok = True
    if not task["is_anomaly"]:
        return {"score": 1.0, "verdict_ok": True, "device_ok": True,
                "cause_ok": True, "verdict_unclear": False,
                "detail": "correctly identified as healthy"}
    dev = (task.get("faulty_device") or "").lower()
    cause = (task.get("root_cause") or "").lower()
    # device: accept the hostname token anywhere in the reply
    # device and cause must be ASSERTED, not merely present. See _asserted_token.
    dev_ok = _asserted_token(t, dev)
    cause_ok = _asserted_token(t, cause.replace("_", " ")) or _asserted_token(t, cause)
    score = (1.0 + (1.0 if dev_ok else 0.0) + (1.0 if cause_ok else 0.0)) / 3.0
    return {"score": round(score, 3), "verdict_ok": verdict_ok, "device_ok": dev_ok,
            "cause_ok": cause_ok, "verdict_unclear": False,
            "detail": "fault found; device/cause partial"}


# ------------------------------------------------------- GSM8K math scoring
# Official protocol (Cobbe et al. 2021): answer is the number after ####.
# Model output is free-form generation; we extract the final numeric answer
# with the same normalisation the official scorer applies (strip $, commas,
# %, whitespace) and compare exactly as strings after normalisation, plus
# a float-tolerant fallback for "18" vs "18.0".
_NUM_TAIL = re.compile(r"-?[\d,]*\.?\d+(?:[eE]-?\d+)?")


def extract_math_answer(text):
    """Extract the final numeric answer from a GSM8K-style generation.

    Strategy: prefer an explicit #### marker (official format), else the last
    number-like token in the text. Returns "" when nothing numeric is found.
    """
    if not text:
        return ""
    m = re.search(r"####\s*([^\n]+)", text)
    cand = m.group(1) if m else text
    # strip code fences / trailing prose, keep numeric tail
    nums = _NUM_TAIL.findall(cand.replace(",", "").replace("$", ""))
    if not nums and cand is not text:
        nums = _NUM_TAIL.findall(text.replace(",", "").replace("$", ""))
    return nums[-1].strip() if nums else ""


def _norm_num(s):
    s = (s or "").strip().replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"\s+", "", s)
    # 18.0 == 18 ; 1,000 == 1000 ; trailing periods from prose
    s = s.rstrip(".")
    try:
        f = float(s)
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return repr(round(f, 6))
    except ValueError:
        return s


def score_gsm8k(pred_text, gold):
    """Exact-match on normalised numbers (official GSM8K rule), float-tolerant."""
    pred = _norm_num(extract_math_answer(pred_text))
    g = _norm_num(str(gold))
    ok = bool(pred) and pred == g
    if not ok and pred and g:
        try:
            ok = abs(float(pred) - float(g)) <= max(1e-6, abs(float(g)) * 1e-6)
        except ValueError:
            ok = False
    return {"correct": ok, "pred": pred, "gold": g,
            "extracted": extract_math_answer(pred_text)[:80]}


if __name__ == "__main__":
    print("normalize_fa:", repr(normalize_fa("گزینه‌هاي ۰ تا ٩")))
    print("token_f1    :", round(token_f1("کتاب های خوب", "کتاب‌هاي خوب"), 3))
    print("wilson 8/32 :", tuple(round(x, 3) for x in wilson(8, 32)))
    print("pass@k 1/4  :", round(pass_at_k(4, 1, 1), 3))
    print("pass_hat_k  :", round(pass_hat_k(4, 1, 2), 3))
    print("ruff        :", {k: v for k, v in lint_python(
        'import os\ndef f(x):\n    if x==None:\n        return "%s" % x\n').items()
        if k in ("n_diagnostics", "codes", "weighted_severity", "n_correctness", "clean")})
    print("devops      :", check_devops(
        "```yaml\nservices:\n  nginx:\n    image: nginx\n```",
        {"must_contain": ["services:", "nginx", "postgres", "volumes:"],
         "must_not_contain": ["version:", "privileged: true"]}))
    print("netdiag     :", score_netdiag(
        "ping 10.0.0.1 ok, ping 8.8.8.8 ok, DNS resolves. The network is healthy, no action needed.",
        {"is_anomaly": False, "faulty_device": None, "root_cause": "none"}))
    print("rustfmt     :", rustfmt_check("fn main(){let x=1;}"))
    print("scorers OK")