#!/usr/bin/env python3
"""
Lint-focused task set.

This is the novelty in the suite: nothing standard exists for "does the model
write code that passes ruff/clippy". The tasks deliberately present correct but
non-idiomatic code and ask for a cleanup, so the linter measures style quality
rather than correctness - correctness is already covered by the code suites.

Each task is scored on four numbers (raw count, severity-weighted, per-linter
detail, noqa usage) because a raw count alone rewards noise: a file missing
every docstring would out-score one with a genuine correctness bug.

`noqa`/`allow` counts are reported separately and treated as gaming. A model
that silences the linter has not satisfied it.

Grouping:
  python_ruff      -> ruff, pinned rule set
  python_fmt       -> ruff format --diff (is it already formatted?)
  rust_clippy      -> clippy correctness+style+complexity+perf
  rust_pedantic    -> clippy pedantic
  rust_nursery    -> clippy nursery (unstable, reported separately)
  rust_fmt         -> rustfmt --check
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")

# (id, prompt, language, group, starter_code or None)
PY_RUFF = [
    ("lint/py/legacy-comprehension",
     "Rewrite this so it is idiomatic modern Python 3.12+. Keep behaviour identical. "
     "Reply with only a ```python block.\n\n"
     "```python\n"
     "def get_ids(items):\n"
     "    result = []\n"
     "    for i in range(len(items)):\n"
     "        if items[i].id is not None:\n"
     "            result.append(items[i].id)\n"
     "    return result\n"
     "```"),
    ("lint/py/explicit-none",
     "Modernise this function: fix comparisons, use f-strings, and simplify the "
     "conditionals. Reply with only a ```python block.\n\n"
     "```python\n"
     "def label(n):\n"
     "    if n == None:\n"
     "        return 'none'\n"
     "    elif n == 0:\n"
     "        return 'zero: %s' % n\n"
     "    else:\n"
     "        return 'value %d' % n\n"
     "```"),
    ("lint/py/glob-and-path",
     "Clean up this path handling: avoid building paths with string concatenation, "
     "use pathlib. Reply with only a ```python block.\n\n"
     "```python\n"
     "import os\n"
     "def build(base, name):\n"
     "    d = base + '/' + name\n"
     "    if not os.path.exists(d):\n"
     "        os.makedirs(d)\n"
     "    return d\n"
     "```"),
    ("lint/py/error-handling",
     "Rewrite this with specific exception types and without a bare except. "
     "Reply with only a ```python block.\n\n"
     "```python\n"
     "def load(path):\n"
     "    try:\n"
     "        f = open(path, 'r')\n"
     "        return f.read()\n"
     "    except:\n"
     "        pass\n"
     "    return ''\n"
     "```"),
    ("lint/py/mutable-default",
     "Fix the issues in this function without changing its documented behaviour. "
     "Reply with only a ```python block.\n\n"
     "```python\n"
     "def collect(item, bucket=[]):\n"
     "    bucket.append(item)\n"
     "    return bucket\n"
     "```"),
    ("lint/py/typing",
     "Add complete type annotations and modern generics. Reply with only a "
     "```python block.\n\n"
     "```python\n"
     "def combine(a, b):\n"
     "    out = {}\n"
     "    for k, v in a:\n"
     "        out[k] = v\n"
     "    return list(out.values())\n"
     "```"),
    ("lint/py/subprocess",
     "Rewrite this so it is safe: use a list argument, check the return code, and "
     "capture output. Reply with only a ```python block.\n\n"
     "```python\n"
     "import os, subprocess\n"
     "def host():\n"
     "    out = subprocess.check_output('ping -c 1 ' + os.environ.get('HOST', ''))\n"
     "    return out\n"
     "```"),
    ("lint/py/unused-and-dead",
     "Remove the dead code and unused variables, keep behaviour identical. "
     "Reply with only a ```python block.\n\n"
     "```python\n"
     "import sys\n"
     "def calc(x):\n"
     "    tmp = 99\n"
     "    y = x * 2\n"
     "    z = y + 1\n"
     "    return z\n"
     "```"),
]

RS_CLIPPY = [
    ("lint/rs/clone-vector",
     "Rewrite this idiomatic Rust without the needless clone. Write only the "
     "functions, no `fn main`, no `#[test]`, no `use`.\n\n"
     "```rust\n"
     "fn total(v: &Vec<i32>) -> i32 {\n"
     "    let owned: Vec<i32> = v.clone();\n"
     "    let mut s = 0;\n"
     "    for x in &owned {\n"
     "        s += x;\n"
     "    }\n"
     "    s\n"
     "}\n"
     "```"),
    ("lint/rs/needless-return",
     "Clean this up to satisfy clippy::pedantic. Write only the function, no "
     "`fn main`, no `#[test]`, no `use`.\n\n"
     "```rust\n"
     "fn double(x: i32) -> i32 {\n"
     "    let y = x * 2;\n"
     "    return y;\n"
     "}\n"
     "```"),
    ("lint/rs/ptr-arg",
     "Rewrite taking a slice reference instead of &Vec, and document the function. "
     "Write only the function, no `fn main`, no `#[test]`, no `use`.\n\n"
     "```rust\n"
     "fn len3(v: &Vec<String>) -> usize {\n"
     "    v.len()\n"
     "}\n"
     "```"),
    ("lint/rs/manual-len",
     "Remove the manual length check and the redundant comparison. Write only the "
     "function, no `fn main`, no `#[test]`, no `use`.\n\n"
     "```rust\n"
     "fn first(v: &[i32]) -> Option<i32> {\n"
     "    if v.len() > 0 {\n"
     "        return Some(v[0]);\n"
     "    }\n"
     "    None\n"
     "}\n"
     "```"),
    ("lint/rs/unnecessary-wrap",
     "Simplify this so it returns the value directly rather than in a redundant "
     "wrapper. Write only the function, no `fn main`, no `#[test]`, no `use`.\n\n"
     "```rust\n"
     "fn identity(x: i32) -> Option<i32> {\n"
     "    if true {\n"
     "        return Some(x);\n"
     "    }\n"
     "    None\n"
     "}\n"
     "```"),
    ("lint/rs/result-vs-option",
     "Use Result instead of Option where this can fail, and document it. Write only "
     "the function, no `fn main`, no `#[test]`, no `use`.\n\n"
     "```rust\n"
     "fn parse(s: &str) -> Option<i32> {\n"
     "    match s.parse::<i32>() {\n"
     "        Ok(v) => Some(v),\n"
     "        Err(_) => None,\n"
     "    }\n"
     "}\n"
     "```"),
]


def rows(pairs, lang, group):
    return [{"id": i, "source": "hand-authored/lint-suite", "category": group,
             "lang": lang, "group": group, "prompt": p}
            for i, p, lang, group in ((x[0], x[1], lang, group) for x in pairs)]


if __name__ == "__main__":
    all_rows = (rows(PY_RUFF, "python", "python_ruff")
                + rows(RS_CLIPPY, "rust", "rust_clippy"))
    with open(os.path.join(OUT, "lint.json"), "w") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=1)
    print(f"  lint                     {len(all_rows):>4} tasks "
          f"({len(PY_RUFF)} ruff, {len(RS_CLIPPY)} clippy) -> {OUT}/lint.json")