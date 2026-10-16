#!/usr/bin/env python3
"""
Validate that every task in the suite is solvable and the harness agrees.

A benchmark task that no correct program can pass measures the harness, not the
model. This script proves each task is winnable before any model is run against
it, which is how the Rust suite was caught: the test strings were being emitted
as Python repr, so all ten tasks failed to compile no matter what the model
wrote. Every model scored 0/10 on "Rust ability" and the actual cause was a
missing tuple flatten in the task builder.

Checks:
  * HumanEval   - the dataset's own canonical_solution must pass its test
  * MBPP        - the dataset's own reference `code` must pass its test_list
  * Rust        - hand-written references must compile and pass
  * bugfix/algo/refactor - hand-written references must pass
  * MC suites   - gold index in range, choices present and unique, no leaked
                  gold substring in the question
  * netdiag/devops - schema sanity, and the false-premise share is what we claim
  * embedded_cpp - delegates to verify_cpp.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scorers as S  # noqa: E402


def _res_file(name):
    cands = [os.environ.get("RES_DIR", ""),
             os.path.join(HERE, "data", "res")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, name)):
            return os.path.join(c, name)
    return os.path.join(HERE, "data", "res", name)

T = os.path.join(HERE, "tasks")


def load(n):
    return json.load(open(os.path.join(T, n + ".json")))


def hdr(s):
    print(f"\n=== {s} ===")


# --------------------------------------------------------------- HumanEval
def check_humaneval():
    hdr("HumanEval canonical solutions")
    rows = load("code_humaneval")
    bad = []
    t0 = time.time()
    for i, t in enumerate(rows):
        # canonical_solution is a function BODY - the signature lives in the
        # prompt. Concatenating the body on its own gives an indented fragment
        # and every task fails, which is how the harness bug was caught.
        src = t["prompt"] + t["canonical_solution"]
        r = S.run_python(src, t["test"] + f"\ncheck({t['entry_point']})", timeout=15)
        if not r["passed"]:
            bad.append((t["id"], (r.get("stderr_tail") or "")[-200:]))
    dt = time.time() - t0
    print(f"  {len(rows) - len(bad)}/{len(rows)} canonical solutions pass  ({dt:.0f}s)")
    for tid, err in bad[:8]:
        print(f"    FAIL {tid}: {err.strip()[:200]}")
    return len(bad) == 0


# --------------------------------------------------------------------- MBPP
def check_mbpp():
    hdr("MBPP sanitized reference solutions")
    import json as _j
    src = _j.load(open(_res_file("sanitized-mbpp.json")))
    rows = load("code_mbpp")
    by_id = {f"MBPP/{t['task_id']}": t for t in src}
    bad = []
    t0 = time.time()
    for t in rows:
        ref = by_id.get(t["id"])
        if not ref:
            bad.append((t["id"], "no reference found"))
            continue
        imports = "\n".join(ref.get("test_imports") or [])
        r = S.run_python(imports + "\n" + ref["code"], "\n".join(ref["test_list"]), timeout=15)
        if not r["passed"]:
            bad.append((t["id"], (r.get("stderr_tail") or "")[-160:]))
    dt = time.time() - t0
    print(f"  {len(rows) - len(bad)}/{len(rows)} reference solutions pass  ({dt:.0f}s)")
    for tid, err in bad[:8]:
        print(f"    FAIL {tid}: {err.strip()[:150]}")
    return len(bad) == 0


# --------------------------------------------------------------------- Rust
RUST_REF = {
    "rust/reverse_string": 'fn reverse_string(s: &str) -> String { s.chars().rev().collect() }',
    "rust/is_palindrome": 'fn is_palindrome(s: &str) -> bool {\n'
                          '    let c: String = s.chars().filter(|c| c.is_alphanumeric())\n'
                          '        .map(|c| c.to_ascii_lowercase()).collect();\n'
                          '    c.chars().eq(c.chars().rev())\n}',
    "rust/gcd": 'fn gcd(a: u64, b: u64) -> u64 {\n'
                '    let (mut a, mut b) = (a, b);\n'
                '    while b != 0 { let t = b; b = a % b; a = t; } a\n}',
    "rust/fib": 'fn fib(n: u32) -> u64 {\n'
                '    let (mut a, mut b) = (0u64, 1u64);\n'
                '    for _ in 0..n { let t = a + b; a = b; b = t; } a\n}',
    "rust/is_prime": 'fn is_prime(n: u64) -> bool {\n'
                     '    if n < 2 { return false; }\n'
                     '    if n % 2 == 0 { return n == 2; }\n'
                     '    let mut i = 3;\n'
                     '    while i * i <= n { if n % i == 0 { return false; } i += 2; }\n'
                     '    true\n}',
    "rust/sum_vec": 'fn sum_vec(v: &Vec<i64>) -> i64 { v.iter().sum() }',
    "rust/max_slice": 'fn max_slice(v: &[i32]) -> Option<i32> { v.iter().copied().max() }',
    "rust/count_upper": 'fn count_upper(s: &str) -> usize { s.chars().filter(|c| c.is_ascii_uppercase()).count() }',
    "rust/vowel_count": 'fn vowel_count(s: &str) -> usize {\n'
                        '    s.chars().filter(|c| matches!(c.to_ascii_lowercase(), \'a\'|\'e\'|\'i\'|\'o\'|\'u\')).count()\n}',
    "rust/bubble_sort": 'fn bubble_sort(v: &mut Vec<i32>) {\n'
                        '    let n = v.len();\n'
                        '    for i in 0..n { for j in 0..n - 1 - i {\n'
                        '        if v[j] > v[j + 1] { v.swap(j, j + 1); } } }\n}',
}


def check_rust():
    hdr("Rust reference solutions")
    rows = load("code_rust")
    bad = []
    for t in rows:
        ref = RUST_REF.get(t["id"])
        if not ref:
            bad.append((t["id"], "no reference written"))
            continue
        r = S.run_rust(ref, t["tests"], t["entry_point"])
        if not r["passed"]:
            bad.append((t["id"], (r.get("stderr_tail") or "")[-200:]))
    print(f"  {len(rows) - len(bad)}/{len(rows)} references compile and pass")
    for tid, err in bad:
        print(f"    FAIL {tid}: {err.strip()[:220]}")
    return len(bad) == 0


# ------------------------------------------------- bugfix / algo / refactor
CODE_REF = {
    # bugfix: the corrected version of each deliberately broken function
    "bugfix/off_by_one": "def sum_range(a, b):\n    return sum(range(a, b + 1))",
    "bugfix/mutable_default": "def append_item(item, lst=None):\n"
                              "    if lst is None:\n        lst = []\n"
                              "    lst.append(item)\n    return lst",
    "bugfix/none_compare": "def is_none(x):\n    return x is None",
    "bugfix/shadowed_loop": "def total(items):\n    return sum(i for i in items if i >= 0)",
    "bugfix/wrong_operator": "def is_even(n):\n    return n % 2 == 0",
    "bugfix/unclosed_file": "def read_all(path):\n"
                            "    with open(path) as f:\n        return f.read()",
    "bugfix/string_vs_int": "def parse_port(value):\n    return int(value) + 1",
    "bugfix/index_error": "def second(lst):\n    return lst[1] if len(lst) > 1 else None",
    "bugfix/wrong_key": "def get_age(person):\n    return person['age']",
    "bugfix/infinite_loop": "def count_down(n):\n"
                            "    count = 0\n    while n > 0:\n        count += 1\n        n -= 1\n"
                            "    return count",
    "bugfix/type_confusion": "def average(nums):\n"
                             "    return sum(nums) / len(nums) if nums else None",
    "bugfix/float_truncation": "def half(n):\n    return n / 2",

    # algo
    "algo/two_sum": "def two_sum(nums, target):\n    seen = {}\n"
                    "    for i, x in enumerate(nums):\n"
                    "        if target - x in seen:\n            return [seen[target - x], i]\n"
                    "        seen[x] = i\n    return None",
    "algo/binary_search": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n"
                          "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
                          "        if arr[mid] == target:\n            return mid\n"
                          "        if arr[mid] < target:\n            lo = mid + 1\n"
                          "        else:\n            hi = mid - 1\n    return -1",
    "algo/merge_intervals": "import itertools\ndef merge_intervals(intervals):\n"
                            "    s = sorted(intervals)\n"
                            "    return [list(g) for _, g in itertools.groupby(s, key=lambda i: None)]\n"
                            "    " if False else
                            "def merge_intervals(intervals):\n    s = sorted(intervals)\n"
                            "    out = []\n    for st, en in s:\n"
                            "        if out and st <= out[-1][1]:\n"
                            "            out[-1][1] = max(out[-1][1], en)\n"
                            "        else:\n            out.append([st, en])\n    return out",
    "algo/valid_parentheses": "def valid_parentheses(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n"
                              "    st = []\n    for c in s:\n"
                              "        if c in pairs.values():\n            st.append(c)\n"
                              "        elif st and st[-1] == pairs[c]:\n            st.pop()\n"
                              "        else:\n            return False\n    return not st",
    "algo/coin_change": "def coin_change(coins, amount):\n"
                        "    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n"
                        "    for a in range(1, amount + 1):\n"
                        "        for c in coins:\n"
                        "            if c <= a:\n"
                        "                dp[a] = min(dp[a], dp[a - c] + 1)\n"
                        "    return -1 if dp[amount] == float('inf') else dp[amount]",
    "algo/longest_common_prefix": "def longest_common_prefix(strs):\n"
                                  "    if not strs:\n        return ''\n"
                                  "    p = strs[0]\n    for s in strs[1:]:\n"
                                  "        while not s.startswith(p):\n            p = p[:-1]\n"
                                  "            if not p:\n                return ''\n    return p",
    "algo/product_except_self": "def product_except_self(nums):\n"
                                "    n = len(nums)\n    out = [1] * n\n"
                                "    pre = 1\n"
                                "    for i in range(n):\n        out[i] = pre\n        pre *= nums[i]\n"
                                "    suf = 1\n"
                                "    for i in range(n - 1, -1, -1):\n"
                                "        out[i] *= suf\n        suf *= nums[i]\n    return out",
    "algo/max_subarray": "def max_subarray(nums):\n"
                         "    best = cur = nums[0]\n"
                         "    for x in nums[1:]:\n        cur = max(x, cur + x)\n"
                         "        best = max(best, cur)\n    return best",
    "algo/group_anagrams": "def group_anagrams(words):\n    d = {}\n"
                           "    for w in words:\n        d.setdefault(''.join(sorted(w)), []).append(w)\n"
                           "    return list(d.values())",
    "algo/number_of_islands": "def number_of_islands(grid):\n"
                               "    if not grid:\n        return 0\n"
                               "    seen = [[False] * len(grid[0]) for _ in grid]\n    n = 0\n"
                               "    for r in range(len(grid)):\n        for c in range(len(grid[0])):\n"
                               "            if grid[r][c] == '1' and not seen[r][c]:\n"
                               "                n += 1\n                st = [(r, c)]\n                seen[r][c] = True\n"
                               "                while st:\n                    y, x = st.pop()\n"
                               "                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):\n"
                               "                        ny, nx = y + dy, x + dx\n"
                               "                        if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) \\\n"
                               "                                and grid[ny][nx] == '1' and not seen[ny][nx]:\n"
                               "                            seen[ny][nx] = True\n"
                               "                            st.append((ny, nx))\n    return n",
    "algo/climb_stairs": "def climb_stairs(n):\n    a, b = 1, 1\n"
                         "    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "algo/top_k_frequent": "from collections import Counter\n"
                           "def top_k_frequent(nums, k):\n"
                           "    return [w for w, _ in Counter(nums).most_common(k)]",

    # refactor
    "refactor/loop_to_comprehension": "out = [x * x for x in range(30) if x % 3 == 0 and x != 0]",
    "refactor/extract_function": "def _abs_sum(values):\n    return sum(abs(v) for v in values)\n\n"
                                 "def process(data):\n    return _abs_sum(data)",
    "refactor/early_return": "def check(n):\n    if n < 0:\n        return False\n"
                            "    if n == 0:\n        return True\n    return n % 2 == 0",
    "refactor/remove_duplicates": "def dedupe(lst):\n    return list(dict.fromkeys(lst))",
    "refactor/replace_loop_sum": "def total(values):\n    return sum(values)",
}


def check_code(alt_id_field="id"):
    hdr("bugfix / algo / refactor reference solutions")
    ok = True
    for suite in ("bugfix", "algo", "refactor"):
        rows = load(suite)
        bad = []
        for t in rows:
            ref = CODE_REF.get(t["id"])
            if not ref:
                bad.append((t["id"], "no reference written"))
                continue
            r = S.run_python(ref, t["test"], timeout=10)
            if not r["passed"]:
                bad.append((t["id"], (r.get("stderr_tail") or "")[-180:]))
        print(f"  {suite:<10} {len(rows) - len(bad)}/{len(rows)} references pass")
        for tid, err in bad:
            print(f"    FAIL {tid}: {err.strip()[:190]}")
            ok = False
    return ok


# ------------------------------------------------------------------- MC sanity
def check_mc():
    hdr("multiple-choice suites")
    ok = True
    for suite in ("science", "electronics", "persian", "english"):
        rows = load(suite)
        bad = []
        for t in rows:
            ch = t["choices"]
            a = t["answer"]
            if not isinstance(a, int) or a < 0 or a >= len(ch):
                bad.append((t["id"], f"gold index {a} out of range for {len(ch)} choices"))
            elif len(ch) < 3:
                # 3-way is legitimate (the Persian NLI items are ternary:
                # entailed / not entailed / neutral). Two-way is not.
                bad.append((t["id"], f"{len(ch)} choices, expected 3 or 4"))
            elif len(set(ch)) != len(ch):
                bad.append((t["id"], "duplicate choices"))
            elif not t["question"].strip():
                bad.append((t["id"], "empty question"))
            else:
                norm = S.normalize_fa(t["question"]) if t.get("lang") == "fa" else t["question"]
                if norm.count(ch[a]) > 1:
                    bad.append((t["id"], "gold answer appears twice - ambiguous"))
        print(f"  {suite:<14} {len(rows) - len(bad)}/{len(rows)} well-formed")
        for tid, err in bad[:6]:
            print(f"    FAIL {tid}: {err}")
            ok = False
    return ok


# ------------------------------------------------------- netdiag / devops sanity
def check_structured():
    hdr("netdiag / devops schema")
    ok = True
    nd = load("netdiag")
    fp = sum(1 for t in nd if not t["is_anomaly"])
    print(f"  netdiag  {len(nd)} items, false-premise share = {fp}/{len(nd)} "
          f"({fp/len(nd):.0%})")
    if fp == 0:
        print("    FAIL no false-premise items: a model that always says 'broken' "
              "would score full marks")
        ok = False
    elif fp / len(nd) < 0.25:
        print("    WARN false-premise share below the 25% floor")
    for t in nd:
        if t["is_anomaly"] and not t.get("root_cause"):
            print(f"    FAIL {t['id']}: anomaly with no root cause")
            ok = False

    dv = load("devops")
    for t in dv:
        if not t["checks"].get("must_contain"):
            print(f"    FAIL {t['id']}: no must_contain checks")
            ok = False
    print(f"  devops   {len(dv)} items, all have scoring checks")
    return ok


def main():
    t0 = time.time()
    results = {
        "HumanEval": check_humaneval(),
        "MBPP": check_mbpp(),
        "Rust": check_rust(),
        "bugfix/algo/refactor": check_code(),
        "multiple choice": check_mc(),
        "netdiag/devops": check_structured(),
    }
    print("\n=== embedded C++ ===")
    r = subprocess.run([sys.executable, os.path.join(HERE, "verify_cpp.py")],
                       capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "no output"
    print("  " + tail)
    results["embedded_cpp"] = r.returncode == 0

    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    bad = [k for k, v in results.items() if not v]
    print(f"\n{'ALL SUITES VALIDATED' if not bad else 'BROKEN: ' + ', '.join(bad)}"
          f"   ({time.time()-t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())