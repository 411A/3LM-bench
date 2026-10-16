#!/usr/bin/env python3
"""
Builds the benchmark task files.

Real datasets (downloaded, unmodified):
  * HumanEval        - code_gen_python      (164 problems)
  * MBPP sanitized   - code_gen_mbpp        (427 problems)
  * Hand-authored    - the rest, following the methodology in
                       docs/METHODOLOGY.md

Everything lands in tasks/*.json as {id, prompt, ...metadata}.
"""
import gzip, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tasks")
# Raw upstream downloads (HumanEval.jsonl.gz, sanitized-mbpp.json).
# Override with RES_DIR env to point at an existing cache.
_candidates = [os.environ.get("RES_DIR", ""),
               os.path.join(HERE, "data", "res")]
RES = next((c for c in _candidates if c and os.path.isdir(c)),
           os.path.join(HERE, "data", "res"))
os.makedirs(OUT, exist_ok=True)
rnd = random.Random(20261012)


def dump(name, rows):
    p = os.path.join(OUT, name + ".json")
    with open(p, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"  {name:<26} {len(rows):>4} tasks -> {p}")


# ---------------------------------------------------------------- code: HumanEval
def humaneval_py():
    rows = []
    with gzip.open(f"{RES}/HumanEval.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            # HumanEval's test field DEFINES check(candidate) but never CALLS it.
            # Without the explicit call, the test body defines a function and
            # exits 0: zero assertions run and any parseable stub scores a pass.
            # Verified: run_python(any_wrong_stub, test) -> passed=True, while
            # the same stub with "\ncheck(entry_point)" appended -> passed=False.
            # This is why verify_tasks.py agreed with everything while the runner
            # was wrong: the verifier appends the call, the runner did not.
            test = d["test"] + f"\ncheck({d['entry_point']})\n"
            rows.append({
                "id": d["task_id"],
                "source": "HumanEval",
                "lang": "python",
                "prompt": d["prompt"],
                "test": test,
                "entry_point": d["entry_point"],
                "canonical_solution": d["canonical_solution"],
            })
    return rows


# ---------------------------------------------------------------- code: MBPP
def mbpp():
    d = json.load(open(f"{RES}/sanitized-mbpp.json"))
    rows = []
    for t in d:
        tests = "\n".join(t["test_list"])
        rows.append({
            "id": f"MBPP/{t['task_id']}",
            "source": "MBPP-sanitized",
            "lang": "python",
            "prompt": (t["prompt"] +
                       "\n\nYour code should satisfy these tests:\n```python\n" + tests + "\n```"),
            "test": tests,
            "entry_point": None,
        })
    return rows


# ---------------------------------------------------------------- Rust (MultiPL-E style)
# Format follows MultiPL-E's eval_rust.py: the model writes ONLY the function,
# the harness appends fn main() + assert_eq! tests. Writing fn main() yourself
# produces a duplicate-main compile error, so the prompt says so explicitly.
def rust_tasks():
    base = [
        ("reverse_string", "Write a function `reverse_string(s: &str) -> String` that returns s reversed.",
         [('assert_eq!(reverse_string("abc"), "cba");', 'assert_eq!(reverse_string(""), "");')]),
        ("is_palindrome", "Write a function `is_palindrome(s: &str) -> bool` returning true if s reads the same backwards, ignoring case.",
         [('assert!(is_palindrome("Aba"));', 'assert!(!is_palindrome("ab"));')]),
        ("gcd", "Write a function `gcd(a: u64, b: u64) -> u64` computing the greatest common divisor.",
         [('assert_eq!(gcd(12, 18), 6);', 'assert_eq!(gcd(7, 13), 1);')]),
        ("fib", "Write a function `fib(n: u32) -> u64` returning the nth Fibonacci number, f(0)=0, f(1)=1.",
         [('assert_eq!(fib(0), 0);', 'assert_eq!(fib(10), 55);')]),
        ("is_prime", "Write a function `is_prime(n: u64) -> bool` returning true if n is prime.",
         [('assert!(is_prime(7));', 'assert!(!is_prime(9));')]),
        ("sum_vec", "Write a function `sum_vec(v: &Vec<i64>) -> i64` summing all elements.",
         [('assert_eq!(sum_vec(&vec![1,2,3]), 6);', 'assert_eq!(sum_vec(&vec![]), 0);')]),
        ("max_slice", "Write a function `max_slice(v: &[i32]) -> Option<i32>` returning the maximum, or None if empty.",
         [('assert_eq!(max_slice(&[1,5,3]), Some(5));', 'assert_eq!(max_slice(&[]), None);')]),
        ("count_upper", "Write a function `count_upper(s: &str) -> usize` counting ASCII uppercase letters.",
         [('assert_eq!(count_upper("AbC"), 2);', 'assert_eq!(count_upper(""), 0);')]),
        ("vowel_count", "Write a function `vowel_count(s: &str) -> usize` counting ASCII vowels.",
         [('assert_eq!(vowel_count("hello"), 2);', 'assert_eq!(vowel_count("xyz"), 0);')]),
        ("bubble_sort", "Write a function `bubble_sort(v: &mut Vec<i32>)` that sorts in place ascending.",
         [('let mut v=vec![3,1,2]; bubble_sort(&mut v); assert_eq!(v, vec![1,2,3]);', '')]),
    ]
    rows = []
    for name, desc, tests in base:
        # Flatten one level. Each entry above is a list of test strings, and
        # keeping the inner tuple would emit Python's repr into the Rust source
        # ("['assert_eq!(...)', '...']"), which fails to compile for reasons that
        # have nothing to do with the model's answer.
        keep = [frag for group in tests for frag in group if frag]
        assert all(isinstance(k, str) for k in keep), f"{name}: non-string test"
        assert keep, f"{name}: no tests survived filtering"
        rows.append({
            "id": f"rust/{name}",
            "source": "hand-authored/MultiPL-E-format",
            "lang": "rust",
            "prompt": (desc +
                       "\n\nWrite only the function definition. Do NOT write `fn main`, "
                       "do NOT write `#[test]`, and do NOT include a `use` statement. "
                       "Output only Rust code."),
            "tests": keep,
            "entry_point": name,
        })
    return rows


# ---------------------------------------------------------------- bug fixing
# Lineage: SWE-bench's FAIL_TO_PASS / PASS_TO_PASS construction, dropped from
# repo scale to single file (SWE-bench needs Docker + repo checkout).
BUGFIX = [
    ("off_by_one", '''def sum_range(a, b):
    """Return the sum of integers from a to b INCLUSIVE."""
    return sum(range(a, b))''',
     "assert sum_range(1, 4) == 10", "b is exclusive; should be b + 1"),
    ("mutable_default", '''def append_item(item, lst=[]):
    lst.append(item)
    return lst''',
     "assert append_item(1) == [1]\nassert append_item(2) == [2]", "mutable default argument"),
    ("none_compare", '''def is_none(x):
    return x == None''',
     "assert is_none(None) is True\nassert is_none(0) is False", "== None should be is None"),
    ("shadowed_loop", '''def total(items):
    total = 0
    for item in items:
        if item < 0:
            continue
        total += item
    return total''',
     "assert total([1, 2, 3]) == 6", "no actual bug here - verify the function is correct"),
    ("wrong_operator", '''def is_even(n):
    return n % 2 == 1''',
     "assert is_even(4) is True\nassert is_even(5) is False", "modulo/equality operator inverted"),
    ("unclosed_file", '''def read_all(path):
    f = open(path)
    data = f.read()
    return data''',
     "assert isinstance(read_all.__doc__, type(None)) or True", "file handle never closed"),
    ("string_vs_int", '''def parse_port(value):
    return int(value) + 1''',
     "assert parse_port('80') == 81", "no actual bug - verify correctness"),
    ("index_error", '''def second(lst):
    return lst[1]''',
     "assert second([1,2,3]) == 2\nassert second([5]) is None", "index 1 out of range for 1-element list"),
    ("wrong_key", '''def get_age(person):
    return person['ages']''',
     "def p(): return {'age': 30}\nassert get_age(p()) == 30", "wrong dict key"),
    ("infinite_loop", '''def count_down(n):
    count = 0
    while n > 0:
        count += 1
    return count''',
     "assert count_down(3) == 3", "n never decremented - infinite loop"),
    ("type_confusion", '''def average(nums):
    total = 0
    for n in nums:
        total += n
    return total / len(nums)''',
     "assert average([2, 4]) == 3\nassert average([]) is None", "empty list divides by zero"),
    ("float_truncation", '''def half(n):
    return n // 2''',
     "assert half(5) == 2.5\nassert half(4) == 2", "// truncates; true division needed"),
]


def bugfix():
    return [{
        "id": f"bugfix/{name}",
        "source": "hand-authored/SWE-bench-lineage",
        "lang": "python",
        "prompt": ("The following Python function is buggy. Fix it.\n\n"
                   f"```python\n{code}\n```\n\n"
                   f"It should satisfy:\n```python\n{test}\n```\n"
                   "Reply with only the corrected function inside a ```python block."),
        "test": test,
        "hint": hint,
    } for name, code, test, hint in BUGFIX]


# ---------------------------------------------------------------- algorithmic
# Every entry names the function AND its exact signature. Without the signature
# the model picks its own name, the test raises NameError, and the item is scored
# as a failure when the algorithm was in fact correct. That is a harness failure
# masquerading as a model failure, and it is invisible unless you read the
# stderr - which is why every task's stderr is stored verbatim.
ALGO = [
    ("two_sum", "Return the two indices that add up to target.",
     "def two_sum(nums, target):", 
     "assert two_sum([2,7,11,15], 9) == [0,1]\nassert two_sum([3,2,4], 6) == [1,2]\nassert two_sum([1,2], 9) is None"),
    ("binary_search", "Return the index of target in the sorted list arr, or -1 if absent.",
     "def binary_search(arr, target):",
     "assert binary_search([1,3,5,7], 5) == 2\nassert binary_search([1,3,5], 4) == -1"),
    ("merge_intervals", "Merge overlapping intervals and return them sorted.",
     "def merge_intervals(intervals):",
     "assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]\nassert merge_intervals([]) == []"),
    ("valid_parentheses", "Return True if the bracket string is balanced.",
     "def valid_parentheses(s):",
     "assert valid_parentheses('([]{})') is True\nassert valid_parentheses('(]') is False"),
    ("coin_change", "Return the fewest coins to make amount, or -1 if impossible.",
     "def coin_change(coins, amount):",
     "assert coin_change([1,2,5], 11) == 3\nassert coin_change([2], 3) == -1"),
    ("longest_common_prefix", "Return the longest common prefix of a list of strings.",
     "def longest_common_prefix(strs):",
     "assert longest_common_prefix(['flower','flow','flight']) == 'fl'\nassert longest_common_prefix([]) == ''"),
    ("product_except_self", "Return the product of all other elements at each index, without division.",
     "def product_except_self(nums):",
     "assert product_except_self([1,2,3,4]) == [24,12,8,6]"),
    ("max_subarray", "Return the maximum sum of a contiguous subarray.",
     "def max_subarray(nums):",
     "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6\nassert max_subarray([-1]) == -1"),
    ("group_anagrams", "Group words that are anagrams of each other.",
     "def group_anagrams(words):",
     # sorted() on lists compares element-wise, so the groups come out in
     # first-word order: 'bat' < 'eat' < 'tan'. Writing the expected list in
     # any other order would fail a correct implementation.
     "r = group_anagrams(['eat','tea','tan','ate','nat','bat'])\n"
     "assert sorted(r) == [['bat'], ['eat','tea','ate'], ['tan','nat']]"),
    ("number_of_islands", "Count islands in a grid given as a list of strings of '1' and '0'.",
     "def number_of_islands(grid):",
     "assert number_of_islands(['110','010','111']) == 1\nassert number_of_islands(['000','000']) == 0"),
    ("climb_stairs", "Ways to climb n stairs taking 1 or 2 at a time.",
     "def climb_stairs(n):",
     "assert climb_stairs(2) == 2\nassert climb_stairs(3) == 3\nassert climb_stairs(10) == 89"),
    ("top_k_frequent", "Return the k most frequent elements, most frequent first.",
     "def top_k_frequent(nums, k):",
     "r = top_k_frequent([1,1,1,2,2,3], 2)\nassert sorted(r) == [1,2]"),
]


def algo():
    return [{
        "id": f"algo/{name}", "source": "hand-authored", "lang": "python",
        "prompt": ("Problem: " + desc +
                   f"\n\nWrite a Python function named `{name}` with this signature:\n"
                   f"```python\n{sig}\n```\n"
                   "Reply with only the function inside a ```python block."),
        "test": test,
    } for name, desc, sig, test in ALGO]


# ---------------------------------------------------------------- refactoring
REFACTOR = [
    ("loop_to_comprehension", '''out = []
for x in range(30):
    if x % 3 == 0 and x != 0:
        out.append(x * x)''',
     "assert out == [x*x for x in range(30) if x % 3 == 0 and x != 0]"),
    ("extract_function", '''def process(data):
    total = 0
    for d in data:
        if d < 0:
            total += -d
        else:
            total += d
    return total''',
     "assert process([-1, 2, -3]) == 6"),
    ("early_return", '''def check(n):
    if n < 0:
        result = False
    elif n == 0:
        result = True
    else:
        result = n % 2 == 0
    return result''',
     "assert check(-1) is False\nassert check(0) is True\nassert check(4) is True\nassert check(3) is False"),
    ("remove_duplicates", '''def dedupe(lst):
    seen = []
    for item in lst:
        if item not in seen:
            seen.append(item)
    return seen''',
     "assert dedupe([1,1,2,3,2]) == [1,2,3]"),
    ("replace_loop_sum", '''def total(values):
    t = 0
    i = 0
    while i < len(values):
        t += values[i]
        i += 1
    return t''',
     "assert total([1,2,3]) == 6"),
]


def refactor():
    return [{
        "id": f"refactor/{name}", "source": "hand-authored", "lang": "python",
        "prompt": ("Refactor this Python code for clarity and efficiency. "
                   "Behaviour MUST stay identical.\n\n```python\n" + code +
                   "\n```\n\nReply with only the refactored code inside a ```python block."),
        "test": test,
    } for name, code, test in REFACTOR]


if __name__ == "__main__":
    print("building tasks...")
    dump("code_humaneval", humaneval_py())
    dump("code_mbpp", mbpp())
    dump("code_rust", rust_tasks())
    dump("bugfix", bugfix())
    dump("algo", algo())
    dump("refactor", refactor())
    print("done")
