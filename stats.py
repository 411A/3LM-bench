#!/usr/bin/env python3
"""
Is a multiple-choice score better than guessing? The statistics that decide it.

THE PROBLEM THIS SOLVES

A multiple-choice accuracy on its own is not interpretable. "electronics 58%,
chance 25%" invites the conclusion that the model knows electronics, but with
n=58 the 95% interval on 58% is roughly [0.45, 0.70] and the one-sided
probability of seeing 58% or better from a pure guesser is about 1e-6 - fine. Now
consider the Persian suite at n=14. A model scoring 43% there is LESS than 2
items above the 1-in-3 chance rate for the NLI items. Its 95% interval runs
straight through chance. The number is real arithmetic and it means nothing.

So every MC figure this harness reports is now emitted as a bundle:

    accuracy            the raw fraction correct
    chance_baseline     mean over items of 1/len(choices)   <- the null model
    best_constant       accuracy of always answering the most common gold letter
    lift_over_chance    accuracy / chance_baseline
    ci95_wilson         interval on the accuracy
    p_value             one-sided P(accuracy >= observed | pure guessing)
    significant         p_value < 0.05
    beats_chance        the 95% interval excludes the chance baseline
    min_acc_to_reject   the accuracy REQUIRED at this n to exclude chance

`min_acc_to_reject` is the honest answer to "how do we know it isn't random?":
at n=14 a model must reach roughly 64% before the data can rule out guessing at
all. At n=58 the bar is far lower. That number belongs next to every accuracy,
because it is what determines whether the accuracy means anything.

WHY AN EXACT BINOMIAL TAIL

No SciPy here (standard library only), and n is small enough (<= a few hundred)
that the exact tail is both cheap and free of the continuity-correction fudge
that makes normal approximations optimistic. We compute
P(X >= c) = sum_{k=c..n} C(n,k) p0^k (1-p0)^(n-k) in log space via lgamma so
that n! overflows never turn a p-value into a wrong answer.

WHY A ONE-SIDED TEST

The only question worth asking is whether the model beats the null model of
uniform random choice. There is no hypothesis that it does WORSE, so a two-sided
test would waste half the power on an alternative nobody cares about. (The other
thing worth watching - answering one letter constantly rather than uniformly at
random - is handled separately and more strictly by `best_constant`, which is a
deterministic baseline no stochastic test is needed for.)
"""
import math

ALPHA = 0.05


def wilson(c, n, z=1.959963985):
    """Wilson score interval. Preferred over the normal approximation because
    every n in this suite is < 100 and the normal interval misbehaves badly at
    small n and at proportions near 0 or 1."""
    if n == 0:
        return (0.0, 1.0)
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_binom_pmf(k, n, p0):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p0) + (n - k) * math.log1p(-p0))


def binom_sf(c, n, p0):
    """P(X >= c) for X ~ Binomial(n, p0). Exact, computed in log space."""
    if n <= 0:
        return 1.0
    if p0 <= 0:
        return 0.0 if c > 0 else 1.0
    if p0 >= 1:
        return 1.0 if c >= n else 0.0
    if c <= 0:
        return 1.0
    if c > n:
        return 0.0
    # Sum upward from c. log-sum-exp so the terms never underflow to a wrong 0.
    terms = []
    for k in range(c, n + 1):
        lp = _log_binom_pmf(k, n, p0)
        if lp > -745:
            terms.append(math.exp(lp))
    if not terms:
        return 0.0
    m = max(terms)
    return min(1.0, m * sum(t / m for t in terms))


def chance_baseline(tasks):
    """Null model: uniform random choice per item.

    Uses 1/len(choices) per item rather than 1/4, because the suites mix 3-option
    and 4-option items (the Persian NLI items have three choices). Reporting a
    flat 25% would understate chance on the 3-option items and make them look
    better than they are.
    """
    if not tasks:
        return 0.0
    return sum(1.0 / len(t["choices"]) for t in tasks) / len(tasks)


def best_constant(tasks):
    """Accuracy of the smartest knowledge-free strategy: always answer the single
    most common gold letter. Deterministic, so no significance test is needed -
    this is the floor a model with zero knowledge can guarantee."""
    from collections import Counter
    if not tasks:
        return 0.0, 0
    c = Counter(t["answer"] for t in tasks)
    top, cnt = c.most_common(1)[0]
    return cnt / len(tasks), top


def mc_bundle(records, tasks):
    """Everything needed to judge whether an MC accuracy means anything.

    `records` are the scored records (may repeat across reps); `tasks` supplies
    the per-item choice counts. When only records are available the choice count
    is recovered from the stored record, which is why records carry `choices`.
    """
    n = len(records)
    c = sum(1 for r in records if r.get("correct"))
    choice_counts = [len(r.get("choices") or []) for r in records]
    choice_counts = [k for k in choice_counts if k >= 2]
    p0 = (sum(1.0 / k for k in choice_counts) / len(choice_counts)) if choice_counts else 0.25

    acc = c / n if n else 0.0
    lo, hi = wilson(c, n)
    pval = binom_sf(c, n, p0) if n else 1.0

    # the smartest zero-knowledge strategy, on the realised gold distribution
    golds = [r.get("gold_index") for r in records if r.get("gold_index") is not None]
    if golds:
        from collections import Counter
        top_cnt = Counter(golds).most_common(1)[0][1]
        const = top_cnt / len(golds)
    else:
        const = max(choice_counts, default=4) and 1.0 / max(choice_counts, default=4)

    need = min_acc_to_reject(n, p0)
    return {
        "n": n,
        "correct": c,
        "accuracy": round(acc, 4),
        "chance_baseline": round(p0, 4),
        "best_constant_baseline": round(const, 4),
        "lift_over_chance": round(acc / p0, 3) if p0 > 0 else None,
        "ci95_wilson": [round(lo, 4), round(hi, 4)],
        "p_value_vs_chance": round(pval, 6),
        "significant_vs_chance": bool(pval < ALPHA),
        "beats_chance": bool(lo > p0),
        "min_acc_to_reject_chance": round(need, 4) if need is not None else None,
        "n_needed_for_50pct_lift": min_n_for_lift(p0, 1.5),
    }


def min_acc_to_reject(n, p0, alpha=ALPHA):
    """Smallest number of correct answers at this n whose one-sided p-value falls
    below alpha. This is the honest 'how well must a model do to prove it is not
    guessing' number. Returns None when even n/n cannot reject (n too small)."""
    for c in range(0, n + 1):
        if binom_sf(c, n, p0) < alpha:
            return c / n
    return None


def min_n_for_lift(p0, lift, alpha=ALPHA):
    """How many items are needed before a model scoring `lift`x the chance rate
    can be distinguished from a guesser. Tells you whether a suite has the power
    to detect the effect size you care about."""
    n = 4
    while n <= 4000:
        p = min(0.999, p0 * lift)
        c = int(math.ceil(p * n))
        if binom_sf(c, n, p0) < alpha:
            return n
        n += max(1, n // 20)
    return None


if __name__ == "__main__":
    print("=" * 74)
    print("What accuracy is required to rule out guessing, by suite size?")
    print("=" * 74)
    print(f"{'n':>4} {'chance':>8} {'min acc to reject':>19} {'n needed for 1.5x lift':>23}")
    for n in (11, 14, 15, 35, 58, 100, 200):
        p0 = 0.25
        need = min_acc_to_reject(n, p0)
        print(f"{n:>4} {p0:>8.0%} {need:>18.1%} {str(min_n_for_lift(p0,1.5)):>23}")
    print()
    print("Three-option items (Persian NLI), chance 33.3%:")
    print(f"{'n':>4} {'min acc to reject':>19}")
    for n in (14, 35, 58):
        print(f"{n:>4} {min_acc_to_reject(n, 1/3):>18.1%}")
    print()
    print("=" * 74)
    print("The suites as built")
    print("=" * 74)
    import json
    import os
    T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
    for name in ("science", "electronics", "persian", "english"):
        tasks = json.load(open(os.path.join(T, name + ".json")))
        p0 = chance_baseline(tasks)
        const, top = best_constant(tasks)
        need = min_acc_to_reject(len(tasks), p0)
        print(f"{name:<13} n={len(tasks):<4} chance={p0:.1%}  "
              f"best-constant={const:.1%} (letter {chr(65+top)})  "
              f"must reach {need:.0%} to prove non-random")