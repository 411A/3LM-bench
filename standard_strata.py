#!/usr/bin/env python3
"""Build stratified 30-item screening instances from the standard task files.

READS tasks/std_*.json (+ data/raw/gsm8k.json for step counts).
WRITES tasks/std_*_s30.json. Never modifies the source files.

Coverage contract (enforced, not hoped):
  * Every stratum contributes >= 3 items (MIN_PER_SUBCAT). The builder FAILS
    loudly if any stratum cannot supply them — a sampler that silently drops
    a subcategory is a broken sampler.
  * Where upstream ships a natural subcategory it is used as the stratum:
      mmlu        6 subjects x 5
      arc         exam families Mercury / state-exams / national-intl x 10
      hellaswag   indomain / zeroshot x 15
      belebele_fa question_number 1 / 2 x 15, all 30 from DISTINCT passages
      gsm8k       solution steps 0-2 / 3-4 / 5+ x 10
  * Where upstream ships NO subcategory (winogrande: nothing;
    truthfulqa MC rows: no topic fields), the stratum is a documented
    surface proxy (length terciles / choice-count bands) and is labelled
    proxy=True. No fake topics are invented.
  * Inside a stratum, distinct keys (exam source, activity label, passage)
    are maximized round-robin so 10 items never come from 1 source.

Files are ordered ROUND-ROBIN across strata, so any prefix (--limit 12 ->
4 per band) stays balanced. Deterministic seed 20261012 + per-suite salt,
recorded per item and in data/manifest.json.

Usage: python3 standard_strata.py [--check]
"""
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "tasks")
RAW = os.path.join(HERE, "data", "raw")
SEED = 20261012
MIN_PER_SUBCAT = 3


def load(name):
    p = os.path.join(TASKS, name + ".json")
    if not os.path.exists(p):
        raise SystemExit(f"missing source (run standard_fetch.py first): {p}")
    return json.load(open(p))


def pick_diverse(rng, items, n, key, min_distinct=None):
    """Take n items, maximizing distinct key() values (round-robin over keys).

    Guarantees: exactly n items (fails if pool smaller), every item unique,
    and at least min_distinct distinct keys (fails loudly otherwise).
    min_distinct=None means "as many as exist, up to n" — maximal but
    satisfiable (e.g. WikiHow has only 14 activity titles, so 15 distinct
    of 15 picks is impossible and must not fail the build).
    """
    if len(items) < n:
        raise SystemExit(f"FAIL: pool {len(items)} < requested {n}")
    by_key = {}
    for r in items:
        by_key.setdefault(key(r), []).append(r)
    for v in by_key.values():
        rng.shuffle(v)
    out, seen = [], set()
    keys = list(by_key)
    rng.shuffle(keys)
    if min_distinct is None:
        min_distinct = min(n, len(keys))
    ki = 0
    while len(out) < n:
        gave = False
        for _ in range(len(keys)):
            k = keys[ki % len(keys)]
            ki += 1
            if by_key[k]:
                out.append(by_key[k].pop())
                seen.add(k)
                gave = True
                if len(out) == n:
                    break
        if not gave:
            raise SystemExit("FAIL: pool exhausted unexpectedly")
    if len(seen) < min_distinct:
        raise SystemExit(
            f"FAIL: only {len(seen)} distinct keys, need {min_distinct}")
    return out


def shuffle_take(rng, items, n):
    idx = list(range(len(items)))
    rng.shuffle(idx)
    if len(idx) < n:
        raise SystemExit(f"FAIL: pool {len(idx)} < requested {n}")
    return [items[i] for i in idx[:n]]


def terciles(rows, key):
    vs = sorted(((key(r), i) for i, r in enumerate(rows)))
    n = len(rows)
    c1, c2 = vs[n // 3][0], vs[2 * n // 3][0]
    return ({"short": [rows[i] for v, i in vs if v <= c1],
             "mid": [rows[i] for v, i in vs if c1 < v <= c2],
             "long": [rows[i] for v, i in vs if v > c2]}, (c1, c2))


def surflen(r):
    return len(r.get("question", "")) + sum(len(c) for c in r.get("choices", []))


NATIONAL = {"TIMSS", "NAEP"}


def arc_source(r):
    return r["id"].split("_")[0]


def build_arc(name, salt):
    rows = load(name)
    rng = random.Random(f"{SEED}-{salt}")
    merc = [r for r in rows if arc_source(r) == "Mercury"]
    nat = [r for r in rows if arc_source(r) in NATIONAL]
    state = [r for r in rows if arc_source(r) != "Mercury"
             and arc_source(r) not in NATIONAL]
    bands = [
        ("Mercury", pick_diverse(rng, merc, 10, arc_source, 1)),
        ("state-exams", pick_diverse(rng, state, 10, arc_source, 5)),
        ("national-intl", pick_diverse(rng, nat, 10, arc_source, 2)),
    ]
    return rows, bands, {"type": "exam-family", "proxy": False}


def build_hellaswag():
    rows = load("std_hellaswag")
    raw = {r["ind"]: r for r in
           json.load(open(os.path.join(RAW, "hellaswag.json")))}
    rng = random.Random(f"{SEED}-hellaswag")

    def split_of(t):
        return raw[int(t["id"].split("-")[1])].get("split_type", "?")

    def act_of(t):
        return raw[int(t["id"].split("-")[1])].get("activity_label", "?")

    bands = []
    for split in ("indomain", "zeroshot"):
        pool = [r for r in rows if split_of(r) == split]
        bands.append((split, pick_diverse(rng, pool, 15, act_of)))
    return rows, bands, {"type": "split_type(indomain/zeroshot)+distinct-activity",
                         "proxy": False,
                         "note": "zeroshot(WikiHow) has only 14 activity "
                                 "titles: all 14 covered"}


def build_winogrande():
    rows = load("std_winogrande")
    rng = random.Random(f"{SEED}-winogrande")
    bands, cuts = terciles(rows, lambda r: len(r.get("question", "")))
    out = [(b, shuffle_take(rng, bands[b], 10)) for b in ("short", "mid", "long")]
    return rows, out, {"type": "surface-length-terciles", "cuts": cuts,
                       "proxy": True,
                       "note": "upstream ships no topic labels; length only"}


def build_truthfulqa():
    rows = load("std_truthfulqa_mc")
    rng = random.Random(f"{SEED}-truthfulqa")
    groups = {"choice2-4": [r for r in rows if len(r["choices"]) <= 4],
              "choice5-7": [r for r in rows if 5 <= len(r["choices"]) <= 7],
              "choice8+": [r for r in rows if len(r["choices"]) >= 8]}
    bands = [(b, shuffle_take(rng, groups[b], 10)) for b in
             ("choice2-4", "choice5-7", "choice8+")]
    return rows, bands, {"type": "choice-count-bands", "proxy": True,
                         "note": "MC rows ship no topic fields; distractors only"}


def build_belebele():
    rows = load("std_belebele_fa")
    raw = json.load(open(os.path.join(RAW, "belebele_fa.json")))
    rng = random.Random(f"{SEED}-belebele")

    def qnum(t):
        return raw[int(t["id"].rsplit("-", 1)[1])].get("question_number")

    bands = []
    used = set()
    for q in (1, 2):
        pool = [r for r in rows if qnum(r) == q
                and r["question"].split("\n\n")[0] not in used]
        band = pick_diverse(
            rng, pool, 15, lambda r: r["question"].split("\n\n")[0], 15)
        used.update(r["question"].split("\n\n")[0] for r in band)
        bands.append((f"qnum{q}", band))
    # 30 distinct passages overall
    passages = [r["question"].split("\n\n")[0] for _, b in bands for r in b]
    if len(set(passages)) < 30:
        raise SystemExit(f"FAIL: only {len(set(passages))} distinct passages")
    return rows, bands, {"type": "question_number+distinct-passage", "proxy": False}


def build_mmlu():
    rows = load("std_mmlu")
    rng = random.Random(f"{SEED}-mmlu")
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append(r)
    bands = [(s, shuffle_take(rng, by_subj[s], 5)) for s in sorted(by_subj)]
    return rows, bands, {"type": "subject", "proxy": False,
                         "subjects": sorted(by_subj)}


def build_gsm8k():
    rows = load("std_gsm8k")
    raw = json.load(open(os.path.join(RAW, "gsm8k.json")))
    rng = random.Random(f"{SEED}-gsm8k")

    def steps(t):
        g = raw[int(t["id"].split("-")[1])]["answer"]
        return len(re.findall(r"<<", g))

    groups = {"steps0-2": [r for r in rows if steps(r) <= 2],
              "steps3-4": [r for r in rows if 3 <= steps(r) <= 4],
              "steps5+": [r for r in rows if steps(r) >= 5]}
    bands = [(b, shuffle_take(rng, groups[b], 10)) for b in
             ("steps0-2", "steps3-4", "steps5+")]
    return rows, bands, {"type": "solution-step-bands", "proxy": False,
                         "note": "steps from <<..>> in official answers; "
                                 "no topic labels upstream"}


BUILDERS = {
    "std_arc_challenge": lambda: build_arc("std_arc_challenge", "arcc"),
    "std_arc_easy": lambda: build_arc("std_arc_easy", "arce"),
    "std_hellaswag": build_hellaswag,
    "std_winogrande": build_winogrande,
    "std_truthfulqa_mc": build_truthfulqa,
    "std_belebele_fa": build_belebele,
    "std_mmlu": build_mmlu,
    "std_gsm8k": build_gsm8k,
}


def main():
    check_only = "--check" in sys.argv
    ok = True
    for name, fn in BUILDERS.items():
        out = os.path.join(TASKS, name + "_s30.json")
        if check_only:
            e = "OK " if os.path.exists(out) else "MISSING "
            print(f"  {e} {out}")
            ok &= os.path.exists(out)
            continue
        rows, bands, meta = fn()
        # hard guarantee: no stratum below MIN_PER_SUBCAT, ever
        for bname, items in bands:
            if len(items) < MIN_PER_SUBCAT:
                raise SystemExit(f"FAIL: {name}/{bname} has {len(items)} "
                                 f"< {MIN_PER_SUBCAT}")
        # round-robin: any prefix stays stratum-balanced
        ordered = []
        for k in range(max(len(b) for _, b in bands)):
            for bi, (_, b) in enumerate(bands):
                if k < len(b):
                    r = dict(b[k])
                    r["stratum"] = bi
                    r["stratum_name"] = bands[bi][0]
                    r["screen_subset"] = True
                    r["screen_seed"] = SEED
                    ordered.append(r)
        with open(out, "w") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=1)
        print(f"  {name}_s30: {len(ordered)} items "
              f"({'+'.join(f'{bn}={len(bi)}' for bn, bi in bands)}) [{meta['type']}]")
        man_path = os.path.join(HERE, "data", "manifest.json")
        man = json.load(open(man_path)) if os.path.exists(man_path) else {}
        import hashlib
        man[name + "_s30"] = {
            "source": name, "kind": "screen30", "n_items": len(ordered),
            "seed": SEED, "min_per_stratum": MIN_PER_SUBCAT,
            "strata": meta,
            "stratum_names": [bn for bn, _ in bands],
            "order": "round-robin",
            "sha16": hashlib.sha256(json.dumps(
                ordered, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16],
        }
        with open(man_path, "w") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
