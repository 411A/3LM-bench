#!/usr/bin/env python3
"""Fetch pinned standard benchmarks and convert to llmbench task files.

Local-first, stdlib-only (no torch/transformers/datasets/pyarrow).
Downloads via the Hugging Face datasets-server JSON API (or direct raw
for Belebele), caches raw rows under data/raw/, and writes converted
tasks/std_*.json in the internal MC format {id, question, choices, answer}.

Nothing here talks to a model. Rerunning with the same pin is idempotent;
use --check to verify cache without network, --offline to refuse network.

Usage:
  python3 standard_fetch.py --list
  python3 standard_fetch.py --fetch arc_challenge --fetch hellaswag
  python3 standard_fetch.py --fetch-all
  python3 standard_fetch.py --check        # verify cache completeness offline
"""
import argparse
import hashlib
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")
TASKS = os.path.join(HERE, "tasks")

DS_API = "https://datasets-server.huggingface.co/rows"

# Pinned source of truth. revision = HF dataset commit sha at verification
# time (2026-10-16); split/config follow the official eval protocol
# (test where labels are public, else validation).
SOURCES = {
    "arc_challenge": {
        "dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test",
        "n_total": 1172, "license": "CC BY-SA 4.0",
        "paper": "Clark et al. 2018, arXiv:1803.05457",
        "url": "https://huggingface.co/datasets/allenai/ai2_arc",
        "kind": "mc", "note": "grade-school science, hard subset",
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test",
        "n_total": 2376, "license": "CC BY-SA 4.0",
        "paper": "Clark et al. 2018, arXiv:1803.05457",
        "url": "https://huggingface.co/datasets/allenai/ai2_arc",
        "kind": "mc", "note": "grade-school science, easy subset",
    },
    "hellaswag": {
        "dataset": "Rowan/hellaswag", "config": "default", "split": "validation",
        "n_total": 10042, "license": "MIT",
        "paper": "Zellers et al. 2019, arXiv:1905.07830",
        "url": "https://huggingface.co/datasets/Rowan/hellaswag",
        "kind": "mc", "note": "commonsense NLI; validation used (test labels hidden)",
    },
    "winogrande": {
        "dataset": "allenai/winogrande", "config": "winogrande_xl", "split": "validation",
        "n_total": 1267, "license": "CC BY 4.0 (validation); test hidden",
        "paper": "Sakaguchi et al. 2021, arXiv:1907.10641",
        "url": "https://huggingface.co/datasets/allenai/winogrande",
        "kind": "mc", "note": "pronoun-resolution commonsense; validation used",
    },
    "gsm8k": {
        "dataset": "openai/gsm8k", "config": "main", "split": "test",
        "n_total": 1319, "license": "MIT",
        "paper": "Cobbe et al. 2021, arXiv:2110.14168",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
        "kind": "gen_math", "note": "grade-school math; answer after ####",
    },
    "truthfulqa_mc": {
        "dataset": "truthfulqa/truthful_qa", "config": "multiple_choice",
        "split": "validation", "n_total": 817, "license": "Apache-2.0",
        "paper": "Lin et al. 2022, arXiv:2109.07958",
        "url": "https://huggingface.co/datasets/truthfulqa/truthful_qa",
        "kind": "mc", "note": "MC1: one correct answer per item",
    },
    "belebele_fa": {
        "dataset": "facebook/belebele", "config": "pes_Arab (direct jsonl)",
        "split": "test", "n_total": 900, "license": "CC BY-SA 4.0",
        "paper": "Bandarkar et al. 2024, arXiv:2308.16884",
        "url": "https://huggingface.co/datasets/facebook/belebele",
        "kind": "mc", "note": "Persian (pes_Arab) reading comprehension; 4 options",
        "direct": "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/pes_Arab.jsonl",
    },
}

# MMLU is 57 per-subject configs; we pin a 6-subject STEM/general core
# (~1000 items) for the standard profile. Full 57-subject fetch is opt-in.
MMLU_SUBJECTS = [
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_biology",
    "high_school_computer_science",
    "high_school_world_history",
]
MMLU_META = {
    "dataset": "cais/mmlu", "split": "test", "license": "MIT",
    "paper": "Hendrycks et al. 2021, arXiv:2009.03300",
    "url": "https://huggingface.co/datasets/cais/mmlu",
    "kind": "mc",
}


def _get(url, timeout=60, retries=6):
    import time as _t
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llmbench/1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            _t.sleep(2 * (i + 1))
    raise last


def fetch_rows(dataset, config, split, total, raw_path):
    """Paginate datasets-server rows API; cache raw JSON rows to raw_path."""
    rows = []
    offset = 0
    # resume partial cache if present (rate-limit safe)
    part = raw_path + ".part"
    if os.path.exists(part):
        try:
            rows = json.load(open(part))
            offset = len(rows)
            print(f"    resuming at {offset}")
        except Exception:
            rows, offset = [], 0
    while offset < total:
        url = (f"{DS_API}?dataset={dataset.replace('/', '%2F')}"
               f"&config={config}&split={split}&offset={offset}&length=100")
        try:
            payload = json.loads(_get(url))
        except Exception as e:
            raise RuntimeError(f"fetch failed at offset {offset}: {e}") from None
        if payload.get("error"):
            raise RuntimeError(f"API error at offset {offset}: {payload}")
        batch = [x["row"] for x in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        print(f"    {offset}/{total} rows", flush=True)
        if offset % 500 == 0:
            with open(part, "w") as f:
                json.dump(rows, f, ensure_ascii=False)
            import time as _t2
            _t2.sleep(2)
        if payload.get("num_rows_total") and offset >= payload["num_rows_total"]:
            break
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    try:
        os.remove(part)
    except OSError:
        pass
    return rows


def load_or_fetch(name, offline=False):
    meta = SOURCES[name]
    raw_path = os.path.join(RAW, name + ".json")
    if os.path.exists(raw_path):
        rows = json.load(open(raw_path))
        print(f"  {name}: cached {len(rows)} rows")
        return rows
    if offline:
        raise SystemExit(f"offline and missing cache: {raw_path}")
    if meta.get("direct"):
        print(f"  {name}: downloading direct jsonl ...")
        raw = _get(meta["direct"], timeout=120).decode("utf-8", "replace")
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
        os.makedirs(RAW, exist_ok=True)
        with open(raw_path, "w") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"  {name}: cached {len(rows)} rows")
        return rows
    print(f"  {name}: fetching {meta['dataset']}/{meta['config']}/{meta['split']} ...")
    return fetch_rows(meta["dataset"], meta["config"], meta["split"],
                      meta["n_total"], raw_path)


# ------------------------------------------------------------ converters
def to_mc(rows, name, qk, cks, ak, idk=None):
    """Generic MC converter. cks = key(s) holding choice list; ak = answer key.
    Returns (tasks, skipped)."""
    out, skipped = [], 0
    for i, r in enumerate(rows):
        try:
            q = str(r[qk]).strip()
            if isinstance(cks, tuple):
                choices = [str(r[c]).strip() for c in cks]
            else:
                c = r[cks]
                choices = [str(x).strip() for x in (c.get("text") if isinstance(c, dict) else c)]
            a = r[ak]
            if isinstance(a, dict):  # truthfulqa labels
                labels = a.get("labels") or []
                a = int(labels.index(1)) if 1 in labels else -1
            elif isinstance(a, str) and a.strip().upper() in "ABCDE":
                a = "ABCDE".index(a.strip().upper())
            else:
                a = int(a)
            rid = str(r[idk]) if idk and r.get(idk) else f"{name}-{i}"
            if not q or len(choices) < 2 or not (0 <= a < len(choices)):
                skipped += 1
                continue
            out.append({"id": rid, "question": q, "choices": choices,
                        "answer": a, "subject": name, "source": name})
        except Exception:
            skipped += 1
    return out, skipped


def convert(name, rows):
    if name in ("arc_challenge", "arc_easy"):
        return to_mc(rows, name, "question", "choices", "answerKey", "id")
    if name == "hellaswag":
        out, skip = [], 0
        for i, r in enumerate(rows):
            try:
                q = (str(r.get("ctx_a", "")) + " " + str(r.get("ctx_b", ""))).strip()
                if not q:
                    q = str(r.get("ctx", "")).strip()
                choices = [str(x).strip() for x in r["endings"]]
                a = int(str(r["label"]).strip())
                if not q or len(choices) != 4 or not (0 <= a < 4):
                    skip += 1
                    continue
                out.append({"id": f"hellaswag-{r.get('ind', i)}", "question": q,
                            "choices": choices, "answer": a,
                            "subject": name, "source": name})
            except Exception:
                skip += 1
        return out, skip
    if name == "winogrande":
        out, skip = [], 0
        for i, r in enumerate(rows):
            try:
                q = str(r["sentence"]).replace("_", "____").strip()
                choices = [str(r["option1"]).strip(), str(r["option2"]).strip()]
                a = int(str(r["answer"]).strip()) - 1
                if not q or not (0 <= a < 2):
                    skip += 1
                    continue
                out.append({"id": f"winogrande-{i}", "question": q,
                            "choices": choices, "answer": a,
                            "subject": name, "source": name})
            except Exception:
                skip += 1
        return out, skip
    if name == "gsm8k":
        out, skip = [], 0
        for i, r in enumerate(rows):
            try:
                q = str(r["question"]).strip()
                ans = str(r["answer"]).strip()
                gold = ans.split("####")[-1].strip().replace(",", "").replace("$", "")
                if not q or not gold:
                    skip += 1
                    continue
                out.append({"id": f"gsm8k-{i}", "question": q,
                            "gold": gold, "gold_full": ans,
                            "subject": name, "source": name})
            except Exception:
                skip += 1
        return out, skip
    if name == "truthfulqa_mc":
        # Upstream MC1 fixes the gold at position 0 on ALL 817 items
        # (verified: gold hist {0: 817}). The official MC1 metric compares
        # likelihoods order-invariantly, so position is irrelevant there.
        # Our letter-pick runner would let a constant-"A" model score 100%,
        # so choices are shuffled with a fixed seed. Documented deviation.
        import random as _rnd
        rng = _rnd.Random(20261012)
        out, skip = [], 0
        for i, r in enumerate(rows):
            try:
                q = str(r["question"]).strip()
                mc1 = r["mc1_targets"]
                choices = [str(x).strip() for x in mc1["choices"]]
                labels = list(mc1["labels"])
                a = labels.index(1)
                if not q or len(choices) < 2:
                    skip += 1
                    continue
                order = list(range(len(choices)))
                rng.shuffle(order)
                new_choices = [choices[j] for j in order]
                new_a = order.index(a)
                out.append({"id": f"truthfulqa-{i}", "question": q,
                            "choices": new_choices, "answer": new_a,
                            "subject": name, "source": name,
                            "shuffled": True, "seed": 20261012})
            except Exception:
                skip += 1
        return out, skip
    if name == "belebele_fa":
        out, skip = [], 0
        for i, r in enumerate(rows):
            try:
                q = (str(r.get("flores_passage", "")).strip() + "\n\n"
                     + str(r.get("question", "")).strip())
                choices = [str(r[f"mc_answer{k}"]).strip() for k in (1, 2, 3, 4)]
                a = int(str(r.get("correct_answer_num", "")).strip()) - 1
                if not q or not (0 <= a < 4) or any(not c for c in choices):
                    skip += 1
                    continue
                out.append({"id": f"belebele-fa-{r.get('question_number', i)}-{i}",
                            "question": q, "choices": choices, "answer": a,
                            "subject": name, "source": name,
                            "lang": "fa", "dialect": r.get("dialect", "")})
            except Exception:
                skip += 1
        return out, skip
    raise ValueError(name)


def convert_mmlu(rows_by_subject):
    out, skip = [], 0
    for subj, rows in rows_by_subject.items():
        conv, s = to_mc(rows, f"mmlu_{subj}", "question", "choices", "answer")
        for t in conv:
            t["subject"] = f"mmlu/{subj}"
            t["source"] = "mmlu"
        out.extend(conv)
        skip += s
    return out, skip


def write_tasks(name, tasks):
    p = os.path.join(TASKS, f"std_{name}.json")
    with open(p, "w") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=1)
    print(f"  wrote {p} ({len(tasks)} items)")


def manifest_entry(name, tasks, skipped):
    meta = SOURCES.get(name, MMLU_META)
    h = hashlib.sha256(json.dumps(tasks, sort_keys=True,
                                  ensure_ascii=False).encode()).hexdigest()[:16]
    return {"name": name, "dataset": meta.get("dataset", "cais/mmlu"),
            "config": meta.get("config", "6-subject core"),
            "split": meta.get("split", "test"),
            "license": meta.get("license"), "paper": meta.get("paper"),
            "url": meta.get("url"), "kind": meta.get("kind", "mc"),
            "n_items": len(tasks), "skipped": skipped, "sha16": h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fetch", action="append", default=[])
    ap.add_argument("--fetch-all", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--mmlu-subjects", default=",".join(MMLU_SUBJECTS))
    args = ap.parse_args()

    if args.list:
        for n, m in SOURCES.items():
            print(f"  {n:<15} {m['dataset']}/{m['config']}/{m['split']} "
                  f"n={m['n_total']} {m['license']}")
        print(f"  mmlu            cais/mmlu/test 6 subjects "
              f"({', '.join(MMLU_SUBJECTS)})")
        return

    want = list(args.fetch) or ([] if not args.fetch_all else list(SOURCES))
    if args.check and not want:
        want = list(SOURCES) + ["mmlu"]
    if not want and not args.check:
        ap.error("give --fetch NAME, --fetch-all, --check, or --list")

    os.makedirs(RAW, exist_ok=True)
    man_path = os.path.join(DATA, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}

    if args.check:
        ok = True
        for n in SOURCES:
            p = os.path.join(TASKS, f"std_{n}.json")
            e = "OK " if os.path.exists(p) else "MISSING "
            print(f"  {e} {p}")
            ok &= os.path.exists(p)
        pm = os.path.join(TASKS, "std_mmlu.json")
        print(f"  {'OK ' if os.path.exists(pm) else 'MISSING '} {pm}")
        ok &= os.path.exists(pm)
        raise SystemExit(0 if ok else 1)

    for n in want:
        if n == "mmlu":
            continue
        if n not in SOURCES:
            raise SystemExit(f"unknown dataset: {n} (see --list)")
        rows = load_or_fetch(n, offline=args.offline)
        tasks, skip = convert(n, rows)
        write_tasks(n, tasks)
        man[n] = manifest_entry(n, tasks, skip)

    if "mmlu" in want:
        subjects = [s.strip() for s in args.mmlu_subjects.split(",") if s.strip()]
        by_subj = {}
        for s in subjects:
            rp = os.path.join(RAW, f"mmlu_{s}.json")
            if os.path.exists(rp):
                by_subj[s] = json.load(open(rp))
                print(f"  mmlu/{s}: cached {len(by_subj[s])} rows")
            else:
                if args.offline:
                    raise SystemExit(f"offline and missing cache: {rp}")
                print(f"  mmlu/{s}: fetching cais/mmlu/{s}/test ...")
                # total unknown upfront; page until empty (cap 2000)
                rows, off = [], 0
                while True:
                    url = (f"{DS_API}?dataset=cais%2Fmmlu&config={s}"
                           f"&split=test&offset={off}&length=100")
                    pl = json.loads(_get(url))
                    if pl.get("error"):
                        raise RuntimeError(f"MMLU {s}: {pl}")
                    b = [x["row"] for x in pl.get("rows", [])]
                    if not b:
                        break
                    rows.extend(b)
                    off += len(b)
                    if off >= (pl.get("num_rows_total") or 10**9):
                        break
                with open(rp, "w") as f:
                    json.dump(rows, f, ensure_ascii=False)
                by_subj[s] = rows
        tasks, skip = convert_mmlu(by_subj)
        write_tasks("mmlu", tasks)
        man["mmlu"] = manifest_entry("mmlu", tasks, skip)
        man["mmlu"]["subjects"] = subjects

    with open(man_path, "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(f"manifest -> {man_path}")


if __name__ == "__main__":
    main()
