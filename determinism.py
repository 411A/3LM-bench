#!/usr/bin/env python3
"""
Determinism verification.

Reproducibility has two independent sources, and conflating them is the mistake to
avoid:

  1. SAMPLING nondeterminism - does the backend honour the seed? If it ignores
     `seed`, every call is a fresh draw and no two runs can ever agree.
     Measured here by hashing N identical requests and counting unique hashes.

  2. NUMERIC nondeterminism - does the same seed produce the same tokens on the
     same hardware? Thread scheduling can change float reduction order, so
     bit-identical output is NOT guaranteed even with a seed. llama.cpp and
     llama-ollama are both CPU-bound here with -t 2, so this is a real risk and
     is measured rather than assumed.

Regex is fully deterministic (same input -> same output, always) and so are
ruff / rustc / g++ / clippy on a fixed input with a fixed toolchain version. The
completion hash is recorded per task so reproducibility of the WHOLE pipeline can
be re-checked offline from the results file alone.

Usage:
  python3 determinism.py --backend ollama --model nemotron-3-nano:4b
  python3 determinism.py --backend llamacpp --gguf ~/ollama-dl/minicpm5-2b.gguf --port 8091
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import Backend  # noqa: E402

PROMPTS = [
    ("short", "Reply with exactly: OK"),
    ("code", "Write a Python function `add(a, b)` returning a + b. "
             "Output only a ```python block."),
    ("long", "Explain, in detail, what a thread-safe LRU cache must guarantee "
             "and what breaks if it does not. Be concrete."),
]


def h(s):
    return hashlib.sha256((s or "").encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=4096)
    args = ap.parse_args()
    port = args.port or (11434 if args.backend == "ollama" else 8090)

    b = Backend(args.backend, args.model, port=port, api_key=args.api_key,
                num_ctx=args.ctx)
    if not b.alive():
        raise SystemExit(f"no server on {b.host}:{port}")

    print(f"=== determinism: {args.model} via {args.backend}:{port} "
          f"({args.reps} identical requests each) ===")
    print(f"{'case':<10} {'temp':>5} {'seed':>6}  {'unique':>7}  {'length':>7}  verdict")
    print("-" * 68)

    all_det = True
    for label, text in PROMPTS:
        msgs = [{"role": "user", "content": text}]
        for temp in (0.7, 0.0):
            hashes, lens = [], []
            for i in range(args.reps):
                r = b.generate(msgs, max_tokens=200, temperature=temp,
                               seed=20261012)
                hashes.append(h(r["content"]))
                lens.append(len(r["content"] or ""))
            uniq = len(set(hashes))
            same_len = len(set(lens)) == 1
            det = (uniq == 1 and same_len)
            all_det &= det
            print(f"{label:<10} {temp:>5} {20261012:>6}  {uniq:>4}/{args.reps} "
                  f"{lens[0]:>7}  "
                  f"{'DETERMINISTIC' if det else 'NON-DETERMINISTIC'}")

    # A different seed must give a DIFFERENT answer at t>0, otherwise the seed is
    # being ignored and the run is accidentally reproducible (all models would
    # then be identical, which is its own catastrophic failure mode).
    print()
    print("=== seed actually reaches the sampler? ===")
    msgs = [{"role": "user", "content": PROMPTS[2][1]}]
    hs = []
    for sd in (20261012, 999888777, 55555):
        r = b.generate(msgs, max_tokens=200, temperature=0.7, seed=sd)
        hs.append(h(r["content"]))
    uniq = len(set(hs))
    print(f"  3 distinct seeds at temp=0.7 -> {uniq} unique outputs "
          f"({'seed honoured' if uniq == 3 else 'SEED IGNORED OR BUFFERED'})")
    if uniq != 3:
        all_det = False

    # MC path: the whole point of the logprob scorer is that argmax over a fixed
    # distribution has no sampling stage at all.
    print()
    print("=== MC path (argmax over top-k, no sampling stage) ===")
    mc = []
    for i in range(3):
        idx, letter, dist, meta, mode, mass = b.mc_logprobs(
            "Which is a prime number?", ["4", "6", "7", "9"])
        mc.append((idx, letter, tuple(sorted((k, round(v, 9)) for k, v in dist.items()))))
    print(f"  3 runs -> {len(set(mc))} unique verdicts "
          f"({'DETERMINISTIC' if len(set(mc)) == 1 else 'NON-DETERMINISTIC'})")
    if len(set(mc)) != 1:
        all_det = False

    print()
    print("VERDICT:", "fully deterministic under a fixed seed" if all_det
          else "NOT reproducible - see above")
    return 0 if all_det else 1


if __name__ == "__main__":
    sys.exit(main())