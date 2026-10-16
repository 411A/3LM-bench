#!/usr/bin/env python3
"""
Controlled throughput measurement, separate from the accuracy suite.

Why separate: gen_tps inside the accuracy run is confounded. Token speed varies
with context length (attention over a 4096-token KV cache is slower than over
300), with batch contents, and with whatever else the box was doing. A speed
number taken from a mixed workload is not comparable across models.

This fixes the conditions - identical prompt, identical target token count,
identical context length - and repeats, so the reported value is a median of
like-for-like measurements rather than an average over heterogeneous work.

Two methods, because the two runtimes measure differently:
  * llama.cpp models -> llama-bench, which reports prompt-processing and token-
    generation rates separately from the same run. That is the cleanest number
    available and it needs no server.
  * Ollama models -> N fixed-length completions through /api/generate, using the
    server's own eval_count and eval_duration, plus a wall-clock cross-check.

Reports median and IQR rather than mean, because decode throughput on a shared
2-core box has a long right tail when the scheduler preempts it.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import Backend  # noqa: E402

LLAMA_BENCH = os.path.expanduser("~/llama/llama-b11558/llama-bench")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "speed.json")

# Identical prompt for every model. Long enough to force a realistic context,
# short enough not to dominate the run.
PROMPT = ("Below is a Python module. Explain, in detail, what each function does, "
          "what invariants it maintains, and what would break if it were called "
          "concurrently.\n\n"
          "def compute_totals(records, rate_table, currency):\n"
          "    totals = {}\n"
          "    for rec in records:\n"
          "        key = (rec['customer'], rec['period'])\n"
          "        amount = rec['amount'] * rate_table[rec['currency']]\n"
          "        totals[key] = totals.get(key, 0.0) + amount\n"
          "    return {k: round(v, 2) for k, v in totals.items()}\n\n"
          "def reconcile(ledger, computed):\n"
          "    diffs = []\n"
          "    for key, amount in computed.items():\n"
          "        actual = ledger.get(key, 0.0)\n"
          "        if abs(actual - amount) > 0.01:\n"
          "            diffs.append((key, actual, amount))\n"
          "    return diffs\n\n")


def stats(xs):
    if not xs:
        return {}
    xs = sorted(xs)
    q1 = st.quantiles(xs, n=4)[0] if len(xs) > 3 else xs[0]
    q3 = st.quantiles(xs, n=4)[2] if len(xs) > 3 else xs[-1]
    return {"n": len(xs), "median": round(st.median(xs), 3),
            "mean": round(st.mean(xs), 3), "min": round(min(xs), 3),
            "max": round(max(xs), 3), "q1": round(q1, 3), "q3": round(q3, 3),
            "stdev": round(st.stdev(xs), 3) if len(xs) > 1 else 0.0}


def bench_llamacpp(gguf, ctx, n_predict, threads, reps):
    cmd = [LLAMA_BENCH, "-m", gguf, "-p", "512", "-n", str(n_predict),
           "-c", str(ctx), "-t", str(threads), "-r", str(reps), "-q"]
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "llama-bench timeout"}
    out = (p.stdout or "").strip().splitlines()
    if not out:
        return {"available": False, "error": (p.stderr or "")[-300:]}
    hdr = out[0].split("|")
    row = out[-1].split("|")
    d = {h.strip(): v.strip() for h, v in zip(hdr, row) if h.strip()}
    pp = tg = None
    for k, v in d.items():
        if "pp512" in k:
            pp = float(v)
        if f"tg{n_predict}" in k or k.startswith("tg"):
            tg = float(v)
    return {"available": True, "model": d.get("model"), "test": d.get("test"),
            "threads": d.get("threads"), "pp512_tps": pp, "tg_tps": tg,
            "raw_header": hdr, "raw_row": row}


def bench_ollama(b, n_predict, reps):
    """Fixed-length generation, repeated. Server-side counters are primary;
    wall clock is the cross-check that catches a server lying about its own rate."""
    gen_tps, wall_tps, prompt_tps, ttfts = [], [], [], []
    for i in range(reps):
        req = urllib.request.Request(
            f"http://{b.host}:{b.port}/api/generate",
            data=json.dumps({"model": b.name, "prompt": PROMPT, "stream": False,
                             "think": False,
                             "options": {"num_ctx": b.num_ctx, "num_predict": n_predict,
                                         "temperature": 0.7, "num_gpu_layers": 0}}).encode(),
            method="POST")
        req.add_header("Content-Type", "application/json")
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        wall = time.monotonic() - t0
        ec = d.get("eval_count") or 0
        pc = d.get("prompt_eval_count") or 0
        ed = (d.get("eval_duration") or 0) / 1e9
        pd = (d.get("prompt_eval_duration") or 0) / 1e9
        if ed and ec:
            gen_tps.append(ec / ed)
        if wall and ec:
            wall_tps.append(ec / wall)
        if pd and pc:
            prompt_tps.append(pc / pd)
        ttfts.append(pd)
        print(f"    rep {i+1}/{reps}: {ec} tok in {wall:.1f}s wall "
              f"-> {ec/wall:.2f} tok/s (server {ec/ed if ed else 0:.2f})", flush=True)
    return {"available": True, "gen_tps": stats(gen_tps),
            "wall_tps": stats(wall_tps), "prompt_tps": stats(prompt_tps),
            "ttft_s": stats(ttfts), "n_predict": n_predict,
            "prompt_chars": len(PROMPT)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "llamacpp"], default="ollama")
    ap.add_argument("--model", default="nemotron-3-nano:4b")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    port = args.port or (11434 if args.backend == "ollama" else 8090)
    label = args.model.replace("/", "_").replace(":", "_")

    def meminfo():
        return {k: int(v.split()[0]) // 1024
                for k, _, v in (l.partition(":") for l in open("/proc/meminfo"))}

    print(f"=== speed bench: {args.model} via {args.backend}:{port} ===")
    mi = meminfo()
    print(f"mem available: {mi['MemAvailable']} MiB | swap free: {mi['SwapFree']} MiB")

    if args.backend == "llamacpp":
        res = bench_llamacpp(args.gguf, args.ctx, args.n_predict, args.threads, args.reps)
        if res.get("available"):
            print(f"  llama-bench pp512: {res['pp512_tps']} tok/s")
            print(f"  llama-bench tg{n_predict}: {res['tg_tps']} tok/s")
    else:
        b = Backend("ollama", args.model, port=port, num_ctx=args.ctx)
        if not b.alive():
            raise SystemExit(f"no ollama on {port}")
        res = bench_ollama(b, args.n_predict, args.reps)
        g = res.get("gen_tps", {})
        print(f"  gen  tok/s: median {g.get('median')}  IQR [{g.get('q1')}, {g.get('q3')}]")
        print(f"  prompt tps: {res.get('prompt_tps', {}).get('median')}")
        print(f"  ttft  ms  : {res.get('ttft_s', {}).get('median')}")

    mi = meminfo()
    payload = {"model": args.model, "backend": args.backend, "port": port,
               "ctx": args.ctx, "n_predict": args.n_predict, "reps": args.reps,
               "threads": args.threads, "label": label, "result": res,
               "mem_available_mib": mi["MemAvailable"],
               "swap_free_mib": mi["SwapFree"]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    allr = json.load(open(OUT)) if os.path.exists(OUT) else {}
    allr[label] = payload
    with open(OUT, "w") as f:
        json.dump(allr, f, indent=1)
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()