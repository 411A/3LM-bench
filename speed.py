#!/usr/bin/env python3
"""speed.py — the dedicated throughput instrument.

Why this exists: a throughput number taken as a by-product of a long,
possibly-chopped benchmark run is not a throughput measurement. The model
may have been sharing the box, running against a memory-fragmented heap, or
measured on the very first token of a cold server. This script measures
speed the way a speed test should be measured:

  * ONE model resident, nothing else running (the caller runs this with the
    benchmark slot free; the preflight in run_llama.sh enforces the same
    rule for benchmark runs).
  * A fresh server, warmed up with a discarded prompt, then N measured
    prompts of identical shape.
  * The MEDIAN of the per-item decode rate is the headline (robust to the
    odd slow item); min/max/p25/p75 are kept so the spread is visible.
  * Everything is written to results/llamacpp/<label>-speed.json with the
    server build, the prompt hash and the exact conditions.

Usage:
  python3 speed.py <model> [<model> ...]      # one model at a time
  python3 speed.py --all                      # every model, sequentially
  python3 speed.py --serve-only <model>       # start the server and stop
"""
import json
import os
import statistics as st
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import Backend, BackendError  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(os.path.expanduser("~"), "ollama-dl")
RESULTS = os.path.join(HERE, "results", "llamacpp")
APIKEY = "local-llama"
CTX = 4096
N_PROMPTS = 12
WARMUP = 2
MAX_TOKENS = 128

# Same table the queue uses (model|file|port).
KNOWN = [
    ("nemotron-3-nano", "nemotron-3-nano.gguf", 8097),
    ("minicpm5-2b", "minicpm5-2b.gguf", 8091),
    ("k2-horizon", "k2-horizon.gguf", 8092),
    ("qwen35-2b", "qwen2b.gguf", 8094),
    ("granite4-3b", "granite.gguf", 8095),
    ("medgemma-4b", "medgemma.gguf", 8096),
]

# One fixed prompt, hashed into the output so the measurement is auditable.
PROMPT = ("Explain, in one short paragraph, why a benchmark harness must "
          "record the exact decoder settings it used. Keep it under 120 words.")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def server_up(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(label, path, port):
    if server_up(port):
        say(f"server already up on :{port} — reusing it")
        return False
    log = os.path.join(HERE, "logs", f"speed-{label}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    launcher = os.path.join(os.path.expanduser("~"), "ollama-work", "run-llama.sh")
    with open(log, "ab") as fh:
        subprocess.Popen(["bash", launcher, path, str(port), str(CTX)],
                         stdout=fh, stderr=fh, start_new_session=True)
    for _ in range(120):
        time.sleep(5)
        if server_up(port):
            say(f"server up on :{port}")
            return True
    raise SystemExit(f"server never became healthy on :{port} (see {log})")


def stop_server(port, started):
    if not started:
        return
    subprocess.run(["pkill", "-f", f"llama-server.*{port}"],
                   capture_output=True)
    time.sleep(3)


def cpu_busy_pct():
    """% of the whole box's CPU used by everything (sampled live over 1s)."""
    def sample():
        f = open("/proc/stat").readline().split()
        vals = [int(x) for x in f[1:9]]
        return sum(vals), vals[3] + vals[4]     # total, idle+iowait
    t1, i1 = sample()
    time.sleep(1)
    t2, i2 = sample()
    if t2 <= t1:
        return 0.0
    return 100.0 * (t2 - t1 - (i2 - i1)) / (t2 - t1)


def wait_quiet(max_tries=60, budget_pct=30.0):
    """The measurement is only valid on an idle box: 2 cores, one model.

    The standing desktop stack (Xorg + terminal + this agent) costs about a
    quarter of the box; the runaway thumbnailer that used to sit here cost
    half of it. The number actually observed is recorded with the result.
    """
    for i in range(max_tries):
        pct = cpu_busy_pct()
        if pct <= budget_pct:
            say(f"box is quiet enough (foreign CPU {pct:.1f}%)")
            return pct
        say(f"waiting for a quiet box (foreign CPU {pct:.1f}%, budget {budget_pct:.0f}%)")
        time.sleep(5)
    say("WARNING: box still busy; measuring anyway and recording the load")
    return cpu_busy_pct()


def measure(label, path, port):
    foreign_pct = wait_quiet()
    started = start_server(label, path, port)
    try:
        b = Backend("llamacpp", label, port=port, api_key=APIKEY, timeout=600)
        for i in range(WARMUP):
            b.generate([{"role": "user", "content": PROMPT}],
                       max_tokens=MAX_TOKENS, temperature=0.7)
        tps, wall = [], []
        for i in range(N_PROMPTS):
            r = b.generate([{"role": "user", "content": PROMPT}],
                           max_tokens=MAX_TOKENS, temperature=0.7)
            tps.append(r["gen_tps"])
            wall.append(r["wall_s"])
            say(f"  [{i+1}/{N_PROMPTS}] {r['gen_tps']} tok/s  wall {r['wall_s']}s")
        summary = {
            "model": label,
            "n_prompts": N_PROMPTS,
            "warmup_prompts": WARMUP,
            "max_tokens": MAX_TOKENS,
            "ctx": CTX,
            "prompt_sha_prefix": __import__("hashlib").sha256(
                PROMPT.encode()).hexdigest()[:16],
            "tps_median": round(st.median(tps), 3),
            "tps_mean": round(sum(tps) / len(tps), 3),
            "tps_min": round(min(tps), 3),
            "tps_max": round(max(tps), 3),
            "tps_p25": round(sorted(tps)[len(tps) // 4], 3),
            "tps_p75": round(sorted(tps)[(3 * len(tps)) // 4], 3),
            "wall_s_median": round(st.median(wall), 2),
            "per_prompt_tps": [round(t, 3) for t in tps],
            "measured_unix": time.time(),
            "load_avg_at_start": open("/proc/loadavg").read().split()[0],
            "foreign_cpu_pct_at_start": round(foreign_pct, 1),
            "server_version": b.server_versions().get("llama_server")
                              or "see logs/speed-%s.log" % label,
        }
        out = os.path.join(RESULTS, f"{label}-speed.json")
        os.makedirs(RESULTS, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(summary, fh, indent=1)
        say(f"{label}: median {summary['tps_median']} tok/s "
            f"(p25 {summary['tps_p25']} – p75 {summary['tps_p75']}, "
            f"min {summary['tps_min']} – max {summary['tps_max']}) -> {out}")
        return summary
    finally:
        stop_server(port, started)


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "--all":
        args = [m for m, _, _ in KNOWN]
    table = {m: (f, p) for m, f, p in KNOWN}
    for label in args:
        if label not in table:
            raise SystemExit(f"unknown model {label}")
        fname, port = table[label]
        path = os.path.join(DL, fname)
        if not os.path.exists(path):
            say(f"{label}: GGUF missing ({path}) — skipped")
            continue
        say(f"=== {label} : isolated throughput ===")
        try:
            measure(label, path, port)
        except (BackendError, SystemExit) as e:
            say(f"{label}: FAILED — {e}")


if __name__ == "__main__":
    main()
