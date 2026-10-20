#!/usr/bin/env bash
# gates.sh — pre-production gate suite. Run BEFORE any real benchmark queue.
#
#   bash gates.sh              # all gates (e2e auto-skips without live servers)
#   bash gates.sh unit policy  # only these gates
#
# Gates: compiler | lint | policy | plan | unit | integration | audit | e2e | perf
# Any gate failure exits non-zero. Nothing here runs a model except e2e,
# which issues exactly 2 single-token probes and skips cleanly offline.
set -uo pipefail
cd "$(dirname "$0")"

GATES="${*:-compiler lint policy plan unit integration audit e2e perf}"
FAIL=()
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
pass() { say "GATE PASS: $1"; }
fail() { say "GATE FAIL: $1"; FAIL+=("$1"); }
want() { case " $GATES " in *" $1 "*) return 0;; *) return 1;; esac; }

if want compiler; then
  say "--- compiler: every .py must compile, every .sh must parse ---"
  if python3 -m compileall -q . 2>&1 | head -5 \
     && for f in *.sh; do bash -n "$f" || exit 1; done; then pass compiler; else fail compiler; fi
fi

if want lint; then
  say "--- lint: ruff F/E9 on our own code (real errors only, not style) ---"
  RBIN="$(command -v ruff || echo "$HOME/.local/bin/ruff")"
  if [ -x "$RBIN" ] && "$RBIN" check --no-cache --isolated \
       --select F,E9 --exclude __pycache__ . 2>&1 | tail -3; then pass lint; else fail lint; fi
fi

if want policy; then
  say "--- policy: repo hygiene rules ---"
  ok=1
  # no absolute private paths in tracked code (the gate excludes itself:
  # it necessarily contains the patterns it searches for)
  if git ls-files '*.py' '*.sh' | grep -v "^gates.sh$" | xargs grep -n "/home/\|/tmp/opencode" 2>/dev/null | grep -v "^Binary"; then
    say "absolute paths found above"; ok=0
  fi
  # no secrets in tracked code
  if git ls-files '*.py' '*.sh' '*.json' | xargs grep -inE "sk-[a-z0-9]{8}|bearer [a-z0-9._-]{10,}|password\s*=\s*['\"][^'\"]" 2>/dev/null | grep -viE "add_header|args\.|getenv|environ|argparse|password_reset"; then
    say "possible secret above"; ok=0
  fi
  # upstream data must not be git-tracked
  if git ls-files | grep -E "^(results|logs|data/raw)/|tasks/std_.*\.json$"; then
    say "generated/data files tracked above"; ok=0
  fi
  # every manifest suite carries a license
  python3 -c "
import json,sys
m=json.load(open('data/manifest.json')); c=json.load(open('benchmarks/catalog.json'))
bad=[k for k,v in m.items() if not v.get('license') and v.get('kind')!='screen30']
sys.exit(1 if bad else 0)" 2>/dev/null || { say "manifest license gap"; ok=0; }
  [ "$ok" = 1 ] && pass policy || fail policy
fi

if want plan; then
  say "--- plan: every command a queue needs must resolve dry ---"
  ok=1
  python3 standard_fetch.py --check >/dev/null 2>&1 || { say "standard cache incomplete"; ok=0; }
  python3 standard_strata.py --check >/dev/null 2>&1 || { say "s30 cache incomplete"; ok=0; }
  python3 run.py --help >/dev/null 2>&1 || { say "run.py --help broken"; ok=0; }
  bash run_llama.sh --list >/dev/null 2>&1 || { say "model discovery broken"; ok=0; }
  bash run_llama.sh qwen35-2b qualifier --dry-run >/dev/null 2>&1 || { say "qualifier dry-run broken"; ok=0; }
  bash run_llama.sh qwen35-2b screening --dry-run >/dev/null 2>&1 || { say "screening dry-run broken"; ok=0; }
  [ "$ok" = 1 ] && pass plan || fail plan
fi

if want unit; then
  say "--- unit: scorer/stat contracts ---"
  if python3 tests/test_units.py 2>&1 | tail -2; then pass unit; else fail unit; fi
fi

if want integration; then
  say "--- integration: stub backend through all 30 runners ---"
  if python3 validate.py 2>&1 | tail -1 | grep -q "VALIDATE OK"; then pass integration; else fail integration; fi
fi

if want audit; then
  say "--- audit: every published number must reproduce from its raw records ---"
  if python3 audit.py 2>&1 | tail -2; then pass audit; else fail audit; fi
fi

if want e2e; then
  say "--- e2e: 1 MC probe per reachable backend (skips cleanly if none) ---"
  python3 - <<'PY'
import sys, urllib.request, json
sys.path.insert(0, ".")
found = []
try:
    urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5)
    found.append("ollama")
except Exception:
    pass
for port in (8091, 8092, 8093, 8094, 8095, 8096, 8097):
    try:
        code = urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                      timeout=3).status
        if code == 200:
            found.append(f"llamacpp:{port}")
    except Exception:
        pass
print("reachable:", found or "none")
if not found:
    print("E2E SKIP: no live servers")
    raise SystemExit(0)
from backend import Backend
if "ollama" in found:
    tags = json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10))
    have = [m["name"] for m in tags.get("models", [])]
    # Pin a known-good general model: the registry's first entry is
    # functiongemma, which refuses general prompts (see run_all.sh) and
    # answers MC with ~zero letter mass. Probing it proves nothing.
    name = next((m for m in ("nemotron-3-nano:4b", "qwen3.5:4b-q4_K_M",
                             "granite4-3b:latest", "medgemma-4b:latest")
                 if m in have), None)
    if name is None:
        print("E2E SKIP: no known-good ollama model present")
        raise SystemExit(0)
    b = Backend("ollama", name, port=11434, timeout=120)
    # Production prompt shape (SYSTEM_MC + labeled options), not a degenerate
    # probe: a bare "2+2?"/system="x" once made a good model answer A, which
    # is model behavior on a bad prompt, not a harness defect.
    r = b.mc_logprobs("What is 2+2?", ["3", "4", "5"],
                      system="Answer with a single letter and nothing else.")
    idx, letter = r[0], r[1]
    mass = r[5] if len(r) > 5 else max(r[2].values())
    assert idx == 1 and letter == "B" and mass > 0.5, \
        f"ollama probe picked {letter} mass={mass}"
    print(f"E2E OK ollama/{name}: picked B, mass={mass:.3f}")
print("E2E DONE")
PY
  [ $? -eq 0 ] && pass e2e || fail e2e
fi

if want perf; then
  say "--- perf: scoring throughput must not regress ---"
  python3 - <<'PY'
import sys, time
sys.path.insert(0, ".")
import run as R
recs = [{"id": f"i{k}", "rep": 0, "category": "code", "task": "code_humaneval",
         "passed": k % 2 == 0, "tokens": {"eval": 1}, "speed": {"gen_tps": 4.0},
         "lint": {"n_diagnostics": 0, "noqa_gaming": False}} for k in range(2000)]
t0 = time.monotonic()
R.aggregate(recs)
dt = time.monotonic() - t0
print(f"aggregate 2000 records: {dt:.2f}s")
sys.exit(0 if dt < 10 else 1)
PY
  [ $? -eq 0 ] && pass perf || fail perf
fi

echo
if [ "${#FAIL[@]}" -gt 0 ]; then say "GATES FAILED: ${FAIL[*]}"; exit 1; fi
say "ALL GATES PASS — clear for production."
