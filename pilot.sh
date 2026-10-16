#!/usr/bin/env bash
# pilot.sh — run one model across every suite, small bounded subsets, observe live.
#
# This is a DIAGNOSTIC sweep, not a benchmark result: every group uses --limit,
# so subsets are front-biased and must never be quoted as scores. Its job is to
# prove each runner/scorer/aggregation path works on a live model and to surface
# harness defects (which are logged, never smoothed over).
#
# Writes only to /tmp/diag/*.json — your results/ table is never touched.
# One model at a time: refuses to start if another run.py holds the machine.
#
# Usage:
#   bash pilot.sh ollama nemotron-3-nano:4b
#   bash pilot.sh llamacpp k2-horizon --port 8092 --api-key local-llama
set -uo pipefail
cd "$(dirname "$0")"

BACKEND="${1:-ollama}"
MODEL="${2:-nemotron-3-nano:4b}"
shift 2 2>/dev/null || true
# any extra args (e.g. --port 8092 --api-key local-llama) pass through to run.py
PORT=""; APIKEY=""
while [ "$#" -gt 0 ]; do
  case "$1" in --port) PORT="$2"; shift 2;; --api-key) APIKEY="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;; esac
done
[ -n "$PORT" ] || { [ "$BACKEND" = ollama ] && PORT=11434 || PORT=8090; }

OUT=/tmp/diag
mkdir -p "$OUT"

if pgrep -f "python3 -u run\.py" >/dev/null 2>&1; then
  if pgrep -f "python3 -u run\.py" | while read -r p; do ps -p "$p" -o args= 2>/dev/null; done | grep -q "run.py"; then
    echo "another run.py is active; refusing to start a second (see status.sh)"
    exit 1
  fi
fi

EXTRA=()
[ -n "$APIKEY" ] && EXTRA+=(--api-key "$APIKEY")

group() { # name tasks limit reps
  local name="$1" tasks="$2" limit="$3" reps="$4"
  local out="$OUT/${BACKEND}-$name.json"
  if [ -s "$out" ]; then
    echo "--- $name: exists ($out), skipping (delete it to re-run) ---"
    return 0
  fi
  echo "=== [$name] tasks=$tasks limit=$limit reps=$reps ==="
  # shellcheck disable=SC2086
  timeout 900 python3 -u run.py --backend "$BACKEND" --model "$MODEL" \
    --port "$PORT" "${EXTRA[@]}" \
    --tasks "$tasks" --limit "$limit" --reps "$reps" \
    --label "diag-$name" --out "$out" 2>&1 \
    | grep -E "^(---|  rep|  \[|.*done in|.*ERROR|.*!!|.*PASS|.*OK |.*X  )" \
    | tail -30
}

group custom-mc  "science,electronics,persian,english"                           5 1
group std-mc1     "std_arc_challenge,std_arc_easy,std_winogrande"                5 1
group std-mc2     "std_truthfulqa_mc,std_belebele_fa,std_mmlu"                   5 1
group std-gen     "std_hellaswag,std_gsm8k"                                     3 1
group code        "code_humaneval,code_mbpp"                                    2 1
group gen1        "bugfix,algo,refactor,code_rust"                              2 1
group gen2        "embedded_cpp,devops,lint,netdiag"                            2 1
group agentic     "agentic"                                                     5 1

echo
echo "==================== DIAGNOSTIC SUMMARY (not scores) ===================="
python3 - "$OUT" <<'PY'
import glob, json, sys
out = sys.argv[1]
tot, bad = 0, 0
for f in sorted(glob.glob(out + "/*.json")):
    d = json.load(open(f))
    recs = d.get("records", [])
    tot += len(recs)
    err = [r for r in recs
           if "ERROR" in str(r.get("id", "")) or r.get("stopped") in ("error", "exception")]
    bad += len(err)
    cats = sorted({r.get("category", "?") for r in recs})
    print(f"  {f.split('/')[-1]:<16} {len(recs):>3} records  cats={','.join(cats)}"
          + (f"  !! {len(err)} ERROR STUBS" if err else ""))
print(f"  TOTAL {tot} records, {bad} error stubs across {len(glob.glob(out + '/*.json'))} groups")
print("  Subsets are --limit front-biased: observe the pipeline, do not quote numbers.")
PY
