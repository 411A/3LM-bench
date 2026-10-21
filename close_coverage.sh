#!/usr/bin/env bash
# close_coverage.sh — fill the instrument-coverage gaps, nothing else.
#
# The coverage matrix (reports/screening.html) shows which model was measured
# with which instrument. After the first screening pass only three of six
# models had a screening run, and only three had the CS-100 decider and the
# 19-chain agentic deep-dive. This driver closes exactly those gaps.
#
# Rules this driver obeys:
#   * A gap is an INCOMPLETE file, not a missing one: a run that died halfway
#     leaves a file that exists but is partial - that is still a gap.
#   * Nothing already measured is re-measured. The check is per (model,
#     instrument) and it reads the file's own coverage_gaps/partial flags.
#   * Instrument order is by VISIBILITY: the screening table (the headline
#     of the report) is closed for every model first, then the CS-100
#     decider, then the 19-chain agentic deep-dive. Stop the queue at any
#     point and the report stays complete and honest.
#   * One model resident at a time, always.
#
#   bash close_coverage.sh              # every gap that is still open
#   bash close_coverage.sh k2-horizon   # one model
#   bash close_coverage.sh --list       # show the gap matrix and exit
set -uo pipefail
cd "$(dirname "$0")"
LOG_DIR="logs/coverage"
mkdir -p "$LOG_DIR"

ALL_MODELS="minicpm5-2b k2-horizon medgemma-4b qwen35-2b granite4-3b nemotron-3-nano"
WANT="${*:-}"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---- the gap test: does this (model, instrument) still need work? ----
complete() { # $1 = results file -> 0 if the file is a finished, gap-free run
  [ -s "$1" ] || return 1
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("partial"):
    sys.exit(1)
if d.get("coverage_gaps"):
    sys.exit(1)
sys.exit(0)
PY
}

if [ "${1:-}" = "--list" ]; then
  printf '%-18s %-12s %-10s %-12s\n' model screening cs-100 agentic-19
  for m in $ALL_MODELS; do
    printf '%-18s ' "$m"
    for spec in "screening:results/llamacpp/$m.json" \
                "cs:results/llamacpp/$m-cs.json" \
                "agentic:results/llamacpp/$m-agentic.json"; do
      name="${spec%%:*}"; file="${spec#*:}"
      if complete "$file"; then printf '%-12s ' "HAVE"; else printf '%-12s ' "GAP"; fi
    done
    printf '\n'
  done
  exit 0
fi

[ -n "$WANT" ] || WANT="$ALL_MODELS"

run_instrument() { # $1 = instrument key, $2 = model
  key="$1"; m="$2"
  case "$key" in
    screening)
      file="results/llamacpp/$m.json"; prof="screening"
      out="results/llamacpp/{label}.json" ;;
    cs)
      file="results/llamacpp/$m-cs.json"; prof="tasks=std_mmlu_cs"
      out="results/llamacpp/{label}-cs.json" ;;
    agentic)
      file="results/llamacpp/$m-agentic.json"; prof="tasks=agentic"
      out="results/llamacpp/{label}-agentic.json" ;;
  esac
  if complete "$file"; then
    say "$key: $m already complete — skipping"
    return 0
  fi
  extra=""
  [ -s "$file" ] && extra="--resume"
  say "$key: $m -> $file ${extra:+(resuming)}"
  # One model resident at a time: if another runner still holds the slot, WAIT
  # for it instead of dying and marking a gap that never existed.
  waited=0
  while pgrep -f "^python3 -u run\.py" >/dev/null 2>&1; do
    if [ "$waited" = 0 ]; then
      say "$key: $m waiting for the benchmark slot to free up"
    fi
    waited=$((waited + 1))
    if [ "$waited" -gt 720 ]; then
      say "$key: $m gave up waiting for the slot"
      return 1
    fi
    sleep 5
  done
  # Run a SNAPSHOT of the launcher, never the file in the tree: bash reads
  # scripts incrementally, so editing run_llama.sh while it runs shifts the
  # read offset and the running job dies with a syntax error (it did, at
  # 05:17 today, right after a completed run). A per-job copy makes that
  # class of accident impossible.
  local runner
  runner="$(mktemp "${TMPDIR:-/tmp}/llmbench-runner-XXXXXX.sh")"
  cp run_llama.sh "$runner"
  # shellcheck disable=SC2086
  bash "$runner" "$m" "$prof" --out "$out" $extra --until-complete 6 \
    > "$LOG_DIR/$m-$key.log" 2>&1
  rc=$?
  rm -f "$runner"
  if [ "$rc" = "75" ]; then
    say "$key: $m PAUSED by the power policy (battery too low)"
    say "  stopping the queue: plug the charger in, then re-run close_coverage.sh"
    exit 0
  fi
  [ "$rc" != 0 ] && say "$key: $m did not finish cleanly (see $LOG_DIR/$m-$key.log)"
  # Re-aggregate with the current definitions, so the corpus never mixes
  # statistical conventions across files.
  python3 rescore.py --files "$file" >/dev/null 2>&1 \
    || say "rescore of $file failed (non-fatal)"
  if complete "$file"; then
    say "$key: $m COMPLETE"
    # Snapshot the evidence: a power cut cannot lose it (the checkpoint
    # protocol), but a dead disk can.
    bash snapshot.sh >/dev/null 2>&1 || say "snapshot failed (non-fatal)"
  else
    say "$key: $m still has gaps — left marked as incomplete (never quoted)"
  fi
}

# ---- instrument first (visibility order), then models ----
for key in screening cs agentic; do
  say "===== instrument: $key ====="
  for m in $WANT; do
    run_instrument "$key" "$m"
  done
done
say "COVERAGE CLOSURE FINISHED"
