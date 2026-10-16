#!/usr/bin/env bash
# Run llmbench across models, strictly ONE AT A TIME.
#
# Why sequential and not a loop over a job list: on 2 cores, two models resident
# at once split the same pipe. Earlier work in this project measured parallel
# downloads starving each other (6 KiB/s vs 447 KB/s); the same effect applies to
# two inference servers, and it would corrupt every throughput number in the
# table, which is one of the things this suite exists to measure.
#
# Memory is the other constraint. A 4B q4 model needs ~3 GB resident and this
# box has 7.6 GB total with ~850 MiB free while anything else is up. So between
# models we unload ollama's model and let the page cache drop, and we refuse to
# start the next model if free memory is under the floor.
#
# Usage:
#   bash run_all.sh <label>   # run ONE model, then stop for review (the default workflow)
#   bash run_all.sh --all     # the whole queue, one model at a time, no parallelism
#   RESUME=1 bash run_all.sh <label|--all>  # skip models that already have results
#
# The default is deliberately one-by-one, not the full queue. Every recent
# hardening round in this project found defects by READING a finished model's
# results before the next model started (a scorer regression, a gaming exploit,
# a metric pinned at 0). A fire-and-forget 40h queue would bake any such defect
# into every row instead of one. Run one model, read its log + results file,
# then launch the next.

set -uo pipefail

cd "$(dirname "$0")"
mkdir -p results logs

MIN_FREE_MIB=900          # below this, ollama will thrash instead of serving

# Two tiers. Tier A is the measured winners and gets the full paired design with
# repetitions, because those are the models whose comparison actually has to hold
# up. Tier B still runs every category but with fewer code items and one rep, so
# the table has a row for them without costing another day.
#
# Standard suites run as stratified s30 instances (30 items each, guaranteed
# subcategory coverage via standard_strata.py) — no caps needed. GSM8K-s30 is
# generation (~45s/item at 3-8 tok/s): 30 items ≈ 23 min. Full official splits
# (HellaSwag 10k etc.) never run in a queue: Tier-3 overnight opt-in only.
#
# Tier A: HumanEval 40 + MBPP 50 + s30, 2 reps, agentic 3
# Tier B: HumanEval 20 + MBPP 20 + s30, 1 rep, agentic 2
TIER_A_MODELS="nemotron-3-nano minicpm5-2b"
TIER_A_CAPS="code_humaneval=40,code_mbpp=50,agentic=8"
TIER_A_REPS=2
TIER_A_AGENTIC=3
TIER_B_CAPS="code_humaneval=20,code_mbpp=20,agentic=8"
TIER_A_REPS=2
TIER_A_AGENTIC=3
TIER_B_CAPS="code_humaneval=20,code_mbpp=20,agentic=8"
TIER_B_REPS=1
TIER_B_AGENTIC=2
# Profile selectors for run.py --tasks:
PROFILE_CUSTOM="code_humaneval,code_mbpp,bugfix,algo,refactor,code_rust,embedded_cpp,science,electronics,persian,english,netdiag,devops,lint,agentic"
PROFILE_S30="std_arc_challenge_s30,std_arc_easy_s30,std_hellaswag_s30,std_winogrande_s30,std_truthfulqa_mc_s30,std_belebele_fa_s30,std_mmlu_s30,std_gsm8k_s30"
# Full official splits (Tier-3 overnight opt-in, NOT the default queue):
PROFILE_STANDARD="std_arc_challenge,std_arc_easy,std_hellaswag,std_winogrande,std_truthfulqa_mc,std_belebele_fa,std_mmlu,std_gsm8k"
PROFILE_ALL="all"
# Default queue: custom suites full + stratified s30. Full splits never run
# here (they would double-count alongside s30 and cost ~30h on HellaSwag).
PROFILE_DEFAULT="$PROFILE_CUSTOM,$PROFILE_S30"

# label | backend | model | port
# Pipe delimiter, not colon: ollama tags contain colons ("nemotron-3-nano:4b"),
# and splitting on colon silently shoves "4b:11434" into the port argument.
#
# qwen35-2b and qwen35-4b are the projector-free rebuilds made with
# `ollama create` from a bare GGUF. That command drops the renderer/parser
# metadata the official manifest carries, so these two have NO tool-calling
# support and will score ~0 on agentic. That is a packaging artefact, not a
# capability difference, and the report labels it as such - otherwise the table
# would claim qwen is bad at agents when the official qwen3.5:4b is not.
MODELS=(
  "nemotron-3-nano|ollama|nemotron-3-nano:4b|11434"
  "minicpm5-2b|llamacpp|minicpm5-2b|8091"
  "qwen3.5-4b|ollama|qwen3.5:4b-q4_K_M|11434"
  "k2-horizon|llamacpp|k2-horizon|8092"
  "granite4-3b|ollama|granite4-3b:latest|11434"
  "medgemma-4b|ollama|medgemma-4b:latest|11434"
  "qwen35-4b|ollama|qwen35-4b:latest|11434"
  "qwen35-2b|ollama|qwen35-2b:latest|11434"
)

# Models excluded on purpose, with reasons recorded so the omission is a decision
# rather than an oversight:
#   functiongemma:latest  - ollama reports "does not support tools" despite the
#                           registry advertising renderer=functiongemma, and it
#                           refuses general prompts. Unusable for this suite.

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Forensic trap: two runs have now died with no DONE/FAILED line and no kernel
# OOM trace (userspace oomd leaves no dmesg entry and the journal is unreadable
# here). Without this trap the next silent death is equally undebuggable.
trap 'rc=$?; say "driver EXIT rc=$rc (line ${LINENO:-?}); last log: $(tail -1 "logs/${label:-unknown}.log" 2>/dev/null)"' EXIT

free_mib() { awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo; }

wait_for_mem() {
  local tries=0
  while [ "$(free_mib)" -lt "$MIN_FREE_MIB" ]; do
    tries=$((tries+1))
    if [ "$tries" -gt 30 ]; then
      say "WARNING: only $(free_mib) MiB free after 60s, starting anyway"
      return 0
    fi
    say "waiting for memory: $(free_mib) MiB free (floor $MIN_FREE_MIB), attempt $tries"
    sleep 5
  done
}

# Never run two models at once. This is the whole point of a sequential driver:
# on 2 cores a second model halves the first one's throughput and every tok/s
# number in the comparison table becomes meaningless. Also blocks if an earlier
# run of this same driver is still going, which is exactly what happens when a
# model list is extended mid-flight.
any_run_alive() {
  # exclude this script's own descendants by matching only the python invocation
  pgrep -f "python3 -u run\.py" >/dev/null 2>&1
}

wait_for_exclusive() {
  local tries=0
  # 12h ceiling: the longest single model here (nemotron, tier A) runs about
  # 7h, and this guard's whole job is to let a queue wait behind it.
  while any_run_alive; do
    tries=$((tries+1))
    if [ "$tries" -gt 1440 ]; then
      say "ERROR: another run.py is still alive after 12h; refusing to start a second"
      return 1
    fi
    [ $((tries % 4)) -eq 1 ] && say "another llmbench run is still active; waiting (${tries}/1440)"
    sleep 30
  done
}

unload() {
  # ollama keeps the model resident for 5 min after last use; ask it to drop it
  curl -s -m 20 http://127.0.0.1:11434/api/generate \
    -d '{"model":"x","keep_alive":0,"prompt":""}' >/dev/null 2>&1 || true
  sleep 3
}

# No two models resident at once. The queue is sequential, but ollama's 5-min
# keep-alive means "previous model finished" does NOT mean "previous model is
# out of RAM" - on 7.6 GB an overlap thrashes and corrupts every timing number.
# Poll /api/ps until the server reports zero loaded models.
wait_for_ps_empty() {
  local tries=0
  while [ "$(curl -s -m 5 http://127.0.0.1:11434/api/ps 2>/dev/null \
              | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("models",[])))' 2>/dev/null)" != "0" ]; do
    tries=$((tries+1))
    if [ "$tries" -gt 12 ]; then
      say "WARNING: /api/ps still non-empty after 60s; continuing anyway"
      curl -s -m 5 http://127.0.0.1:11434/api/ps 2>/dev/null | head -c 300
      echo
      return 0
    fi
    say "waiting for previous model to unload ($(curl -s -m 5 http://127.0.0.1:11434/api/ps 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("models",[])))' 2>/dev/null) resident)"
    sleep 5
  done
}

start_llamacpp() {
  local gguf="$1" port="$2" label="$3"
  if ! curl -s -m 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    say "starting llama-server for $label on :$port"
    nohup bash "$HOME/ollama-work/run-llama.sh" "$gguf" "$port" 4096 \
      >> "logs/llama-$label.log" 2>&1 &
    for _ in $(seq 1 90); do
      sleep 5
      curl -s -m 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    done
  fi
}

# Per-model subsample caps. code_humaneval 40 and code_mbpp 60 keep the estimate
# useful while bounding the run; every small hand-authored set runs in full
# because they are cheap and their whole value is coverage.
CAPS="code_humaneval=40,code_mbpp=60,agentic=8"

if [ "$#" -gt 0 ]; then
  if [ "$1" = "--all" ]; then
    FILTER=""; RUN_ALL=1; shift
  else
    FILTER="$1"; shift
  fi
else
  FILTER=""; RUN_ALL="${RUN_ALL:-}"
fi

if [ -z "$FILTER" ] && [ "${RUN_ALL:-}" != "1" ]; then
  say "no model given - refusing to run the whole queue by default (see header)."
  say "available models:"
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r label backend model port <<<"$entry"
    if [[ " $TIER_A_MODELS " == *" $label "* ]]; then tier=A; else tier=B; fi
    say "  $label  (tier $tier, $backend)"
  done
  say 'run one:  bash run_all.sh <label>   (e.g. bash run_all.sh nemotron-3-nano)'
  say 'run all:  bash run_all.sh --all'
  exit 1
fi

# EXCLUDE is for handing off between drivers: a model already being run by an
# earlier driver must not be re-run here, and RESUME cannot express that because
# its results file does not exist yet.
EXCLUDE="${EXCLUDE:-}"

is_excluded() {
  case " $EXCLUDE " in *" $1 "*) return 0 ;; esac
  return 1
}

say "llmbench: tier A = [$TIER_A_MODELS] (${TIER_A_CAPS}, ${TIER_A_REPS} reps)"
[ -n "$EXCLUDE" ] && say "  excluding (handled by another driver): $EXCLUDE"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r label backend model port <<<"$entry"

  if [ -n "$FILTER" ] && [[ "$label" != *"$FILTER"* ]]; then
    say "skip $label (filtered out)"
    continue
  fi
  if is_excluded "$label"; then
    say "skip $label (handled by the other driver)"
    continue
  fi
  # Backend-separated results: results/<backend>/<label>.json, never mixed.
  out="results/$backend/$label.json"
  if [ -n "${RESUME:-}" ] && [ -s "$out" ]; then
    say "skip $label (already have results)"
    continue
  fi

  if [[ " $TIER_A_MODELS " == *" $label "* ]]; then
    CAPS="$TIER_A_CAPS"; REPS="$TIER_A_REPS"; AGENTIC_REPS="$TIER_A_AGENTIC"; TIER=A
  else
    CAPS="$TIER_B_CAPS"; REPS="$TIER_B_REPS"; AGENTIC_REPS="$TIER_B_AGENTIC"; TIER=B
  fi

  say "================================================================"
  say "MODEL $label  tier $TIER  ($backend:$port)  reps=$REPS agentic=$AGENTIC_REPS"
  say "  caps: $CAPS"
  if ! wait_for_exclusive; then
    say "skipped $label (another run holds the machine)"
    continue
  fi
  wait_for_mem
  unload
  if [ "$backend" = "ollama" ]; then
    wait_for_ps_empty
  fi

  if [ "$backend" = "llamacpp" ]; then
    case "$label" in
      minicpm5-2b) gguf="$HOME/ollama-dl/minicpm5-2b.gguf" ;;
      k2-horizon)  gguf="$HOME/ollama-dl/k2-horizon.gguf" ;;
      *) say "no gguf mapping for $label"; continue ;;
    esac
    unload                       # ollama's copy must go first
    start_llamacpp "$gguf" "$port" "$label"
    extra=(--api-key local-llama)
  else
    extra=()
  fi

  log="logs/$label.log"
  say "writing to $log"
  # Retry once on a transient backend failure: a model swap on 7.6 GB of RAM can
  # OOM a runner mid-task, and losing a multi-hour run to that is not worth it.
  for attempt in 1 2; do
    python3 -u run.py \
      --backend "$backend" --model "$model" --port "$port" \
      --tasks "$PROFILE_DEFAULT" \
      --reps "$REPS" --per-task-cap "$CAPS" --agentic-reps "$AGENTIC_REPS" \
      --label "$label" ${extra[@]} > "$log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && break
    # rc=2 is an argparse usage error: it will fail identically forever, so
    # retrying just doubles the wait before the same message.
    [ $rc -eq 2 ] && break
    if [ $attempt -eq 1 ]; then
      say "  attempt 1 failed (rc=$rc); unloading and retrying once"
      tail -4 "$log" | sed 's/^/    /'
      unload
      sleep 10
      if [ "$backend" = "llamacpp" ]; then
        pkill -f "llama-server.*$port" 2>/dev/null || true
        sleep 5
        start_llamacpp "$gguf" "$port" "$label"
      fi
    fi
  done
  if [ $rc -ne 0 ]; then
    say "FAILED $label after 2 attempts (rc=$rc); tail of log:"
    tail -5 "$log" | sed 's/^/    /'
  else
    say "DONE $label -> $out"
  fi

  if [ "$backend" = "llamacpp" ]; then
    pkill -f "llama-server.*$port" 2>/dev/null || true
  fi
  unload
  sleep 5
done

say "all models finished"
python3 report.py | tee results/REPORT.md