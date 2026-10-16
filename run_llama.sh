#!/usr/bin/env bash
# run_llama.sh — pick a local GGUF, serve it with llama.cpp, benchmark it live.
#
# The whole session streams to your terminal AND to logs/<label>.log,
# so you watch every item in real time and keep the full record.
#
# Usage:
#   bash run_llama.sh                        # choose model + profile from menus
#   bash run_llama.sh qwen2b                 # model by name, profile menu
#   bash run_llama.sh qwen2b pilot           # no menus at all
#   bash run_llama.sh --list                 # show available models and exit
#   bash run_llama.sh qwen2b pilot --dry-run # print what would run, start nothing
#
# Profiles:
#   pilot      bounded diagnostic sweep (writes /tmp/diag only, never results/)
#   screening  tier-B caps across all suites -> results/<label>.json
#   custom     standard suites only (ARC/HellaSwag/GSM8K/...) -> results/<label>.json
#   tasks=a,b  explicit task list, e.g. tasks=english,std_gsm8k (with --limit N)
set -uo pipefail
cd "$(dirname "$0")"

DL="$HOME/ollama-dl"
BIN="$HOME/llama/llama-b11558/llama-server"
CTX=4096
APIKEY="local-llama"

# label | gguf file | default port
KNOWN=(
  "nemotron-3-nano|nemotron-3-nano.gguf|8097"
  "minicpm5-2b|minicpm5-2b.gguf|8091"
  "k2-horizon|k2-horizon.gguf|8092"
  "qwen35-4b|qwen4b.gguf|8093"
  "qwen35-2b|qwen2b.gguf|8094"
  "granite4-3b|granite.gguf|8095"
  "medgemma-4b|medgemma.gguf|8096"
)

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { say "ERROR: $*"; exit 1; }

list_models() {
  echo "GGUFs in $DL:"
  for entry in "${KNOWN[@]}"; do
    IFS='|' read -r label file port <<<"$entry"
    if [ -f "$DL/$file" ]; then
      printf '  %-15s %-22s %5s MB  (port %s)\n' "$label" "$file" \
        "$(du -mL "$DL/$file" | cut -f1)" "$port"
    else
      printf '  %-15s MISSING %s\n' "$label" "$file"
    fi
  done
  extras=$(comm -23 <(ls "$DL"/*.gguf 2>/dev/null | xargs -n1 basename | sort -u) \
    <(printf '%s\n' "${KNOWN[@]}" | cut -d'|' -f2 | sort -u))
  [ -n "$extras" ] && { echo "Unmapped GGUFs (usable via tasks= profile with explicit port):"; echo "$extras" | sed 's/^/  /'; }
}

[ ! -x "$BIN" ] && die "llama-server not found at $BIN"

# ---- arg parsing (menus for anything not given) ----
DRYRUN=0; LIMIT=""; OUT_OVERRIDE=""
WANT=""; PROFILE=""
ARGS=("$@")
i=0
while [ $i -lt $# ]; do
  a="${ARGS[$i]}"
  case "$a" in
    --list) list_models; exit 0;;
    --dry-run) DRYRUN=1;;
    --limit) i=$((i+1)); LIMIT="${ARGS[$i]:?--limit needs a number}";;
    --limit=*) LIMIT="${a#--limit=}";;
    --out) i=$((i+1)); OUT_OVERRIDE="${ARGS[$i]:?--out needs a path}";;    pilot|screening|custom) PROFILE="$a";;
    tasks=*) PROFILE="$a";;
    --*) die "unknown flag $a";;
    *) [ -z "$WANT" ] && WANT="$a" || die "extra arg $a";;
  esac
  i=$((i+1))
done

pick() { # $1=prompt, $2..=options; echoes choice
  local prompt="$1"; shift
  local opts=("$@")
  if [ "${#opts[@]}" -eq 1 ]; then echo "${opts[0]}"; return; fi
  echo "$prompt" >&2
  select o in "${opts[@]}"; do [ -n "$o" ] && { echo "$o"; break; }; done
}

if [ -z "$WANT" ]; then
  avail=()
  for entry in "${KNOWN[@]}"; do
    IFS='|' read -r label file _ <<<"$entry"
    [ -f "$DL/$file" ] && avail+=("$label")
  done
  [ "${#avail[@]}" -gt 0 ] || die "no GGUF found in $DL"
  if [ -n "${LLAMA_MODEL:-}" ]; then
    WANT="$LLAMA_MODEL"
  elif [ -t 0 ]; then
    WANT=$(pick "Choose a model:" "${avail[@]}")
  else
    die "no model given and stdin is not a tty (pass a model name or set LLAMA_MODEL)"
  fi
fi
if [ -z "$PROFILE" ]; then
  if [ -n "${LLAMA_PROFILE:-}" ]; then
    PROFILE="$LLAMA_PROFILE"
  elif [ -t 0 ]; then
    PROFILE=$(pick "Choose a profile:" pilot screening custom)
  else
    die "no profile given and stdin is not a tty (pass a profile or set LLAMA_PROFILE)"
  fi
fi
[ "$PROFILE" = custom ] && PROFILE="tasks=std_arc_challenge,std_arc_easy,std_hellaswag,std_winogrande,std_truthfulqa_mc,std_belebele_fa,std_mmlu,std_gsm8k"

# resolve label -> gguf + port
GGUF=""; PORT=""
for entry in "${KNOWN[@]}"; do
  IFS='|' read -r label file p <<<"$entry"
  if [ "$label" = "$WANT" ]; then GGUF="$DL/$file"; PORT="$p"; fi
done
[ -z "$GGUF" ] && { [ -f "$DL/$WANT" ] && GGUF="$DL/$WANT" || GGUF="$WANT"; PORT=8090; WANT=$(basename "$GGUF" .gguf); }
[ -f "$GGUF" ] || die "model file not found: $GGUF"
LABEL="$WANT"

# ---- profile -> run.py args ----
TASKS="all"; CAPS=""; REPS=1; AGENTIC=2; OUT="results/$LABEL.json"; DIAG=0
case "$PROFILE" in
  pilot) TASKS="all"; LIMIT="${LIMIT:-5}"; REPS=1; OUT="/tmp/diag/llama-$LABEL.json"; DIAG=1;;
  screening)
    CAPS="code_humaneval=20,code_mbpp=20,std_hellaswag=100,std_arc_easy=100,std_arc_challenge=100,std_mmlu=100,std_belebele_fa=100,std_winogrande=100,std_truthfulqa_mc=100,std_gsm8k=30,agentic=8"
    REPS=1; AGENTIC=2;;
  tasks=*) TASKS="${PROFILE#tasks=}"; LIMIT="${LIMIT:-0}";;
  *) die "unknown profile $PROFILE";;
esac
# A bounded --limit subset is front-biased and must never silently overwrite a
# real results file. Route it to /tmp/diag unless --out says otherwise.
if [ -n "${LIMIT:-}" ] && [ "${LIMIT:-0}" != 0 ] && [ "$PROFILE" != pilot ]; then
  OUT="/tmp/diag/llama-$LABEL-custom.json"; DIAG=1
fi
[ -n "${OUT_OVERRIDE:-}" ] && OUT="$OUT_OVERRIDE" && DIAG=0
[ -s "$OUT" ] && [ "$DIAG" = 0 ] && say "WARNING: $OUT exists and will be resumed/overwritten on config change"

say "model   : $LABEL ($GGUF, $(du -hL "$GGUF" | cut -f1)) via llama.cpp :$PORT"
say "profile : $PROFILE  tasks=$TASKS  limit=${LIMIT:-none}  reps=$REPS"
say "out     : $OUT"
[ "$DRYRUN" = 1 ] && { say "(dry run — nothing started)"; exit 0; }

# ---- never two runners; never two models resident ----
# Anchored pattern: matches only a real `python3 -u run.py ...` process,
# never a shell whose command line merely mentions it.
if pgrep -f "^python3 -u run\.py" >/dev/null 2>&1; then
  die "another run.py is active — one model at a time (see status.sh)"
fi

ollama_loaded() {
  curl -s -m 5 http://127.0.0.1:11434/api/ps 2>/dev/null \
    | python3 -c 'import json,sys
try: n = len(json.load(sys.stdin).get("models", []))
except Exception: n = 0
raise SystemExit(0 if n == 0 else 1)' 2>/dev/null
}

# ollama keep-alive would hold GBs alongside llama-server: ask it to drop them
if ! ollama_loaded; then
  say "unloading ollama models first (7.6 GB cannot hold two runtimes)"
  curl -s -m 20 http://127.0.0.1:11434/api/generate \
    -d '{"model":"x","keep_alive":0,"prompt":""}' >/dev/null 2>&1 || true
  for _ in $(seq 1 12); do
    sleep 5
    ollama_loaded && break
  done
  ollama_loaded || say "WARNING: ollama still reports loaded models; continuing anyway"
fi

# ---- server (reuse if already healthy on this port) ----
# NOTE: /health returns 503 while the model is still loading, and plain
# `curl -s` exits 0 for a 503 — so an empty check declares a loading server
# ready and the first probe dies. Require HTTP 200 explicitly.
server_up() {
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:$PORT/health")" = 200 ]
}
STARTED_BY_ME=0
if server_up; then
  say "llama-server already up on :$PORT — reusing it"
else
  say "starting llama-server for $LABEL on :$PORT (ctx $CTX)"
  mkdir -p logs
  # shellcheck disable=SC2086
  nohup bash "$HOME/ollama-work/run-llama.sh" "$GGUF" "$PORT" "$CTX" \
    >> "logs/llama-$LABEL.log" 2>&1 &
  for _ in $(seq 1 60); do sleep 5; server_up && break; done
  server_up || die "server never became healthy — see logs/llama-$LABEL.log"
  STARTED_BY_ME=1
fi

cleanup() {
  if [ "$STARTED_BY_ME" = 1 ]; then
    say "stopping my llama-server on :$PORT"
    pkill -f "llama-server.*$PORT" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---- benchmark: everything streams live AND lands in the log file ----
mkdir -p logs "$(dirname "$OUT")"
LOG="logs/llama-run-$LABEL.log"
say "logging live to $LOG (Ctrl+C stops the run; server cleanup still runs)"
# shellcheck disable=SC2086
python3 -u run.py --backend llamacpp --model "$LABEL" --port "$PORT" \
  --api-key "$APIKEY" --model-path "$GGUF" \
  --tasks "$TASKS" --reps "$REPS" \
  ${CAPS:+--per-task-cap "$CAPS"} \
  ${LIMIT:+--limit "$LIMIT"} \
  --agentic-reps "$AGENTIC" --label "$LABEL" --out "$OUT" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
[ "$DIAG" = 1 ] && say "diagnostic run — $OUT is NOT a benchmark score (front-biased --limit subsets)"
say "done rc=$rc -> $OUT  (full log: $LOG)"
exit "$rc"
