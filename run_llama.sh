#!/usr/bin/env bash
# run_llama.sh — pick a local GGUF, serve it with llama.cpp, benchmark it live.
#
# The whole session streams to your terminal AND to logs/<label>.log,
# so you watch every item in real time and keep the full record.
#
# Usage:
#   bash run_llama.sh                        # choose model + profile from menus
#   bash run_llama.sh all qualifier          # EVERY local GGUF, one at a time
#   bash run_llama.sh qwen35-2b,nemotron-3-nano screening
#   bash run_llama.sh qwen2b                 # model by name, profile menu
#   bash run_llama.sh qwen2b pilot           # no menus at all
#   bash run_llama.sh --list                 # show available models and exit
#   bash run_llama.sh qwen2b pilot --dry-run # print what would run, start nothing
#
# Profiles:
#   pilot      bounded diagnostic sweep (writes /tmp/diag only, never results/)
#   qualifier  ~30 min/model World-Cup qualifier -> results/<label>-q.json,
#              ranked by qualify.py into CUT vs ADVANCE
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
DRYRUN=0; LIMIT=""; OUT_OVERRIDE=""; RESUME_MODE=0
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
    --out) i=$((i+1)); OUT_OVERRIDE="${ARGS[$i]:?--out needs a path}";;
    --resume) RESUME_MODE=1;;    pilot|screening|custom|qualifier) PROFILE="$a";;
    tasks=*) PROFILE="$a";;
    --*) die "unknown flag $a";;
    *) [ -z "$WANT" ] && WANT="$a" || die "extra arg $a";;
  esac
  i=$((i+1))
done

pick() { # $1=prompt, $2..=label::desc entries; echoes chosen label
  local prompt="$1"; shift
  local -a labels=() shown=()
  local e l d
  for e in "$@"; do
    l="${e%%::*}"; d="${e#*::}"
    labels+=("$l"); shown+=("$l -- $d")
  done
  if [ "${#labels[@]}" -eq 1 ]; then echo "${labels[0]}"; return; fi
  echo "$prompt" >&2
  local o
  select o in "${shown[@]}"; do
    if [ -n "$o" ]; then echo "${o%% -- *}"; break; fi
  done
}

if [ -z "$WANT" ]; then
  entries=()
  for entry in "${KNOWN[@]}"; do
    IFS='|' read -r label file p <<<"$entry"
    if [ -f "$DL/$file" ]; then
      note="$(du -hL "$DL/$file" | cut -f1)B, port $p"
      [ "$label" = nemotron-3-nano ] && note="$note, ex-Ollama blob (same weights)"
      entries+=("$label::$note")
    fi
  done
  [ "${#entries[@]}" -gt 0 ] || die "no GGUF found in $DL"
  if [ -n "${LLAMA_MODEL:-}" ]; then
    WANT="$LLAMA_MODEL"
  elif [ -t 0 ]; then
    WANT=$(pick "Choose a model:" "${entries[@]}")
  else
    die "no model given and stdin is not a tty (pass a model name or set LLAMA_MODEL)"
  fi
fi
# NOTE: the profile menu lives after the model menu (model list section
# below) so interactive runs choose model first, then profile.

# resolve label -> gguf + port. No mixing: every llamacpp result lives under
# results/llamacpp/ (ollama runs use results/ollama/ via run_all.sh),
# so a backend is never silently compared against another.
resolve_model() { # $1=label; sets GGUF PORT LABEL, returns 1 if unknown
  GGUF=""; PORT=""
  local entry label file p
  for entry in "${KNOWN[@]}"; do
    IFS='|' read -r label file p <<<"$entry"
    if [ "$label" = "$1" ]; then GGUF="$DL/$file"; PORT="$p"; fi
  done
  if [ -z "$GGUF" ]; then
    if [ -f "$DL/$1" ]; then GGUF="$DL/$1"; PORT=8090;
    elif [ -f "$1" ]; then GGUF="$1"; PORT=8090;
    else return 1; fi
  fi
  [ -f "$GGUF" ] || return 1
  LABEL="$1"
  return 0
}

all_labels() {
  local entry label file _
  for entry in "${KNOWN[@]}"; do
    IFS='|' read -r label file _ <<<"$entry"
    [ -f "$DL/$file" ] && echo "$label"
  done
}

# ---- profile -> run.py args (model-independent part) ----
profile_args() { # sets TASKS CAPS REPS AGENTIC LIMIT(for pilot/qualifier/tasks)
  TASKS="all"; CAPS=""; REPS=1; AGENTIC=2; TIMEOUT=""
  case "$PROFILE" in
    pilot) TASKS="all"; LIMIT="${LIMIT:-5}"; REPS=1; TIMEOUT=300;;
    qualifier)
      # WORLD CUP QUALIFIER: ~30 min/model. Stratified s30 subsets, identical
      # for every model (deterministic seed), so the ranking is fair. MC kills
      # the guessers, 3+3 code + 12 math + 3 chains test generation/tool use.
      # Survivors go to screening/full. Results ARE comparable across models
      # (identical items) but NOT against full-split numbers.
      TASKS="science,electronics,persian,english,std_arc_easy_s30,std_winogrande_s30,std_truthfulqa_mc_s30,std_hellaswag_s30,std_belebele_fa_s30,std_mmlu_s30,std_gsm8k_s30,code_humaneval,code_mbpp,agentic"
      LIMIT="${LIMIT:-12}"
      CAPS="code_humaneval=3,code_mbpp=3,agentic=3"
      REPS=1; AGENTIC=1; TIMEOUT=300;;
    screening)
      # Default comparable run: custom suites full + stratified s30 (30 each
      # by construction, no caps needed). Full official splits are Tier-3
      # opt-in via tasks=std_hellaswag,... (overnight).
      TASKS="code_humaneval,code_mbpp,bugfix,algo,refactor,code_rust,embedded_cpp,science,electronics,persian,english,netdiag,devops,lint,agentic,std_arc_challenge_s30,std_arc_easy_s30,std_hellaswag_s30,std_winogrande_s30,std_truthfulqa_mc_s30,std_belebele_fa_s30,std_mmlu_s30,std_gsm8k_s30"
      CAPS="code_humaneval=20,code_mbpp=20,agentic=8"
      REPS=1; AGENTIC=2;;
    tasks=*) TASKS="${PROFILE#tasks=}"; LIMIT="${LIMIT:-0}";;
    *) die "unknown profile $PROFILE";;
  esac
}

# per-model OUT: backend folder + qualifier suffix; bounded --limit subsets
# never silently overwrite real results (diag-safe), except the qualifier,
# whose identical subsets per model ARE the ranking instrument.
model_out() { # sets OUT DIAG
  OUT="results/llamacpp/$LABEL.json"; DIAG=0
  case "$PROFILE" in
    pilot) OUT="/tmp/diag/llamacpp-$LABEL.json"; DIAG=1;;
    qualifier) OUT="results/llamacpp/$LABEL-q.json";;
    tasks=*)
      if [ -n "${LIMIT:-}" ] && [ "${LIMIT:-0}" != 0 ]; then
        OUT="/tmp/diag/llamacpp-$LABEL-custom.json"; DIAG=1
      fi;;
  esac
  [ -n "${OUT_OVERRIDE:-}" ] && OUT="$OUT_OVERRIDE" && DIAG=0
}

server_up() {
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:$PORT/health")" = 200 ]
}
STARTED_BY_ME=0
SPAWNED=0
cleanup() {
  # SPAWNED covers the startup window: ^C while waiting for /health must
  # still kill the child (an orphaned server once squatted 2h and filled
  # swap). Reused servers (never spawned) are left alone.
  if [ "$STARTED_BY_ME" = 1 ] || [ "$SPAWNED" = 1 ]; then
    say "stopping llama-server on :$PORT"
    pkill -f "llama-server.*$PORT" 2>/dev/null || true
    STARTED_BY_ME=0
    SPAWNED=0
  fi
}
trap cleanup EXIT
# One Ctrl+C stops the WHOLE queue (previously it only killed the inner
# run.py and the loop marched on to the next model, forcing four more ^Cs).
trap 'exit 130' INT TERM

free_mib() { awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo; }

wait_for_mem() {
  local tries=0
  while [ "$(free_mib)" -lt 900 ]; do
    tries=$((tries+1))
    if [ "$tries" -gt 24 ]; then
      say "WARNING: only $(free_mib) MiB free after 2 min (floor 900); "
      say "  starting anyway — OOM risk, watch for Killed servers"
      return 0
    fi
    say "waiting for memory: $(free_mib) MiB free (floor 900), attempt $tries"
    sleep 5
  done
}

run_one() { # $1=label, $2=k/n for banner
  resolve_model "$1" || { say "FAILED $1: unknown model or missing file"; return 1; }
  profile_args
  model_out
  say "================================================================"
  say "MODEL $LABEL ($2)  $GGUF ($(du -hL "$GGUF" | cut -f1))  :$PORT"
  say "profile : $PROFILE  tasks=$TASKS  limit=${LIMIT:-none}  reps=$REPS"
  say "out     : $OUT"
  [ -s "$OUT" ] && [ "$DIAG" = 0 ] \
    && say "WARNING: $OUT exists and will be resumed/overwritten on config change"
  if [ "$DRYRUN" = 1 ]; then say "(dry run — nothing started)"; return 0; fi

  wait_for_mem

  # Stray servers from killed runs squat on GBs and corrupt every timing
  # number. Mine is identified by PORT; anything else gets named, not killed.
  strays=$(pgrep -af "llama-server --model" 2>/dev/null \
    | grep -v "port $PORT\$" | grep -v "port $PORT " || true)
  [ -n "$strays" ] && say "WARNING: stray llama-server(s) holding RAM:" \
    && echo "$strays" | sed 's/^/    /' | cut -c1-120

  if server_up; then
    say "llama-server already up on :$PORT — reusing it"
  else
    say "starting llama-server for $LABEL on :$PORT (ctx $CTX)"
    mkdir -p logs
    SPAWNED=1
    nohup bash "$HOME/ollama-work/run-llama.sh" "$GGUF" "$PORT" "$CTX" \
      >> "logs/llama-$LABEL.log" 2>&1 &
    for _ in $(seq 1 60); do sleep 5; server_up && break; done
    server_up || { say "FAILED $LABEL: server never healthy — see logs/llama-$LABEL.log"; return 1; }
    STARTED_BY_ME=1
  fi

  mkdir -p logs "$(dirname "$OUT")"
  local LOG="logs/llama-run-$LABEL.log"
  say "logging live to $LOG"
  local RESUME_FLAG=()
  [ "$RESUME_MODE" = 1 ] && RESUME_FLAG=(--resume "$OUT")
  # shellcheck disable=SC2086
  python3 -u run.py --backend llamacpp --model "$LABEL" --port "$PORT" \
    --api-key "$APIKEY" --model-path "$GGUF" \
    --tasks "$TASKS" --reps "$REPS" \
    ${CAPS:+--per-task-cap "$CAPS"} \
    ${LIMIT:+--limit "$LIMIT"} \
    ${TIMEOUT:+--timeout "$TIMEOUT"} \
    "${RESUME_FLAG[@]}" \
    --agentic-reps "$AGENTIC" --label "$LABEL" --out "$OUT" 2>&1 | tee "$LOG"
  local rc=${PIPESTATUS[0]}
  cleanup
  [ "$DIAG" = 1 ] && say "diagnostic run — $OUT is NOT a benchmark score (front-biased --limit subsets)"
  say "done rc=$rc -> $OUT"
  return "$rc"
}

# ---- model list: menu / all / comma list ----
if [ -z "$WANT" ]; then
  entries=()
  for label in $(all_labels); do
    for entry in "${KNOWN[@]}"; do
      IFS='|' read -r l file p <<<"$entry"
      if [ "$l" = "$label" ]; then
        note="$(du -hL "$DL/$file" | cut -f1)B, port $p"
        [ "$label" = nemotron-3-nano ] && note="$note, ex-Ollama blob (same weights)"
        entries+=("$label::$note")
      fi
    done
  done
  entries+=("all::run every model above, one at a time, same profile")
  [ "${#entries[@]}" -gt 0 ] || die "no GGUF found in $DL"
  if [ -n "${LLAMA_MODEL:-}" ]; then
    WANT="$LLAMA_MODEL"
  elif [ -t 0 ]; then
    WANT=$(pick "Choose a model:" "${entries[@]}")
  else
    die "no model given and stdin is not a tty (pass a model name, 'all', or set LLAMA_MODEL)"
  fi
fi
if [ -z "$PROFILE" ]; then
  if [ -n "${LLAMA_PROFILE:-}" ]; then
    PROFILE="$LLAMA_PROFILE"
  elif [ -t 0 ]; then
    PROFILE=$(pick "Choose a profile:" \
      "pilot::all 23 suites x ~5 items -> /tmp only, ~45 min, NOT scores" \
      "qualifier::14 fast suites, ~30 min/model -> results, ranks + CUT/ADVANCE" \
      "screening::tier-B caps, all suites -> results/, hours, comparable" \
      "custom::8 standard suites, FULL splits -> results/, overnight")
  else
    die "no profile given and stdin is not a tty (pass a profile or set LLAMA_PROFILE)"
  fi
fi
[ "$PROFILE" = custom ] && PROFILE="tasks=std_arc_challenge,std_arc_easy,std_hellaswag,std_winogrande,std_truthfulqa_mc,std_belebele_fa,std_mmlu,std_gsm8k"

# ---- expand WANT to the queue: all | a,b | single ----
MODELS=()
if [ "$WANT" = all ]; then
  while read -r l; do MODELS+=("$l"); done < <(all_labels)
elif [[ "$WANT" == *,* ]]; then
  IFS=',' read -ra MODELS <<<"$WANT"
else
  MODELS=("$WANT")
fi
[ "${#MODELS[@]}" -gt 0 ] || die "empty model queue"
if [ "$RESUME_MODE" = 1 ] && [ "${#MODELS[@]}" -gt 1 ]; then
  die "--resume takes one model (one file): rerun each failed model separately"
fi

if [ "$DRYRUN" = 1 ]; then
  for m in "${MODELS[@]}"; do
    resolve_model "$m" || die "unknown model: $m"
    profile_args; model_out
    say "would run: $LABEL  profile=$PROFILE  tasks=$TASKS  "
    say "           limit=${LIMIT:-none}  reps=$REPS  out=$OUT"
  done
  say "(dry run — nothing started)"
  exit 0
fi

# ---- never two runners; never two models resident ----
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

# ollama keep-alive would hold GBs alongside llama-server: drop once, up front
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

# ---- the queue: strictly one model at a time ----
DONE=(); FAILED=()
k=0
for m in "${MODELS[@]}"; do
  k=$((k + 1))
  if run_one "$m" "$k/${#MODELS[@]}"; then
    DONE+=("$m")
  else
    FAILED+=("$m")
  fi
done

echo
say "==================== QUEUE FINISHED ===================="
[ "${#DONE[@]}" -gt 0 ] && say "DONE:   ${DONE[*]}"
[ "${#FAILED[@]}" -gt 0 ] && say "FAILED: ${FAILED[*]}"
[ "${#FAILED[@]}" -gt 0 ] && exit 1
exit 0
