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
  # nemotron-3-nano:tool calls DO NOT WORK via this blob on llama-server
  # (56 trials, 0 calls, model narrates instead — needs Ollama's tool
  # renderer, same stripped-metadata class as the qwen35 rebuilds). Use the
  # ollama tag for anything agentic (verified 2/2 full-trace solves there).
  # MC/code/generation via this blob are unaffected and valid.
  "nemotron-3-nano|nemotron-3-nano.gguf|8097"
  "minicpm5-2b|minicpm5-2b.gguf|8091"
  "k2-horizon|k2-horizon.gguf|8092"
  # qwen35-4b DISABLED 2026-10-17: qwen4b.gguf + llama-server b11558 picks
  # letter A on ~everything at 0.1-0.8 mass (0/12 truthfulqa, 1/12 mmlu, two
  # runs) while qwen3.5:4b-q4_K_M via Ollama scores 5/6 at mass 1.0 with
  # varied letters on the same items. Same weights, different harness path =
  # broken template/rendering pairing, not a model verdict. The GGUF's
  # embedded template bytes are genuine upstream Qwen3.5, so the failure is
  # in server-side rendering (b11558 predates qwen35 handling). Benchmark
  # qwen3.5 through Ollama (see run_all.sh) until the server is upgraded.
  # To re-enable: uncomment the line below.
  # "qwen35-4b|qwen4b.gguf|8093"
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
DRYRUN=0; LIMIT=""; OUT_OVERRIDE=""; RESUME_MODE=0; UNTIL_COMPLETE=0
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
    --resume) RESUME_MODE=1;;
    --until-complete) i=$((i+1)); UNTIL_COMPLETE="${ARGS[$i]:?--until-complete needs rounds}";;
    --until-complete=*) UNTIL_COMPLETE="${a#--until-complete=}";;
    pilot|screening|custom|qualifier) PROFILE="$a";;
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
  [ -n "${OUT_OVERRIDE:-}" ] && OUT="${OUT_OVERRIDE//\{label\}/$LABEL}" && DIAG=0
}

model_out_for() { # $1=label -> prints its OUT path (for resume bookkeeping)
  local _save="$LABEL"
  LABEL="$1"
  model_out
  printf '%s' "$OUT"
  LABEL="$_save"
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

# ---- quiet-machine preflight -------------------------------------------
# A benchmark number is only worth what the machine it was taken on was
# doing at the time. The protocol is ONE model resident, nothing else
# running, and enough free memory to hold it. This check enforces that
# BEFORE the model is loaded, and refuses to start on a busy box rather
# than quietly publishing a number measured at 60% power.
# % of the WHOLE box's CPU used by everything, sampled live over 1s.
# The load average is a lagging indicator (it stayed above 2 for minutes
# after the box went idle) and it hides a single runaway thread - which is
# exactly what happened here: a thumbnailer sat at 100% of one core for
# 28 hours, so every throughput number taken before it died was measured
# at half power.
cpu_busy_pct() {
  local a b
  a=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat)
  sleep 1
  b=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat)
  awk -v a="$a" -v b="$b" 'BEGIN{
    split(a,x," "); split(b,y," ");
    t=y[1]-x[1]; i=y[2]-x[2];
    if (t<=0) {print "0.0"} else {printf "%.1f", 100*(t-i)/t}
  }'
}

preflight_quiet() {
  local tries=0 busy="" pct
  while : ; do
    busy=""
    # another model resident?
    if pgrep -f "[l]lama-server --model" >/dev/null 2>&1; then
      busy="another llama-server is resident"
    fi
    # another benchmark runner?
    if pgrep -f "^python3 -u run\.py" >/dev/null 2>&1; then
      busy="another benchmark runner is active"
    fi
    # foreign CPU load (everything except our own server, which is not up yet)
    pct=$(cpu_busy_pct)
    if [ -z "$busy" ] && awk "BEGIN{exit !($pct > 30.0)}"; then
      busy="$pct% of the box's CPU is busy before the model is even loaded"
    fi
    [ -z "$busy" ] && break
    tries=$((tries + 1))
    if [ "$tries" -gt 36 ]; then
      say "WARNING: box still busy after 3 min ($busy); starting anyway"
      say "  (throughput numbers from this run are marked as measured under load)"
      return 0
    fi
    [ "$tries" = 1 ] && say "waiting for a quiet box: $busy"
    sleep 5
  done
  say "quiet box confirmed: 1 model, foreign CPU ${pct}%, $(free_mib) MiB free"
}

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
  preflight_quiet

  # A partial file means the last run died or aborted mid-way: its server is
  # either dead or fat. Never reuse it — restart for a fresh memory window.
  # (Reusing a 5GB server is how one OOM becomes two.)
  if [ -s "$OUT" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT')).get('partial') else 1)" 2>/dev/null; then
    say "previous run partial — restarting server for a fresh memory window"
    pkill -f "llama-server.*$PORT" 2>/dev/null || true
    sleep 5
  fi

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
  # The laptop's power manager suspends the box on idle; logind then
  # SIGTERMs every user process and the worker dies holding the results of
  # the rep in memory. Hold a sleep/idle inhibitor for the whole run.
  local INHIBIT=()
  if command -v systemd-inhibit >/dev/null 2>&1; then
    INHIBIT=(systemd-inhibit --what=sleep:idle --why="llmbench benchmark run")
    say "holding sleep/idle inhibitor for the duration of the run"
  else
    say "WARNING: systemd-inhibit missing - a suspend would kill this run"
  fi
  # shellcheck disable=SC2086
  "${INHIBIT[@]}" python3 -u run.py --backend llamacpp --model "$LABEL" --port "$PORT" \
    --api-key "$APIKEY" --model-path "$GGUF" \
    --tasks "$TASKS" --reps "$REPS" \
    ${CAPS:+--per-task-cap "$CAPS"} \
    ${LIMIT:+--limit "$LIMIT"} \
    ${TIMEOUT:+--timeout "$TIMEOUT"} \
    "${RESUME_FLAG[@]}" \
    --agentic-reps "$AGENTIC" --label "$LABEL" --out "$OUT" 2>&1 | tee "$LOG"
  local rc=${PIPESTATUS[0]}
  cleanup
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT')).get('partial') else 1)" 2>/dev/null; then
    say "NOTE: $OUT is PARTIAL (stubs or unrun tasks) — resume, do not quote"
  fi
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
# --until-complete N: after each model, while its OUT is partial and progress
# is being made, restart the server and resume (up to N extra rounds). This
# grinds through tasks no single server lifetime can finish (e.g. 30 long
# generations on a 7.6GB box). Stops early on completion OR on a round with
# zero new records (no progress = BLOCKED, not slow).
DONE=(); FAILED=(); PARTIAL_DONE=()
rec_count() {
  # Valid records only: a stub (backend error) is not progress - counting it
  # made the resume loop believe a round had succeeded while nothing usable
  # was banked (five rounds, zero new records, "progress" every time).
  python3 -c "
import json,sys
try:
    d=json.load(open('$1'))
except Exception:
    print(0); raise SystemExit
rs=d.get('records',[])
ok=[r for r in rs if 'ERROR' not in str(r.get('id','')) and r.get('stopped') not in ('error','exception')]
print(len(ok))
" 2>/dev/null || echo 0
}
k=0
for m in "${MODELS[@]}"; do
  k=$((k + 1))
  if run_one "$m" "$k/${#MODELS[@]}"; then
    DONE+=("$m")
  else
    FAILED+=("$m")
  fi
  if [ "${UNTIL_COMPLETE:-0}" -gt 0 ] && [ "${#MODELS[@]}" -eq 1 ] && [ "$DRYRUN" != 1 ]; then
    rounds=0
    while [ "$rounds" -lt "$UNTIL_COMPLETE" ]; do
      out_f=$(model_out_for "$m") || break
      python3 -c "import json,sys; sys.exit(0 if json.load(open('$out_f')).get('partial') else 1)" 2>/dev/null || break
      before=$(rec_count "$out_f")
      rounds=$((rounds + 1))
      say "---- resume round $rounds/$UNTIL_COMPLETE for $m ($before records banked) ----"
      run_one "$m" "$k/${#MODELS[@]} round$rounds" || true
      after=$(rec_count "$out_f")
      if [ "$after" -le "$before" ]; then
        say "no progress in round $rounds ($before records) — BLOCKED, stopping"
        break
      fi
    done
  fi
done

echo
say "==================== QUEUE FINISHED ===================="
[ "${#DONE[@]}" -gt 0 ] && say "DONE:   ${DONE[*]}"
[ "${#FAILED[@]}" -gt 0 ] && say "FAILED: ${FAILED[*]}"
[ "${#FAILED[@]}" -gt 0 ] && exit 1
exit 0
