#!/usr/bin/env bash
# One-screen status of the llmbench queue.
set -uo pipefail
cd "$(dirname "$0")"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

echo "=== memory ==="
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/{
  printf "  %-14s %6.0f MiB\n", $1, $2/1024 }' /proc/meminfo

echo
echo "=== processes ==="
if pgrep -f "python3 -u run\.py" >/dev/null 2>&1; then
  ps -eo pid,etime,args 2>/dev/null | grep "[p]ython3 -u run.py" \
    | sed 's/--per-task-cap.*--label/... --label/' | cut -c1-120 | sed 's/^/  /'
else
  echo "  no benchmark running"
fi

echo
echo "=== results so far ==="
for f in results/*.json results/*/*.json; do
  [ -e "$f" ] || continue
  case "$f" in */speed.json) continue ;; esac
  python3 - "$f" <<'PY'
import json, sys, os
p = sys.argv[1]
name = os.path.basename(p)[:-5]
try:
    d = json.load(open(p))
except Exception as e:
    print(f"  {name:<20} unreadable: {e}"); raise SystemExit
recs = d.get("records", [])
cats = {}
for r in recs:
    cats[r.get("category","?")] = cats.get(r.get("category","?"), 0) + 1
print(f"  {name:<20} {len(recs):>5} records  {len(cats):>2} categories  "
      f"elapsed {d.get('elapsed_s',0)/3600:.2f} h  cfg {d.get('config_hash','-')}")
PY
done

echo
echo "=== current log tail ==="
for lg in logs/*.log; do
  [ "$(basename "$lg")" = "queue.log" ] && continue
  [ "$(basename "$lg")" = "driver.log" ] && continue
  # only the most recently modified model log
  :
done
newest=$(ls -t logs/*.log 2>/dev/null | grep -vE 'queue|driver' | head -1)
if [ -n "$newest" ]; then
  say "$(basename "$newest") :"
  tail -6 "$newest" | sed 's/^/    /'
fi

echo
echo "=== overall ranking (by category score) ==="
python3 report.py 2>/dev/null | grep -E '^\| ' | head -40 || echo "  (needs at least one finished run)"