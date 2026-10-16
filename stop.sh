#!/usr/bin/env bash
# Stop everything llmbench: the queue driver AND any runner it launched.
# Lesson: killing only "python3 -u run.py" lets run_all.sh's retry loop
# relaunch a fresh 7h run behind your back (and truncate the log doing it).
# Always stop the DRIVER first, then the runner, then unload the model.
set -uo pipefail
pkill -f "bash run_all\.sh" 2>/dev/null || true
sleep 1
pkill -f "python3 -u run\.py" 2>/dev/null || true
sleep 3
if pgrep -f "bash run_all\.sh|python3 -u run\.py" >/dev/null 2>&1; then
  echo "STILL ALIVE:"; pgrep -af "run_all\.sh|run\.py" | grep -v grep
  exit 1
fi
curl -s -m 20 http://127.0.0.1:11434/api/generate \
  -d '{"model":"x","keep_alive":0,"prompt":""}' >/dev/null 2>&1 || true
echo "llmbench stopped; model unload requested."
