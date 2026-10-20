#!/usr/bin/env bash
# snapshot.sh — timestamped archive of everything that cannot be regenerated.
#
# A power cut does NOT lose results (the checkpoint protocol survives it -
# see crashtest.py). A dead disk does. This copies the evidence (result
# files, the report, the manifests) into ~/llmbench-backups/ under a
# timestamp, so a single archive can restore the state of the project.
#
#   bash snapshot.sh            # snapshot now
#   bash snapshot.sh --list     # list the archives
#   bash snapshot.sh --keep 10  # keep only the 10 newest
set -uo pipefail
cd "$(dirname "$0")"

BACKUP_DIR="$HOME/llmbench-backups"
KEEP="${SNAPSHOT_KEEP:-20}"
mkdir -p "$BACKUP_DIR"

case "${1:-}" in
  --list)
    ls -lh --time-style=long-iso "$BACKUP_DIR" 2>/dev/null | tail -n +2
    exit 0
    ;;
  --keep)
    KEEP="${2:-20}"
    ;;
esac

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$BACKUP_DIR/llmbench-$STAMP.tar.gz"

# What is worth keeping: the results (the evidence), the report, the task
# definitions and the manifests. Not the logs (huge) or the model weights.
tar -czf "$OUT" \
    results \
    reports \
    tasks \
    benchmarks \
    data/manifest.json \
    --exclude='results/attic/*' \
    --exclude='*.tmp' 2>/dev/null

if [ -s "$OUT" ]; then
  say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
  say "snapshot: $OUT ($(du -h "$OUT" | cut -f1))"
  # Keep only the newest N archives.
  ls -1t "$BACKUP_DIR"/llmbench-*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) \
    | while read -r old; do rm -f "$old"; say "pruned old snapshot $old"; done
else
  echo "snapshot failed: $OUT" >&2
  exit 1
fi
