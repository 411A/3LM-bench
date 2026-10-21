#!/bin/bash
# remeasure_debiased.sh — re-measure ONLY the cells whose instance changed.
#
# The position-debiasing of the MMLU-6 / HellaSwag / WinoGrande s30 instances
# changed the CONTENT of those items (same questions, same option sets, new
# option order and gold letter). Their records were dropped with
# drop_tasks.py, so the coverage gaps are explicit and the resume re-runs
# exactly those items - 90 MC items per screening file, 36 per qualifier
# file, 90 for k2's text instrument. Nothing else is touched.
#
#   bash remeasure_debiased.sh
set -uo pipefail
cd "$(dirname "$0")"
LOG_DIR="logs/debiased"
mkdir -p "$LOG_DIR"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# The five models with letter-logprob MC cells (k2 has none: its llama.cpp
# build cannot score letters, so its MC lives in the text instrument).
MODELS="nemotron-3-nano qwen35-2b granite4-3b minicpm5-2b medgemma-4b"

for m in $MODELS; do
  say "screening re-measure: $m (3 MC suites, 90 items)"
  bash run_llama.sh "$m" screening --resume --until-complete 4 \
    > "$LOG_DIR/$m-screening.log" 2>&1 \
    || say "  screening re-measure: $m did not finish cleanly"
  say "  -> $(python3 - "$m" <<'PY'
import json, sys
d = json.load(open(f"results/llamacpp/{sys.argv[1]}.json"))
g = d.get("coverage_gaps")
print("COMPLETE" if not g else f"still missing: {g}")
PY
)"
done

for m in $MODELS k2-horizon; do
  say "qualifier re-measure: $m (3 MC suites, 12 items each)"
  bash run_llama.sh "$m" qualifier --resume --until-complete 4 \
    > "$LOG_DIR/$m-qualifier.log" 2>&1 \
    || say "  qualifier re-measure: $m did not finish cleanly"
  say "  -> $(python3 - "$m" <<'PY'
import json, sys
d = json.load(open(f"results/llamacpp/{sys.argv[1]}-q.json"))
g = d.get("coverage_gaps")
print("COMPLETE" if not g else f"still missing: {g}")
PY
)"
done

say "text-instrument re-measure: k2-horizon (3 suites, 90 items)"
bash run_llama.sh k2-horizon \
  "tasks=std_hellaswag_s30_txt,std_winogrande_s30_txt,std_mmlu_s30_txt" \
  --out 'results/llamacpp/{label}-txt.json' --resume --until-complete 4 \
  > "$LOG_DIR/k2-horizon-txt.log" 2>&1 \
  || say "  text re-measure did not finish cleanly"
say "  -> $(python3 - <<'PY'
import json
d = json.load(open("results/llamacpp/k2-horizon-txt.json"))
g = d.get("coverage_gaps")
print("COMPLETE" if not g else f"still missing: {g}")
PY
)"
say "RE-MEASURE FINISHED"
