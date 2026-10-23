# Repository layout and data flow

The repository is intentionally a small, flat Python application rather than
an installed package. `run.py`, verification utilities, and shell drivers
import one another as root-level modules and assume the repository root as
their working directory. Keeping these entrypoints in place avoids changing
CLI paths and benchmark execution behavior merely to make the tree look nested.

## Top-level directories

| Path | Contents | Git policy |
|---|---|---|
| `benchmarks/` | Machine-readable benchmark catalog and source/license metadata | Tracked |
| `data/manifest.json` | Hashes, counts, licenses, and metadata for acquired source data | Tracked |
| `data/res/` | Small HumanEval and sanitized MBPP source inputs used by task builders | Tracked under upstream terms; see `THIRD_PARTY_NOTICES.md` |
| `data/raw/` | Downloaded standard benchmark source files | Ignored; acquire with `standard_fetch.py` |
| `tasks/` | Custom tasks and generated task instances | Custom files tracked; `std_*.json` generated files ignored |
| `results/`, `logs/` | Model outputs and execution logs | Ignored; local evidence only |
| `reports/` | Committed HTML report snapshot and its local chart library | Tracked; snapshot requires local result files to regenerate |
| `tests/` | Model-free unit contracts and report contracts against local raw results | Tracked |
| `docs/` | Methodology, contributor guidance, architecture, and archived reviews | Tracked |

## Main execution path

1. `standard_fetch.py` acquires pinned external source data; `standard_strata.py`
   produces balanced standard screening instances. `build_tasks*.py` generates
   custom task files from bundled inputs and authored prompts.
2. `verify_*.py`, `verify_tasks.py`, and the controls in `gates.sh` check task
   shape, scoring, and negative controls before inference.
3. `run.py` uses `backend.py` to call a local Ollama or llama.cpp server;
   `scorers.py` and `agentic.py` score outputs. `run_llama.sh` and `run_all.sh`
   provide queueing and restart behavior.
4. `report.py`, `charts.py`, and `audit.py` aggregate, render, and verify the
   local result files. The checked-in `reports/screening.html` is a snapshot;
   results are intentionally not included in a fresh clone.

## Local evidence and report regeneration

`.gitignore` excludes standard downloads, generated standard task JSON, raw
result JSON, logs, caches, and local model weights. Do not remove these
exclusions just to make a fresh clone reproduce a local report. First restore
the benchmark result files in the expected `results/<backend>/` layout, then
run `python3 charts.py`. The generator refuses to replace the report if it finds
no model results.

`tests/test_units.py` is suitable for clean-checkout CI. `tests/test_report.py`
and the complete `bash gates.sh` suite additionally need the benchmark cache
and result evidence described above; some gates also need optional local
toolchains or a reachable inference server.

## Documentation map

- `README.md` — overview, quick start, methodology summary, current result
  snapshot, and command index.
- `docs/METHODOLOGY.md` — detailed protocol and research rationale.
- `CONTRIBUTING.md` — development checks and change requirements.
- `THIRD_PARTY_NOTICES.md` — licenses for bundled data inputs and assets.
- `docs/archive/` — dated audit records and historical prompts; these are not
  the current protocol authority.
