# Contributing

Thanks for helping improve llmbench. Keep changes reproducible and keep
benchmark protocol changes separate from code cleanup.

## Before opening a pull request

Run the model-free checks:

```bash
python3 -m py_compile ./*.py tests/*.py
for script in ./*.sh; do bash -n "$script"; done
python3 tests/test_units.py
python3 run.py --help
```

If you have the local benchmark caches and complete result files, also run:

```bash
python3 tests/test_report.py
python3 audit.py
bash gates.sh
```

The report contracts and full gate suite rely on local files excluded from Git;
see [Repository layout and data flow](docs/REPOSITORY_LAYOUT.md). CI runs the
model-free checks only and never downloads benchmark data or starts inference.

## Change guidelines

- Preserve the distinction between standard/custom suites and between
  letter-logprob/text-match MC instruments.
- Missing measurements must remain missing, not be converted to zero.
- If task prompts, IDs, golds, or scoring change, explain the protocol effect
  and identify which stored results become stale. Do not silently reuse old
  scores for changed items.
- Keep model inference opt-in. Tests and CI must not call a hosted model API or
  require a running local model.
- Keep command-line entrypoints and shared top-level imports at the repository
  root unless the import/path migration is tested across the shell workflows.
- Add or update regression tests for scoring, coverage, report, and resume
  behavior affected by a change.
- Preserve third-party notices. Do not commit run logs, raw standard dataset
  downloads, model weights, secrets, or personal machine artifacts.

## Pull request description

State what changed, how it was tested, whether a benchmark protocol or published
result changed, and whether any rerun is required. Link the relevant methodology
or catalog entry when changing an instrument.
