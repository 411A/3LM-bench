# REVIEW4 — why a "complete" cell kept vanishing, and the three killers (2026-10-20)

Context: nemotron-3-nano's `std_gsm8k_s30` cell was the only hole left in the
screening matrix. Six attempts to fill it produced **zero** usable records.
This review explains why, with evidence, and lists the fixes.

## The symptom

```
[05:21:36] !! backend error: RemoteDisconnected ... /v1/chat/completions
[05:21:36] !! server is dead — skipping retry, recording stub
[05:48:17] !! OOM-GUARD: server RSS 5817 MiB over budget 5500 at run_gsm8k-item-7
[07:17:32] [1/30] gsm8k-197  OK  pred=6277  gold=6277  3.16tok/s
Terminated   python3 -u run.py ...
```

Each round looked like a different failure. They were three independent
killers, and only the first was visible in the logs.

## Killer 1 — the laptop sleeps mid-run (SIGTERM), and the results die with it

Evidence:

```
$ dmesg | grep "PM: suspend"
[637714.985138] ACPI: PM: Preparing to enter system sleep state S3
[651045.250719] ACPI: PM: Preparing to enter system sleep state S3
[661762.835791] ACPI: PM: Preparing to enter system sleep state S3   <-- 07:17 today
[661763.404616] PM: suspend exit
```

`xfce4-power-manager` (PID 2063) holds the `sleep` inhibitor and suspends the
box on idle. On suspend, `systemd-logind` applies `KillUserProcesses=yes`
(the Debian default; `/etc/systemd/logind.conf` has it commented out) and
**SIGTERMs every user process** — including the benchmark worker. The shell
survived, so the queue kept starting new rounds against a machine that was
about to sleep again. The "Terminated" line above is that SIGTERM.

Why it was invisible: the *only* symptom was one missing line in the log, and
it looked like an operator action.

## Killer 2 — results were held in RAM until the whole rep finished

`run.py` wrote its checkpoint **per rep**, but a rep of 30 GSM8K generations
takes ~25 minutes. Any kill inside that window threw away every *completed*
item of the rep — including the item that had just been scored (item 1 at
07:17:32 was OK and never reached disk).

## Killer 3 — the resume loop counted stubs as progress

`rec_count()` counted *records*, and a backend-error stub is a record. So
every round looked like progress even though nothing usable was banked:
5 rounds, 7 stub records, 0 valid records, loop still claiming success.

## Killer 4 (latent) — `partial` was decided by stub-presence, not coverage

`"partial": any("ERROR" in id ...)`. A stub that a later attempt replaced
with a real record would have kept the file "partial" forever, and a task
that never ran at all was invisible unless it left a stub behind.

## The fixes (each with a regression test)

| # | Fix | Where | Test |
|---|-----|-------|------|
| 1 | Every task now reports **item-by-item**; each finished item is checkpointed to disk immediately (`_ProgressList` streams each appended record into the checkpoint hook). | `run.py` | `progress list streams every appended record` |
| 2 | **SIGTERM/SIGINT → checkpoint, then exit 143.** A suspend now costs the item in flight, not the rep. | `run.py` | `SIGTERM exits 143`, `SIGTERM path is reachable` |
| 3 | The run holds a **sleep/idle inhibitor** (`systemd-inhibit --what=sleep:idle`) for its whole duration, so the power manager cannot suspend the box mid-benchmark. | `run_llama.sh` | verified live: `systemd-inhibit --list` shows `sleep:idle … block` |
| 4 | `partial` is now **coverage-based**: a (task, rep) pair is complete only if it has as many valid (non-stub) records as items expected. A replaced stub stops counting; a task that never ran is always a gap. | `run.py` `coverage_report()` | 5 coverage unit tests |
| 5 | The resume loop counts **valid records only**; a round that banks nothing usable is BLOCKED, not "progress". | `run_llama.sh` | verified on the live run |
| 6 | Checkpoints now carry `partial` + `coverage_gaps` (they previously wrote neither, so a checkpointed file looked complete to the resume loop). | `run.py` | live run |

## Why the underlying OOM still happens (and why that is fine)

The server's RSS grows ~130 MB per long generation and the kernel OOM-killer
takes it at ~4.9 GB anon-rss (`dmesg`: `Killed process … (llama-server)`).
`nemotron_h` is a hybrid (Mamba-style SSM + attention) 4B with a 262K native
context; on a 7.6 GB box a single server lifetime cannot hold 30 sustained
512-token generations.

This is a **deployment finding about the model on this hardware**, not a
harness defect. The harness response is the correct one:
per-item OOM guard → clean abort with checkpoints → fresh server → resume.
That is the `--until-complete` loop: grind the work in slices the hardware
can survive, never lose a scored item, never present a stub as a score.

## What this review does NOT claim

- It does not claim the memory growth is root-caused. It isn't. It is
  bounded, guarded, observed, and survivable — and the *evidence* is kept.
- It does not claim the previous 5 rounds were "model failures". They were
  infrastructure failures, and the report said so (the cell was missing, not
  zero).
- It does not change any published number. Every other cell is untouched.
