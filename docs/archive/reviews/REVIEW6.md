# REVIEW6 — the benchmark was eating 3.2 GB of RAM in its own sandbox (2026-10-21)

### The symptom

A question that had been open all day: *"Memory usage is not very high but
swap is! Why?"*

```
MemTotal 7.99 GB   MemFree 226 MB   MemAvailable 4.89 GB
SwapTotal 7.81 GB  SwapFree 3.04 GB  →  4.78 GB "in use"
SwapCached 22 MB
vmstat si/so:  0/0  0/0  0/0        ← no page moved in or out
```

No swapping was happening. The swap was a *residue* - but the residue had a
cause, and the cause was the harness.

### The cause

`/tmp` on this box is a **3.8 GB tmpfs**: it is RAM, not disk.

```python
# scorers.py (before)
TMP = tempfile.mkdtemp(prefix="llmbench-")
```

The code sandbox compiled every candidate program **into that directory** and
never deleted anything: the C++ path left a ~4.3 MB binary per item, the
Rust path another, the Python path a source file. After a week of runs:

```
$ ls -d /tmp/llmbench-* | wc -l
629
$ du -sh /tmp/llmbench-yc19yonx
107M        # 25 binaries x 4.3 MB
```

**629 leftover directories ≈ 3.2 GB of RAM**, held by the kernel as tmpfs
pages. That is what:

- made `MemFree` look tiny (226 MB) while `MemAvailable` said 4.9 GB;
- forced the kernel to swap out real RAM to back the tmpfs (the 4.78 GB);
- added ~130 MB of memory pressure per code item, feeding the OOM kills;
- and made every throughput number a number measured on a half-starved box.

The `si/so` counters were 0 because the swap had nothing left to do: the
pages were stale, and Linux does not pull swapped pages back in while RAM
looks plentiful.

### The fix

1. **The sandbox lives on disk**, never in `/tmp`:
   `~/.llmbench-tmp` (override with `LLMBENCH_TMP`). 87 GB free on /home.
2. **Every artifact is deleted after it has been run** - the compiled
   binary, the source file, on the success path, the compile-error path and
   the timeout path.
3. **Stale directories are swept** at start-up (older than 6 h), so a
   crashed run cannot leak.
4. **Two unit contracts lock it in**: the sandbox must be off `/tmp`, and it
   must be empty after a run.

### The effect (measured immediately after the cleanup)

| | Before | After |
|---|---|---|
| `/tmp` used | 3.9 GB (100%) | 443 MB (12%) |
| `MemFree` | 226 MB | 5.55 GB |
| Swap used | 4.78 GB | **1.33 GB** (the kernel pulled the pages back) |
| Leftover sandbox dirs | 629 | 0 |

### What this review does NOT claim

- It does not claim the memory *growth* of `llama-server` (~130 MB per long
  generation) is explained - that one is still unexplained and still
  bounded by the per-item OOM guard.
- It does not claim the earlier numbers were wrong. Capability numbers are
  unaffected (determinism was verified under exactly these conditions);
  throughput numbers were *pessimistic*, i.e. measured on a box that was
  3.2 GB short of RAM.
- It does not claim /tmp was a bad choice by the operating system. The
  mistake was ours: writing multi-megabyte build artifacts into a RAM-backed
  directory and never cleaning up.
