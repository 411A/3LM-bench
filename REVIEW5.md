# REVIEW5 — the box was running at half power, and nobody was measuring it (2026-10-20)

### The finding

While building the dedicated throughput instrument, the box refused to go
quiet. The reason was not the benchmark:

```
PID   %CPU  COMMAND
1074583 100.2  glycin-svg        <- 28h03m of CPU time, 2 days uptime
792      14.3  Xorg
647908   14.3  opencode
```

`glycin-svg` (the desktop thumbnailer) had been pegging **one of the box's two
cores at 100%** for two days. Every measurement in this project before
16:05 was taken with roughly half the machine gone.

### What it did and did not affect

| Quantity | Affected? | Why |
|----------|-----------|-----|
| Accuracy, Wilson CIs, item-level outcomes | **No** | The model still produced the same answers. The determinism check is the proof: 30/30 byte-identical completions for the same 30 GSM8K items, measured in different rounds, under different load conditions, across two separate server lifetimes. |
| Throughput (tok/s) | **Yes — systematically depressed** | The decode rate is a direct function of available CPU. nemotron measures **3.83 tok/s isolated** vs **2.50 tok/s** in-run on the same box: a ~35% penalty. |
| Wall-clock per item / timeouts | Marginally | Requests took ~15-30 s against a 900 s budget; no request was ever close to the limit. |
| Memory behaviour | No | The OOM kills were a different, separately-investigated defect (REVIEW4). |

So every *capability* number in the project stands. Every *speed* number that
was taken as a by-product of a long run was conservative (pessimistic) and is
no longer published as a speed measurement.

### The fixes

1. **Preflight now samples live CPU** (`cpu_busy_pct` in `run_llama.sh`).
   The load average was useless here: it is a lagging indicator (it read
   2.2 for minutes after the box went idle) and it cannot see a single
   runaway thread hiding inside an otherwise-idle box. The preflight now
   measures the fraction of the whole box that is busy *before* the model is
   loaded - any CPU in use at that moment is foreign CPU - and waits (up to
   3 minutes) instead of publishing a number taken on a busy machine. The
   observed value is written into the run log.
2. **`speed.py` is now a dedicated instrument.** One model, fresh server,
   two warm-up prompts discarded, then the **median** of 12 identical
   prompts. min/p25/p75/max are kept so the spread is visible, and the
   foreign-CPU percentage at measurement time is recorded with the result.
3. **The report publishes only isolated throughput.** The efficiency chart
   now takes its x-axis from `results/llamacpp/<label>-speed.json`; the
   in-run `mean_gen_tps` is kept in the data but is no longer the published
   number, and the axis is labelled "decode tok/s (isolated pass)".
4. **The runaway is dead.** `glycin-svg` was killed; the desktop stack
   (Xorg + terminal + this agent) costs a standing ~22% of a 2-core box,
   which is recorded with every speed measurement so the number is
   auditable rather than merely claimed.

### Why the speed number needs its own instrument at all

A tok/s taken as a by-product of a long benchmark run inherits everything
that run inherited: memory pressure (the server grows ~130 MB per long
generation, so a cell measured late in a run runs against a more fragmented
heap), the cold first token of a fresh server, and any foreign load. The
data shows the first two are small (the chopped run's first item measured
2.94 tok/s against a 2.91 median - no penalty) and the third was huge and
invisible. Hence: measure speed where speed is the question.

### What this review does NOT claim

- It does not claim the earlier capability runs need redoing. They do not.
- It does not claim the box will stay quiet. The preflight now *detects* a
  busy box instead of silently measuring on one, and the number records the
  foreign load it was taken under.
- It does not claim the in-run tok/s values are wrong; they are pessimistic
  measurements of a busier box, and they stay in the result files as such.
