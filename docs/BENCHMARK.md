# Published benchmark protocol

## Recorded experiment

Start: 2026-10-01 09:00 Asia/Shanghai. Completed: 2026-10-02 08:32 Asia/Shanghai.
Two workloads, three implementations each, 50,000 unique formal shots each:
100,000 distinct physical samples and 300,000 measured decoder evaluations.
Inputs include 64 additional held-out warmup shots at indices 0–63.
Formal shot index 0 corresponds to exported input index 64.

Both selected physical error rates are **p=0.001**, SI1000, X memory. BB uses
18 rounds; superdense Color uses 19. The initial p=.002 pilot and predeclared
.001/.003 alternatives were selected on accuracy/flag criteria, not speed.
The final comparison fixes beam=5, pqlimit=100000 and one identity detector
order for all implementations. These settings differ from the original paper's
ensemble benchmark settings; this release compares implementations at the same
configuration rather than claiming the paper's reported decoding accuracy.

## Source and hardware

The original baseline uses original_tesseract.cc and tesseract.h extracted
verbatim from the archived upstream source. The production CPU and GPU source
hashes, frozen binary identities and package versions are recorded in
[provenance.json](../benchmarks/results/2026-10/provenance.json).
That file is historical provenance: it also lists development files and binary
hashes that are not distributed as executables in the source release.

CPU: AMD Ryzen 7 4800HS, one core; GPU: GTX 1660 Ti, 6 GB with one host core.
CPU workers were compiled in Ubuntu 24.04/WSL2; GPU Python ran natively on
Windows 11. CUDA kernels use precise division and disable fused multiply-add.
Each GPU resident workspace has 100000 entries per shot; BB uses CTA640/chunk64,
Color uses wide-state CTA256/chunk256. No formal shot needed CPU replay.

## Timing and outcomes

The primary metric is **average seconds per shot**: the sum of successful
1,000-shot block wall times divided by 50,000. CPU IPC/result serialization,
GPU upload, kernel execution, synchronization, gather and live fallback are
inside the timing interval. Graph construction, NVRTC compilation, workspace
allocation, 64-shot warmup, checkpoint IO and accuracy calculation are outside.
Setup/warmup wall times are reported separately. This is batch throughput, not
a per-shot real-time latency distribution.

A shot is a logical failure if any predicted logical observable differs from
the sampled truth. All generated formal shots enter the denominator, including
empty syndromes and flagged low-confidence results. BB has 12 observables,
Color has one. Consequently the two codes' LERs are separate workload outcomes.

All three versions have exactly matching per-shot success/failure indicators:
BB 20/50000, Color 1264/50000. This experiment records aggregate and per-shot
logical outcomes; it does not repeat the prior complete correction/cost-bit
comparison gate on every formal shot. All figures describe observed results.

Vertical figure error bars are Wilson 95% binomial intervals. The 50,000 shots
are unique; they are not repetitions of a small batch. Timing is one measured
pass per configuration and should not be interpreted as a repeated-run timing
confidence interval.

The BB GPU run resumed after a buffer-capacity failure at 9000 shots. A host
allocation bound was corrected to include all possible chunk origins. No
kernel, algorithm, input or search parameter changed. Successful blocks were
retained; the interrupted block reran. Downtime and failed-attempt overhead are
excluded from the main throughput figure. See
[repair_history.json](../benchmarks/results/2026-10/repair_history.json).

## Evidence layout

- `benchmarks/results/2026-10/summary.json`: table and figure source.
- `{bb,color}_blocks.json`: 50 block records per implementation with counts, wall times, failure/flag totals and backend statistics; duplicated per-shot arrays are omitted.
- `{bb,color}_shot_decisions.npz`: compact per-shot failure, flag and predicted-logical-mask arrays for all implementations.
- `benchmarks/circuits/{bb,color}.stim`: the two actual formal-run circuits.
- `benchmarks/inputs/manifest.json`: original array hashes, seeds and dimensions.

Full sampled input arrays are intentionally not distributed, either in Git or as Release attachments. The runner regenerates samples from the supplied circuits and recorded seeds. Use the same generated input directory for all three implementations. Exact historical input bytes and outcome counts are not guaranteed across sampler versions. Original array hashes remain as provenance. Arrays are little-endian as exported on x86-64.

The source runner adapts the measured experiment for Linux and Windows/WSL. CPU decoder source and GPU kernel modules are unchanged; host paths have been reorganized. Historical scheduling and unrelated development experiments are omitted. Verification and plotting never decode.
