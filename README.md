# Packrat on a 12-core desktop

A reproduction of the BERT experiments from [Packrat](https://github.com/msr-fiddle/packrat)
(Bhardwaj et al., ICML 2025), on a desktop instead of a server. Packrat's claim: on CPUs,
splitting the cores across several model instances with fewer threads each can serve a batch
with lower latency than one instance using all the threads.

**Result:** the claim holds here, with a smaller gain. At batch 32, four 2-thread instances beat
one 8-thread instance by 15% in latency (23% in throughput), in all three repeats. The fully
thin split (eight 1-thread instances) never wins.

## Setup

| | |
|---|---|
| CPU | Ryzen 9 3900X, 12 cores / 24 threads, one thread pinned per physical core |
| Memory | 16 GB DDR4 at 2133 MT/s, dual channel, XMP off |
| OS | Windows 11, WSL2, Ubuntu 22.04 |
| Software | Python 3.10, torch 1.12.1 (CPU), transformers 4.26.0: Packrat's pinned versions |
| Model | BERT-base, eager mode (see finding 4), the benchmark's 16-token input |
| Measurement | 10 warmup + 100 timed iterations per config, 3 repeats, mean with range |

## Experiment 1a: one instance, 1 to 12 threads

Packrat's premise: more threads give diminishing returns. Mean latency, ms, 3 repeats.

| Batch | 1 thread | 4 | 8 | 12 | Speedup at 8 | at 12 |
|---|---|---|---|---|---|---|
| 1 | 127 | 45 | 31 | 33 | 4.1× | 3.9× |
| 8 | 672 | 198 | 131 | 139 | 5.1× | 4.8× |
| 32 | 2532 | 734 | 473 | 533 | 5.4× | 4.7× |

8 threads buy 4 to 5.4×, not 8×. Threads 9 to 12 add nothing; at batch 32 they are 13% slower.

## Experiment 1b: one 8-thread instance vs eight 1-thread instances

Packrat's `thin` mode (`run.py --instance_id 3`). Latency of a split = its slowest instance,
since the batch is done when its last piece is. Mean of 3 repeats, ms.

| Batch | 1×8 | 8×1 | 8×1 ÷ 1×8 |
|---|---|---|---|
| 8 | 172 | 245 | 1.42 |
| 16 | 279 | 337 | 1.21 |
| 32 | 544 | 549 | 1.01 |
| 64 | 964 | 985 | 1.02 |

Fully thin loses at small batches and ties at larger ones (also a tie at 128 to 512, 1 to 2 repeats).

## Experiment 1c: all splits of 8 cores

`scripts/splits.py`: a small harness that starts N instances × T threads, pins each instance to
its own cores, warms up, then releases every instance through a barrier so the timed loops start
together. Its 1×8 agrees with Packrat's own harness within 3% in the same session. Mean of 3
repeats, ms, with the per-repeat range for the winning split.

| Batch | 1×8 | 2×4 | 4×2 | 8×1 | Best split vs 1×8 |
|---|---|---|---|---|---|
| 8 | 158 | 146 | 165 | 216 | none (2×4: 0.76 to 1.04) |
| 16 | 268 | 268 | 253 | 329 | none (4×2: 0.89 to 1.00) |
| 32 | 562 | 506 | **480** | 513 | **4×2, 15% faster, every repeat (0.82 to 0.87)** |
| 64 | 956 | 975 | 888 | 1010 | 4×2, 7% faster, every repeat (0.91 to 0.95), within noise |

Throughput at batch 32: 4×2 serves 70 samples/s against 57 for 1×8.

Both extremes lose, the middle wins, which is the paper's mechanism. The paper reports 1.43 to 1.83×
averaged across batch sizes on servers; here the best batch gives 1.18×.

## Prediction, written before the runs

`results/predictions.md`. Called before 1b: fat wins or ties at batch 8 (right); batch 32 close
(right in direction, the absolute range missed because the whole machine ran about 15% slower that
day). No call was made on the 2×4 and 4×2 splits, so the 4×2 win was not predicted.

## Findings in the Packrat tool

1. `benchmarks/bench.py` line 110 writes `[min, max, avg]` under the header `min, avg, max`:
   the "avg" column holds the tail.
2. On this machine `utils/topology.py` allocates CPUs 0 to 11, which are 6 physical cores with their
   hyperthread siblings (WSL numbers siblings adjacently). Twelve threads ran on six cores until
   overridden (`scripts/packrat-local-changes.patch`).
3. The `interleave` mapping takes `cores[-0:]` on a one-socket machine, which is the whole list, so
   every core appears twice.
4. `benchmarks/bert.py` compares the `Optimizations` enum to the string `"script"`, so the
   TorchScript trace never runs and results labeled `script` are eager.

## Caveats

- WSL2: no control over turbo, SMT or clock; the L3 cluster layout is hidden. With turbo on, the
  1-thread baseline runs at a higher clock than 8 threads, which makes speedups look worse than at a
  fixed clock.
- Daytime run-to-run noise of about 10% on this machine; differences under that are not findings.
- In Experiment 1b the eight instances start their timed loops at different times, which favors thin;
  thin still did not win. Experiment 1c fixes this with a barrier.
- One 16-token input. Short inputs mean small matrix multiplies and little weight reuse per read.
- The explanation for the smaller gain (two desktop memory channels feeding several copies of the
  weights) is a hypothesis, not a measurement.

## Next

1. The 2133 vs 3200 MT/s test: one BIOS toggle, same grid. If the gap between splits changes with
   memory speed, memory bandwidth is doing the work.
2. The same grid on all 12 cores (1×12, 2×6, 3×4, 4×3, 6×2, 12×1).
3. LLM decode on a small model: does the multi-instance advantage survive one-token-at-a-time
   generation?

## Rerun
Packrat’s harness (their repo, with scripts/packrat-local-changes.patch applied)

python run.py --benchmark bert --instance_id 1 # Experiment 1a
python run.py --benchmark bert --instance_id 3 # Experiment 1b

this repo

python scripts/splits.py --out results/splits.jsonl # Experiment 1c, about 45 minutes
python scripts/report_splits.py results/splits.jsonl

