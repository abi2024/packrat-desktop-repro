#!/usr/bin/env python3
"""One BERT batch split across N instances x T threads, each instance pinned to its own cores.

Same input text as packrat/benchmarks/bert.py (16 tokens). Every instance warms up, then all
instances pass a barrier so the timed loops start together. Each instance's per-iteration
latencies are kept, so the report can give p50/p95/p99, not only the mean.

Usage (from ~/packrat, with packrat-env active):
  python splits.py --splits 1x8,2x4,4x2,8x1 --batches 8,16,32,64 --repeats 3 --out results/splits.jsonl
Cores default to 0,2,4,...,22: one CPU per physical core on this WSL box (see lscpu -e).
"""
import argparse
import json
import os
import time
from multiprocessing import Barrier, Process, Queue, set_start_method

TEXT = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"


def worker(inst, cores, threads, batch, warmup, iters, barrier, q):
    os.sched_setaffinity(0, set(cores))
    import torch
    from transformers import BertModel, BertTokenizer, logging

    logging.set_verbosity_error()
    torch.set_num_interop_threads(1)
    torch.set_num_threads(threads)
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    enc = tok([TEXT] * batch, padding=True, return_tensors="pt", return_token_type_ids=True)
    model = BertModel.from_pretrained("bert-base-uncased").eval()
    lat = []
    with torch.no_grad():
        for _ in range(warmup):
            model(enc["input_ids"], token_type_ids=enc["token_type_ids"])
        barrier.wait()
        for _ in range(iters):
            t0 = time.perf_counter()
            model(enc["input_ids"], token_type_ids=enc["token_type_ids"])
            lat.append((time.perf_counter() - t0) * 1000)
    q.put({"instance": inst, "cores": cores, "threads": threads, "batch": batch, "latencies_ms": lat})


def run_config(cores, n, t, batch, warmup, iters):
    assert n * t <= len(cores), f"{n}x{t} needs {n * t} cores, only {len(cores)} given"
    assert batch % n == 0, f"batch {batch} does not split evenly across {n} instances"
    barrier, q = Barrier(n), Queue()
    procs = [
        Process(target=worker, args=(i, cores[i * t:(i + 1) * t], t, batch // n, warmup, iters, barrier, q))
        for i in range(n)
    ]
    for p in procs:
        p.start()
    rows = [q.get() for _ in procs]
    for p in procs:
        p.join()
    return sorted(rows, key=lambda r: r["instance"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", default=",".join(str(c) for c in range(0, 24, 2)))
    ap.add_argument("--splits", default="1x8,2x4,4x2,8x1", help="instances x threads, comma-separated")
    ap.add_argument("--batches", default="8,16,32,64")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default="results/splits.jsonl")
    a = ap.parse_args()

    cores = [int(c) for c in a.cores.split(",")]
    splits = [tuple(int(x) for x in s.split("x")) for s in a.splits.split(",")]
    batches = [int(b) for b in a.batches.split(",")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    for r in range(1, a.repeats + 1):
        for B in batches:
            for n, t in splits:
                t0 = time.time()
                rows = run_config(cores, n, t, B, a.warmup, a.iters)
                rec = {"repeat": r, "batch": B, "instances": n, "threads": t, "cores": cores[: n * t],
               "warmup": a.warmup, "iters": a.iters, "wall_s": round(time.time() - t0, 1),
               "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                slowest = max(sum(x["latencies_ms"]) / len(x["latencies_ms"]) for x in rows)
                print(f"r{r} batch {B:>3} {n}x{t}: slowest-instance mean {slowest:7.1f} ms  ({rec['wall_s']}s)", flush=True)


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    main()
