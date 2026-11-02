#!/usr/bin/env python3
"""Table from results/splits.jsonl.

Latency of a config = the slowest instance's mean per-iteration latency (the batch is done when
its last piece is). p99 = the slowest instance's 99th percentile. Throughput = sum over instances
of samples per second. Ratio = latency / latency of the 1xT split at the same batch (above 1 = worse than fat).
"""
import json
import sys

import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "results/splits.jsonl"
recs = [json.loads(l) for l in open(path)]

out = []
for rec in recs:
    inst_means = [np.mean(x["latencies_ms"]) for x in rec["rows"]]
    worst = rec["rows"][int(np.argmax(inst_means))]
    thr = sum(x["batch"] * 1000.0 / np.mean(x["latencies_ms"]) for x in rec["rows"])
    out.append({"batch": rec["batch"], "split": f'{rec["instances"]}x{rec["threads"]}', "repeat": rec["repeat"],
                "lat_ms": max(inst_means), "p99_ms": np.percentile(worst["latencies_ms"], 99), "thr_sps": thr})
d = pd.DataFrame(out)

summary = d.groupby(["batch", "split"]).agg(lat_ms=("lat_ms", "mean"), lat_min=("lat_ms", "min"), lat_max=("lat_ms", "max"),
                                            p99_ms=("p99_ms", "mean"), thr_sps=("thr_sps", "mean"), n=("repeat", "count"))
fat = summary.xs(summary.reset_index().split[summary.reset_index().split.str.startswith("1x")].iloc[0], level="split")["lat_ms"]
summary["vs_fat"] = summary["lat_ms"] / summary.index.get_level_values("batch").map(fat)
pd.set_option("display.width", 160)
print(summary.round(2))
print("\nPer repeat, latency ms:")
print(d.pivot_table(index=["batch", "split"], columns="repeat", values="lat_ms").round(1))
