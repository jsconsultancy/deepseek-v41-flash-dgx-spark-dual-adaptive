"""Per-graph kernel-family sums from a bench_round_breakdown_tp.py trace (rank0.json).

The 2026-10-07 TensorFold comparison needed exactly this: the engine's coarse timers cover a whole
step, but the draft pass is its own CUDA graph with a `graph id`, and the question (what does the
drafter read, and where is the next millisecond) is only answerable per graph.

    python3 tools/decode_graph_families.py results/<run>/rank0.json [--draft <graph id>] \\
        [--steps 12] [--engine ours]

Family names follow results/round-breakdown-20261007/analyze.py so the numbers line up with it.
Graph ids are capture-order dependent: print them all with `--list` and pick the draft graph by its
replay count (one per draft pass, i.e. one per measured burst) and its kernel shapes.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics


def family(name: str, engine: str) -> str:
    if engine == "ours":
        if "moe_up" in name or "moe_down" in name:
            return "expert_projection"
        if "_fp8_linear" in name or "_fp8_grouped" in name:
            return "fp8_dense_projection"
        if "wmma_tensorop_bf16" in name:
            return "bf16_matmul"
        if "sgemm" in name or "gemmSN" in name:
            return "fp32_matmul"
        if "exl3m_" in name:
            return "exl3_dense_projection"
    else:
        if "dsv41_x3ld::ld_kernel" in name:
            return "expert_projection"
        if "dsv41_dense3::lanes_linear" in name or "::linear_kernel<" in name:
            return "exl3_dense_projection"
        if "dsv41_mhc::boundary" in name:
            return "mhc_boundary"
    if "roce::" in name or "tfroce::" in name or "nccl" in name.lower():
        return "communication"
    if any(x in name for x in ("prefetch", "l2pace", "tfl2pf", "_touch")):
        return "prefetch"
    return "other"


def graph_events(events):
    """graph id -> its kernel events, plus each id's replay count."""
    groups = collections.defaultdict(list)
    for e in events:
        gid = (e.get("args") or {}).get("graph id")
        if gid is not None:
            groups[gid].append(e)
    return groups


def summarize(ev, engine: str):
    nodes = collections.defaultdict(list)
    for e in ev:
        nodes[(e["args"].get("graph node id"), e["name"])].append(e)
    counts = {len(v) for v in nodes.values()}
    replays = min(counts) if counts else 0
    if len(counts) != 1:
        # A graph with a data-dependent trip count still replays whole; report the common case and
        # say so rather than pretending the average is a replay count.
        replays = statistics.mode(counts) if counts else 0
    fam = collections.Counter()
    for e in ev:
        fam[family(e["name"], engine)] += e["dur"] / 1000
    spans = []
    by_node = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in ev:
        by_node[e["args"].get("graph node id")][e["name"]].append(e)
    n = replays or 1
    for i in range(n):
        starts, ends = [], []
        for per_name in by_node.values():
            for lst in per_name.values():
                if i < len(lst):
                    starts.append(lst[i]["ts"])
                    ends.append(lst[i]["ts"] + lst[i]["dur"])
        if starts:
            spans.append((max(ends) - min(starts)) / 1000)
    return {"replays": replays, "nodes": len(nodes), "span_ms": statistics.mean(spans) if spans else float("nan"),
            "families_per_pass": {k: v / n for k, v in fam.items()},
            "top": collections.Counter({e["name"]: e["dur"] for e in ev}).most_common(8),
            "top_counts": collections.Counter(e["name"] for e in ev)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--draft", type=int, help="graph id of the draft pass; omit to list them")
    ap.add_argument("--steps", type=int, default=12, help="measured verification windows")
    ap.add_argument("--engine", default="ours", choices=("ours", "tensorfold"))
    a = ap.parse_args()
    trace = json.loads(open(a.trace).read())
    events = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    groups = graph_events(events)
    info = {gid: summarize(ev, a.engine) for gid, ev in groups.items()}
    if a.draft is None:
        print(f"{len(events)} kernel events, {len(groups)} graphs")
        for gid in sorted(groups, key=lambda g: -info[g]["replays"]):
            s = info[gid]
            print(f"  graph {gid}: {s['replays']:4d} replays, {s['nodes']:5d} nodes, "
                  f"span {s['span_ms']:8.3f} ms, " +
                  ", ".join(f"{k} {v:.2f}" for k, v in sorted(s["families_per_pass"].items())))
        return 0
    s = info[a.draft]
    print(f"draft graph {a.draft}: {s['replays']} replays, {s['nodes']} nodes, span {s['span_ms']:.3f} ms/pass")
    print("families ms/pass:")
    for k, v in sorted(s["families_per_pass"].items(), key=lambda kv: -kv[1]):
        print(f"  {v:8.3f}  {k}")
    print(f"kernel-sum total {sum(s['families_per_pass'].values()):.3f} ms/pass")
    print("top kernels (total ms over the trace, calls):")
    for name, ms in s["top"]:
        print(f"  {ms:8.2f}  {s['top_counts'][name]:4d}x  {name[:88]}")
    if a.steps:
        main = [g for g in info if g != a.draft and info[g]["replays"] > 0]
        tot = sum(info[g]["span_ms"] * info[g]["replays"] for g in main) / a.steps
        print(f"\nnon-draft graphs, {a.steps} windows: {tot:.2f} ms/window "
              f"(over {len(main)} graphs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
