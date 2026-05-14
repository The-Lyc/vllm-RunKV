"""Compare per-layer IO duration across three runs:
  - dryrun   : replay disabled (baseline)
  - runkv    : replay enabled (with HS overhead)
  - tightllm : replay enabled (no HS overhead)

For each layer compute mean io_dur (= load_ready - load_start, ms) over all
non-warmup records. Δ = run.io_dur - dryrun.io_dur. We also bucket records by
replay_ratio to see whether Δ correlates with replay_ratio.

Outputs (under --output-dir):
  per_layer.csv           # one row per layer
  delta_vs_ratio.csv      # mean Δ per replay_ratio bucket
  summary.txt             # human-readable

Defaults point at the 20260512_1114 experiments and the matching dryrun
20260512_111010 file.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(p: Path, warmup: int = 1) -> list[dict]:
    out = []
    for line in p.open():
        r = json.loads(line)
        if r.get("load_start_ms_from_anchor") is None: continue
        if r.get("load_ready_ms_from_anchor") is None: continue
        if r.get("step", 0) < warmup: continue
        if r.get("layer_idx") in (None, 0): continue  # skip embedding/first layer
        out.append({
            "step": r["step"],
            "layer": r["layer_idx"],
            "rr": r.get("replay_ratio") or 0.0,
            "ntok": r.get("num_actual_tokens") or 0,
            "io": r["load_ready_ms_from_anchor"] - r["load_start_ms_from_anchor"],
        })
    return out


def per_layer_mean(recs: list[dict]) -> dict[int, float]:
    g: dict[int, list[float]] = defaultdict(list)
    for r in recs:
        g[r["layer"]].append(r["io"])
    return {L: statistics.fmean(v) for L, v in g.items() if v}


def bucket_by_ratio(recs: list[dict], baseline: dict[int, float], step: float = 0.05):
    """Group by (replay_ratio bucket); within each bucket compute mean delta
    averaged across layers."""
    buckets: dict[float, list[float]] = defaultdict(list)
    counts: dict[float, int] = defaultdict(int)
    ntoks: dict[float, list[int]] = defaultdict(list)
    for r in recs:
        b = round(r["rr"] / step) * step
        base = baseline.get(r["layer"])
        if base is None: continue
        buckets[b].append(r["io"] - base)
        ntoks[b].append(r["ntok"])
        counts[b] += 1
    rows = []
    for b in sorted(buckets):
        rows.append({
            "ratio_bucket": round(b, 3),
            "n": counts[b],
            "mean_delta_ms": statistics.fmean(buckets[b]),
            "mean_ntok": statistics.fmean(ntoks[b]),
        })
    return rows


def write_per_layer_csv(path: Path, layers: list[int],
                        means: dict[str, dict[int, float]]) -> None:
    labels = list(means)  # preserves order
    fieldnames = ["layer"] + [f"io_{l}" for l in labels] + \
                 [f"delta_{l}_vs_dryrun" for l in labels if l != "dryrun"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for L in layers:
            row = {"layer": L}
            for lab in labels:
                row[f"io_{lab}"] = means[lab].get(L)
            base = means["dryrun"].get(L)
            for lab in labels:
                if lab == "dryrun": continue
                v = means[lab].get(L)
                row[f"delta_{lab}_vs_dryrun"] = (v - base) if (v is not None and base is not None) else None
            w.writerow(row)


def write_bucket_csv(path: Path, bucket_tables: dict[str, list[dict]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "ratio_bucket", "n", "mean_delta_ms", "mean_ntok"])
        for lab, rows in bucket_tables.items():
            for r in rows:
                w.writerow([lab, r["ratio_bucket"], r["n"],
                            f"{r['mean_delta_ms']:.3f}", f"{r['mean_ntok']:.0f}"])


def fmt_per_layer(layers, means) -> str:
    labels = list(means)
    head = "  layer   " + "  ".join(f"io_{l:>8s}" for l in labels)
    head += "   " + "  ".join(f"Δ_{l}".ljust(11) for l in labels if l != "dryrun")
    lines = [head]
    for L in layers:
        parts = [f"  {L:>5d}  "]
        for lab in labels:
            v = means[lab].get(L)
            parts.append(f"{v:>11.3f}" if v is not None else "         NaN")
        base = means["dryrun"].get(L)
        for lab in labels:
            if lab == "dryrun": continue
            v = means[lab].get(L)
            d = (v - base) if (v is not None and base is not None) else None
            parts.append(f" {d:+10.3f}" if d is not None else "       NaN")
        lines.append("  ".join(parts))
    return "\n".join(lines)


def fmt_buckets(label: str, rows: list[dict]) -> str:
    lines = [f"  [{label}] Δ io_dur vs dryrun  (bucketed by replay_ratio)",
             "    ratio   n      mean_Δ(ms)   mean_ntok"]
    for r in rows:
        lines.append(f"    {r['ratio_bucket']:>5.2f}  {r['n']:>5d}    {r['mean_delta_ms']:>+9.3f}   {r['mean_ntok']:>9.0f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", default="exp_results/opt_feedback_observation/opt_component_mfu_1000_20260512_111010.flat.jsonl")
    ap.add_argument("--runkv", default="exp_results/opt_feedback_observation/opt_component_mfu_1000_20260512_1114.flat.jsonl")
    ap.add_argument("--tightllm", default="exp_results/tightllm_observation/opt_component_mfu_1000_20260512_1114.flat.jsonl")
    ap.add_argument("--output-dir", default="exp_results/analysis/io_delta/20260512_1114")
    ap.add_argument("--bucket-step", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    recs = {
        "dryrun":   load(Path(args.dryrun), args.warmup_steps),
        "runkv":    load(Path(args.runkv), args.warmup_steps),
        "tightllm": load(Path(args.tightllm), args.warmup_steps),
    }
    means = {lab: per_layer_mean(rs) for lab, rs in recs.items()}
    layers = sorted({L for m in means.values() for L in m})

    base = means["dryrun"]
    buckets = {
        "runkv":    bucket_by_ratio(recs["runkv"], base, args.bucket_step),
        "tightllm": bucket_by_ratio(recs["tightllm"], base, args.bucket_step),
    }

    write_per_layer_csv(out / "per_layer.csv", layers, means)
    write_bucket_csv(out / "delta_vs_ratio.csv", buckets)

    # Overall per-record stats for context
    summary = []
    summary.append("=" * 78)
    summary.append("Per-layer mean io_dur (ms) and Δ vs dryrun")
    summary.append("=" * 78)
    summary.append(f"  dryrun   = {args.dryrun}")
    summary.append(f"  runkv    = {args.runkv}")
    summary.append(f"  tightllm = {args.tightllm}")
    for lab, rs in recs.items():
        ios = [r["io"] for r in rs]
        ntoks = [r["ntok"] for r in rs]
        summary.append(f"  [{lab}] records={len(rs)}  mean_io={statistics.fmean(ios):.3f}ms  "
                       f"mean_ntok={statistics.fmean(ntoks):.0f}")
    summary.append("")
    summary.append(fmt_per_layer(layers, means))
    summary.append("")
    summary.append("=" * 78)
    summary.append("Δ vs dryrun bucketed by replay_ratio")
    summary.append("=" * 78)
    summary.append("  (If Δ correlates negatively with replay_ratio → IO truly drops with replay.)")
    summary.append("  (NOTE: num_actual_tokens grows with replay_ratio, which inflates io_dur,")
    summary.append("         so a flat/positive Δ does NOT necessarily mean replay is useless.)")
    summary.append("")
    summary.append(fmt_buckets("runkv", buckets["runkv"]))
    summary.append("")
    summary.append(fmt_buckets("tightllm", buckets["tightllm"]))
    summary.append("")

    text = "\n".join(summary)
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"\n[done] outputs written to {out}")


if __name__ == "__main__":
    main()
