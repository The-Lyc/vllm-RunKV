#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RunKV vs Vanilla vLLM 结果分析与可视化脚本
用于生成论文级别的图表和表格

Tips:
  - 默认会加载 results-dir 下所有时间戳的结果（可能混在一起）
  - 使用 --timestamp YYYYMMDD_HHMMSS 或 --select latest-per-type/latest-common-per-type
    可自动选取最新的一组（或每个测试类型的最新一组）结果进行分析
"""

import argparse
import glob
import importlib.util
import json
import os
import statistics
from collections import defaultdict

import regex as re

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None
    print("Warning: numpy not installed, using basic statistics")

# 尝试导入可选的绘图库
try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping plot generation")

HAS_PANDAS = importlib.util.find_spec("pandas") is not None
if not HAS_PANDAS:
    print("Warning: pandas not installed, using basic tables")


_TS_RE = re.compile(r"_(\d{8}_\d{6})\.json$")


def _parse_result_filename(filename: str):
    """Parse result filename into (bench_type, tag, config, timestamp).

    Expected patterns (examples):
      - throughput_vanilla_in128_out128_20260209_101859.json
      - latency_runkv_bs4_in512_20260204_104119.json
      - serving_vanilla_qps8_20260204_104119.json
      - sharegpt_vanilla_20260204_104119.json
    """
    m = _TS_RE.search(filename)
    if not m:
        return None
    timestamp = m.group(1)
    base = filename[: m.start()]  # strip "_{timestamp}.json"
    parts = base.split("_")
    if len(parts) < 2:
        return None
    bench_type, tag = parts[0], parts[1]
    config = "_".join(parts[2:]) if len(parts) > 2 else ""
    return bench_type, tag, config, timestamp


def _iter_result_files(results_dir: str):
    for json_file in glob.glob(f"{results_dir}/*.json"):
        filename = os.path.basename(json_file)
        if "summary" in filename:
            continue
        parsed = _parse_result_filename(filename)
        if not parsed:
            continue
        bench_type, tag, config, timestamp = parsed
        if tag not in ("vanilla", "runkv"):
            continue
        yield json_file, filename, bench_type, tag, config, timestamp


def _select_timestamps(files, selection: str):
    """Return a set of (bench_type, tag, timestamp) tuples to include."""
    if selection == "all":
        return None

    # Group timestamps by bench_type and tag.
    ts_by_type_tag = defaultdict(set)
    for _, _, bench_type, tag, _, timestamp in files:
        ts_by_type_tag[(bench_type, tag)].add(timestamp)

    if selection == "latest-per-type":
        selected = set()
        for (bench_type, tag), timestamps in ts_by_type_tag.items():
            if timestamps:
                selected.add((bench_type, tag, max(timestamps)))
        return selected

    if selection == "latest-common-per-type":
        # For each benchmark type, prefer the latest timestamp that exists for
        # both tags.
        selected = set()
        types = sorted({bench_type for (bench_type, _) in ts_by_type_tag})
        for bench_type in types:
            v_ts = ts_by_type_tag.get((bench_type, "vanilla"), set())
            r_ts = ts_by_type_tag.get((bench_type, "runkv"), set())
            common = v_ts & r_ts
            if common:
                chosen = max(common)
                selected.add((bench_type, "vanilla", chosen))
                selected.add((bench_type, "runkv", chosen))
            else:
                # Fallback: latest per tag for this type (keeps partial runs usable).
                if v_ts:
                    selected.add((bench_type, "vanilla", max(v_ts)))
                if r_ts:
                    selected.add((bench_type, "runkv", max(r_ts)))
        return selected

    raise ValueError(f"Unknown selection: {selection}")


def load_results(
    results_dir: str, *, timestamp: str | None = None, selection: str = "all"
) -> dict:
    """加载benchmark结果（可按时间戳筛选）"""
    results = {"vanilla": defaultdict(list), "runkv": defaultdict(list)}

    all_files = list(_iter_result_files(results_dir))
    include_tuples = _select_timestamps(all_files, selection=selection)

    for json_file, filename, bench_type, tag, config, file_timestamp in all_files:
        if timestamp is not None and file_timestamp != timestamp:
            continue
        if (
            include_tuples is not None
            and (bench_type, tag, file_timestamp) not in include_tuples
        ):
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            # Skip error markers written by the benchmark runner script.
            if isinstance(data, dict) and (
                data.get("_status") in ("error", "skipped")
                or data.get("status") in ("error", "skipped")
            ):
                continue

            # 解析benchmark类型和配置
            if bench_type == "throughput":
                results[tag]["throughput"].append(
                    {
                        "file": filename,
                        "timestamp": file_timestamp,
                        "config": config,
                        "tokens_per_second": data.get("tokens_per_second", 0),
                        "requests_per_second": data.get("requests_per_second", 0),
                        "elapsed_time": data.get("elapsed_time", 0),
                        "num_requests": data.get("num_requests", 0),
                        "total_num_tokens": data.get("total_num_tokens", 0),
                    }
                )
            elif bench_type == "latency":
                results[tag]["latency"].append(
                    {
                        "file": filename,
                        "timestamp": file_timestamp,
                        "config": config,
                        "avg_latency": data.get("avg_latency", 0),
                        "latencies": data.get("latencies", []),
                        "percentiles": data.get("percentiles", {}),
                    }
                )
            elif bench_type == "serving":
                results[tag]["serving"].append(
                    {
                        "file": filename,
                        "timestamp": file_timestamp,
                        "config": config,
                        "data": data,
                    }
                )

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def compute_speedup(results: dict) -> dict:
    """计算RunKV相对于Vanilla的加速比"""
    speedup = {}

    # 吞吐量加速比
    if results["vanilla"]["throughput"] and results["runkv"]["throughput"]:
        vanilla_tps = statistics.mean(
            [r["tokens_per_second"] for r in results["vanilla"]["throughput"]]
        )
        runkv_tps = statistics.mean(
            [r["tokens_per_second"] for r in results["runkv"]["throughput"]]
        )
        speedup["throughput_speedup"] = (
            runkv_tps / vanilla_tps if vanilla_tps > 0 else 0
        )
        speedup["vanilla_tps"] = vanilla_tps
        speedup["runkv_tps"] = runkv_tps

    # 延迟改进
    if results["vanilla"]["latency"] and results["runkv"]["latency"]:
        vanilla_lat = statistics.mean(
            [r["avg_latency"] for r in results["vanilla"]["latency"]]
        )
        runkv_lat = statistics.mean(
            [r["avg_latency"] for r in results["runkv"]["latency"]]
        )
        speedup["latency_improvement"] = vanilla_lat / runkv_lat if runkv_lat > 0 else 0
        speedup["vanilla_latency"] = vanilla_lat
        speedup["runkv_latency"] = runkv_lat

    return speedup


def generate_throughput_table(results: dict) -> str:
    """生成吞吐量对比表格 (LaTeX格式)"""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Throughput Comparison: RunKV vs Vanilla vLLM}")
    lines.append(r"\label{tab:throughput}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Vanilla (tok/s) & RunKV (tok/s) & Speedup \\")
    lines.append(r"\midrule")

    vanilla_by_config = {
        r.get("config", ""): r["tokens_per_second"]
        for r in results["vanilla"]["throughput"]
    }
    runkv_by_config = {
        r.get("config", ""): r["tokens_per_second"]
        for r in results["runkv"]["throughput"]
    }

    all_configs = sorted(
        {c for c in vanilla_by_config if c} | {c for c in runkv_by_config if c}
    )
    for config in all_configs:
        vanilla_tps = vanilla_by_config.get(config, 0)
        runkv_tps = runkv_by_config.get(config, 0)
        speedup = runkv_tps / vanilla_tps if vanilla_tps > 0 else 0
        lines.append(
            f"{config} & {vanilla_tps:.1f} & {runkv_tps:.1f} & {speedup:.2f}x \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_latency_table(results: dict) -> str:
    """生成延迟对比表格 (LaTeX格式)"""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Latency Comparison: RunKV vs Vanilla vLLM}")
    lines.append(r"\label{tab:latency}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Configuration & \multicolumn{2}{c}{Avg Latency (s)} & "
        r"\multicolumn{2}{c}{P99 Latency (s)} & Improvement \\"
    )
    lines.append(r" & Vanilla & RunKV & Vanilla & RunKV & \\")
    lines.append(r"\midrule")

    vanilla_by_config = {
        r.get("config", ""): {
            "avg": r["avg_latency"],
            "p99": r["percentiles"].get(99, r["percentiles"].get("99", 0)),
        }
        for r in results["vanilla"]["latency"]
    }
    runkv_by_config = {
        r.get("config", ""): {
            "avg": r["avg_latency"],
            "p99": r["percentiles"].get(99, r["percentiles"].get("99", 0)),
        }
        for r in results["runkv"]["latency"]
    }

    all_configs = sorted(
        {c for c in vanilla_by_config if c} | {c for c in runkv_by_config if c}
    )
    for config in all_configs:
        v = vanilla_by_config.get(config, {"avg": 0, "p99": 0})
        r = runkv_by_config.get(config, {"avg": 0, "p99": 0})
        improvement = v["avg"] / r["avg"] if r["avg"] > 0 else 0
        lines.append(
            f"{config} & {v['avg']:.3f} & {r['avg']:.3f} & "
            f"{v['p99']:.3f} & {r['p99']:.3f} & {improvement:.2f}x \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def plot_throughput_comparison(results: dict, output_dir: str):
    """绘制吞吐量对比图"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    vanilla_by_config = {
        r.get("config", ""): r["tokens_per_second"]
        for r in results["vanilla"]["throughput"]
    }
    runkv_by_config = {
        r.get("config", ""): r["tokens_per_second"]
        for r in results["runkv"]["throughput"]
    }
    configs = sorted(
        {c for c in vanilla_by_config if c} | {c for c in runkv_by_config if c}
    )

    if not configs:
        return

    vanilla_tps = [vanilla_by_config.get(c, 0) for c in configs]
    runkv_tps = [runkv_by_config.get(c, 0) for c in configs]

    x = np.arange(len(configs)) if HAS_NUMPY else list(range(len(configs)))
    width = 0.35

    x1 = (x - width / 2) if HAS_NUMPY else [xi - width / 2 for xi in x]
    x2 = (x + width / 2) if HAS_NUMPY else [xi + width / 2 for xi in x]
    ax.bar(
        x1, vanilla_tps[: len(configs)], width, label="Vanilla vLLM", color="steelblue"
    )
    ax.bar(x2, runkv_tps[: len(configs)], width, label="RunKV", color="darkorange")

    ax.set_ylabel("Tokens per Second")
    ax.set_title("Throughput Comparison: RunKV vs Vanilla vLLM")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300)
    plt.savefig(f"{output_dir}/throughput_comparison.pdf")
    plt.close()
    print(f"Saved throughput comparison plot to {output_dir}/throughput_comparison.png")


def plot_latency_comparison(results: dict, output_dir: str):
    """绘制延迟对比图"""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：平均延迟
    vanilla_by_config = {
        r.get("config", ""): r["avg_latency"] for r in results["vanilla"]["latency"]
    }
    runkv_by_config = {
        r.get("config", ""): r["avg_latency"] for r in results["runkv"]["latency"]
    }
    configs = sorted(
        {c for c in vanilla_by_config if c} | {c for c in runkv_by_config if c}
    )
    vanilla_lat = [vanilla_by_config.get(c, 0) for c in configs]
    runkv_lat = [runkv_by_config.get(c, 0) for c in configs]

    if configs:
        x = np.arange(len(configs)) if HAS_NUMPY else list(range(len(configs)))
        width = 0.35

        x1 = (x - width / 2) if HAS_NUMPY else [xi - width / 2 for xi in x]
        x2 = (x + width / 2) if HAS_NUMPY else [xi + width / 2 for xi in x]
        axes[0].bar(
            x1, vanilla_lat[: len(configs)], width, label="Vanilla", color="steelblue"
        )
        axes[0].bar(
            x2, runkv_lat[: len(configs)], width, label="RunKV", color="darkorange"
        )
        axes[0].set_ylabel("Average Latency (s)")
        axes[0].set_title("Average Latency Comparison")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(configs, rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

    # 右图：延迟分布 (box plot)
    if results["vanilla"]["latency"] and results["runkv"]["latency"]:
        all_vanilla = []
        all_runkv = []
        for r in results["vanilla"]["latency"]:
            all_vanilla.extend(r.get("latencies", []))
        for r in results["runkv"]["latency"]:
            all_runkv.extend(r.get("latencies", []))

        if all_vanilla and all_runkv:
            data = [all_vanilla, all_runkv]
            bp = axes[1].boxplot(data, labels=["Vanilla", "RunKV"], patch_artist=True)
            bp["boxes"][0].set_facecolor("steelblue")
            bp["boxes"][1].set_facecolor("darkorange")
            axes[1].set_ylabel("Latency (s)")
            axes[1].set_title("Latency Distribution")
            axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_comparison.png", dpi=300)
    plt.savefig(f"{output_dir}/latency_comparison.pdf")
    plt.close()
    print(f"Saved latency comparison plot to {output_dir}/latency_comparison.png")


def plot_speedup_chart(speedup: dict, output_dir: str):
    """绘制加速比汇总图"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = []
    values = []
    colors = []

    if "throughput_speedup" in speedup:
        metrics.append("Throughput\nSpeedup")
        values.append(speedup["throughput_speedup"])
        colors.append("darkorange")

    if "latency_improvement" in speedup:
        metrics.append("Latency\nImprovement")
        values.append(speedup["latency_improvement"])
        colors.append("forestgreen")

    if not metrics:
        return

    bars = ax.bar(metrics, values, color=colors, edgecolor="black", linewidth=1.2)

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.2f}x",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.axhline(
        y=1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (1.0x)"
    )
    ax.set_ylabel("Speedup Factor", fontsize=12)
    ax.set_title(
        "RunKV Performance Improvement over Vanilla vLLM",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/speedup_summary.png", dpi=300)
    plt.savefig(f"{output_dir}/speedup_summary.pdf")
    plt.close()
    print(f"Saved speedup summary plot to {output_dir}/speedup_summary.png")


def generate_markdown_report(results: dict, speedup: dict, output_file: str):
    """生成Markdown格式的报告"""
    lines = []
    lines.append("# RunKV vs Vanilla vLLM Benchmark Report\n")
    lines.append("Generated automatically from benchmark results.\n")

    lines.append("## Summary\n")
    if "throughput_speedup" in speedup:
        lines.append(f"- **Throughput Speedup**: {speedup['throughput_speedup']:.2f}x")
        lines.append(f"  - Vanilla: {speedup['vanilla_tps']:.1f} tokens/s")
        lines.append(f"  - RunKV: {speedup['runkv_tps']:.1f} tokens/s")
    if "latency_improvement" in speedup:
        lines.append(
            f"- **Latency Improvement**: {speedup['latency_improvement']:.2f}x"
        )
        lines.append(f"  - Vanilla: {speedup['vanilla_latency']:.3f}s")
        lines.append(f"  - RunKV: {speedup['runkv_latency']:.3f}s")

    lines.append("\n## Detailed Results\n")

    # 吞吐量表格
    lines.append("### Throughput\n")
    lines.append("| Configuration | Vanilla (tok/s) | RunKV (tok/s) | Speedup |")
    lines.append("|---------------|-----------------|---------------|---------|")

    vanilla_tp = {
        r.get("config", ""): r["tokens_per_second"]
        for r in results["vanilla"]["throughput"]
    }
    runkv_tp = {
        r.get("config", ""): r["tokens_per_second"]
        for r in results["runkv"]["throughput"]
    }

    all_tp_configs = sorted({c for c in vanilla_tp if c} | {c for c in runkv_tp if c})
    for config in all_tp_configs:
        v_tps = vanilla_tp.get(config, 0)
        r_tps = runkv_tp.get(config, 0)
        sp = r_tps / v_tps if v_tps > 0 else 0
        lines.append(f"| {config} | {v_tps:.1f} | {r_tps:.1f} | {sp:.2f}x |")

    # 延迟表格
    lines.append("\n### Latency\n")
    lines.append("| Configuration | Vanilla Avg (s) | RunKV Avg (s) | Improvement |")
    lines.append("|---------------|-----------------|---------------|-------------|")

    vanilla_lat = {
        r.get("config", ""): r["avg_latency"] for r in results["vanilla"]["latency"]
    }
    runkv_lat = {
        r.get("config", ""): r["avg_latency"] for r in results["runkv"]["latency"]
    }

    all_lat_configs = sorted(
        {c for c in vanilla_lat if c} | {c for c in runkv_lat if c}
    )
    for config in all_lat_configs:
        v_lat = vanilla_lat.get(config, 0)
        r_lat = runkv_lat.get(config, 0)
        imp = v_lat / r_lat if r_lat > 0 else 0
        lines.append(f"| {config} | {v_lat:.3f} | {r_lat:.3f} | {imp:.2f}x |")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved markdown report to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RunKV benchmark results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./benchmark_results",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save analysis outputs",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Only analyze files with this timestamp (YYYYMMDD_HHMMSS)",
    )
    group.add_argument(
        "--select",
        type=str,
        default="all",
        choices=["all", "latest-per-type", "latest-common-per-type"],
        help="Auto-select which timestamps to analyze",
    )
    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")
    results = load_results(
        args.results_dir, timestamp=args.timestamp, selection=args.select
    )

    print("Computing speedup metrics...")
    speedup = compute_speedup(results)

    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")

    if "throughput_speedup" in speedup:
        print(f"Throughput Speedup: {speedup['throughput_speedup']:.2f}x")
        print(f"  - Vanilla: {speedup['vanilla_tps']:.1f} tokens/s")
        print(f"  - RunKV:   {speedup['runkv_tps']:.1f} tokens/s")

    if "latency_improvement" in speedup:
        print(f"\nLatency Improvement: {speedup['latency_improvement']:.2f}x")
        print(f"  - Vanilla: {speedup['vanilla_latency']:.3f}s")
        print(f"  - RunKV:   {speedup['runkv_latency']:.3f}s")

    print(f"{'=' * 60}\n")

    # 生成可视化
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating visualizations...")
    plot_throughput_comparison(results, args.output_dir)
    plot_latency_comparison(results, args.output_dir)
    plot_speedup_chart(speedup, args.output_dir)

    # 生成表格
    print("Generating LaTeX tables...")
    latex_throughput = generate_throughput_table(results)
    latex_latency = generate_latency_table(results)

    with open(f"{args.output_dir}/tables.tex", "w") as f:
        f.write("% Throughput Table\n")
        f.write(latex_throughput)
        f.write("\n\n% Latency Table\n")
        f.write(latex_latency)
    print(f"Saved LaTeX tables to {args.output_dir}/tables.tex")

    # 生成Markdown报告
    generate_markdown_report(results, speedup, f"{args.output_dir}/report.md")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
