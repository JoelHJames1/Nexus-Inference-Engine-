#!/usr/bin/env python3
"""
NEXUS Benchmark Comparison Tool

Parses benchmark JSON results from NEXUS, llama.cpp, and MLX,
generates comparison tables in Markdown format, and computes
speedup ratios.

Usage:
    python tools/benchmark.py --nexus results/nexus.json \
                              --llamacpp results/llamacpp.json \
                              --mlx results/mlx.json \
                              [--output comparison.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Normalized benchmark result ─────────────────────────────────────────────

@dataclass
class BenchmarkData:
    """Normalized representation of a single benchmark run."""
    engine: str = ""
    model_name: str = ""
    device: str = ""
    chip: str = ""
    os_version: str = ""
    ram_gb: float = 0.0
    gpu_cores: int = 0

    # Memory (bytes)
    peak_rss_bytes: int = 0
    weight_bytes: int = 0
    kv_cache_bytes: int = 0

    # Latency (ms)
    ttft_ms: float = 0.0
    mean_itl_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0

    # Throughput (tok/s)
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0


# ── JSON parsers for each engine format ─────────────────────────────────────

def parse_nexus(data: dict) -> BenchmarkData:
    """Parse NEXUS benchmark JSON."""
    b = BenchmarkData(engine="NEXUS")
    b.model_name = data.get("model_name", "")

    sys_info = data.get("system", {})
    b.device = sys_info.get("device_name", "")
    b.chip = sys_info.get("chip_name", "")
    b.os_version = sys_info.get("os_version", "")
    b.ram_gb = sys_info.get("ram_bytes", 0) / (1024**3)
    b.gpu_cores = sys_info.get("gpu_cores", 0)

    mem = data.get("memory", {})
    b.peak_rss_bytes = mem.get("peak_rss_bytes", 0)
    b.weight_bytes = mem.get("weight_bytes", 0)
    b.kv_cache_bytes = mem.get("kv_cache_bytes", 0)

    lat = data.get("latency", {})
    b.ttft_ms = lat.get("ttft_ms", 0.0)
    b.mean_itl_ms = lat.get("mean_itl_ms", 0.0)
    b.p50_itl_ms = lat.get("p50_itl_ms", 0.0)
    b.p90_itl_ms = lat.get("p90_itl_ms", 0.0)
    b.p99_itl_ms = lat.get("p99_itl_ms", 0.0)

    tp = data.get("throughput", {})
    b.prefill_tok_s = tp.get("prefill_tok_per_s", 0.0)
    b.decode_tok_s = tp.get("decode_tok_per_s", 0.0)
    b.prompt_tokens = tp.get("prompt_tokens", 0)
    b.generated_tokens = tp.get("generated_tokens", 0)

    return b


def parse_llamacpp(data: dict) -> BenchmarkData:
    """
    Parse llama.cpp benchmark JSON.

    Expected format (from llama-bench --output json):
    {
        "model": "...",
        "n_prompt": 512,
        "n_gen": 128,
        "t_prompt_ms": ...,
        "t_gen_ms": ...,
        "prompt_tps": ...,
        "gen_tps": ...,
        ...
    }
    """
    b = BenchmarkData(engine="llama.cpp")

    # llama-bench may produce a list of results or a single object
    if isinstance(data, list):
        data = data[0] if data else {}

    b.model_name = data.get("model", data.get("model_filename", ""))

    b.prompt_tokens = data.get("n_prompt", 0)
    b.generated_tokens = data.get("n_gen", 0)

    # Throughput
    b.prefill_tok_s = data.get("prompt_tps", 0.0)
    b.decode_tok_s = data.get("gen_tps", 0.0)

    # Latency: derive from timing if available
    t_prompt_ms = data.get("t_prompt_ms", 0.0)
    t_gen_ms = data.get("t_gen_ms", 0.0)

    if t_prompt_ms > 0:
        b.ttft_ms = t_prompt_ms
    elif b.prefill_tok_s > 0 and b.prompt_tokens > 0:
        b.ttft_ms = (b.prompt_tokens / b.prefill_tok_s) * 1000.0

    if b.decode_tok_s > 0:
        b.mean_itl_ms = 1000.0 / b.decode_tok_s

    # Memory: llama.cpp may report mem_per_token or model_size
    b.weight_bytes = data.get("model_size", 0)
    b.peak_rss_bytes = data.get("peak_rss", data.get("mem_required", 0))

    return b


def parse_mlx(data: dict) -> BenchmarkData:
    """
    Parse MLX benchmark JSON.

    Expected format (from mlx_lm.generate with --verbose):
    {
        "model": "...",
        "prompt_tokens": 100,
        "generation_tokens": 128,
        "prompt_tps": 500.0,
        "generation_tps": 80.0,
        "peak_memory_gb": 8.5,
        ...
    }
    """
    b = BenchmarkData(engine="MLX")
    b.model_name = data.get("model", "")

    b.prompt_tokens = data.get("prompt_tokens", data.get("n_prompt", 0))
    b.generated_tokens = data.get("generation_tokens", data.get("n_gen", 0))

    b.prefill_tok_s = data.get("prompt_tps", 0.0)
    b.decode_tok_s = data.get("generation_tps", 0.0)

    # Latency derivation
    if b.prefill_tok_s > 0 and b.prompt_tokens > 0:
        b.ttft_ms = (b.prompt_tokens / b.prefill_tok_s) * 1000.0
    if b.decode_tok_s > 0:
        b.mean_itl_ms = 1000.0 / b.decode_tok_s

    # Memory
    peak_gb = data.get("peak_memory_gb", 0.0)
    b.peak_rss_bytes = int(peak_gb * 1024**3) if peak_gb else 0

    return b


# ── Comparison table generation ─────────────────────────────────────────────

def format_bytes(b: int) -> str:
    """Human-readable byte string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    elif b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def speedup(a: float, b: float) -> str:
    """Compute speedup ratio a/b as a formatted string."""
    if b <= 0 or a <= 0:
        return "N/A"
    ratio = a / b
    return f"{ratio:.2f}x"


def latency_speedup(a: float, b: float) -> str:
    """For latency, lower is better, so speedup = b/a."""
    if a <= 0 or b <= 0:
        return "N/A"
    ratio = b / a
    return f"{ratio:.2f}x"


def generate_comparison(results: list[BenchmarkData],
                        baseline_engine: str = "NEXUS") -> str:
    """Generate a Markdown comparison table."""
    if not results:
        return "No results to compare.\n"

    lines: list[str] = []
    lines.append("# NEXUS Benchmark Comparison Report\n")

    # Find baseline
    baseline: Optional[BenchmarkData] = None
    for r in results:
        if r.engine == baseline_engine:
            baseline = r
            break
    if baseline is None:
        baseline = results[0]

    # System info (from baseline)
    if baseline.device or baseline.chip:
        lines.append("## System\n")
        lines.append(f"- **Device**: {baseline.device}")
        lines.append(f"- **Chip**: {baseline.chip}")
        lines.append(f"- **OS**: {baseline.os_version}")
        lines.append(f"- **RAM**: {baseline.ram_gb:.0f} GB")
        if baseline.gpu_cores:
            lines.append(f"- **GPU Cores**: {baseline.gpu_cores}")
        lines.append("")

    # Model info
    model_names = set(r.model_name for r in results if r.model_name)
    if model_names:
        lines.append(f"## Model\n")
        lines.append(f"- **Models tested**: {', '.join(sorted(model_names))}")
        lines.append("")

    engines = [r.engine for r in results]
    header = "| Metric | " + " | ".join(engines)
    if len(results) > 1:
        # Add speedup columns vs baseline
        for r in results:
            if r.engine != baseline.engine:
                header += f" | {baseline.engine} vs {r.engine}"
    header += " |"

    sep_parts = ["| --- |"] + [" --- |"] * len(engines)
    if len(results) > 1:
        sep_parts += [" --- |"] * (len(results) - 1)
    sep = "".join(sep_parts)

    # ── Throughput table ────────────────────────────────────────────────────
    lines.append("## Throughput\n")
    lines.append(header)
    lines.append(sep)

    # Prefill row
    row = "| Prefill (tok/s) |"
    for r in results:
        row += f" {r.prefill_tok_s:.1f} |"
    for r in results:
        if r.engine != baseline.engine:
            row += f" {speedup(baseline.prefill_tok_s, r.prefill_tok_s)} |"
    lines.append(row)

    # Decode row
    row = "| Decode (tok/s) |"
    for r in results:
        row += f" {r.decode_tok_s:.1f} |"
    for r in results:
        if r.engine != baseline.engine:
            row += f" {speedup(baseline.decode_tok_s, r.decode_tok_s)} |"
    lines.append(row)

    lines.append("")

    # ── Latency table ───────────────────────────────────────────────────────
    lines.append("## Latency\n")
    lines.append(header)
    lines.append(sep)

    latency_metrics = [
        ("TTFT (ms)", "ttft_ms"),
        ("Mean ITL (ms)", "mean_itl_ms"),
        ("P50 ITL (ms)", "p50_itl_ms"),
        ("P90 ITL (ms)", "p90_itl_ms"),
        ("P99 ITL (ms)", "p99_itl_ms"),
    ]

    for label, attr in latency_metrics:
        row = f"| {label} |"
        for r in results:
            val = getattr(r, attr, 0.0)
            row += f" {val:.2f} |" if val > 0 else " N/A |"
        for r in results:
            if r.engine != baseline.engine:
                bval = getattr(baseline, attr, 0.0)
                rval = getattr(r, attr, 0.0)
                row += f" {latency_speedup(bval, rval)} |"
        lines.append(row)

    lines.append("")

    # ── Memory table ────────────────────────────────────────────────────────
    lines.append("## Memory\n")

    mem_header = "| Metric | " + " | ".join(engines) + " |"
    mem_sep = "| --- |" + " --- |" * len(engines)
    lines.append(mem_header)
    lines.append(mem_sep)

    mem_metrics = [
        ("Peak RSS", "peak_rss_bytes"),
        ("Weights", "weight_bytes"),
        ("KV Cache", "kv_cache_bytes"),
    ]

    for label, attr in mem_metrics:
        row = f"| {label} |"
        for r in results:
            val = getattr(r, attr, 0)
            row += f" {format_bytes(val)} |" if val > 0 else " N/A |"
        lines.append(row)

    lines.append("")

    # ── Summary ─────────────────────────────────────────────────────────────
    if len(results) > 1:
        lines.append("## Speedup Summary\n")
        lines.append(f"Baseline: **{baseline.engine}**\n")

        for r in results:
            if r.engine == baseline.engine:
                continue
            lines.append(f"### {baseline.engine} vs {r.engine}\n")

            prefill_sp = speedup(baseline.prefill_tok_s, r.prefill_tok_s)
            decode_sp = speedup(baseline.decode_tok_s, r.decode_tok_s)
            ttft_sp = latency_speedup(baseline.ttft_ms, r.ttft_ms)

            mem_ratio = "N/A"
            if baseline.peak_rss_bytes > 0 and r.peak_rss_bytes > 0:
                ratio = r.peak_rss_bytes / baseline.peak_rss_bytes
                mem_ratio = f"{ratio:.2f}x"

            lines.append(f"| Metric | Speedup |")
            lines.append(f"| --- | --- |")
            lines.append(f"| Prefill throughput | {prefill_sp} |")
            lines.append(f"| Decode throughput | {decode_sp} |")
            lines.append(f"| TTFT | {ttft_sp} faster |")
            lines.append(f"| Memory (RSS ratio) | {mem_ratio} |")
            lines.append("")

    lines.append("---")
    lines.append("*Generated by NEXUS benchmark comparison tool*\n")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    """Load and return JSON data from a file."""
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NEXUS Benchmark Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/benchmark.py --nexus bench_nexus.json
  python tools/benchmark.py --nexus bench_nexus.json --llamacpp bench_llama.json
  python tools/benchmark.py --nexus bench_nexus.json --mlx bench_mlx.json --output report.md
        """,
    )

    parser.add_argument("--nexus", type=str, help="NEXUS benchmark JSON file")
    parser.add_argument("--llamacpp", type=str, help="llama.cpp benchmark JSON file")
    parser.add_argument("--mlx", type=str, help="MLX benchmark JSON file")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output Markdown file (default: stdout)",
    )
    parser.add_argument(
        "--baseline", type=str, default="NEXUS",
        choices=["NEXUS", "llama.cpp", "MLX"],
        help="Engine to use as baseline for speedup ratios (default: NEXUS)",
    )

    args = parser.parse_args()

    if not args.nexus and not args.llamacpp and not args.mlx:
        parser.print_help()
        print("\nError: at least one benchmark result file is required.",
              file=sys.stderr)
        sys.exit(1)

    results: list[BenchmarkData] = []

    if args.nexus:
        data = load_json(args.nexus)
        results.append(parse_nexus(data))

    if args.llamacpp:
        data = load_json(args.llamacpp)
        results.append(parse_llamacpp(data))

    if args.mlx:
        data = load_json(args.mlx)
        results.append(parse_mlx(data))

    report = generate_comparison(results, baseline_engine=args.baseline)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
