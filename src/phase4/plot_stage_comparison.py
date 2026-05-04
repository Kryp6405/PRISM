#!/usr/bin/env python3
"""
Plot Phase 4 stage-comparison bar charts.

Expected comparison:
  workloads:
    - baseline
    - encoder_heavy
    - decode_heavy

  stages:
    - AGG
    - E/PD
    - E/P/D

Each plot:
  x-axis = stage type
  y-axis = metric

This script scans AIPerf artifact directories and extracts metrics from:
  - profile_export_aiperf.json
  - profile_export_aiperf.csv
  - profile_export.jsonl fallback

Usage:
  python scripts/phase4/plot_stage_comparison.py \
    --artifact-root artifacts/p4 \
    --out-dir artifacts/p4/plots/stage_comparison

Optional:
  python scripts/phase4/plot_stage_comparison.py \
    --artifact-root artifacts/p4 \
    --out-dir artifacts/p4/plots/stage_comparison \
    --show-table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


STAGE_ORDER = ["AGG", "E/PD", "E/P/D"]

STAGE_PATTERNS = {
    "AGG": [
        "aggregated-native-vllm",
        "aggregated_native_vllm",
        "aggregated",
    ],
    "E/PD": [
        "e-pd-native-vllm",
        "e_pd",
        "e-pd",
    ],
    "E/P/D": [
        "e-p-d-native-vllm",
        "e_p_d",
        "e-p-d",
    ],
}

WORKLOAD_ORDER = ["baseline", "encoder_heavy", "decode_heavy"]

# Add aliases here if your folder names differ.
WORKLOAD_ALIASES = {
    "baseline": [
        "baseline",
        "simple_baseline",
        "simple_baseline_v2",
    ],
    "encoder_heavy": [
        "encoder_heavy",
        "vision_heavy",
        "image_heavy",
        "high_res",
        "2048",
    ],
    "decode_heavy": [
        "decode_heavy",
        "long_decode",
        "output_heavy",
        "1024",
    ],
}

# Standardized metric names we care about.
METRIC_NAME_MAP = {
    "request_latency_ms_avg": [
        "Request Latency (ms)",
        "request_latency_ms",
        "request_latency",
        "request_latency_avg",
    ],
    "request_latency_ms_p50": [
        "Request Latency (ms) p50",
        "request_latency_p50",
        "request_latency_ms_p50",
    ],
    "request_latency_ms_p90": [
        "Request Latency (ms) p90",
        "request_latency_p90",
        "request_latency_ms_p90",
    ],
    "request_latency_ms_p99": [
        "Request Latency (ms) p99",
        "request_latency_p99",
        "request_latency_ms_p99",
    ],
    "request_throughput_rps": [
        "Request Throughput (requests/sec)",
        "request_throughput",
        "request_throughput_rps",
    ],
    "output_token_throughput_tps": [
        "Output Token Throughput (tokens/sec)",
        "output_token_throughput",
        "output_token_throughput_tps",
    ],
    "image_throughput_ips": [
        "Image Throughput (images/sec)",
        "image_throughput",
        "image_throughput_ips",
    ],
    "input_tokens_avg": [
        "Input Sequence Length (tokens)",
        "input_sequence_length",
        "input_tokens",
    ],
    "output_tokens_avg": [
        "Output Sequence Length (tokens)",
        "output_sequence_length",
        "output_tokens",
    ],
    "request_count": [
        "Request Count (requests)",
        "request_count",
    ],
}

METRIC_LABELS = {
    "request_latency_ms_avg": "Avg Request Latency (ms)",
    "request_latency_ms_p50": "P50 Request Latency (ms)",
    "request_latency_ms_p90": "P90 Request Latency (ms)",
    "request_latency_ms_p99": "P99 Request Latency (ms)",
    "request_throughput_rps": "Request Throughput (req/s)",
    "output_token_throughput_tps": "Output Token Throughput (tok/s)",
    "image_throughput_ips": "Image Throughput (img/s)",
    "input_tokens_avg": "Avg Input Tokens",
    "output_tokens_avg": "Avg Output Tokens",
    "request_count": "Request Count",
}

DEFAULT_METRICS = [
    "request_latency_ms_avg",
    "request_latency_ms_p50",
    "request_latency_ms_p90",
    "request_latency_ms_p99",
    "request_throughput_rps",
    "output_token_throughput_tps",
    "image_throughput_ips",
    "input_tokens_avg",
    "output_tokens_avg",
]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)

    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL", "NAN"}:
        return None

    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def infer_stage(path: Path) -> str | None:
    text = str(path).lower()

    # Check more specific pattern first.
    for stage in ["E/P/D", "E/PD", "AGG"]:
        for pattern in STAGE_PATTERNS[stage]:
            if pattern.lower() in text:
                return stage

    return None


def infer_workload(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    full = str(path).lower()

    for workload in WORKLOAD_ORDER:
        for alias in WORKLOAD_ALIASES[workload]:
            alias_l = alias.lower()
            if alias_l in parts or alias_l in full:
                return workload

    return None


def extract_concurrency(path: Path) -> int | None:
    match = re.search(r"concurrency(\d+)", str(path))
    if match:
        return int(match.group(1))
    return None


def load_json(path: Path) -> Any | None:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out[key] = v
            out.update(flatten_json(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}.{i}" if prefix else str(i)
            out[key] = v
            out.update(flatten_json(v, key))

    return out


def find_metric_in_flat(flat: dict[str, Any], metric_key: str) -> float | None:
    aliases = METRIC_NAME_MAP[metric_key]
    normalized_aliases = {normalize_key(a) for a in aliases}

    # First exact-ish normalized key match.
    for k, v in flat.items():
        nk = normalize_key(k)
        if nk in normalized_aliases:
            val = safe_float(v)
            if val is not None:
                return val

    # Then fuzzy suffix/contains match.
    for k, v in flat.items():
        nk = normalize_key(k)
        for alias in normalized_aliases:
            if nk.endswith(alias) or alias in nk:
                val = safe_float(v)
                if val is not None:
                    return val

    return None


def parse_profile_export_json(path: Path) -> dict[str, float]:
    data = load_json(path)
    if data is None:
        return {}

    flat = flatten_json(data)
    metrics: dict[str, float] = {}

    for metric_key in METRIC_NAME_MAP:
        val = find_metric_in_flat(flat, metric_key)
        if val is not None:
            metrics[metric_key] = val

    return metrics


def parse_profile_export_csv(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}

    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return metrics

    if not rows:
        return metrics

    # AIPerf CSV can be either metric rows or one summary row.
    # Case 1: rows like Metric,avg,min,max,p99,p90,p50,std
    fieldnames = rows[0].keys()

    metric_field = None
    for candidate in ["Metric", "metric", "name", "Metric Name"]:
        if candidate in fieldnames:
            metric_field = candidate
            break

    if metric_field:
        for row in rows:
            metric_name = str(row.get(metric_field, "")).strip()
            avg = safe_float(row.get("avg") or row.get("Avg") or row.get("mean") or row.get("Mean"))
            p50 = safe_float(row.get("p50") or row.get("P50"))
            p90 = safe_float(row.get("p90") or row.get("P90"))
            p99 = safe_float(row.get("p99") or row.get("P99"))

            if "Request Latency" in metric_name:
                if avg is not None:
                    metrics["request_latency_ms_avg"] = avg
                if p50 is not None:
                    metrics["request_latency_ms_p50"] = p50
                if p90 is not None:
                    metrics["request_latency_ms_p90"] = p90
                if p99 is not None:
                    metrics["request_latency_ms_p99"] = p99
            elif "Request Throughput" in metric_name and avg is not None:
                metrics["request_throughput_rps"] = avg
            elif "Output Token Throughput" in metric_name and avg is not None:
                metrics["output_token_throughput_tps"] = avg
            elif "Image Throughput" in metric_name and avg is not None:
                metrics["image_throughput_ips"] = avg
            elif "Input Sequence Length" in metric_name and avg is not None:
                metrics["input_tokens_avg"] = avg
            elif "Output Sequence Length" in metric_name and avg is not None:
                metrics["output_tokens_avg"] = avg
            elif "Request Count" in metric_name and avg is not None:
                metrics["request_count"] = avg

        return metrics

    # Case 2: one wide summary row.
    flat: dict[str, Any] = {}
    for row in rows:
        for k, v in row.items():
            flat[k] = v

    for metric_key in METRIC_NAME_MAP:
        val = find_metric_in_flat(flat, metric_key)
        if val is not None:
            metrics[metric_key] = val

    return metrics


def parse_profile_export_jsonl(path: Path) -> dict[str, float]:
    """
    Fallback parser from per-record JSONL.
    Only computes simple latency if obvious fields exist.
    """
    latencies_ms: list[float] = []
    output_tokens: list[float] = []
    input_tokens: list[float] = []

    try:
        lines = path.read_text().splitlines()
    except Exception:
        return {}

    for line in lines:
        if not line.strip():
            continue

        try:
            obj = json.loads(line)
        except Exception:
            continue

        flat = flatten_json(obj)

        latency = None
        for key in flat:
            nk = normalize_key(key)
            if "latency" in nk and ("ms" in nk or "duration" in nk):
                latency = safe_float(flat[key])
                if latency is not None:
                    break

        if latency is not None:
            latencies_ms.append(latency)

        for key in flat:
            nk = normalize_key(key)
            if "output" in nk and "token" in nk:
                val = safe_float(flat[key])
                if val is not None:
                    output_tokens.append(val)
                    break

        for key in flat:
            nk = normalize_key(key)
            if "input" in nk and "token" in nk:
                val = safe_float(flat[key])
                if val is not None:
                    input_tokens.append(val)
                    break

    metrics: dict[str, float] = {}

    if latencies_ms:
        latencies_ms_sorted = sorted(latencies_ms)
        metrics["request_latency_ms_avg"] = sum(latencies_ms) / len(latencies_ms)
        metrics["request_latency_ms_p50"] = percentile(latencies_ms_sorted, 50)
        metrics["request_latency_ms_p90"] = percentile(latencies_ms_sorted, 90)
        metrics["request_latency_ms_p99"] = percentile(latencies_ms_sorted, 99)

    if output_tokens:
        metrics["output_tokens_avg"] = sum(output_tokens) / len(output_tokens)

    if input_tokens:
        metrics["input_tokens_avg"] = sum(input_tokens) / len(input_tokens)

    return metrics


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")

    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def parse_artifact_dir(run_dir: Path) -> dict[str, float]:
    candidates = [
        run_dir / "profile_export_aiperf.json",
        run_dir / "profile_export.json",
        run_dir / "profile_export_aiperf.csv",
        run_dir / "profile_export.csv",
        run_dir / "profile_export.jsonl",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        if candidate.suffix == ".json":
            metrics = parse_profile_export_json(candidate)
        elif candidate.suffix == ".csv":
            metrics = parse_profile_export_csv(candidate)
        elif candidate.suffix == ".jsonl":
            metrics = parse_profile_export_jsonl(candidate)
        else:
            metrics = {}

        if metrics:
            return metrics

    return {}


def discover_runs(artifact_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []

    # AIPerf output dirs usually contain profile_export_aiperf.json/csv.
    candidate_dirs: set[Path] = set()

    for pattern in [
        "**/profile_export_aiperf.json",
        "**/profile_export_aiperf.csv",
        "**/profile_export.json",
        "**/profile_export.csv",
        "**/profile_export.jsonl",
    ]:
        for file in artifact_root.glob(pattern):
            candidate_dirs.add(file.parent)

    for run_dir in sorted(candidate_dirs):
        stage = infer_stage(run_dir)
        workload = infer_workload(run_dir)
        concurrency = extract_concurrency(run_dir)

        if stage is None or workload is None:
            continue

        # Since you fixed concurrency=4, ignore other concurrency dirs unless needed.
        if concurrency is not None and concurrency != 4:
            continue

        metrics = parse_artifact_dir(run_dir)
        if not metrics:
            continue

        runs.append(
            {
                "run_dir": run_dir,
                "stage": stage,
                "workload": workload,
                "concurrency": concurrency,
                "metrics": metrics,
            }
        )

    return runs


def choose_latest_runs(runs: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Keep latest run per (workload, stage), based on path mtime.
    """
    chosen: dict[tuple[str, str], dict[str, Any]] = {}

    for run in runs:
        key = (run["workload"], run["stage"])
        old = chosen.get(key)

        if old is None:
            chosen[key] = run
            continue

        old_mtime = old["run_dir"].stat().st_mtime
        new_mtime = run["run_dir"].stat().st_mtime

        if new_mtime > old_mtime:
            chosen[key] = run

    return chosen


def plot_metric(
    chosen: dict[tuple[str, str], dict[str, Any]],
    workload: str,
    metric_key: str,
    out_dir: Path,
) -> None:
    stages = STAGE_ORDER
    values: list[float | None] = []

    for stage in stages:
        run = chosen.get((workload, stage))
        if run is None:
            values.append(None)
        else:
            values.append(run["metrics"].get(metric_key))

    if all(v is None for v in values):
        return

    x = list(range(len(stages)))
    y = [0.0 if v is None else float(v) for v in values]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, y)

    plt.xticks(x, stages)
    plt.ylabel(METRIC_LABELS.get(metric_key, metric_key))
    plt.xlabel("Stage Type")
    plt.title(f"{pretty_workload_name(workload)} — {METRIC_LABELS.get(metric_key, metric_key)}")
    plt.grid(axis="y", alpha=0.3)

    max_y = max(y) if y else 0.0

    for bar, val in zip(bars, values):
        height = bar.get_height()

        if val is None:
            label = "missing"
            height_for_text = max_y * 0.03 if max_y > 0 else 0.01
        else:
            label = format_metric_value(val)
            height_for_text = height

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height_for_text,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    filename = f"{workload}_{metric_key}.png"
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_summary_grid(
    chosen: dict[tuple[str, str], dict[str, Any]],
    workload: str,
    metric_keys: list[str],
    out_dir: Path,
) -> None:
    available = []
    for metric_key in metric_keys:
        if any(
            chosen.get((workload, stage), {}).get("metrics", {}).get(metric_key) is not None
            for stage in STAGE_ORDER
        ):
            available.append(metric_key)

    if not available:
        return

    n = len(available)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, metric_key in zip(axes_list, available):
        values = []
        for stage in STAGE_ORDER:
            run = chosen.get((workload, stage))
            val = None if run is None else run["metrics"].get(metric_key)
            values.append(val)

        y = [0.0 if v is None else float(v) for v in values]
        x = list(range(len(STAGE_ORDER)))

        bars = ax.bar(x, y)
        ax.set_xticks(x)
        ax.set_xticklabels(STAGE_ORDER)
        ax.set_ylabel(METRIC_LABELS.get(metric_key, metric_key))
        ax.set_title(METRIC_LABELS.get(metric_key, metric_key))
        ax.grid(axis="y", alpha=0.3)

        max_y = max(y) if y else 0.0
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label = "missing" if val is None else format_metric_value(val)
            text_y = height if val is not None else (max_y * 0.03 if max_y > 0 else 0.01)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for ax in axes_list[len(available):]:
        ax.axis("off")

    fig.suptitle(f"{pretty_workload_name(workload)} — Stage Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(out_dir / f"{workload}_summary_grid.png", dpi=200)
    plt.close()


def pretty_workload_name(workload: str) -> str:
    return {
        "baseline": "Baseline: 512×512 input, 64 output tokens",
        "encoder_heavy": "Encoder-heavy: 2048×2048 input, 128 output tokens",
        "decode_heavy": "Decode-heavy: 512×512 input, 1024 output tokens",
    }.get(workload, workload)


def format_metric_value(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:,.1f}"
    if abs(v) >= 10:
        return f"{v:,.2f}"
    return f"{v:,.3f}"


def write_summary_csv(
    chosen: dict[tuple[str, str], dict[str, Any]],
    metric_keys: list[str],
    out_path: Path,
) -> None:
    fields = ["workload", "stage", "run_dir"] + metric_keys

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for workload in WORKLOAD_ORDER:
            for stage in STAGE_ORDER:
                run = chosen.get((workload, stage))
                row: dict[str, Any] = {
                    "workload": workload,
                    "stage": stage,
                    "run_dir": "" if run is None else str(run["run_dir"]),
                }

                for metric_key in metric_keys:
                    if run is None:
                        row[metric_key] = ""
                    else:
                        row[metric_key] = run["metrics"].get(metric_key, "")

                writer.writerow(row)


def print_table(chosen: dict[tuple[str, str], dict[str, Any]], metric_keys: list[str]) -> None:
    for workload in WORKLOAD_ORDER:
        print()
        print(f"=== {pretty_workload_name(workload)} ===")

        for stage in STAGE_ORDER:
            run = chosen.get((workload, stage))
            print(f"\n[{stage}]")

            if run is None:
                print("  missing")
                continue

            print(f"  run_dir: {run['run_dir']}")
            for metric_key in metric_keys:
                val = run["metrics"].get(metric_key)
                if val is not None:
                    print(f"  {metric_key}: {format_metric_value(float(val))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/p4"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/p4/plots/stage_comparison"))
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metric keys to plot.",
    )
    parser.add_argument("--show-table", action="store_true")
    parser.add_argument(
        "--all-runs-csv",
        type=Path,
        default=None,
        help="Optional path to dump discovered run-level metrics.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.artifact_root)
    chosen = choose_latest_runs(runs)

    if args.show_table:
        print_table(chosen, args.metrics)

    write_summary_csv(chosen, args.metrics, args.out_dir / "stage_comparison_summary.csv")

    for workload in WORKLOAD_ORDER:
        for metric_key in args.metrics:
            plot_metric(chosen, workload, metric_key, args.out_dir)

        plot_summary_grid(chosen, workload, args.metrics, args.out_dir)

    if args.all_runs_csv is not None:
        args.all_runs_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.all_runs_csv.open("w", newline="") as f:
            fields = ["workload", "stage", "concurrency", "run_dir"] + args.metrics
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for run in runs:
                row: dict[str, Any] = {
                    "workload": run["workload"],
                    "stage": run["stage"],
                    "concurrency": run["concurrency"],
                    "run_dir": str(run["run_dir"]),
                }
                for metric_key in args.metrics:
                    row[metric_key] = run["metrics"].get(metric_key, "")
                writer.writerow(row)

    print(f"Wrote plots to: {args.out_dir}")
    print(f"Wrote summary CSV to: {args.out_dir / 'stage_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
