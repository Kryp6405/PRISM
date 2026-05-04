#!/usr/bin/env python3
"""
Phase 4 stage-comparison plots.

Creates workload-local plots:

  artifacts/p4/baseline/plots/
  artifacts/p4/encoder_heavy/plots/
  artifacts/p4/decode_heavy/plots/

Main latency plot:
  x-axis groups: AGG, E/PD, E/P/D
  grouped bars: avg latency, p90 latency, p99 latency

Other plots:
  x-axis: AGG, E/PD, E/P/D
  y-axis: metric value
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
    "request_count": "Successful Request Count",
}

LATENCY_METRICS = [
    "request_latency_ms_avg",
    "request_latency_ms_p90",
    "request_latency_ms_p99",
]

LATENCY_LEGEND_LABELS = {
    "request_latency_ms_avg": "Avg",
    "request_latency_ms_p90": "P90",
    "request_latency_ms_p99": "P99",
}

SINGLE_METRICS = [
    "request_throughput_rps",
    "output_token_throughput_tps",
    "image_throughput_ips",
    "input_tokens_avg",
    "output_tokens_avg",
    "request_count",
]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value):
            return None
        return value

    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL", "NAN"}:
        return None

    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def format_metric_value(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:,.1f}"
    if abs(v) >= 10:
        return f"{v:,.2f}"
    return f"{v:,.3f}"


def pretty_workload_name(workload: str) -> str:
    return {
        "baseline": "Baseline: 512×512 input, 64 output tokens",
        "encoder_heavy": "Encoder-heavy: 2048×2048 input, 128 output tokens",
        "decode_heavy": "Decode-heavy: 512×512 input, 1024 output tokens",
    }.get(workload, workload)


def infer_stage(path: Path) -> str | None:
    text = str(path).lower()

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
    return int(match.group(1)) if match else None


def normalize_metric_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


def parse_csv_metrics(path: Path) -> dict[str, float]:
    """
    Parse AIPerf console table CSV.

    This is the safest source for:
      - Input Sequence Length (tokens)
      - Output Sequence Length (tokens)
      - Request Count (requests)

    because the avg column is already per-request.
    """
    metrics: dict[str, float] = {}

    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return metrics

    if not rows:
        return metrics

    fieldnames = set(rows[0].keys())
    metric_field = None

    for candidate in ["Metric", "metric", "name", "Metric Name"]:
        if candidate in fieldnames:
            metric_field = candidate
            break

    if not metric_field:
        return metrics

    for row in rows:
        metric_name = normalize_metric_name(str(row.get(metric_field, "")))

        avg = safe_float(row.get("avg") or row.get("Avg") or row.get("mean") or row.get("Mean"))
        p50 = safe_float(row.get("p50") or row.get("P50"))
        p90 = safe_float(row.get("p90") or row.get("P90"))
        p99 = safe_float(row.get("p99") or row.get("P99"))

        if metric_name == "Request Latency (ms)" or "Request Latency" in metric_name:
            if avg is not None:
                metrics["request_latency_ms_avg"] = avg
            if p50 is not None:
                metrics["request_latency_ms_p50"] = p50
            if p90 is not None:
                metrics["request_latency_ms_p90"] = p90
            if p99 is not None:
                metrics["request_latency_ms_p99"] = p99

        elif metric_name == "Request Throughput (requests/sec)" or "Request Throughput" in metric_name:
            if avg is not None:
                metrics["request_throughput_rps"] = avg

        elif metric_name == "Output Token Throughput (tokens/sec)" or "Output Token Throughput" in metric_name:
            if avg is not None:
                metrics["output_token_throughput_tps"] = avg

        elif metric_name == "Image Throughput (images/sec)" or "Image Throughput" in metric_name:
            if avg is not None:
                metrics["image_throughput_ips"] = avg

        elif metric_name == "Input Sequence Length (tokens)" or "Input Sequence Length" in metric_name:
            if avg is not None:
                metrics["input_tokens_avg"] = avg

        elif metric_name == "Output Sequence Length (tokens)" or "Output Sequence Length" in metric_name:
            if avg is not None:
                metrics["output_tokens_avg"] = avg

        elif metric_name == "Request Count (requests)" or "Request Count" in metric_name:
            if avg is not None:
                metrics["request_count"] = avg

    return metrics


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


def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def find_numeric_field(flat: dict[str, Any], aliases: list[str]) -> float | None:
    normalized_aliases = [normalize_key(a) for a in aliases]

    for key, value in flat.items():
        nk = normalize_key(key)
        if any(nk == alias or nk.endswith(alias) for alias in normalized_aliases):
            val = safe_float(value)
            if val is not None:
                return val

    return None


def parse_json_metrics(path: Path) -> dict[str, float]:
    """
    JSON fallback parser.

    Important:
      JSON often contains total_input_tokens / total_output_tokens.
      We parse them separately and normalize later with request_count.
    """
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return {}

    flat = flatten_json(data)
    metrics: dict[str, float] = {}

    alias_map = {
        "request_latency_ms_avg": [
            "request_latency_ms_avg",
            "request_latency_avg",
            "request_latency_ms",
            "latency_ms_avg",
        ],
        "request_latency_ms_p50": [
            "request_latency_ms_p50",
            "request_latency_p50",
            "latency_ms_p50",
        ],
        "request_latency_ms_p90": [
            "request_latency_ms_p90",
            "request_latency_p90",
            "latency_ms_p90",
        ],
        "request_latency_ms_p99": [
            "request_latency_ms_p99",
            "request_latency_p99",
            "latency_ms_p99",
        ],
        "request_throughput_rps": [
            "request_throughput_rps",
            "request_throughput",
            "requests_per_second",
        ],
        "output_token_throughput_tps": [
            "output_token_throughput_tps",
            "output_token_throughput",
            "tokens_per_second",
        ],
        "image_throughput_ips": [
            "image_throughput_ips",
            "image_throughput",
            "images_per_second",
        ],
        "request_count": [
            "request_count",
            "successful_request_count",
            "success_records",
            "valid_requests",
        ],
        "input_tokens_avg": [
            "input_tokens_avg",
            "avg_input_tokens",
            "input_sequence_length_avg",
        ],
        "output_tokens_avg": [
            "output_tokens_avg",
            "avg_output_tokens",
            "output_sequence_length_avg",
        ],
        "input_tokens_total": [
            "total_input_tokens",
            "input_tokens_total",
            "prompt_tokens_total",
        ],
        "output_tokens_total": [
            "total_output_tokens",
            "output_tokens_total",
            "completion_tokens_total",
        ],
    }

    for metric_key, aliases in alias_map.items():
        val = find_numeric_field(flat, aliases)
        if val is not None:
            metrics[metric_key] = val

    return metrics

def maybe_normalize_token_average(metric_key: str, value: float, metrics: dict[str, float]) -> float:
    """
    If token average looks like a total, divide by request_count.

    Examples:
      input:  44751 / 50 ≈ 895
      output: 3200 / 50 = 64
      output: 6400 / 50 = 128
      output: 51200 / 50 = 1024
    """
    if metric_key not in {"input_tokens_avg", "output_tokens_avg"}:
        return value

    request_count = metrics.get("request_count")
    if request_count is None or request_count <= 0:
        return value

    divided = value / request_count

    # If value is clearly too large to be a per-request sequence length
    # and dividing gives a plausible per-request sequence length, normalize.
    if value > 20000 and 1 <= divided <= 20000:
        return divided

    # Also catch output totals like 3200=64*50 and 6400=128*50.
    if metric_key == "output_tokens_avg" and value > 1500 and 1 <= divided <= 20000:
        return divided

    return value

def normalize_token_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """
    Normalize token metrics after parsing.

    Priority:
      1. If explicit totals exist and request_count exists, compute averages.
      2. If avg fields look like totals, divide by request_count.
    """
    request_count = metrics.get("request_count")

    if request_count is not None and request_count > 0:
        if "input_tokens_total" in metrics:
            metrics["input_tokens_avg"] = metrics["input_tokens_total"] / request_count
        elif "input_tokens_avg" in metrics:
            metrics["input_tokens_avg"] = maybe_normalize_token_average(
                "input_tokens_avg",
                metrics["input_tokens_avg"],
                metrics,
            )

        if "output_tokens_total" in metrics:
            metrics["output_tokens_avg"] = metrics["output_tokens_total"] / request_count
        elif "output_tokens_avg" in metrics:
            metrics["output_tokens_avg"] = maybe_normalize_token_average(
                "output_tokens_avg",
                metrics["output_tokens_avg"],
                metrics,
            )

    for key in ["input_tokens_avg", "output_tokens_avg"]:
        if key in metrics and "request_count" in metrics:
            metrics[key] = maybe_normalize_token_average(key, metrics[key], metrics)

    # Do not expose total fields downstream.
    metrics.pop("input_tokens_total", None)
    metrics.pop("output_tokens_total", None)

    return metrics


def merge_metrics(primary: dict[str, float], fallback: dict[str, float]) -> dict[str, float]:
    """
    Primary wins.

    We use CSV as primary and JSON as fallback.
    """
    merged = dict(fallback)
    merged.update(primary)
    return merged


def parse_artifact_dir(run_dir: Path) -> dict[str, float]:
    """
    Parse both CSV and JSON when available.

    CSV is trusted first for table metrics. JSON fills gaps.
    """
    csv_metrics: dict[str, float] = {}
    json_metrics: dict[str, float] = {}

    for candidate in [run_dir / "profile_export_aiperf.csv", run_dir / "profile_export.csv"]:
        if candidate.exists():
            csv_metrics = parse_csv_metrics(candidate)
            if csv_metrics:
                break

    for candidate in [run_dir / "profile_export_aiperf.json", run_dir / "profile_export.json"]:
        if candidate.exists():
            json_metrics = parse_json_metrics(candidate)
            if json_metrics:
                break

    metrics = merge_metrics(primary=csv_metrics, fallback=json_metrics)
    metrics = normalize_token_metrics(metrics)

    return metrics


def discover_runs(artifact_root: Path) -> list[dict[str, Any]]:
    candidate_dirs: set[Path] = set()

    for pattern in [
        "**/profile_export_aiperf.csv",
        "**/profile_export.csv",
        "**/profile_export_aiperf.json",
        "**/profile_export.json",
    ]:
        for file in artifact_root.glob(pattern):
            candidate_dirs.add(file.parent)

    runs: list[dict[str, Any]] = []

    for run_dir in sorted(candidate_dirs):
        stage = infer_stage(run_dir)
        workload = infer_workload(run_dir)
        concurrency = extract_concurrency(run_dir)

        if stage is None or workload is None:
            continue

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
    chosen: dict[tuple[str, str], dict[str, Any]] = {}

    for run in runs:
        key = (run["workload"], run["stage"])
        old = chosen.get(key)

        if old is None:
            chosen[key] = run
            continue

        if run["run_dir"].stat().st_mtime > old["run_dir"].stat().st_mtime:
            chosen[key] = run

    return chosen


def workload_plot_dir(artifact_root: Path, workload: str) -> Path:
    out_dir = artifact_root / workload / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_latency_grouped(
    chosen: dict[tuple[str, str], dict[str, Any]],
    artifact_root: Path,
    workload: str,
) -> None:
    out_dir = workload_plot_dir(artifact_root, workload)

    x = list(range(len(STAGE_ORDER)))
    width = 0.24
    offsets = [-width, 0, width]

    plt.figure(figsize=(9, 5.5))

    has_any = False

    for metric_key, offset in zip(LATENCY_METRICS, offsets):
        values = []
        for stage in STAGE_ORDER:
            run = chosen.get((workload, stage))
            val = None if run is None else run["metrics"].get(metric_key)
            values.append(val)

        if any(v is not None for v in values):
            has_any = True

        y = [0.0 if v is None else float(v) for v in values]
        bars = plt.bar(
            [i + offset for i in x],
            y,
            width=width,
            label=LATENCY_LEGEND_LABELS[metric_key],
        )

        for bar, val in zip(bars, values):
            if val is None:
                continue
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                format_metric_value(float(val)),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    if not has_any:
        plt.close()
        return

    plt.xticks(x, STAGE_ORDER)
    plt.xlabel("Stage Type")
    plt.ylabel("Request Latency (ms)")
    plt.title(f"{pretty_workload_name(workload)} — Request Latency")
    plt.legend(title="Latency")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_dir / "latency_grouped_avg_p90_p99.png", dpi=200)
    plt.close()


def plot_single_metric(
    chosen: dict[tuple[str, str], dict[str, Any]],
    artifact_root: Path,
    workload: str,
    metric_key: str,
) -> None:
    out_dir = workload_plot_dir(artifact_root, workload)

    values = []
    for stage in STAGE_ORDER:
        run = chosen.get((workload, stage))
        val = None if run is None else run["metrics"].get(metric_key)
        values.append(val)

    if all(v is None for v in values):
        return

    x = list(range(len(STAGE_ORDER)))
    y = [0.0 if v is None else float(v) for v in values]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, y)

    plt.xticks(x, STAGE_ORDER)
    plt.xlabel("Stage Type")
    plt.ylabel(METRIC_LABELS.get(metric_key, metric_key))
    plt.title(f"{pretty_workload_name(workload)} — {METRIC_LABELS.get(metric_key, metric_key)}")
    plt.grid(axis="y", alpha=0.3)

    max_y = max(y) if y else 0.0
    for bar, val in zip(bars, values):
        if val is None:
            label = "missing"
            text_y = max_y * 0.03 if max_y > 0 else 0.01
        else:
            label = format_metric_value(float(val))
            text_y = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    filename = f"{metric_key}.png"
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def write_summary_csv(
    chosen: dict[tuple[str, str], dict[str, Any]],
    artifact_root: Path,
    workload: str,
) -> None:
    out_dir = workload_plot_dir(artifact_root, workload)
    metric_keys = LATENCY_METRICS + SINGLE_METRICS

    with (out_dir / "stage_comparison_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["workload", "stage", "run_dir"] + metric_keys,
        )
        writer.writeheader()

        for stage in STAGE_ORDER:
            run = chosen.get((workload, stage))
            row: dict[str, Any] = {
                "workload": workload,
                "stage": stage,
                "run_dir": "" if run is None else str(run["run_dir"]),
            }

            for metric in metric_keys:
                row[metric] = "" if run is None else run["metrics"].get(metric, "")

            writer.writerow(row)


def write_global_summary_csv(
    chosen: dict[tuple[str, str], dict[str, Any]],
    artifact_root: Path,
) -> None:
    out_dir = artifact_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = LATENCY_METRICS + SINGLE_METRICS

    with (out_dir / "stage_comparison_summary_all_workloads.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["workload", "stage", "run_dir"] + metric_keys,
        )
        writer.writeheader()

        for workload in WORKLOAD_ORDER:
            for stage in STAGE_ORDER:
                run = chosen.get((workload, stage))
                row: dict[str, Any] = {
                    "workload": workload,
                    "stage": stage,
                    "run_dir": "" if run is None else str(run["run_dir"]),
                }

                for metric in metric_keys:
                    row[metric] = "" if run is None else run["metrics"].get(metric, "")

                writer.writerow(row)


def print_table(chosen: dict[tuple[str, str], dict[str, Any]]) -> None:
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
            for metric in LATENCY_METRICS + SINGLE_METRICS:
                val = run["metrics"].get(metric)
                if val is not None:
                    print(f"  {metric}: {format_metric_value(float(val))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/p4"))
    parser.add_argument("--show-table", action="store_true")
    args = parser.parse_args()

    runs = discover_runs(args.artifact_root)
    chosen = choose_latest_runs(runs)

    if args.show_table:
        print_table(chosen)

    for workload in WORKLOAD_ORDER:
        plot_latency_grouped(chosen, args.artifact_root, workload)

        for metric in SINGLE_METRICS:
            plot_single_metric(chosen, args.artifact_root, workload, metric)

        write_summary_csv(chosen, args.artifact_root, workload)

    write_global_summary_csv(chosen, args.artifact_root)

    print("Wrote workload-local plots:")
    for workload in WORKLOAD_ORDER:
        print(f"  {args.artifact_root / workload / 'plots'}")
    print(f"Wrote global summary: {args.artifact_root / 'plots' / 'stage_comparison_summary_all_workloads.csv'}")


if __name__ == "__main__":
    main()
