#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Phase 4 plotting config
# ---------------------------------------------------------------------------

PIPELINES = ["aggregated_native_vllm", "e_pd", "e_p_d"]

PIPELINE_TO_SUMMARY = {
    "aggregated_native_vllm": "aggregated_native_vllm_summary.json",
    "e_pd": "e_pd_summary.json",
    "e_p_d": "e_p_d_summary.json",
}

PIPELINE_LABELS = {
    "aggregated_native_vllm": "Aggregated native vLLM",
    "e_pd": "E/PD",
    "e_p_d": "E/P/D",
}

DEFAULT_GPU_BUDGET = {
    "aggregated_native_vllm": 8,
    "e_pd": 8,
    "e_p_d": 8,
}

DEFAULT_WORKLOADS = [
    "simple_baseline",
    "vision_medium_1img_1024",
    "vision_heavy_1img_2048",
    "ocr_heavy_1img_2048",
    "decode_heavy_1img_1024_out1024",
    "long_prompt_reasoning_1img_1024",
    "stress_multi_image_4img_1024",
    "saturation_mixed_1img_1024",
]

WORKLOAD_LABELS = {
    "simple_baseline": "Simple baseline",
    "vision_medium_1img_1024": "Vision medium 1×1024",
    "vision_heavy_1img_2048": "Vision heavy 1×2048",
    "ocr_heavy_1img_2048": "OCR heavy 1×2048",
    "decode_heavy_1img_1024_out1024": "Decode heavy 1024 output",
    "long_prompt_reasoning_1img_1024": "Long prompt reasoning",
    "stress_multi_image_4img_1024": "Multi-image 4×1024",
    "saturation_mixed_1img_1024": "Saturation mixed 1×1024",
}


METRICS = [
    {
        "keys": ("throughput", "request_throughput"),
        "ylabel": "Request Throughput (requests/sec)",
        "title": "Request Throughput vs Concurrency",
        "outname": "01_request_throughput.png",
    },
    {
        "keys": ("output_token_throughput",),
        "ylabel": "Output Token Throughput (tokens/sec)",
        "title": "Output Token Throughput vs Concurrency",
        "outname": "02_output_token_throughput.png",
    },
    {
        "keys": ("total_token_throughput",),
        "ylabel": "Total Token Throughput (tokens/sec)",
        "title": "Total Token Throughput vs Concurrency",
        "outname": "03_total_token_throughput.png",
    },
    {
        "keys": ("e2e_ms_p50", "e2e_ms"),
        "ylabel": "p50 E2E Latency (ms)",
        "title": "p50 E2E Latency vs Concurrency",
        "outname": "04_e2e_latency_p50.png",
    },
    {
        "keys": ("e2e_ms_p95",),
        "ylabel": "p95 E2E Latency (ms)",
        "title": "p95 E2E Latency vs Concurrency",
        "outname": "05_e2e_latency_p95.png",
    },
    {
        "keys": ("e2e_ms_p99",),
        "ylabel": "p99 E2E Latency (ms)",
        "title": "p99 E2E Latency vs Concurrency",
        "outname": "06_e2e_latency_p99.png",
    },
    {
        "keys": ("ttft_ms_p50", "ttft_ms"),
        "ylabel": "p50 TTFT (ms)",
        "title": "p50 Time to First Token vs Concurrency",
        "outname": "07_ttft_p50.png",
    },
    {
        "keys": ("ttft_ms_p95",),
        "ylabel": "p95 TTFT (ms)",
        "title": "p95 Time to First Token vs Concurrency",
        "outname": "08_ttft_p95.png",
    },
    {
        "keys": ("ttft_ms_p99",),
        "ylabel": "p99 TTFT (ms)",
        "title": "p99 Time to First Token vs Concurrency",
        "outname": "09_ttft_p99.png",
    },
    {
        "keys": ("itl_ms_p50", "itl_ms"),
        "ylabel": "p50 Inter-token Latency (ms)",
        "title": "p50 ITL vs Concurrency",
        "outname": "10_itl_p50.png",
    },
    {
        "keys": ("itl_ms_p95",),
        "ylabel": "p95 Inter-token Latency (ms)",
        "title": "p95 ITL vs Concurrency",
        "outname": "11_itl_p95.png",
    },
    {
        "keys": ("itl_ms_p99",),
        "ylabel": "p99 Inter-token Latency (ms)",
        "title": "p99 ITL vs Concurrency",
        "outname": "12_itl_p99.png",
    },
    {
        "keys": ("prefill_ms_p50",),
        "ylabel": "p50 Prefill Time (ms)",
        "title": "p50 Prefill Time vs Concurrency",
        "outname": "13_prefill_p50.png",
    },
    {
        "keys": ("prefill_ms_p95",),
        "ylabel": "p95 Prefill Time (ms)",
        "title": "p95 Prefill Time vs Concurrency",
        "outname": "14_prefill_p95.png",
    },
    {
        "keys": ("decode_ms_p50",),
        "ylabel": "p50 Decode Time (ms)",
        "title": "p50 Decode Time vs Concurrency",
        "outname": "15_decode_p50.png",
    },
    {
        "keys": ("decode_ms_p95",),
        "ylabel": "p95 Decode Time (ms)",
        "title": "p95 Decode Time vs Concurrency",
        "outname": "16_decode_p95.png",
    },
    {
        "keys": ("queue_ms_p95",),
        "ylabel": "p95 Queue Time (ms)",
        "title": "p95 Queue Time vs Concurrency",
        "outname": "17_queue_p95.png",
    },
    {
        "keys": ("kv_cache_usage_max", "kv_cache_usage_peak"),
        "ylabel": "KV Cache Usage Max (%)",
        "title": "KV Cache Usage vs Concurrency",
        "outname": "18_kv_cache_usage_max.png",
    },
    {
        "keys": ("num_requests_running_max",),
        "ylabel": "Max Running Requests",
        "title": "Max Running Requests vs Concurrency",
        "outname": "19_running_requests_max.png",
    },
    {
        "keys": ("num_requests_waiting_max",),
        "ylabel": "Max Waiting Requests",
        "title": "Max Waiting Requests vs Concurrency",
        "outname": "20_waiting_requests_max.png",
    },
    {
        "keys": ("gpu_util_avg",),
        "ylabel": "Cluster GPU Utilization Avg (%)",
        "title": "Cluster GPU Utilization Average vs Concurrency",
        "outname": "21_cluster_gpu_util_avg.png",
    },
    {
        "keys": ("gpu_util_p95",),
        "ylabel": "Cluster GPU Utilization p95 (%)",
        "title": "Cluster GPU Utilization p95 vs Concurrency",
        "outname": "22_cluster_gpu_util_p95.png",
    },
    {
        "keys": ("gpu_util_peak", "gpu_util_max"),
        "ylabel": "Cluster GPU Utilization Peak (%)",
        "title": "Cluster GPU Utilization Peak vs Concurrency",
        "outname": "23_cluster_gpu_util_peak.png",
    },
    {
        "keys": ("gpu_mem_used_gb_avg", "gpu_mem_used_avg_gb"),
        "ylabel": "Cluster GPU Memory Avg (GB/GPU)",
        "title": "Cluster GPU Memory Average vs Concurrency",
        "outname": "24_cluster_gpu_mem_avg.png",
    },
    {
        "keys": ("gpu_mem_used_gb_peak", "gpu_mem_used_peak_gb"),
        "ylabel": "Cluster GPU Memory Peak (GB/GPU)",
        "title": "Cluster GPU Memory Peak vs Concurrency",
        "outname": "25_cluster_gpu_mem_peak.png",
    },
    {
        "keys": ("cluster_total_power_avg_w", "gpu_power_w_avg", "gpu_power_avg_w"),
        "ylabel": "Cluster Power Avg (W)",
        "title": "Cluster Power Average vs Concurrency",
        "outname": "26_cluster_power_avg.png",
    },
    {
        "keys": ("cluster_total_power_peak_w", "gpu_power_w_peak", "gpu_power_peak_w"),
        "ylabel": "Cluster Power Peak (W)",
        "title": "Cluster Power Peak vs Concurrency",
        "outname": "27_cluster_power_peak.png",
    },
    {
        "keys": ("gpu_temp_c_peak", "gpu_temp_peak_c"),
        "ylabel": "GPU Temperature Peak (°C)",
        "title": "GPU Temperature Peak vs Concurrency",
        "outname": "28_gpu_temp_peak.png",
    },
    {
        "keys": ("gpu_mem_peak_skew_gb",),
        "ylabel": "GPU Memory Peak Skew (GB)",
        "title": "GPU Memory Imbalance vs Concurrency",
        "outname": "29_gpu_mem_peak_skew.png",
    },
    {
        "keys": ("gpu_util_peak_skew",),
        "ylabel": "GPU Utilization Peak Skew (%)",
        "title": "GPU Utilization Imbalance vs Concurrency",
        "outname": "30_gpu_util_peak_skew.png",
    },
    {
        "keys": ("image_latency_ms_avg",),
        "ylabel": "Image Latency Avg (ms/image)",
        "title": "Image Latency vs Concurrency",
        "outname": "31_image_latency_avg.png",
    },
    {
        "keys": ("image_latency_ms_p95",),
        "ylabel": "Image Latency p95 (ms/image)",
        "title": "p95 Image Latency vs Concurrency",
        "outname": "32_image_latency_p95.png",
    },
    {
        "keys": ("image_throughput_avg",),
        "ylabel": "Image Throughput (images/sec)",
        "title": "Image Throughput vs Concurrency",
        "outname": "33_image_throughput.png",
    },
    {
        "keys": ("mm_cache_hit_rate",),
        "ylabel": "MM Cache Hit Rate",
        "title": "MM Cache Hit Rate vs Concurrency",
        "outname": "34_mm_cache_hit_rate.png",
    },
    {
        "keys": ("prefix_cache_hit_rate",),
        "ylabel": "Prefix Cache Hit Rate",
        "title": "Prefix Cache Hit Rate vs Concurrency",
        "outname": "35_prefix_cache_hit_rate.png",
    },
    {
        "keys": ("input_sequence_length_avg",),
        "ylabel": "Input Sequence Length (tokens)",
        "title": "Input Sequence Length vs Concurrency",
        "outname": "36_input_sequence_length.png",
    },
    {
        "keys": ("output_sequence_length_avg", "output_token_count_avg"),
        "ylabel": "Output Sequence Length (tokens)",
        "title": "Output Sequence Length vs Concurrency",
        "outname": "37_output_sequence_length.png",
    },
    {
        "keys": ("throughput_per_gpu",),
        "ylabel": "Requests/sec/GPU",
        "title": "Request Throughput per GPU vs Concurrency",
        "outname": "38_throughput_per_gpu.png",
    },
    {
        "keys": ("output_tps_per_gpu",),
        "ylabel": "Output Tokens/sec/GPU",
        "title": "Output Token Throughput per GPU vs Concurrency",
        "outname": "39_output_tps_per_gpu.png",
    },
    {
        "keys": ("requests_per_watt",),
        "ylabel": "Requests/sec/Watt",
        "title": "Requests per Watt vs Concurrency",
        "outname": "40_requests_per_watt.png",
    },
    {
        "keys": ("output_tps_per_watt",),
        "ylabel": "Output Tokens/sec/Watt",
        "title": "Output Tokens per Watt vs Concurrency",
        "outname": "41_output_tps_per_watt.png",
    },
]


CROSS_WORKLOAD_METRICS = [
    (
        "throughput",
        "Request Throughput (requests/sec)",
        "Request Throughput vs Concurrency",
        "cross_01_request_throughput.png",
    ),
    (
        "output_token_throughput",
        "Output Token Throughput (tokens/sec)",
        "Output Token Throughput vs Concurrency",
        "cross_02_output_tps.png",
    ),
    (
        "total_token_throughput",
        "Total Token Throughput (tokens/sec)",
        "Total Token Throughput vs Concurrency",
        "cross_03_total_token_tps.png",
    ),
    (
        "e2e_ms_p95",
        "p95 E2E Latency (ms)",
        "p95 E2E Latency vs Concurrency",
        "cross_04_e2e_p95.png",
    ),
    (
        "ttft_ms_p95",
        "p95 TTFT (ms)",
        "p95 TTFT vs Concurrency",
        "cross_05_ttft_p95.png",
    ),
    (
        "itl_ms_p95",
        "p95 ITL (ms)",
        "p95 ITL vs Concurrency",
        "cross_06_itl_p95.png",
    ),
    (
        "gpu_util_avg",
        "Cluster GPU Util Avg (%)",
        "GPU Utilization vs Concurrency",
        "cross_07_gpu_util_avg.png",
    ),
    (
        "gpu_mem_used_gb_peak",
        "GPU Memory Peak (GB/GPU)",
        "GPU Memory Peak vs Concurrency",
        "cross_08_gpu_mem_peak.png",
    ),
    (
        "cluster_total_power_avg_w",
        "Cluster Power Avg (W)",
        "Cluster Power vs Concurrency",
        "cross_09_cluster_power_avg.png",
    ),
    (
        "kv_cache_usage_max",
        "KV Cache Usage Max (%)",
        "KV Cache Usage vs Concurrency",
        "cross_10_kv_cache_usage_max.png",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return None


def load_summary(path: Path, pipeline: str, workload: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    data = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []

    for r in data.get("runs", []):
        if r.get("error"):
            continue

        row = dict(r)
        row["pipeline"] = pipeline
        row["pipeline_label"] = PIPELINE_LABELS[pipeline]
        row["workload_folder"] = workload
        row["workload_label"] = WORKLOAD_LABELS.get(workload, workload)

        gpu_budget = (
            get_value(row, "gpu_budget")
            or get_value(row, "cluster_num_gpus")
            or DEFAULT_GPU_BUDGET[pipeline]
        )

        row["gpu_budget"] = gpu_budget

        throughput = get_value(row, "throughput", "request_throughput")
        output_tps = get_value(row, "output_token_throughput")
        total_tps = get_value(row, "total_token_throughput")

        if throughput is not None and gpu_budget:
            row["throughput_per_gpu"] = throughput / gpu_budget

        if output_tps is not None and gpu_budget:
            row["output_tps_per_gpu"] = output_tps / gpu_budget

        if total_tps is not None and gpu_budget:
            row["total_tps_per_gpu"] = total_tps / gpu_budget

        rows.append(row)

    return rows


def load_workload_rows(root: Path, workload: str) -> list[dict[str, Any]]:
    analysis_dir = root / workload / "analysis"
    rows: list[dict[str, Any]] = []

    for pipeline in PIPELINES:
        path = analysis_dir / PIPELINE_TO_SUMMARY[pipeline]
        rows.extend(load_summary(path, pipeline, workload))

    # Compatibility fallback: if there is only one generic summary.json.
    fallback = analysis_dir / "summary.json"
    if not rows and fallback.exists():
        rows.extend(load_summary(fallback, "aggregated_native_vllm", workload))

    return rows


def available_concurrencies(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({
        int(r["concurrency"])
        for r in rows
        if r.get("concurrency") is not None
    })


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pipeline_metric(
    rows: list[dict[str, Any]],
    metric_keys: tuple[str, ...],
    ylabel: str,
    title: str,
    outpath: Path,
    log_y: bool = False,
) -> None:
    concurrencies = available_concurrencies(rows)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    plotted = False

    for pipeline in PIPELINES:
        subset = [
            r for r in rows
            if r.get("pipeline") == pipeline
            and r.get("concurrency") is not None
        ]
        subset.sort(key=lambda x: int(x["concurrency"]))

        xs: list[int] = []
        ys: list[float] = []

        for r in subset:
            val = get_value(r, *metric_keys)
            if val is None:
                continue

            xs.append(int(r["concurrency"]))
            ys.append(val)

        if xs and ys:
            ax.plot(xs, ys, marker="o", label=PIPELINE_LABELS[pipeline])
            plotted = True

    ax.set_title(title)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)

    if concurrencies and all(c > 0 for c in concurrencies):
        ax.set_xscale("log", base=2)
        ax.set_xticks(concurrencies)
        ax.set_xticklabels([str(c) for c in concurrencies])

    if log_y:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)

    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_workload(workload: str, rows: list[dict[str, Any]], root: Path) -> None:
    plots_dir = root / workload / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    workload_label = WORKLOAD_LABELS.get(workload, workload)

    for spec in METRICS:
        plot_pipeline_metric(
            rows,
            tuple(spec["keys"]),
            spec["ylabel"],
            f"{workload_label}: {spec['title']}",
            plots_dir / spec["outname"],
        )


def plot_cross_workload(
    all_rows_by_workload: dict[str, list[dict[str, Any]]],
    root: Path,
    metric_key: str,
    ylabel: str,
    title: str,
    outname: str,
) -> None:
    workloads = [w for w in all_rows_by_workload if all_rows_by_workload.get(w)]

    if not workloads:
        return

    fig, axes = plt.subplots(
        1,
        len(workloads),
        figsize=(5 * len(workloads), 4.5),
        sharey=False,
    )

    if len(workloads) == 1:
        axes = [axes]

    for ax, workload in zip(axes, workloads):
        rows = all_rows_by_workload[workload]
        concurrencies = available_concurrencies(rows)

        plotted = False

        for pipeline in PIPELINES:
            subset = [
                r for r in rows
                if r.get("pipeline") == pipeline
                and r.get("concurrency") is not None
            ]
            subset.sort(key=lambda x: int(x["concurrency"]))

            xs: list[int] = []
            ys: list[float] = []

            for r in subset:
                val = get_value(r, metric_key)
                if val is None:
                    continue

                xs.append(int(r["concurrency"]))
                ys.append(val)

            if xs and ys:
                ax.plot(xs, ys, marker="o", label=PIPELINE_LABELS[pipeline])
                plotted = True

        ax.set_title(WORKLOAD_LABELS.get(workload, workload))
        ax.set_xlabel("Concurrency")

        if concurrencies and all(c > 0 for c in concurrencies):
            ax.set_xscale("log", base=2)
            ax.set_xticks(concurrencies)
            ax.set_xticklabels([str(c) for c in concurrencies])

        ax.grid(True, alpha=0.3)

        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()

    outdir = root / "analysis" / "plots_cross_workload"
    outdir.mkdir(parents=True, exist_ok=True)

    fig.savefig(outdir / outname, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/p4"),
        help="Root artifact directory, usually artifacts/p4.",
    )

    parser.add_argument(
        "--workloads",
        nargs="*",
        default=DEFAULT_WORKLOADS,
        help="Workload folders under artifacts/p4.",
    )

    args = parser.parse_args()

    root = args.root
    workloads = args.workloads

    all_rows_by_workload: dict[str, list[dict[str, Any]]] = {}

    for workload in workloads:
        rows = load_workload_rows(root, workload)
        all_rows_by_workload[workload] = rows

        if rows:
            plot_workload(workload, rows, root)
            print(f"Wrote workload plots for {workload} to {root / workload / 'analysis' / 'plots'}")
        else:
            print(f"No rows found for workload: {workload}")

    for metric_key, ylabel, title, outname in CROSS_WORKLOAD_METRICS:
        plot_cross_workload(
            all_rows_by_workload,
            root,
            metric_key,
            ylabel,
            title,
            outname,
        )

    print(f"Wrote cross-workload plots to {root / 'analysis' / 'plots_cross_workload'}")


if __name__ == "__main__":
    main()
