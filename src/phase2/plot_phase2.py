#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path("artifacts/p2/analysis")
PLOTS_DIR = ROOT / "plots_style1"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILES = {
    "aggregated": ROOT / "aggregated_summary.json",
    "e_pd": ROOT / "encoder_only_summary.json",
    "e_p_d": ROOT / "full_disagg_summary.json",
}

PIPELINE_LABELS = {
    "aggregated": "Aggregated",
    "e_pd": "E/PD",
    "e_p_d": "E/P/D",
}

GPU_BUDGET = {
    "aggregated": 1,
    "e_pd": 2,
    "e_p_d": 3,
}

GENERATIONS = ["short", "medium", "long"]
GEN_LABELS = {
    "short": "Short output",
    "medium": "Medium output",
    "long": "Long output",
}


def get_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return None


def load_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for pipeline, path in SUMMARY_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing summary file: {path}")

        data = json.loads(path.read_text())
        for r in data.get("runs", []):
            row = dict(r)
            row["pipeline"] = pipeline
            row["pipeline_label"] = PIPELINE_LABELS[pipeline]
            row["gpu_budget"] = GPU_BUDGET[pipeline]

            # Derived normalized metrics.
            throughput = get_value(row, "throughput")
            output_tps = get_value(row, "output_token_throughput")
            if throughput is not None:
                row["throughput_per_gpu"] = throughput / row["gpu_budget"]
            if output_tps is not None:
                row["output_tps_per_gpu"] = output_tps / row["gpu_budget"]

            rows.append(row)

    return rows


def available_concurrencies(rows: list[dict[str, Any]]) -> list[int]:
    vals = sorted({
        int(r["concurrency"])
        for r in rows
        if r.get("concurrency") is not None
    })
    return vals


def plot_metric_style1(
    rows: list[dict[str, Any]],
    metric_keys: tuple[str, ...],
    ylabel: str,
    title: str,
    outname: str,
    log_y: bool = False,
) -> None:
    concurrencies = available_concurrencies(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, gen in zip(axes, GENERATIONS):
        plotted = False

        for pipeline in ["aggregated", "e_pd", "e_p_d"]:
            subset = [
                r for r in rows
                if r.get("pipeline") == pipeline
                and r.get("generation") == gen
                and r.get("concurrency") is not None
            ]
            subset.sort(key=lambda x: int(x["concurrency"]))

            xs = []
            ys = []
            for r in subset:
                val = get_value(r, *metric_keys)
                if val is None:
                    continue
                xs.append(int(r["concurrency"]))
                ys.append(val)

            if xs and ys:
                ax.plot(xs, ys, marker="o", label=PIPELINE_LABELS[pipeline])
                plotted = True

        ax.set_title(GEN_LABELS[gen])
        ax.set_xlabel("Concurrency")
        ax.set_xscale("log", base=2)
        ax.set_xticks(concurrencies)
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.grid(True, alpha=0.3)

        if log_y:
            ax.set_yscale("log")

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
    fig.savefig(PLOTS_DIR / outname, dpi=180)
    plt.close(fig)


def plot_role_metric(
    rows: list[dict[str, Any]],
    pipeline: str,
    role_to_metric: dict[str, tuple[str, ...]],
    ylabel: str,
    title: str,
    outname: str,
) -> None:
    concurrencies = available_concurrencies(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, gen in zip(axes, GENERATIONS):
        plotted = False

        subset = [
            r for r in rows
            if r.get("pipeline") == pipeline
            and r.get("generation") == gen
            and r.get("concurrency") is not None
        ]
        subset.sort(key=lambda x: int(x["concurrency"]))

        for role, metric_keys in role_to_metric.items():
            xs = []
            ys = []

            for r in subset:
                val = get_value(r, *metric_keys)
                if val is None:
                    continue
                xs.append(int(r["concurrency"]))
                ys.append(val)

            if xs and ys:
                ax.plot(xs, ys, marker="o", label=role)
                plotted = True

        ax.set_title(GEN_LABELS[gen])
        ax.set_xlabel("Concurrency")
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
    fig.savefig(PLOTS_DIR / outname, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = load_runs()

    # -------------------------------------------------------------------------
    # Primary cross-pipeline performance plots
    # -------------------------------------------------------------------------
    plot_metric_style1(
        rows,
        ("throughput",),
        "Throughput (requests/sec)",
        "Throughput vs Concurrency",
        "01_throughput.png",
    )

    plot_metric_style1(
        rows,
        ("e2e_ms",),
        "E2E Latency Avg (ms)",
        "Average E2E Latency vs Concurrency",
        "02_e2e_latency_avg.png",
    )

    plot_metric_style1(
        rows,
        ("e2e_ms_p95",),
        "E2E Latency p95 (ms)",
        "p95 E2E Latency vs Concurrency",
        "03_e2e_latency_p95.png",
    )

    plot_metric_style1(
        rows,
        ("e2e_ms_p99",),
        "E2E Latency p99 (ms)",
        "p99 E2E Latency vs Concurrency",
        "04_e2e_latency_p99.png",
    )

    plot_metric_style1(
        rows,
        ("output_token_throughput",),
        "Output Token Throughput (tokens/sec)",
        "Output Token Throughput vs Concurrency",
        "05_output_token_throughput.png",
    )

    plot_metric_style1(
        rows,
        ("throughput_per_gpu",),
        "Throughput per GPU (requests/sec/GPU)",
        "Throughput per GPU vs Concurrency",
        "06_throughput_per_gpu.png",
    )

    plot_metric_style1(
        rows,
        ("output_tps_per_gpu",),
        "Output Token Throughput per GPU (tokens/sec/GPU)",
        "Output Token Throughput per GPU vs Concurrency",
        "07_output_tps_per_gpu.png",
    )

    # -------------------------------------------------------------------------
    # Workload sanity plots
    # -------------------------------------------------------------------------
    plot_metric_style1(
        rows,
        ("output_sequence_length_avg", "output_token_count_avg"),
        "Avg Output Sequence Length (tokens)",
        "Observed Output Sequence Length vs Concurrency",
        "08_output_sequence_length.png",
    )

    plot_metric_style1(
        rows,
        ("input_sequence_length_avg",),
        "Avg Input Sequence Length (tokens)",
        "Observed Input Sequence Length vs Concurrency",
        "09_input_sequence_length.png",
    )

    # -------------------------------------------------------------------------
    # GPU telemetry cross-pipeline plots
    # Supports both old and new parser key names.
    # -------------------------------------------------------------------------
    plot_metric_style1(
        rows,
        ("gpu_util_avg",),
        "GPU Utilization Avg (%)",
        "GPU Utilization Average vs Concurrency",
        "10_gpu_util_avg.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_util_p95",),
        "GPU Utilization p95 (%)",
        "GPU Utilization p95 vs Concurrency",
        "11_gpu_util_p95.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_util_peak",),
        "GPU Utilization Peak (%)",
        "GPU Utilization Peak vs Concurrency",
        "12_gpu_util_peak.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_mem_used_gb_peak", "gpu_mem_used_peak_gb"),
        "GPU Memory Peak (GB)",
        "GPU Memory Peak vs Concurrency",
        "13_gpu_mem_peak.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_mem_used_gb_avg", "gpu_mem_used_avg_gb"),
        "GPU Memory Avg (GB)",
        "GPU Memory Average vs Concurrency",
        "14_gpu_mem_avg.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_power_w_avg", "gpu_power_avg_w"),
        "GPU Power Avg (W)",
        "GPU Power Average vs Concurrency",
        "15_gpu_power_avg.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_power_w_peak", "gpu_power_peak_w"),
        "GPU Power Peak (W)",
        "GPU Power Peak vs Concurrency",
        "16_gpu_power_peak.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_sm_util_avg",),
        "SM Utilization Avg (%)",
        "SM Utilization Average vs Concurrency",
        "17_gpu_sm_util_avg.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_mem_util_avg",),
        "Memory Controller Utilization Avg (%)",
        "Memory Controller Utilization Average vs Concurrency",
        "18_gpu_mem_util_avg.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_temp_c_peak", "gpu_temp_peak_c"),
        "GPU Temperature Peak (°C)",
        "GPU Temperature Peak vs Concurrency",
        "19_gpu_temp_peak.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_mem_peak_skew_gb",),
        "GPU Memory Peak Skew (GB)",
        "GPU Memory Imbalance vs Concurrency",
        "20_gpu_mem_peak_skew.png",
    )

    plot_metric_style1(
        rows,
        ("gpu_util_peak_skew",),
        "GPU Utilization Peak Skew (%)",
        "GPU Utilization Imbalance vs Concurrency",
        "21_gpu_util_peak_skew.png",
    )

    # -------------------------------------------------------------------------
    # Optional server-side metrics.
    # These may only exist for aggregated runs, so treat as supplemental.
    # -------------------------------------------------------------------------
    plot_metric_style1(
        rows,
        ("frontend_ttft_s_avg",),
        "Frontend TTFT Avg (s)",
        "Frontend TTFT vs Concurrency",
        "22_frontend_ttft_avg.png",
    )

    plot_metric_style1(
        rows,
        ("frontend_itl_s_avg",),
        "Frontend ITL Avg (s)",
        "Frontend Inter-token Latency vs Concurrency",
        "23_frontend_itl_avg.png",
    )

    plot_metric_style1(
        rows,
        ("frontend_request_duration_s_avg",),
        "Frontend Request Duration Avg (s)",
        "Frontend Request Duration vs Concurrency",
        "24_frontend_request_duration_avg.png",
    )

    plot_metric_style1(
        rows,
        ("frontend_inflight_requests_avg",),
        "Frontend Inflight Requests Avg",
        "Frontend Inflight Requests vs Concurrency",
        "25_frontend_inflight_requests_avg.png",
    )

    plot_metric_style1(
        rows,
        ("frontend_queued_requests_avg",),
        "Frontend Queued Requests Avg",
        "Frontend Queued Requests vs Concurrency",
        "26_frontend_queued_requests_avg.png",
    )

    # -------------------------------------------------------------------------
    # Role-aware E/PD plots.
    # -------------------------------------------------------------------------
    plot_role_metric(
        rows,
        "e_pd",
        {
            "Encoder": ("encoder_gpu_util_avg",),
            "P/D": ("pd_gpu_util_avg",),
        },
        "GPU Utilization Avg (%)",
        "E/PD Role GPU Utilization vs Concurrency",
        "27_e_pd_role_gpu_util_avg.png",
    )

    plot_role_metric(
        rows,
        "e_pd",
        {
            "Encoder": ("encoder_gpu_mem_used_gb_peak",),
            "P/D": ("pd_gpu_mem_used_gb_peak",),
        },
        "GPU Memory Peak (GB)",
        "E/PD Role Memory Peak vs Concurrency",
        "28_e_pd_role_mem_peak.png",
    )

    plot_role_metric(
        rows,
        "e_pd",
        {
            "Encoder": ("encoder_gpu_power_w_avg",),
            "P/D": ("pd_gpu_power_w_avg",),
        },
        "GPU Power Avg (W)",
        "E/PD Role Power Average vs Concurrency",
        "29_e_pd_role_power_avg.png",
    )

    # -------------------------------------------------------------------------
    # Role-aware E/P/D plots.
    # -------------------------------------------------------------------------
    plot_role_metric(
        rows,
        "e_p_d",
        {
            "Encoder": ("encoder_gpu_util_avg",),
            "Prefill": ("prefill_gpu_util_avg",),
            "Decode": ("decode_gpu_util_avg",),
        },
        "GPU Utilization Avg (%)",
        "E/P/D Role GPU Utilization vs Concurrency",
        "30_e_p_d_role_gpu_util_avg.png",
    )

    plot_role_metric(
        rows,
        "e_p_d",
        {
            "Encoder": ("encoder_gpu_mem_used_gb_peak",),
            "Prefill": ("prefill_gpu_mem_used_gb_peak",),
            "Decode": ("decode_gpu_mem_used_gb_peak",),
        },
        "GPU Memory Peak (GB)",
        "E/P/D Role Memory Peak vs Concurrency",
        "31_e_p_d_role_mem_peak.png",
    )

    plot_role_metric(
        rows,
        "e_p_d",
        {
            "Encoder": ("encoder_gpu_power_w_avg",),
            "Prefill": ("prefill_gpu_power_w_avg",),
            "Decode": ("decode_gpu_power_w_avg",),
        },
        "GPU Power Avg (W)",
        "E/P/D Role Power Average vs Concurrency",
        "32_e_p_d_role_power_avg.png",
    )

    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
