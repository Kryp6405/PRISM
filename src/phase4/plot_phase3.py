#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PIPELINES = ["aggregated", "e_pd", "e_p_d"]

PIPELINE_TO_SUMMARY = {
    "aggregated": "aggregated_summary.json",
    "e_pd": "encoder_only_summary.json",
    "e_p_d": "full_disagg_summary.json",
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

DEFAULT_WORKLOADS = [
    "simple_baseline",
    "med_res",
    "high_res",
    "ocr",
    "decode_heavy",
]

WORKLOAD_LABELS = {
    "simple_baseline": "Simple baseline",
    "med_res": "Medium-res",
    "high_res": "High-res",
    "ocr": "OCR / figure",
    "decode_heavy": "Decode-heavy",
}


METRICS = [
    {
        "keys": ("throughput",),
        "ylabel": "Throughput (requests/sec)",
        "title": "Throughput vs Concurrency",
        "outname": "01_throughput.png",
    },
    {
        "keys": ("e2e_ms",),
        "ylabel": "Average E2E Latency (ms)",
        "title": "Average E2E Latency vs Concurrency",
        "outname": "02_e2e_latency_avg.png",
    },
    {
        "keys": ("e2e_ms_p95",),
        "ylabel": "p95 E2E Latency (ms)",
        "title": "p95 E2E Latency vs Concurrency",
        "outname": "03_e2e_latency_p95.png",
    },
    {
        "keys": ("e2e_ms_p99",),
        "ylabel": "p99 E2E Latency (ms)",
        "title": "p99 E2E Latency vs Concurrency",
        "outname": "04_e2e_latency_p99.png",
    },
    {
        "keys": ("output_token_throughput",),
        "ylabel": "Output Token Throughput (tokens/sec)",
        "title": "Output Token Throughput vs Concurrency",
        "outname": "05_output_token_throughput.png",
    },
    {
        "keys": ("throughput_per_gpu",),
        "ylabel": "Throughput per GPU (requests/sec/GPU)",
        "title": "Throughput per GPU vs Concurrency",
        "outname": "06_throughput_per_gpu.png",
    },
    {
        "keys": ("output_tps_per_gpu",),
        "ylabel": "Output Token Throughput per GPU (tokens/sec/GPU)",
        "title": "Output Token Throughput per GPU vs Concurrency",
        "outname": "07_output_tps_per_gpu.png",
    },
    {
        "keys": ("output_sequence_length_avg", "output_token_count_avg"),
        "ylabel": "Observed Output Length (tokens)",
        "title": "Observed Output Length vs Concurrency",
        "outname": "08_output_sequence_length.png",
    },
    {
        "keys": ("input_sequence_length_avg",),
        "ylabel": "Observed Input Sequence Length (tokens)",
        "title": "Observed Input Sequence Length vs Concurrency",
        "outname": "09_input_sequence_length.png",
    },
    {
        "keys": ("gpu_util_avg",),
        "ylabel": "GPU Utilization Avg (%)",
        "title": "GPU Utilization Average vs Concurrency",
        "outname": "10_gpu_util_avg.png",
    },
    {
        "keys": ("gpu_util_p95",),
        "ylabel": "GPU Utilization p95 (%)",
        "title": "GPU Utilization p95 vs Concurrency",
        "outname": "11_gpu_util_p95.png",
    },
    {
        "keys": ("gpu_util_peak",),
        "ylabel": "GPU Utilization Peak (%)",
        "title": "GPU Utilization Peak vs Concurrency",
        "outname": "12_gpu_util_peak.png",
    },
    {
        "keys": ("gpu_mem_used_gb_peak", "gpu_mem_used_peak_gb"),
        "ylabel": "GPU Memory Peak (GB)",
        "title": "GPU Memory Peak vs Concurrency",
        "outname": "13_gpu_mem_peak.png",
    },
    {
        "keys": ("gpu_mem_used_gb_avg", "gpu_mem_used_avg_gb"),
        "ylabel": "GPU Memory Avg (GB)",
        "title": "GPU Memory Average vs Concurrency",
        "outname": "14_gpu_mem_avg.png",
    },
    {
        "keys": ("gpu_power_w_avg", "gpu_power_avg_w"),
        "ylabel": "GPU Power Avg (W)",
        "title": "GPU Power Average vs Concurrency",
        "outname": "15_gpu_power_avg.png",
    },
    {
        "keys": ("gpu_power_w_peak", "gpu_power_peak_w"),
        "ylabel": "GPU Power Peak (W)",
        "title": "GPU Power Peak vs Concurrency",
        "outname": "16_gpu_power_peak.png",
    },
    {
        "keys": ("gpu_sm_util_avg",),
        "ylabel": "SM Utilization Avg (%)",
        "title": "SM Utilization Average vs Concurrency",
        "outname": "17_gpu_sm_util_avg.png",
    },
    {
        "keys": ("gpu_mem_util_avg",),
        "ylabel": "Memory Controller Utilization Avg (%)",
        "title": "Memory Controller Utilization vs Concurrency",
        "outname": "18_gpu_mem_util_avg.png",
    },
    {
        "keys": ("gpu_temp_c_peak", "gpu_temp_peak_c"),
        "ylabel": "GPU Temperature Peak (°C)",
        "title": "GPU Temperature Peak vs Concurrency",
        "outname": "19_gpu_temp_peak.png",
    },
    {
        "keys": ("gpu_mem_peak_skew_gb",),
        "ylabel": "GPU Memory Peak Skew (GB)",
        "title": "GPU Memory Imbalance vs Concurrency",
        "outname": "20_gpu_mem_peak_skew.png",
    },
    {
        "keys": ("gpu_util_peak_skew",),
        "ylabel": "GPU Utilization Peak Skew (%)",
        "title": "GPU Utilization Imbalance vs Concurrency",
        "outname": "21_gpu_util_peak_skew.png",
    },
    {
        "keys": ("requests_per_watt",),
        "ylabel": "Requests/sec/Watt",
        "title": "Requests per Watt vs Concurrency",
        "outname": "22_requests_per_watt.png",
    },
    {
        "keys": ("output_tps_per_watt",),
        "ylabel": "Output tokens/sec/Watt",
        "title": "Output Token Throughput per Watt vs Concurrency",
        "outname": "23_output_tps_per_watt.png",
    },
]


CROSS_WORKLOAD_METRICS = [
    ("throughput", "Throughput (requests/sec)", "Throughput vs Concurrency", "cross_01_throughput.png"),
    ("e2e_ms_p95", "p95 E2E Latency (ms)", "p95 E2E Latency vs Concurrency", "cross_02_e2e_p95.png"),
    ("output_token_throughput", "Output Token Throughput (tokens/sec)", "Output Token Throughput vs Concurrency", "cross_03_output_tps.png"),
    ("gpu_util_avg", "GPU Utilization Avg (%)", "GPU Utilization Avg vs Concurrency", "cross_04_gpu_util_avg.png"),
    ("gpu_mem_used_gb_peak", "GPU Memory Peak (GB)", "GPU Memory Peak vs Concurrency", "cross_05_gpu_mem_peak.png"),
]


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
        row["gpu_budget"] = GPU_BUDGET[pipeline]
        row["workload_folder"] = workload
        row["workload_label"] = WORKLOAD_LABELS.get(workload, workload)

        throughput = get_value(row, "throughput")
        output_tps = get_value(row, "output_token_throughput")

        if throughput is not None and row["gpu_budget"]:
            row["throughput_per_gpu"] = throughput / row["gpu_budget"]

        if output_tps is not None and row["gpu_budget"]:
            row["output_tps_per_gpu"] = output_tps / row["gpu_budget"]

        rows.append(row)

    return rows


def load_workload_rows(root: Path, workload: str) -> list[dict[str, Any]]:
    analysis_dir = root / workload / "analysis"
    rows: list[dict[str, Any]] = []

    for pipeline in PIPELINES:
        path = analysis_dir / PIPELINE_TO_SUMMARY[pipeline]
        rows.extend(load_summary(path, pipeline, workload))

    return rows


def available_concurrencies(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({
        int(r["concurrency"])
        for r in rows
        if r.get("concurrency") is not None
    })


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
    ax.set_xscale("log", base=2)

    if concurrencies:
        ax.set_xticks(concurrencies)
        ax.set_xticklabels([str(c) for c in concurrencies])

    if log_y:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)

    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_role_metric(
    rows: list[dict[str, Any]],
    pipeline: str,
    role_to_metric: dict[str, tuple[str, ...]],
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    concurrencies = available_concurrencies(rows)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    subset = [
        r for r in rows
        if r.get("pipeline") == pipeline
        and r.get("concurrency") is not None
    ]
    subset.sort(key=lambda x: int(x["concurrency"]))

    plotted = False

    for role, metric_keys in role_to_metric.items():
        xs: list[int] = []
        ys: list[float] = []

        for r in subset:
            val = get_value(r, *metric_keys)
            if val is None:
                continue
            xs.append(int(r["concurrency"]))
            ys.append(val)

        if xs and ys:
            ax.plot(xs, ys, marker="o", label=role)
            plotted = True

    ax.set_title(title)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)

    if concurrencies:
        ax.set_xticks(concurrencies)
        ax.set_xticklabels([str(c) for c in concurrencies])

    ax.grid(True, alpha=0.3)

    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

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

    plot_role_metric(
        rows,
        "e_pd",
        {
            "Encoder": ("encoder_gpu_util_avg",),
            "P/D": ("pd_gpu_util_avg",),
        },
        "GPU Utilization Avg (%)",
        f"{workload_label}: E/PD Role GPU Utilization",
        plots_dir / "30_e_pd_role_gpu_util_avg.png",
    )

    plot_role_metric(
        rows,
        "e_pd",
        {
            "Encoder": ("encoder_gpu_mem_used_gb_peak", "encoder_gpu_mem_used_peak_gb"),
            "P/D": ("pd_gpu_mem_used_gb_peak", "pd_gpu_mem_used_peak_gb"),
        },
        "GPU Memory Peak (GB)",
        f"{workload_label}: E/PD Role Memory Peak",
        plots_dir / "31_e_pd_role_mem_peak.png",
    )

    plot_role_metric(
        rows,
        "e_pd",
        {
            "Encoder": ("encoder_gpu_power_w_avg", "encoder_gpu_power_avg_w"),
            "P/D": ("pd_gpu_power_w_avg", "pd_gpu_power_avg_w"),
        },
        "GPU Power Avg (W)",
        f"{workload_label}: E/PD Role Power Average",
        plots_dir / "32_e_pd_role_power_avg.png",
    )

    plot_role_metric(
        rows,
        "e_p_d",
        {
            "Encoder": ("encoder_gpu_util_avg",),
            "Prefill": ("prefill_gpu_util_avg",),
            "Decode": ("decode_gpu_util_avg",),
        },
        "GPU Utilization Avg (%)",
        f"{workload_label}: E/P/D Role GPU Utilization",
        plots_dir / "40_e_p_d_role_gpu_util_avg.png",
    )

    plot_role_metric(
        rows,
        "e_p_d",
        {
            "Encoder": ("encoder_gpu_mem_used_gb_peak", "encoder_gpu_mem_used_peak_gb"),
            "Prefill": ("prefill_gpu_mem_used_gb_peak", "prefill_gpu_mem_used_peak_gb"),
            "Decode": ("decode_gpu_mem_used_gb_peak", "decode_gpu_mem_used_peak_gb"),
        },
        "GPU Memory Peak (GB)",
        f"{workload_label}: E/P/D Role Memory Peak",
        plots_dir / "41_e_p_d_role_mem_peak.png",
    )

    plot_role_metric(
        rows,
        "e_p_d",
        {
            "Encoder": ("encoder_gpu_power_w_avg", "encoder_gpu_power_avg_w"),
            "Prefill": ("prefill_gpu_power_w_avg", "prefill_gpu_power_avg_w"),
            "Decode": ("decode_gpu_power_w_avg", "decode_gpu_power_avg_w"),
        },
        "GPU Power Avg (W)",
        f"{workload_label}: E/P/D Role Power Average",
        plots_dir / "42_e_p_d_role_power_avg.png",
    )


def plot_cross_workload(
    all_rows_by_workload: dict[str, list[dict[str, Any]]],
    root: Path,
    metric_key: str,
    ylabel: str,
    title: str,
    outname: str,
) -> None:
    workloads = [w for w in DEFAULT_WORKLOADS if all_rows_by_workload.get(w)]

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
        ax.set_xscale("log", base=2)

        if concurrencies:
            ax.set_xticks(concurrencies)
            ax.set_xticklabels([str(c) for c in concurrencies])

        ax.grid(True, alpha=0.3)

        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()

    outdir = root / "analysis" / "plots_cross_workload"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / outname, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/p3"))
    parser.add_argument(
        "--workloads",
        nargs="*",
        default=DEFAULT_WORKLOADS,
        help="Workload folders under artifacts/p3.",
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
