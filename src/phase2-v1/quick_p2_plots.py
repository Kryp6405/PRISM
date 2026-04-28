#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("artifacts/p2/analysis")
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "aggregated": ROOT / "aggregated_summary.json",
    "encoder_only": ROOT / "encoder_only_summary.json",
    "full_disagg": ROOT / "full_disagg_summary.json",
}

PIPELINES = ["aggregated", "encoder_only", "full_disagg"]
GENERATIONS = ["short", "medium", "long"]


def load_runs(summary_path: Path, pipeline_name: str) -> list[dict]:
    data = json.loads(summary_path.read_text())
    runs = []
    for r in data["runs"]:
        row = dict(r)
        row["pipeline"] = pipeline_name
        runs.append(row)
    return runs


def all_runs() -> list[dict]:
    rows: list[dict] = []
    for pipeline, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing summary file: {path}")
        rows.extend(load_runs(path, pipeline))
    return rows


def valid_rows(rows: list[dict], metric: str) -> list[dict]:
    out = []
    for r in rows:
        v = r.get(metric)
        c = r.get("concurrency")
        if v is None or c is None:
            continue
        out.append(r)
    return out


def plot_metric_by_generation(
    runs: list[dict],
    metric: str,
    ylabel: str,
    filename_prefix: str,
) -> None:
    for gen in GENERATIONS:
        plt.figure(figsize=(7, 4.5))
        plotted_any = False

        for pipeline in PIPELINES:
            rows = [
                r for r in runs
                if r.get("pipeline") == pipeline and r.get("generation") == gen and r.get(metric) is not None
            ]
            rows.sort(key=lambda x: x["concurrency"])
            if not rows:
                continue

            xs = [r["concurrency"] for r in rows]
            ys = [r[metric] for r in rows]
            plt.plot(xs, ys, marker="o", label=pipeline)
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("Concurrency")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Concurrency ({gen})")
        plt.xticks(sorted({r["concurrency"] for r in runs if r.get("concurrency") is not None}))
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{filename_prefix}_{gen}.png", dpi=180)
        plt.close()


def plot_aggregated_server_metric(
    runs: list[dict],
    metric: str,
    ylabel: str,
    filename_prefix: str,
) -> None:
    for gen in GENERATIONS:
        rows = [
            r for r in runs
            if r.get("pipeline") == "aggregated"
            and r.get("generation") == gen
            and r.get(metric) is not None
        ]
        rows.sort(key=lambda x: x["concurrency"])
        if not rows:
            continue

        plt.figure(figsize=(7, 4.5))
        xs = [r["concurrency"] for r in rows]
        ys = [r[metric] for r in rows]
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Concurrency")
        plt.ylabel(ylabel)
        plt.title(f"Aggregated: {ylabel} vs Concurrency ({gen})")
        plt.xticks(sorted({r["concurrency"] for r in runs if r.get("concurrency") is not None}))
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{filename_prefix}_{gen}.png", dpi=180)
        plt.close()


def plot_encoder_only_role_metric(
    runs: list[dict],
    metric_suffix: str,
    ylabel: str,
    filename_prefix: str,
) -> None:
    roles = [
        ("encoder", f"encoder_gpu_{metric_suffix}"),
        ("pd", f"pd_gpu_{metric_suffix}"),
    ]

    for gen in GENERATIONS:
        plt.figure(figsize=(7, 4.5))
        plotted_any = False

        for role_name, metric in roles:
            rows = [
                r for r in runs
                if r.get("pipeline") == "encoder_only"
                and r.get("generation") == gen
                and r.get(metric) is not None
            ]
            rows.sort(key=lambda x: x["concurrency"])
            if not rows:
                continue

            xs = [r["concurrency"] for r in rows]
            ys = [r[metric] for r in rows]
            plt.plot(xs, ys, marker="o", label=role_name)
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("Concurrency")
        plt.ylabel(ylabel)
        plt.title(f"Encoder-only: {ylabel} by GPU role ({gen})")
        plt.xticks(sorted({r["concurrency"] for r in runs if r.get("concurrency") is not None}))
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{filename_prefix}_{gen}.png", dpi=180)
        plt.close()


def plot_full_disagg_role_metric(
    runs: list[dict],
    metric_suffix: str,
    ylabel: str,
    filename_prefix: str,
) -> None:
    roles = [
        ("encoder", f"encoder_gpu_{metric_suffix}"),
        ("prefill", f"prefill_gpu_{metric_suffix}"),
        ("decode", f"decode_gpu_{metric_suffix}"),
    ]

    for gen in GENERATIONS:
        plt.figure(figsize=(7, 4.5))
        plotted_any = False

        for role_name, metric in roles:
            rows = [
                r for r in runs
                if r.get("pipeline") == "full_disagg"
                and r.get("generation") == gen
                and r.get(metric) is not None
            ]
            rows.sort(key=lambda x: x["concurrency"])
            if not rows:
                continue

            xs = [r["concurrency"] for r in rows]
            ys = [r[metric] for r in rows]
            plt.plot(xs, ys, marker="o", label=role_name)
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("Concurrency")
        plt.ylabel(ylabel)
        plt.title(f"Full-disagg: {ylabel} by GPU role ({gen})")
        plt.xticks(sorted({r["concurrency"] for r in runs if r.get("concurrency") is not None}))
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{filename_prefix}_{gen}.png", dpi=180)
        plt.close()


def main() -> None:
    runs = all_runs()

    # Core cross-pipeline plots
    plot_metric_by_generation(runs, "throughput", "Throughput (req/s)", "throughput")
    plot_metric_by_generation(runs, "e2e_ms", "E2E Latency (ms)", "e2e")
    plot_metric_by_generation(runs, "output_token_throughput", "Output Token Throughput (tokens/s)", "output_tps")
    plot_metric_by_generation(runs, "output_sequence_length_avg", "Avg Output Sequence Length", "output_len")

    # GPU telemetry cross-pipeline plots
    plot_metric_by_generation(runs, "gpu_util_avg", "GPU Util Avg (%)", "gpu_util_avg")
    plot_metric_by_generation(runs, "gpu_util_peak", "GPU Util Peak (%)", "gpu_util_peak")
    plot_metric_by_generation(runs, "gpu_mem_used_gb_avg", "GPU Memory Used Avg (GB)", "gpu_mem_avg")
    plot_metric_by_generation(runs, "gpu_mem_used_gb_peak", "GPU Memory Used Peak (GB)", "gpu_mem_peak")
    plot_metric_by_generation(runs, "gpu_power_w_avg", "GPU Power Avg (W)", "gpu_power_avg")
    plot_metric_by_generation(runs, "gpu_power_w_peak", "GPU Power Peak (W)", "gpu_power_peak")
    plot_metric_by_generation(runs, "gpu_temp_c_peak", "GPU Temperature Peak (C)", "gpu_temp_peak")
    plot_metric_by_generation(runs, "gpu_active_count_avg", "Active GPU Count", "gpu_active_count")
    plot_metric_by_generation(runs, "gpu_mem_peak_skew_gb", "GPU Memory Peak Skew (GB)", "gpu_mem_skew")
    plot_metric_by_generation(runs, "gpu_util_peak_skew", "GPU Util Peak Skew", "gpu_util_skew")

    # Aggregated-only server metrics where available
    plot_aggregated_server_metric(runs, "frontend_ttft_s_avg", "Frontend TTFT (s)", "agg_frontend_ttft")
    plot_aggregated_server_metric(runs, "frontend_itl_s_avg", "Frontend ITL (s)", "agg_frontend_itl")
    plot_aggregated_server_metric(runs, "frontend_request_duration_s_avg", "Frontend Request Duration (s)", "agg_frontend_reqdur")
    plot_aggregated_server_metric(runs, "frontend_inflight_requests_avg", "Frontend Inflight Requests Avg", "agg_frontend_inflight")
    plot_aggregated_server_metric(runs, "frontend_queued_requests_avg", "Frontend Queued Requests Avg", "agg_frontend_queued")
    plot_aggregated_server_metric(runs, "frontend_output_tokens_rate", "Frontend Output Tokens Rate", "agg_frontend_output_tokens_rate")
    plot_aggregated_server_metric(runs, "frontend_requests_rate", "Frontend Requests Rate", "agg_frontend_requests_rate")

    # Role-aware GPU plots
    plot_encoder_only_role_metric(runs, "util_avg", "GPU Util Avg (%)", "encoder_only_role_util_avg")
    plot_encoder_only_role_metric(runs, "mem_used_gb_peak", "GPU Memory Peak (GB)", "encoder_only_role_mem_peak")
    plot_encoder_only_role_metric(runs, "power_w_avg", "GPU Power Avg (W)", "encoder_only_role_power_avg")

    plot_full_disagg_role_metric(runs, "util_avg", "GPU Util Avg (%)", "full_disagg_role_util_avg")
    plot_full_disagg_role_metric(runs, "mem_used_gb_peak", "GPU Memory Peak (GB)", "full_disagg_role_mem_peak")
    plot_full_disagg_role_metric(runs, "power_w_avg", "GPU Power Avg (W)", "full_disagg_role_power_avg")

    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
