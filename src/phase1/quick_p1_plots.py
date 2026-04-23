#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("artifacts/p1/analysis")

FILES = {
    "aggregated": ROOT / "aggregated_summary.json",
    "encoder_only": ROOT / "encoder_only_summary.json",
    "full_disagg": ROOT / "full_disagg_summary.json",
}


def infer_generation(name: str) -> str:
    if "_short_" in name:
        return "short"
    if "_long_" in name:
        return "long"
    return "unknown"


def load_runs(summary_path: Path, pipeline_name: str):
    data = json.loads(summary_path.read_text())
    runs = []
    for r in data["runs"]:
        name = r.get("artifact_name", "")
        runs.append({
            "pipeline": pipeline_name,
            "generation": infer_generation(name),
            "concurrency": r.get("concurrency"),
            "throughput": r.get("throughput"),
            "e2e_ms": r.get("e2e_ms"),
            "output_token_throughput": r.get("output_token_throughput"),
            "output_sequence_length_avg": r.get("output_sequence_length_avg"),
        })
    return runs


def plot_metric(all_runs, generation, metric, ylabel, outname):
    plt.figure(figsize=(7, 4.5))
    pipelines = ["aggregated", "encoder_only", "full_disagg"]

    for pipeline in pipelines:
        rows = [r for r in all_runs if r["generation"] == generation and r["pipeline"] == pipeline]
        rows.sort(key=lambda x: x["concurrency"])
        xs = [r["concurrency"] for r in rows]
        ys = [r[metric] for r in rows]
        plt.plot(xs, ys, marker="o", label=pipeline)

    plt.xlabel("Concurrency")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Concurrency ({generation})")
    plt.xticks([1, 4, 16])
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / outname, dpi=160)
    plt.close()


def main():
    all_runs = []
    for pipeline, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing summary file: {path}")
        all_runs.extend(load_runs(path, pipeline))

    plot_metric(all_runs, "short", "throughput", "Throughput (req/s)", "quick_throughput_short.png")
    plot_metric(all_runs, "long", "throughput", "Throughput (req/s)", "quick_throughput_long.png")

    plot_metric(all_runs, "short", "e2e_ms", "E2E Latency (ms)", "quick_e2e_short.png")
    plot_metric(all_runs, "long", "e2e_ms", "E2E Latency (ms)", "quick_e2e_long.png")

    plot_metric(all_runs, "short", "output_token_throughput", "Output Token Throughput (tokens/s)", "quick_output_tps_short.png")
    plot_metric(all_runs, "long", "output_token_throughput", "Output Token Throughput (tokens/s)", "quick_output_tps_long.png")

    plot_metric(all_runs, "short", "output_sequence_length_avg", "Avg Output Sequence Length", "quick_output_len_short.png")
    plot_metric(all_runs, "long", "output_sequence_length_avg", "Avg Output Sequence Length", "quick_output_len_long.png")

    print(f"Wrote plots to {ROOT}")


if __name__ == "__main__":
    main()
