#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------

def is_aiperf_artifact_dir(path: Path) -> bool:
    return (
        (path / "profile_export_aiperf.json").exists()
        or (path / "profile_export_aiperf.csv").exists()
        or (path / "profile_export.jsonl").exists()
        or (path / "gpu_telemetry_export.jsonl").exists()
        or (path / "gpu_telemetry_all_nodes.csv").exists()
        or (path / "server_metrics_export.json").exists()
        or (path / "server_metrics_export.csv").exists()
    )


def parse_one(parser_script: Path, artifact_dir: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(parser_script), str(artifact_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return {
            "artifact_dir": str(artifact_dir),
            "artifact_name": artifact_dir.name,
            "error": result.stderr.strip() or f"parser exited with {result.returncode}",
            "stdout": result.stdout.strip(),
        }

    try:
        return json.loads(result.stdout)
    except Exception as e:
        return {
            "artifact_dir": str(artifact_dir),
            "artifact_name": artifact_dir.name,
            "error": f"invalid parser JSON output: {e}",
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def sort_key(row: dict[str, Any]) -> tuple:
    mode_order = {
        "aggregated_native_vllm": 0,
        "aggregated": 0,
        "e_pd": 1,
        "encoder_only": 1,
        "e_p_d": 2,
        "full_disagg": 2,
        None: 99,
    }

    try:
        c = int(row.get("concurrency"))
    except Exception:
        c = 999999

    return (
        str(row.get("workload_type") or ""),
        c,
        mode_order.get(row.get("mode"), 99),
        row.get("artifact_name", row.get("artifact_dir", "")),
    )


def aggregate_count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}

    for r in rows:
        val = r.get(key)
        k = str(val if val is not None else "unknown")
        out[k] = out.get(k, 0) + 1

    return out


def collect_metric_availability(rows: list[dict[str, Any]]) -> dict[str, int]:
    important_metrics = [
        # Throughput
        "throughput",
        "request_throughput",
        "output_token_throughput",
        "total_token_throughput",
        "throughput_per_gpu",
        "output_tps_per_gpu",
        "total_tps_per_gpu",

        # Core latency
        "e2e_ms",
        "e2e_ms_p50",
        "e2e_ms_p95",
        "e2e_ms_p99",

        # TTFT / ITL
        "ttft_ms",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "ttft_ms_p99",
        "itl_ms",
        "itl_ms_p50",
        "itl_ms_p95",
        "itl_ms_p99",

        # Prefill / decode / queue
        "prefill_ms_avg",
        "prefill_ms_p50",
        "prefill_ms_p95",
        "prefill_ms_p99",
        "decode_ms_avg",
        "decode_ms_p50",
        "decode_ms_p95",
        "decode_ms_p99",
        "queue_ms_avg",
        "queue_ms_p50",
        "queue_ms_p95",
        "queue_ms_p99",

        # Token / request stats
        "input_sequence_length_avg",
        "input_sequence_length_p95",
        "output_sequence_length_avg",
        "output_sequence_length_p95",
        "output_token_count_avg",
        "output_token_count_p95",
        "request_count",
        "benchmark_duration_sec",
        "total_input_sequence_length",
        "total_output_sequence_length",
        "total_output_tokens",
        "total_usage_prompt_tokens",
        "total_usage_completion_tokens",
        "osl_mismatch_count",

        # VLM / multimodal
        "num_images_avg",
        "image_latency_ms_avg",
        "image_latency_ms_p95",
        "image_throughput_avg",
        "mm_cache_hit_rate",

        # Cluster GPU telemetry
        "cluster_num_hosts",
        "cluster_num_gpus",
        "gpu_active_count_avg",
        "gpu_util_avg",
        "gpu_util_p50",
        "gpu_util_p95",
        "gpu_util_p99",
        "gpu_util_peak",
        "gpu_mem_used_gb_avg",
        "gpu_mem_used_gb_p95",
        "gpu_mem_used_gb_peak",
        "gpu_power_w_avg",
        "gpu_power_w_p95",
        "gpu_power_w_peak",
        "cluster_total_power_avg_w",
        "cluster_total_power_peak_w",
        "gpu_temp_c_peak",
        "gpu_mem_peak_skew_gb",
        "gpu_util_peak_skew",
        "gpu_power_avg_skew_w",

        # vLLM server metrics
        "kv_cache_usage_avg",
        "kv_cache_usage_max",
        "kv_cache_usage_p95",
        "num_requests_running_avg",
        "num_requests_running_max",
        "num_requests_waiting_avg",
        "num_requests_waiting_max",
        "num_preemptions_total",

        # Cache
        "prefix_cache_hits_total",
        "prefix_cache_queries_total",
        "prefix_cache_hit_rate",
        "mm_cache_hits_total",
        "mm_cache_queries_total",
        "mm_cache_hit_rate",

        # Server-side token metrics
        "server_prompt_tokens_total",
        "server_generation_tokens_total",
        "server_prompt_tokens_rate",
        "server_generation_tokens_rate",

        # Dynamo/frontend legacy compatibility
        "frontend_ttft_s_avg",
        "frontend_itl_s_avg",
        "frontend_request_duration_s_avg",
        "frontend_inflight_requests_avg",
        "frontend_queued_requests_avg",
    ]

    return {
        metric: sum(1 for r in rows if r.get(metric) is not None)
        for metric in important_metrics
    }


def select_compact_run_fields(row: dict[str, Any]) -> dict[str, Any]:
    """
    A compact table-like view that is easier to inspect than the full run row.
    The full rows are still preserved under `runs`.
    """
    keys = [
        "artifact_name",
        "artifact_dir",
        "mode",
        "generation",
        "workload_type",
        "concurrency",
        "gpu_budget",

        "request_count",
        "benchmark_duration_sec",
        "throughput",
        "output_token_throughput",
        "total_token_throughput",
        "throughput_per_gpu",
        "output_tps_per_gpu",

        "e2e_ms",
        "e2e_ms_p50",
        "e2e_ms_p95",
        "e2e_ms_p99",
        "ttft_ms",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "ttft_ms_p99",
        "itl_ms",
        "itl_ms_p50",
        "itl_ms_p95",
        "itl_ms_p99",

        "prefill_ms_p95",
        "decode_ms_p95",
        "queue_ms_p95",

        "input_sequence_length_avg",
        "output_sequence_length_avg",
        "output_token_count_avg",
        "num_images_avg",
        "image_latency_ms_avg",
        "image_throughput_avg",

        "cluster_num_hosts",
        "cluster_num_gpus",
        "gpu_active_count_avg",
        "gpu_util_avg",
        "gpu_util_peak",
        "gpu_mem_used_gb_avg",
        "gpu_mem_used_gb_peak",
        "gpu_power_w_avg",
        "gpu_power_w_peak",
        "cluster_total_power_avg_w",
        "cluster_total_power_peak_w",
        "gpu_mem_peak_skew_gb",
        "gpu_util_peak_skew",

        "kv_cache_usage_avg",
        "kv_cache_usage_max",
        "num_requests_running_avg",
        "num_requests_running_max",
        "num_requests_waiting_avg",
        "num_requests_waiting_max",
        "mm_cache_hit_rate",
        "prefix_cache_hit_rate",

        "sources",
        "error",
    ]

    return {k: row.get(k) for k in keys if k in row}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Directory containing AIPerf artifact directories for one workload.",
    )

    parser.add_argument(
        "--run-prefix",
        type=str,
        default="",
        help="Optional artifact directory prefix filter. Leave empty to parse all artifact dirs.",
    )

    parser.add_argument(
        "--parser-script",
        type=Path,
        default=Path("src/phase4/parse_aiperf_phase4.py"),
        help="Path to the Phase 4 parser script.",
    )

    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Return an empty summary instead of failing if artifact root does not exist.",
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="Also include compact_runs with the most important fields.",
    )

    args = parser.parse_args()

    if not args.artifact_root.exists():
        if args.allow_empty:
            artifact_dirs: list[Path] = []
        else:
            raise FileNotFoundError(f"artifact root does not exist: {args.artifact_root}")
    else:
        artifact_dirs = sorted(
            p for p in args.artifact_root.iterdir()
            if p.is_dir()
            and (not args.run_prefix or p.name.startswith(args.run_prefix))
            and is_aiperf_artifact_dir(p)
        )

    rows = [parse_one(args.parser_script, d) for d in artifact_dirs]
    rows = sorted(rows, key=sort_key)

    errors = [r for r in rows if r.get("error")]

    output: dict[str, Any] = {
        "artifact_root": str(args.artifact_root),
        "run_prefix": args.run_prefix,
        "run_count": len(rows),
        "error_count": len(errors),
        "counts_by_workload": aggregate_count(rows, "workload_type"),
        "counts_by_mode": aggregate_count(rows, "mode"),
        "counts_by_concurrency": aggregate_count(rows, "concurrency"),
        "metric_availability": collect_metric_availability(rows),
        "runs": rows,
    }

    if args.compact:
        output["compact_runs"] = [select_compact_run_fields(r) for r in rows]

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
