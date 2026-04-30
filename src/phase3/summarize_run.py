#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def is_aiperf_artifact_dir(path: Path) -> bool:
    return (
        (path / "profile_export_aiperf.json").exists()
        or (path / "profile_export.jsonl").exists()
        or (path / "profile_export_aiperf.csv").exists()
        or (path / "gpu_telemetry_export.jsonl").exists()
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


def sort_key(row: dict[str, Any]) -> tuple:
    mode_order = {
        "aggregated": 0,
        "encoder_only": 1,
        "full_disagg": 2,
        None: 99,
    }

    mode = row.get("mode")
    concurrency = row.get("concurrency")

    try:
        c = int(concurrency)
    except Exception:
        c = 999999

    return (
        c,
        mode_order.get(mode, 99),
        row.get("artifact_name", row.get("artifact_dir", "")),
    )


def aggregate_run_count_by_workload(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        w = r.get("workload_type") or "unknown"
        out[w] = out.get(w, 0) + 1
    return out


def aggregate_run_count_by_mode(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        m = r.get("mode") or "unknown"
        out[m] = out.get(m, 0) + 1
    return out


def aggregate_run_count_by_concurrency(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        c = r.get("concurrency")
        key = str(c if c is not None else "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def collect_metric_availability(rows: list[dict[str, Any]]) -> dict[str, int]:
    important_metrics = [
        "throughput",
        "e2e_ms",
        "e2e_ms_p50",
        "e2e_ms_p95",
        "e2e_ms_p99",
        "output_token_throughput",
        "throughput_per_gpu",
        "output_tps_per_gpu",
        "input_sequence_length_avg",
        "output_sequence_length_avg",
        "request_count",
        "benchmark_duration_sec",
        "gpu_util_avg",
        "gpu_util_p95",
        "gpu_util_peak",
        "gpu_mem_used_gb_avg",
        "gpu_mem_used_gb_peak",
        "gpu_power_w_avg",
        "gpu_power_w_peak",
        "gpu_sm_util_avg",
        "gpu_mem_util_avg",
        "gpu_temp_c_peak",
        "gpu_mem_peak_skew_gb",
        "gpu_util_peak_skew",
        "frontend_ttft_s_avg",
        "frontend_itl_s_avg",
        "frontend_request_duration_s_avg",
        "frontend_inflight_requests_avg",
        "frontend_queued_requests_avg",
    ]

    out: dict[str, int] = {}
    for metric in important_metrics:
        out[metric] = sum(1 for r in rows if r.get(metric) is not None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--parser-script", type=Path, default=Path("src/phase3/parse_aiperf.py"))
    parser.add_argument("--allow-empty", action="store_true")
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
            and p.name.startswith(args.run_prefix)
            and is_aiperf_artifact_dir(p)
        )

    rows = [parse_one(args.parser_script, d) for d in artifact_dirs]
    rows = sorted(rows, key=sort_key)

    errors = [r for r in rows if r.get("error")]

    output = {
        "artifact_root": str(args.artifact_root),
        "run_prefix": args.run_prefix,
        "run_count": len(rows),
        "error_count": len(errors),
        "counts_by_workload": aggregate_run_count_by_workload(rows),
        "counts_by_mode": aggregate_run_count_by_mode(rows),
        "counts_by_concurrency": aggregate_run_count_by_concurrency(rows),
        "metric_availability": collect_metric_availability(rows),
        "runs": rows,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
