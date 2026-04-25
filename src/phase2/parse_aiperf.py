#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def infer_concurrency_from_name(name: str) -> int | None:
    m = re.search(r"concurrency(\d+)", name)
    return int(m.group(1)) if m else None


def infer_generation_from_name(name: str) -> str | None:
    for g in ("short", "medium", "long"):
        if f"_{g}_" in name:
            return g
    return None


def parse_profile_export_aiperf_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())

    return {
        "throughput": safe_float(data.get("request_throughput", {}).get("avg")),
        "ttft_ms": None,
        "ttft_ms_p50": None,
        "ttft_ms_p95": None,
        "ttft_ms_p99": None,
        "tbt_ms": None,
        "tbt_ms_p50": None,
        "tbt_ms_p95": None,
        "tbt_ms_p99": None,
        "e2e_ms": safe_float(data.get("request_latency", {}).get("avg")),
        "e2e_ms_p50": safe_float(data.get("request_latency", {}).get("p50")),
        "e2e_ms_p95": safe_float(data.get("request_latency", {}).get("p95")),
        "e2e_ms_p99": safe_float(data.get("request_latency", {}).get("p99")),
        "output_token_throughput": safe_float(data.get("output_token_throughput", {}).get("avg")),
        "request_count": safe_float(data.get("request_count", {}).get("avg")),
        "output_token_count_avg": safe_float(data.get("output_token_count", {}).get("avg")),
        "input_sequence_length_avg": safe_float(data.get("input_sequence_length", {}).get("avg")),
        "output_sequence_length_avg": safe_float(data.get("output_sequence_length", {}).get("avg")),
        "benchmark_duration_sec": safe_float(data.get("benchmark_duration", {}).get("avg")),
        "total_output_tokens": safe_float(data.get("total_output_tokens", {}).get("avg")),
    }


def parse_profile_export_jsonl(path: Path) -> dict[str, Any]:
    ttft_vals: list[float] = []
    tbt_vals: list[float] = []
    e2e_vals: list[float] = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue

        metrics = row.get("metrics", {})

        ttft = safe_float(metrics.get("time_to_first_token", {}).get("value"))
        tbt = safe_float(metrics.get("inter_token_latency", {}).get("value"))
        e2e = safe_float(metrics.get("request_latency", {}).get("value"))

        if ttft is not None:
            ttft_vals.append(ttft)
        if tbt is not None:
            tbt_vals.append(tbt)
        if e2e is not None:
            e2e_vals.append(e2e)

    return {
        "ttft_ms": mean(ttft_vals) if ttft_vals else None,
        "ttft_ms_p50": percentile(ttft_vals, 50),
        "ttft_ms_p95": percentile(ttft_vals, 95),
        "ttft_ms_p99": percentile(ttft_vals, 99),
        "tbt_ms": mean(tbt_vals) if tbt_vals else None,
        "tbt_ms_p50": percentile(tbt_vals, 50),
        "tbt_ms_p95": percentile(tbt_vals, 95),
        "tbt_ms_p99": percentile(tbt_vals, 99),
        "e2e_ms": mean(e2e_vals) if e2e_vals else None,
        "e2e_ms_p50": percentile(e2e_vals, 50),
        "e2e_ms_p95": percentile(e2e_vals, 95),
        "e2e_ms_p99": percentile(e2e_vals, 99),
    }


def parse_gpu_telemetry_jsonl(path: Path) -> dict[str, Any]:
    gpu_indices: set[int] = set()

    gpu_util_vals = []
    gpu_mem_vals = []
    gpu_power_vals = []
    gpu_temp_vals = []
    mem_util_vals = []
    sm_util_vals = []
    decoder_util_vals = []
    encoder_util_vals = []
    jpg_util_vals = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue

        gpu_index = row.get("gpu_index")
        if isinstance(gpu_index, int):
            gpu_indices.add(gpu_index)

        td = row.get("telemetry_data", {})

        for src, dest in [
            (td.get("gpu_utilization"), gpu_util_vals),
            (td.get("gpu_memory_used"), gpu_mem_vals),
            (td.get("gpu_power_usage"), gpu_power_vals),
            (td.get("gpu_temperature"), gpu_temp_vals),
            (td.get("mem_utilization"), mem_util_vals),
            (td.get("sm_utilization"), sm_util_vals),
            (td.get("decoder_utilization"), decoder_util_vals),
            (td.get("encoder_utilization"), encoder_util_vals),
            (td.get("jpg_utilization"), jpg_util_vals),
        ]:
            v = safe_float(src)
            if v is not None:
                dest.append(v)

    return {
        "gpu_count_observed": len(gpu_indices) if gpu_indices else None,
        "gpu_util_avg": mean(gpu_util_vals) if gpu_util_vals else None,
        "gpu_util_peak": max(gpu_util_vals) if gpu_util_vals else None,
        "gpu_mem_used_avg_gb": mean(gpu_mem_vals) if gpu_mem_vals else None,
        "gpu_mem_used_peak_gb": max(gpu_mem_vals) if gpu_mem_vals else None,
        "gpu_power_avg_w": mean(gpu_power_vals) if gpu_power_vals else None,
        "gpu_power_peak_w": max(gpu_power_vals) if gpu_power_vals else None,
        "gpu_temp_avg_c": mean(gpu_temp_vals) if gpu_temp_vals else None,
        "gpu_temp_peak_c": max(gpu_temp_vals) if gpu_temp_vals else None,
        "gpu_mem_util_avg": mean(mem_util_vals) if mem_util_vals else None,
        "gpu_mem_util_peak": max(mem_util_vals) if mem_util_vals else None,
        "gpu_sm_util_avg": mean(sm_util_vals) if sm_util_vals else None,
        "gpu_sm_util_peak": max(sm_util_vals) if sm_util_vals else None,
        "gpu_decoder_util_avg": mean(decoder_util_vals) if decoder_util_vals else None,
        "gpu_decoder_util_peak": max(decoder_util_vals) if decoder_util_vals else None,
        "gpu_encoder_util_avg": mean(encoder_util_vals) if encoder_util_vals else None,
        "gpu_encoder_util_peak": max(encoder_util_vals) if encoder_util_vals else None,
        "gpu_jpg_util_avg": mean(jpg_util_vals) if jpg_util_vals else None,
        "gpu_jpg_util_peak": max(jpg_util_vals) if jpg_util_vals else None,
    }


def parse_server_metrics_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    configured = summary.get("endpoints_configured", []) or []
    successful = summary.get("endpoints_successful", []) or []

    return {
        "server_metrics_enabled": bool(configured),
        "server_metrics_endpoint_count": len(configured),
        "server_metrics_success_count": len(successful),
    }


def parse_server_metrics_csv(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "frontend_ttft_s_avg": None,
        "frontend_ttft_s_p50": None,
        "frontend_ttft_s_p95": None,
        "frontend_ttft_s_p99": None,
        "frontend_itl_s_avg": None,
        "frontend_itl_s_p50": None,
        "frontend_itl_s_p95": None,
        "frontend_itl_s_p99": None,
        "frontend_request_duration_s_avg": None,
        "frontend_request_duration_s_p50": None,
        "frontend_request_duration_s_p95": None,
        "frontend_request_duration_s_p99": None,
        "frontend_output_seq_tokens_avg": None,
        "frontend_output_seq_tokens_p50": None,
        "frontend_output_seq_tokens_p95": None,
        "frontend_input_seq_tokens_avg": None,
        "frontend_input_seq_tokens_p50": None,
        "frontend_input_seq_tokens_p95": None,
        "frontend_output_tokens_total": None,
        "frontend_output_tokens_rate": None,
        "frontend_requests_total": None,
        "frontend_requests_rate": None,
        "frontend_inflight_requests_avg": None,
        "frontend_queued_requests_avg": None,
    }

    metric_to_fields = {
        "dynamo_frontend_time_to_first_token_seconds": [
            ("avg", "frontend_ttft_s_avg"),
            ("p50_estimate", "frontend_ttft_s_p50"),
            ("p95_estimate", "frontend_ttft_s_p95"),
            ("p99_estimate", "frontend_ttft_s_p99"),
        ],
        "dynamo_frontend_inter_token_latency_seconds": [
            ("avg", "frontend_itl_s_avg"),
            ("p50_estimate", "frontend_itl_s_p50"),
            ("p95_estimate", "frontend_itl_s_p95"),
            ("p99_estimate", "frontend_itl_s_p99"),
        ],
        "dynamo_frontend_request_duration_seconds": [
            ("avg", "frontend_request_duration_s_avg"),
            ("p50_estimate", "frontend_request_duration_s_p50"),
            ("p95_estimate", "frontend_request_duration_s_p95"),
            ("p99_estimate", "frontend_request_duration_s_p99"),
        ],
        "dynamo_frontend_output_sequence_tokens": [
            ("avg", "frontend_output_seq_tokens_avg"),
            ("p50_estimate", "frontend_output_seq_tokens_p50"),
            ("p95_estimate", "frontend_output_seq_tokens_p95"),
        ],
        "dynamo_frontend_input_sequence_tokens": [
            ("avg", "frontend_input_seq_tokens_avg"),
            ("p50_estimate", "frontend_input_seq_tokens_p50"),
            ("p95_estimate", "frontend_input_seq_tokens_p95"),
        ],
    }

    with path.open(newline="") as f:
        # remove comment lines, then feed remaining rows to DictReader
        lines = [line for line in f if line.strip() and not line.startswith("#")]
        reader = csv.DictReader(lines)

        for row in reader:
            metric = (row.get("Metric") or "").strip()
            row_type = (row.get("Type") or "").strip()

            if metric == "dynamo_frontend_output_tokens" and row_type == "counter":
                summary["frontend_output_tokens_total"] = safe_float(row.get("total"))
                summary["frontend_output_tokens_rate"] = safe_float(row.get("rate"))

            elif metric == "dynamo_frontend_requests" and row_type == "counter":
                summary["frontend_requests_total"] = safe_float(row.get("total"))
                summary["frontend_requests_rate"] = safe_float(row.get("rate"))

            elif metric == "dynamo_frontend_inflight_requests" and row_type == "gauge":
                summary["frontend_inflight_requests_avg"] = safe_float(row.get("avg"))

            elif metric == "dynamo_frontend_queued_requests" and row_type == "gauge":
                summary["frontend_queued_requests_avg"] = safe_float(row.get("avg"))

            elif metric in metric_to_fields and row_type == "histogram":
                for csv_field, out_field in metric_to_fields[metric]:
                    val = safe_float(row.get(csv_field))
                    if val is not None:
                        summary[out_field] = val

    return summary


def merge_fill_none(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for k, v in new.items():
        if k not in base or base[k] in (None, ""):
            if v not in (None, ""):
                base[k] = v
    return base


def parse_artifact_dir(artifact_dir: Path) -> dict[str, Any]:
    summary = {
        "artifact_dir": str(artifact_dir),
        "artifact_name": artifact_dir.name,
        "generation": infer_generation_from_name(artifact_dir.name),
        "concurrency": infer_concurrency_from_name(artifact_dir.name),

        "throughput": None,
        "ttft_ms": None,
        "ttft_ms_p50": None,
        "ttft_ms_p95": None,
        "ttft_ms_p99": None,
        "tbt_ms": None,
        "tbt_ms_p50": None,
        "tbt_ms_p95": None,
        "tbt_ms_p99": None,
        "e2e_ms": None,
        "e2e_ms_p50": None,
        "e2e_ms_p95": None,
        "e2e_ms_p99": None,
        "output_token_throughput": None,
        "request_count": None,
        "output_token_count_avg": None,
        "input_sequence_length_avg": None,
        "output_sequence_length_avg": None,
        "benchmark_duration_sec": None,
        "total_output_tokens": None,

        "gpu_count_observed": None,
        "gpu_util_avg": None,
        "gpu_util_peak": None,
        "gpu_mem_used_avg_gb": None,
        "gpu_mem_used_peak_gb": None,
        "gpu_power_avg_w": None,
        "gpu_power_peak_w": None,
        "gpu_temp_avg_c": None,
        "gpu_temp_peak_c": None,
        "gpu_mem_util_avg": None,
        "gpu_mem_util_peak": None,
        "gpu_sm_util_avg": None,
        "gpu_sm_util_peak": None,
        "gpu_decoder_util_avg": None,
        "gpu_decoder_util_peak": None,
        "gpu_encoder_util_avg": None,
        "gpu_encoder_util_peak": None,
        "gpu_jpg_util_avg": None,
        "gpu_jpg_util_peak": None,

        "server_metrics_enabled": None,
        "server_metrics_endpoint_count": None,
        "server_metrics_success_count": None,

        "frontend_ttft_s_avg": None,
        "frontend_ttft_s_p50": None,
        "frontend_ttft_s_p95": None,
        "frontend_ttft_s_p99": None,
        "frontend_itl_s_avg": None,
        "frontend_itl_s_p50": None,
        "frontend_itl_s_p95": None,
        "frontend_itl_s_p99": None,
        "frontend_request_duration_s_avg": None,
        "frontend_request_duration_s_p50": None,
        "frontend_request_duration_s_p95": None,
        "frontend_request_duration_s_p99": None,
        "frontend_output_seq_tokens_avg": None,
        "frontend_output_seq_tokens_p50": None,
        "frontend_output_seq_tokens_p95": None,
        "frontend_input_seq_tokens_avg": None,
        "frontend_input_seq_tokens_p50": None,
        "frontend_input_seq_tokens_p95": None,
        "frontend_output_tokens_total": None,
        "frontend_output_tokens_rate": None,
        "frontend_requests_total": None,
        "frontend_requests_rate": None,
        "frontend_inflight_requests_avg": None,
        "frontend_queued_requests_avg": None,

        "sources": [],
    }

    aiperf_json = artifact_dir / "profile_export_aiperf.json"
    jsonl = artifact_dir / "profile_export.jsonl"
    gpu_jsonl = artifact_dir / "gpu_telemetry_export.jsonl"
    server_json = artifact_dir / "server_metrics_export.json"
    server_csv = artifact_dir / "server_metrics_export.csv"

    if aiperf_json.exists():
        summary = merge_fill_none(summary, parse_profile_export_aiperf_json(aiperf_json))
        summary["sources"].append(aiperf_json.name)

    if jsonl.exists():
        summary = merge_fill_none(summary, parse_profile_export_jsonl(jsonl))
        summary["sources"].append(jsonl.name)

    if gpu_jsonl.exists():
        summary = merge_fill_none(summary, parse_gpu_telemetry_jsonl(gpu_jsonl))
        summary["sources"].append(gpu_jsonl.name)

    if server_json.exists():
        summary = merge_fill_none(summary, parse_server_metrics_json(server_json))
        summary["sources"].append(server_json.name)

    if server_csv.exists():
        summary = merge_fill_none(summary, parse_server_metrics_csv(server_csv))
        summary["sources"].append(server_csv.name)

    csv_path = artifact_dir / "profile_export_aiperf.csv"
    if csv_path.exists():
        summary["sources"].append(csv_path.name)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(parse_artifact_dir(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
