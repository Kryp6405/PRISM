#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
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


def infer_mode_from_name(name: str) -> str | None:
    if "-aggregated-" in name:
        return "aggregated"
    if "-encoder-only-" in name:
        return "encoder_only"
    if "-full-disagg-" in name:
        return "full_disagg"
    return None


def gpu_role_map_for_mode(mode: str | None) -> dict[int, str]:
    if mode == "encoder_only":
        return {0: "encoder", 1: "pd"}
    if mode == "full_disagg":
        return {0: "encoder", 1: "prefill", 2: "decode"}
    return {}


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


def _gpu_metric_summary(vals: list[float], prefix: str) -> dict[str, Any]:
    if not vals:
        return {
            f"{prefix}_avg": None,
            f"{prefix}_std": None,
            f"{prefix}_p50": None,
            f"{prefix}_p95": None,
            f"{prefix}_peak": None,
        }
    return {
        f"{prefix}_avg": mean(vals),
        f"{prefix}_std": pstdev(vals) if len(vals) > 1 else 0.0,
        f"{prefix}_p50": percentile(vals, 50),
        f"{prefix}_p95": percentile(vals, 95),
        f"{prefix}_peak": max(vals),
    }


def parse_gpu_telemetry_jsonl(path: Path, mode: str | None) -> dict[str, Any]:
    gpu_role_map = gpu_role_map_for_mode(mode)

    per_gpu: dict[int, dict[str, list[float]]] = {}
    all_gpu_indices: set[int] = set()

    def ensure_gpu(g: int) -> None:
        if g not in per_gpu:
            per_gpu[g] = {
                "gpu_util": [],
                "gpu_mem_used_gb": [],
                "gpu_power_w": [],
                "gpu_temp_c": [],
                "gpu_mem_util": [],
                "gpu_sm_util": [],
                "gpu_decoder_util": [],
                "gpu_encoder_util": [],
                "gpu_jpg_util": [],
            }

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue

        gpu_index = row.get("gpu_index")
        if not isinstance(gpu_index, int):
            continue

        all_gpu_indices.add(gpu_index)
        ensure_gpu(gpu_index)

        td = row.get("telemetry_data", {})
        mapping = {
            "gpu_util": td.get("gpu_utilization"),
            "gpu_mem_used_gb": td.get("gpu_memory_used"),
            "gpu_power_w": td.get("gpu_power_usage"),
            "gpu_temp_c": td.get("gpu_temperature"),
            "gpu_mem_util": td.get("mem_utilization"),
            "gpu_sm_util": td.get("sm_utilization"),
            "gpu_decoder_util": td.get("decoder_utilization"),
            "gpu_encoder_util": td.get("encoder_utilization"),
            "gpu_jpg_util": td.get("jpg_utilization"),
        }
        for k, raw in mapping.items():
            v = safe_float(raw)
            if v is not None:
                per_gpu[gpu_index][k].append(v)

    out: dict[str, Any] = {
        "gpu_count_observed": len(all_gpu_indices) if all_gpu_indices else None,
    }

    # Aggregate across all GPU samples
    for metric_key, prefix in [
        ("gpu_util", "gpu_util"),
        ("gpu_mem_used_gb", "gpu_mem_used_gb"),
        ("gpu_power_w", "gpu_power_w"),
        ("gpu_temp_c", "gpu_temp_c"),
        ("gpu_mem_util", "gpu_mem_util"),
        ("gpu_sm_util", "gpu_sm_util"),
        ("gpu_decoder_util", "gpu_decoder_util"),
        ("gpu_encoder_util", "gpu_encoder_util"),
        ("gpu_jpg_util", "gpu_jpg_util"),
    ]:
        all_vals: list[float] = []
        for g in sorted(per_gpu):
            all_vals.extend(per_gpu[g][metric_key])
        out.update(_gpu_metric_summary(all_vals, prefix))

    # Active GPU count: util > 10 or mem > 1 GB on a sample-average basis per gpu
    active_count = 0
    for g in sorted(per_gpu):
        util_avg = mean(per_gpu[g]["gpu_util"]) if per_gpu[g]["gpu_util"] else 0.0
        mem_avg = mean(per_gpu[g]["gpu_mem_used_gb"]) if per_gpu[g]["gpu_mem_used_gb"] else 0.0
        if util_avg > 10.0 or mem_avg > 1.0:
            active_count += 1
    out["gpu_active_count_avg"] = active_count if per_gpu else None

    # Per-GPU summaries
    per_gpu_mem_peaks = []
    per_gpu_util_peaks = []
    for g in sorted(per_gpu):
        for metric_key, prefix in [
            ("gpu_util", f"gpu{g}_util"),
            ("gpu_mem_used_gb", f"gpu{g}_mem_used_gb"),
            ("gpu_power_w", f"gpu{g}_power_w"),
            ("gpu_temp_c", f"gpu{g}_temp_c"),
            ("gpu_mem_util", f"gpu{g}_mem_util"),
            ("gpu_sm_util", f"gpu{g}_sm_util"),
        ]:
            out.update(_gpu_metric_summary(per_gpu[g][metric_key], prefix))

        util_peak = max(per_gpu[g]["gpu_util"]) if per_gpu[g]["gpu_util"] else None
        mem_peak = max(per_gpu[g]["gpu_mem_used_gb"]) if per_gpu[g]["gpu_mem_used_gb"] else None
        if util_peak is not None:
            per_gpu_util_peaks.append(util_peak)
        if mem_peak is not None:
            per_gpu_mem_peaks.append(mem_peak)

    # Imbalance metrics
    out["gpu_util_peak_skew"] = (
        max(per_gpu_util_peaks) - min(per_gpu_util_peaks)
        if len(per_gpu_util_peaks) >= 2 else None
    )
    out["gpu_mem_peak_skew_gb"] = (
        max(per_gpu_mem_peaks) - min(per_gpu_mem_peaks)
        if len(per_gpu_mem_peaks) >= 2 else None
    )

    # Role-labeled summaries for disaggregated setups
    for gpu_idx, role in gpu_role_map.items():
        if gpu_idx not in per_gpu:
            continue
        for metric_key, prefix in [
            ("gpu_util", f"{role}_gpu_util"),
            ("gpu_mem_used_gb", f"{role}_gpu_mem_used_gb"),
            ("gpu_power_w", f"{role}_gpu_power_w"),
            ("gpu_temp_c", f"{role}_gpu_temp_c"),
            ("gpu_mem_util", f"{role}_gpu_mem_util"),
            ("gpu_sm_util", f"{role}_gpu_sm_util"),
        ]:
            out.update(_gpu_metric_summary(per_gpu[gpu_idx][metric_key], prefix))

    return out


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

    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("Endpoint,Type,Metric,Unit,avg,min,max,std,"):
            header = line.split(",")
            i += 1
            while i < len(lines) and not lines[i].startswith("Endpoint,Type,Metric,Unit,"):
                row_vals = lines[i].split(",")
                if len(row_vals) == len(header):
                    row = dict(zip(header, row_vals))
                    metric = row.get("Metric", "")
                    row_type = row.get("Type", "")
                    if metric == "dynamo_frontend_inflight_requests" and row_type == "gauge":
                        summary["frontend_inflight_requests_avg"] = safe_float(row.get("avg"))
                    elif metric == "dynamo_frontend_queued_requests" and row_type == "gauge":
                        summary["frontend_queued_requests_avg"] = safe_float(row.get("avg"))
                i += 1
            continue

        if line.startswith("Endpoint,Type,Metric,Unit,total,rate,rate_avg,"):
            header = line.split(",")
            i += 1
            while i < len(lines) and not lines[i].startswith("Endpoint,Type,Metric,Unit,"):
                row_vals = lines[i].split(",")
                if len(row_vals) == len(header):
                    row = dict(zip(header, row_vals))
                    metric = row.get("Metric", "")
                    row_type = row.get("Type", "")
                    if metric == "dynamo_frontend_output_tokens" and row_type == "counter":
                        summary["frontend_output_tokens_total"] = safe_float(row.get("total"))
                        summary["frontend_output_tokens_rate"] = safe_float(row.get("rate"))
                    elif metric == "dynamo_frontend_requests" and row_type == "counter":
                        summary["frontend_requests_total"] = safe_float(row.get("total"))
                        summary["frontend_requests_rate"] = safe_float(row.get("rate"))
                i += 1
            continue

        if line.startswith("Endpoint,Type,Metric,Unit,count,count_rate,sum,sum_rate,avg,"):
            header = line.split(",")
            i += 1
            while i < len(lines) and not lines[i].startswith("Endpoint,Type,Metric,Unit,"):
                row_vals = lines[i].split(",")
                if len(row_vals) > len(header):
                    row_vals = row_vals[: len(header) - 1] + [",".join(row_vals[len(header) - 1 :])]
                if len(row_vals) == len(header):
                    row = dict(zip(header, row_vals))
                    metric = row.get("Metric", "")
                    row_type = row.get("Type", "")
                    if row_type == "histogram":
                        if metric == "dynamo_frontend_time_to_first_token_seconds":
                            summary["frontend_ttft_s_avg"] = safe_float(row.get("avg"))
                            summary["frontend_ttft_s_p50"] = safe_float(row.get("p50_estimate"))
                            summary["frontend_ttft_s_p95"] = safe_float(row.get("p95_estimate"))
                            summary["frontend_ttft_s_p99"] = safe_float(row.get("p99_estimate"))
                        elif metric == "dynamo_frontend_inter_token_latency_seconds":
                            summary["frontend_itl_s_avg"] = safe_float(row.get("avg"))
                            summary["frontend_itl_s_p50"] = safe_float(row.get("p50_estimate"))
                            summary["frontend_itl_s_p95"] = safe_float(row.get("p95_estimate"))
                            summary["frontend_itl_s_p99"] = safe_float(row.get("p99_estimate"))
                        elif metric == "dynamo_frontend_request_duration_seconds":
                            summary["frontend_request_duration_s_avg"] = safe_float(row.get("avg"))
                            summary["frontend_request_duration_s_p50"] = safe_float(row.get("p50_estimate"))
                            summary["frontend_request_duration_s_p95"] = safe_float(row.get("p95_estimate"))
                            summary["frontend_request_duration_s_p99"] = safe_float(row.get("p99_estimate"))
                        elif metric == "dynamo_frontend_output_sequence_tokens":
                            summary["frontend_output_seq_tokens_avg"] = safe_float(row.get("avg"))
                            summary["frontend_output_seq_tokens_p50"] = safe_float(row.get("p50_estimate"))
                            summary["frontend_output_seq_tokens_p95"] = safe_float(row.get("p95_estimate"))
                        elif metric == "dynamo_frontend_input_sequence_tokens":
                            summary["frontend_input_seq_tokens_avg"] = safe_float(row.get("avg"))
                            summary["frontend_input_seq_tokens_p50"] = safe_float(row.get("p50_estimate"))
                            summary["frontend_input_seq_tokens_p95"] = safe_float(row.get("p95_estimate"))
                i += 1
            continue

        i += 1

    return summary


def merge_fill_none(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for k, v in new.items():
        if k not in base or base[k] in (None, ""):
            if v not in (None, ""):
                base[k] = v
    return base


def parse_artifact_dir(artifact_dir: Path) -> dict[str, Any]:
    mode = infer_mode_from_name(artifact_dir.name)

    summary = {
        "artifact_dir": str(artifact_dir),
        "artifact_name": artifact_dir.name,
        "mode": mode,
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
        "gpu_active_count_avg": None,
        "gpu_util_avg": None,
        "gpu_util_std": None,
        "gpu_util_p50": None,
        "gpu_util_p95": None,
        "gpu_util_peak": None,
        "gpu_mem_used_gb_avg": None,
        "gpu_mem_used_gb_std": None,
        "gpu_mem_used_gb_p50": None,
        "gpu_mem_used_gb_p95": None,
        "gpu_mem_used_gb_peak": None,
        "gpu_power_w_avg": None,
        "gpu_power_w_std": None,
        "gpu_power_w_p50": None,
        "gpu_power_w_p95": None,
        "gpu_power_w_peak": None,
        "gpu_temp_c_avg": None,
        "gpu_temp_c_std": None,
        "gpu_temp_c_p50": None,
        "gpu_temp_c_p95": None,
        "gpu_temp_c_peak": None,
        "gpu_mem_util_avg": None,
        "gpu_mem_util_std": None,
        "gpu_mem_util_p50": None,
        "gpu_mem_util_p95": None,
        "gpu_mem_util_peak": None,
        "gpu_sm_util_avg": None,
        "gpu_sm_util_std": None,
        "gpu_sm_util_p50": None,
        "gpu_sm_util_p95": None,
        "gpu_sm_util_peak": None,
        "gpu_decoder_util_avg": None,
        "gpu_decoder_util_std": None,
        "gpu_decoder_util_p50": None,
        "gpu_decoder_util_p95": None,
        "gpu_decoder_util_peak": None,
        "gpu_encoder_util_avg": None,
        "gpu_encoder_util_std": None,
        "gpu_encoder_util_p50": None,
        "gpu_encoder_util_p95": None,
        "gpu_encoder_util_peak": None,
        "gpu_jpg_util_avg": None,
        "gpu_jpg_util_std": None,
        "gpu_jpg_util_p50": None,
        "gpu_jpg_util_p95": None,
        "gpu_jpg_util_peak": None,
        "gpu_util_peak_skew": None,
        "gpu_mem_peak_skew_gb": None,

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
        summary = merge_fill_none(summary, parse_gpu_telemetry_jsonl(gpu_jsonl, mode))
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
