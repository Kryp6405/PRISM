#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


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


def stat_summary(vals: list[float], prefix: str) -> dict[str, Any]:
    if not vals:
        return {
            f"{prefix}_avg": None,
            f"{prefix}_std": None,
            f"{prefix}_p50": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_peak": None,
        }

    return {
        f"{prefix}_avg": mean(vals),
        f"{prefix}_std": pstdev(vals) if len(vals) > 1 else 0.0,
        f"{prefix}_p50": percentile(vals, 50),
        f"{prefix}_p95": percentile(vals, 95),
        f"{prefix}_p99": percentile(vals, 99),
        f"{prefix}_min": min(vals),
        f"{prefix}_max": max(vals),
        f"{prefix}_peak": max(vals),
    }


def get_stat(obj: dict[str, Any] | None, stat: str = "avg") -> float | None:
    if not isinstance(obj, dict):
        return None
    return safe_float(obj.get(stat))


def infer_concurrency_from_name(name: str) -> int | None:
    m = re.search(r"concurrency(\d+)", name)
    return int(m.group(1)) if m else None


def infer_generation_from_name(name: str) -> str | None:
    for g in ("short", "medium", "long"):
        if f"_{g}_" in name:
            return g
    return None


def infer_mode_from_name(name: str) -> str | None:
    if "aggregated" in name:
        return "aggregated"
    if "encoder-only" in name or "encoder_only" in name:
        return "encoder_only"
    if "full-disagg" in name or "full_disagg" in name:
        return "full_disagg"
    return None


def gpu_budget_for_mode(mode: str | None) -> int | None:
    if mode == "aggregated":
        return 1
    if mode == "encoder_only":
        return 2
    if mode == "full_disagg":
        return 3
    return None


def gpu_role_map_for_mode(mode: str | None) -> dict[int, str]:
    if mode == "aggregated":
        return {0: "aggregated"}
    if mode == "encoder_only":
        return {0: "encoder", 1: "pd"}
    if mode == "full_disagg":
        return {0: "encoder", 1: "prefill", 2: "decode"}
    return {}


def used_gpu_indices_for_mode(mode: str | None) -> list[int]:
    return sorted(gpu_role_map_for_mode(mode).keys())


# ---------------------------------------------------------------------------
# Request-side AIPerf summary
# ---------------------------------------------------------------------------

def parse_profile_export_aiperf_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())

    out: dict[str, Any] = {}

    out["aiperf_version"] = data.get("aiperf_version")
    out["benchmark_id"] = data.get("benchmark_id")
    out["was_cancelled"] = data.get("was_cancelled")
    out["start_time"] = data.get("start_time")
    out["end_time"] = data.get("end_time")

    # Main request-side metrics.
    out["throughput"] = get_stat(data.get("request_throughput"))
    out["request_throughput"] = out["throughput"]

    latency = data.get("request_latency", {})
    out["e2e_ms"] = get_stat(latency, "avg")
    out["e2e_ms_p50"] = get_stat(latency, "p50")
    out["e2e_ms_p95"] = get_stat(latency, "p95")
    out["e2e_ms_p99"] = get_stat(latency, "p99")
    out["e2e_ms_min"] = get_stat(latency, "min")
    out["e2e_ms_max"] = get_stat(latency, "max")
    out["e2e_ms_std"] = get_stat(latency, "std")

    out["output_token_throughput"] = get_stat(data.get("output_token_throughput"))
    out["request_count"] = get_stat(data.get("request_count"))
    out["benchmark_duration_sec"] = get_stat(data.get("benchmark_duration"))
    out["total_output_tokens"] = get_stat(data.get("total_output_tokens"))
    out["total_isl"] = get_stat(data.get("total_isl"))
    out["total_osl"] = get_stat(data.get("total_osl"))

    # Sequence/token sanity metrics.
    isl = data.get("input_sequence_length", {})
    osl = data.get("output_sequence_length", {})
    otc = data.get("output_token_count", {})
    num_images = data.get("num_images", {})

    out["input_sequence_length_avg"] = get_stat(isl, "avg")
    out["input_sequence_length_p50"] = get_stat(isl, "p50")
    out["input_sequence_length_p95"] = get_stat(isl, "p95")

    out["output_sequence_length_avg"] = get_stat(osl, "avg")
    out["output_sequence_length_p50"] = get_stat(osl, "p50")
    out["output_sequence_length_p95"] = get_stat(osl, "p95")
    out["output_sequence_length_p99"] = get_stat(osl, "p99")
    out["output_sequence_length_min"] = get_stat(osl, "min")
    out["output_sequence_length_max"] = get_stat(osl, "max")
    out["output_sequence_length_std"] = get_stat(osl, "std")

    out["output_token_count_avg"] = get_stat(otc, "avg")
    out["output_token_count_p50"] = get_stat(otc, "p50")
    out["output_token_count_p95"] = get_stat(otc, "p95")
    out["output_token_count_p99"] = get_stat(otc, "p99")

    out["num_images_avg"] = get_stat(num_images, "avg")

    # HTTP / loadgen metrics useful for debugging, not main paper plots.
    for metric_name, out_prefix in [
        ("http_req_duration", "http_req_duration_ms"),
        ("http_req_waiting", "http_req_waiting_ms"),
        ("http_req_sending", "http_req_sending_ms"),
        ("http_req_receiving", "http_req_receiving_ms"),
        ("http_req_connecting", "http_req_connecting_ms"),
        ("http_req_dns_lookup", "http_req_dns_lookup_ms"),
        ("http_req_data_sent", "http_req_data_sent_kb"),
        ("http_req_data_received", "http_req_data_received_kb"),
        ("http_req_connection_reused", "http_req_connection_reused"),
    ]:
        metric = data.get(metric_name)
        if isinstance(metric, dict):
            out[f"{out_prefix}_avg"] = get_stat(metric, "avg")
            out[f"{out_prefix}_p50"] = get_stat(metric, "p50")
            out[f"{out_prefix}_p95"] = get_stat(metric, "p95")
            out[f"{out_prefix}_p99"] = get_stat(metric, "p99")

    # Input config from AIPerf.
    input_config = data.get("input_config", {})
    loadgen = input_config.get("loadgen", {}) if isinstance(input_config, dict) else {}
    endpoint = input_config.get("endpoint", {}) if isinstance(input_config, dict) else {}
    input_cfg = input_config.get("input", {}) if isinstance(input_config, dict) else {}

    out["input_config_concurrency"] = safe_float(loadgen.get("concurrency"))
    out["input_config_request_count"] = safe_float(loadgen.get("request_count"))
    out["input_config_arrival_pattern"] = loadgen.get("arrival_pattern")
    out["endpoint_type"] = endpoint.get("type")
    out["endpoint_use_server_token_count"] = endpoint.get("use_server_token_count")

    try:
        out["image_width_mean"] = safe_float(input_cfg["image"]["width"]["mean"])
    except Exception:
        out["image_width_mean"] = None

    try:
        out["image_height_mean"] = safe_float(input_cfg["image"]["height"]["mean"])
    except Exception:
        out["image_height_mean"] = None

    try:
        out["prompt_output_tokens_mean"] = safe_float(input_cfg["prompt"]["output_tokens"]["mean"])
    except Exception:
        out["prompt_output_tokens_mean"] = None

    try:
        out["prompt_output_tokens_stddev"] = safe_float(input_cfg["prompt"]["output_tokens"]["stddev"])
    except Exception:
        out["prompt_output_tokens_stddev"] = None

    return out


# ---------------------------------------------------------------------------
# Request-level JSONL fallback
# ---------------------------------------------------------------------------

def parse_profile_export_jsonl(path: Path) -> dict[str, Any]:
    request_latency_vals: list[float] = []
    http_duration_vals: list[float] = []
    http_waiting_vals: list[float] = []
    output_token_vals: list[float] = []
    input_seq_vals: list[float] = []
    output_seq_vals: list[float] = []

    ttft_vals: list[float] = []
    tbt_vals: list[float] = []

    cancelled_count = 0
    total_count = 0

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except Exception:
            continue

        total_count += 1
        metadata = row.get("metadata", {})
        if metadata.get("was_cancelled") is True:
            cancelled_count += 1

        metrics = row.get("metrics", {})

        def metric_value(name: str) -> float | None:
            obj = metrics.get(name)
            if isinstance(obj, dict):
                return safe_float(obj.get("value"))
            return None

        for name, target in [
            ("request_latency", request_latency_vals),
            ("http_req_duration", http_duration_vals),
            ("http_req_waiting", http_waiting_vals),
            ("output_token_count", output_token_vals),
            ("input_sequence_length", input_seq_vals),
            ("output_sequence_length", output_seq_vals),
        ]:
            v = metric_value(name)
            if v is not None:
                target.append(v)

        # These may be absent unless streaming / server support emits them.
        for name in ("time_to_first_token", "ttft"):
            v = metric_value(name)
            if v is not None:
                ttft_vals.append(v)

        for name in ("inter_token_latency", "time_between_tokens", "tbt", "itl"):
            v = metric_value(name)
            if v is not None:
                tbt_vals.append(v)

    out: dict[str, Any] = {
        "profile_jsonl_request_count": total_count,
        "profile_jsonl_cancelled_count": cancelled_count,
    }

    # Only fill these as fallback-style names so we do not overwrite the
    # authoritative profile_export_aiperf.json values unless needed downstream.
    if request_latency_vals:
        out.update(stat_summary(request_latency_vals, "jsonl_e2e_ms"))
    if http_duration_vals:
        out.update(stat_summary(http_duration_vals, "jsonl_http_req_duration_ms"))
    if http_waiting_vals:
        out.update(stat_summary(http_waiting_vals, "jsonl_http_req_waiting_ms"))
    if output_token_vals:
        out.update(stat_summary(output_token_vals, "jsonl_output_token_count"))
    if input_seq_vals:
        out.update(stat_summary(input_seq_vals, "jsonl_input_sequence_length"))
    if output_seq_vals:
        out.update(stat_summary(output_seq_vals, "jsonl_output_sequence_length"))

    if ttft_vals:
        out["ttft_ms"] = mean(ttft_vals)
        out["ttft_ms_p50"] = percentile(ttft_vals, 50)
        out["ttft_ms_p95"] = percentile(ttft_vals, 95)
        out["ttft_ms_p99"] = percentile(ttft_vals, 99)
    else:
        out["ttft_ms"] = None
        out["ttft_ms_p50"] = None
        out["ttft_ms_p95"] = None
        out["ttft_ms_p99"] = None

    if tbt_vals:
        out["tbt_ms"] = mean(tbt_vals)
        out["tbt_ms_p50"] = percentile(tbt_vals, 50)
        out["tbt_ms_p95"] = percentile(tbt_vals, 95)
        out["tbt_ms_p99"] = percentile(tbt_vals, 99)
    else:
        out["tbt_ms"] = None
        out["tbt_ms_p50"] = None
        out["tbt_ms_p95"] = None
        out["tbt_ms_p99"] = None

    return out


# ---------------------------------------------------------------------------
# GPU telemetry from profile_export_aiperf.json summary
# ---------------------------------------------------------------------------

GPU_METRIC_NAME_MAP = {
    "gpu_utilization": "gpu_util",
    "gpu_memory_used": "gpu_mem_used_gb",
    "gpu_power_usage": "gpu_power_w",
    "gpu_temperature": "gpu_temp_c",
    "mem_utilization": "gpu_mem_util",
    "sm_utilization": "gpu_sm_util",
    "decoder_utilization": "gpu_decoder_util",
    "encoder_utilization": "gpu_encoder_util",
    "jpg_utilization": "gpu_jpg_util",
    "energy_consumption": "gpu_energy_mj",
    "power_violation": "gpu_power_violation_us",
}


def extract_gpu_metric_stats(metric_obj: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(metric_obj, dict):
        return out

    for stat in ("avg", "std", "p50", "p95", "p99", "min", "max"):
        val = safe_float(metric_obj.get(stat))
        out[f"{prefix}_{stat}"] = val

    # Peak alias for plotting.
    out[f"{prefix}_peak"] = out.get(f"{prefix}_max")
    return out


def parse_gpu_telemetry_from_aiperf_json(path: Path, mode: str | None) -> dict[str, Any]:
    data = json.loads(path.read_text())
    telemetry_data = data.get("telemetry_data", {})
    endpoints = telemetry_data.get("endpoints", {})

    role_map = gpu_role_map_for_mode(mode)
    used_indices = used_gpu_indices_for_mode(mode)

    # Currently AIPerf nests this as endpoints -> localhost -> gpus.
    all_gpus: dict[int, dict[str, Any]] = {}

    if isinstance(endpoints, dict):
        for endpoint_data in endpoints.values():
            if not isinstance(endpoint_data, dict):
                continue
            gpus = endpoint_data.get("gpus", {})
            if not isinstance(gpus, dict):
                continue

            for gpu_key, gpu_obj in gpus.items():
                if not isinstance(gpu_obj, dict):
                    continue
                idx = gpu_obj.get("gpu_index")
                if isinstance(idx, int):
                    all_gpus[idx] = gpu_obj

    out: dict[str, Any] = {
        "gpu_telemetry_source": "profile_export_aiperf.json",
        "gpu_count_observed": len(all_gpus) if all_gpus else None,
        "gpu_count_used": len(used_indices) if used_indices else None,
    }

    # Per-GPU stats.
    for idx, gpu_obj in sorted(all_gpus.items()):
        metrics = gpu_obj.get("metrics", {})
        out[f"gpu{idx}_name"] = gpu_obj.get("gpu_name")
        out[f"gpu{idx}_uuid"] = gpu_obj.get("gpu_uuid")

        for raw_name, prefix_name in GPU_METRIC_NAME_MAP.items():
            metric_obj = metrics.get(raw_name)
            out.update(extract_gpu_metric_stats(metric_obj, f"gpu{idx}_{prefix_name}"))

    # Aggregate across used GPUs only, not all observed GPUs.
    # This is important because AIPerf may observe idle GPUs 2/3 even in aggregated runs.
    used_metric_samples_by_stat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for idx in used_indices:
        gpu_obj = all_gpus.get(idx)
        if not gpu_obj:
            continue
        metrics = gpu_obj.get("metrics", {})
        for raw_name, prefix_name in GPU_METRIC_NAME_MAP.items():
            metric_obj = metrics.get(raw_name)
            if not isinstance(metric_obj, dict):
                continue
            for stat in ("avg", "p50", "p95", "p99", "min", "max"):
                val = safe_float(metric_obj.get(stat))
                if val is not None:
                    used_metric_samples_by_stat[prefix_name][stat].append(val)

    for prefix_name, stat_vals in used_metric_samples_by_stat.items():
        for stat, vals in stat_vals.items():
            if not vals:
                continue
            # Across used GPUs:
            # avg stats are averaged, p95/max stats use max to preserve hot-stage behavior.
            if stat in ("avg", "p50"):
                out[f"{prefix_name}_{stat}"] = mean(vals)
            else:
                out[f"{prefix_name}_{stat}"] = max(vals)

        # Standard naming aliases.
        out[f"{prefix_name}_peak"] = out.get(f"{prefix_name}_max")

    # Backward-compatible aliases used by older plotting scripts.
    if "gpu_mem_used_gb_avg" in out:
        out["gpu_mem_used_avg_gb"] = out["gpu_mem_used_gb_avg"]
    if "gpu_mem_used_gb_peak" in out:
        out["gpu_mem_used_peak_gb"] = out["gpu_mem_used_gb_peak"]
    if "gpu_power_w_avg" in out:
        out["gpu_power_avg_w"] = out["gpu_power_w_avg"]
    if "gpu_power_w_peak" in out:
        out["gpu_power_peak_w"] = out["gpu_power_w_peak"]
    if "gpu_temp_c_avg" in out:
        out["gpu_temp_avg_c"] = out["gpu_temp_c_avg"]
    if "gpu_temp_c_peak" in out:
        out["gpu_temp_peak_c"] = out["gpu_temp_c_peak"]

    # Role-labeled stats.
    for idx, role in role_map.items():
        gpu_obj = all_gpus.get(idx)
        if not gpu_obj:
            continue
        metrics = gpu_obj.get("metrics", {})

        for raw_name, prefix_name in GPU_METRIC_NAME_MAP.items():
            metric_obj = metrics.get(raw_name)
            out.update(extract_gpu_metric_stats(metric_obj, f"{role}_{prefix_name}"))

        # Backward-compatible aliases for common role metrics.
        if f"{role}_gpu_mem_used_gb_peak" in out:
            out[f"{role}_gpu_mem_used_peak_gb"] = out[f"{role}_gpu_mem_used_gb_peak"]
        if f"{role}_gpu_power_w_avg" in out:
            out[f"{role}_gpu_power_avg_w"] = out[f"{role}_gpu_power_w_avg"]

    # Skew/imbalance across used GPUs.
    used_util_peaks: list[float] = []
    used_mem_peaks: list[float] = []
    used_power_avgs: list[float] = []

    for idx in used_indices:
        util = safe_float(out.get(f"gpu{idx}_gpu_util_peak"))
        mem = safe_float(out.get(f"gpu{idx}_gpu_mem_used_gb_peak"))
        power = safe_float(out.get(f"gpu{idx}_gpu_power_w_avg"))

        if util is not None:
            used_util_peaks.append(util)
        if mem is not None:
            used_mem_peaks.append(mem)
        if power is not None:
            used_power_avgs.append(power)

    out["gpu_util_peak_skew"] = (
        max(used_util_peaks) - min(used_util_peaks)
        if len(used_util_peaks) >= 2 else 0.0 if len(used_util_peaks) == 1 else None
    )
    out["gpu_mem_peak_skew_gb"] = (
        max(used_mem_peaks) - min(used_mem_peaks)
        if len(used_mem_peaks) >= 2 else 0.0 if len(used_mem_peaks) == 1 else None
    )
    out["gpu_power_avg_skew_w"] = (
        max(used_power_avgs) - min(used_power_avgs)
        if len(used_power_avgs) >= 2 else 0.0 if len(used_power_avgs) == 1 else None
    )

    # Active count among used GPUs.
    active = 0
    for idx in used_indices:
        util_avg = safe_float(out.get(f"gpu{idx}_gpu_util_avg")) or 0.0
        mem_peak = safe_float(out.get(f"gpu{idx}_gpu_mem_used_gb_peak")) or 0.0
        if util_avg > 1.0 or mem_peak > 1.0:
            active += 1
    out["gpu_active_count_avg"] = active if used_indices else None

    # Energy/efficiency placeholders from telemetry.
    # Energy values in AIPerf summary are per-GPU metrics; sum across used GPUs if available.
    energy_vals = []
    for idx in used_indices:
        val = safe_float(out.get(f"gpu{idx}_gpu_energy_mj_avg"))
        if val is not None:
            energy_vals.append(val)

    out["gpu_energy_mj_sum"] = sum(energy_vals) if energy_vals else None

    return out


# ---------------------------------------------------------------------------
# GPU telemetry from raw gpu_telemetry_export.jsonl fallback
# ---------------------------------------------------------------------------

def parse_gpu_telemetry_jsonl(path: Path, mode: str | None) -> dict[str, Any]:
    role_map = gpu_role_map_for_mode(mode)
    used_indices = used_gpu_indices_for_mode(mode)

    per_gpu: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue

        idx = row.get("gpu_index")
        if not isinstance(idx, int):
            continue

        td = row.get("telemetry_data", {})
        for raw_name, prefix_name in GPU_METRIC_NAME_MAP.items():
            v = safe_float(td.get(raw_name))
            if v is not None:
                per_gpu[idx][prefix_name].append(v)

    out: dict[str, Any] = {
        "gpu_telemetry_source": "gpu_telemetry_export.jsonl",
        "gpu_count_observed": len(per_gpu) if per_gpu else None,
        "gpu_count_used": len(used_indices) if used_indices else None,
    }

    # Per-GPU raw summaries.
    for idx in sorted(per_gpu):
        for prefix_name, vals in per_gpu[idx].items():
            out.update(stat_summary(vals, f"gpu{idx}_{prefix_name}"))

    # Aggregate across used GPUs only.
    for prefix_name in set(GPU_METRIC_NAME_MAP.values()):
        all_vals: list[float] = []
        for idx in used_indices:
            all_vals.extend(per_gpu[idx].get(prefix_name, []))
        if all_vals:
            out.update(stat_summary(all_vals, prefix_name))

    # Backward-compatible aliases.
    if "gpu_mem_used_gb_avg" in out:
        out["gpu_mem_used_avg_gb"] = out["gpu_mem_used_gb_avg"]
    if "gpu_mem_used_gb_peak" in out:
        out["gpu_mem_used_peak_gb"] = out["gpu_mem_used_gb_peak"]
    if "gpu_power_w_avg" in out:
        out["gpu_power_avg_w"] = out["gpu_power_w_avg"]
    if "gpu_power_w_peak" in out:
        out["gpu_power_peak_w"] = out["gpu_power_w_peak"]
    if "gpu_temp_c_avg" in out:
        out["gpu_temp_avg_c"] = out["gpu_temp_c_avg"]
    if "gpu_temp_c_peak" in out:
        out["gpu_temp_peak_c"] = out["gpu_temp_c_peak"]

    # Role-labeled summaries.
    for idx, role in role_map.items():
        for prefix_name, vals in per_gpu[idx].items():
            out.update(stat_summary(vals, f"{role}_{prefix_name}"))

        if f"{role}_gpu_mem_used_gb_peak" in out:
            out[f"{role}_gpu_mem_used_peak_gb"] = out[f"{role}_gpu_mem_used_gb_peak"]
        if f"{role}_gpu_power_w_avg" in out:
            out[f"{role}_gpu_power_avg_w"] = out[f"{role}_gpu_power_w_avg"]

    used_util_peaks = [
        safe_float(out.get(f"gpu{idx}_gpu_util_peak"))
        for idx in used_indices
    ]
    used_util_peaks = [x for x in used_util_peaks if x is not None]

    used_mem_peaks = [
        safe_float(out.get(f"gpu{idx}_gpu_mem_used_gb_peak"))
        for idx in used_indices
    ]
    used_mem_peaks = [x for x in used_mem_peaks if x is not None]

    out["gpu_util_peak_skew"] = (
        max(used_util_peaks) - min(used_util_peaks)
        if len(used_util_peaks) >= 2 else 0.0 if len(used_util_peaks) == 1 else None
    )
    out["gpu_mem_peak_skew_gb"] = (
        max(used_mem_peaks) - min(used_mem_peaks)
        if len(used_mem_peaks) >= 2 else 0.0 if len(used_mem_peaks) == 1 else None
    )

    active = 0
    for idx in used_indices:
        util_avg = safe_float(out.get(f"gpu{idx}_gpu_util_avg")) or 0.0
        mem_peak = safe_float(out.get(f"gpu{idx}_gpu_mem_used_gb_peak")) or 0.0
        if util_avg > 1.0 or mem_peak > 1.0:
            active += 1
    out["gpu_active_count_avg"] = active if used_indices else None

    return out


# ---------------------------------------------------------------------------
# Server metrics
# ---------------------------------------------------------------------------

SERVER_METRIC_MAP = {
    "dynamo_frontend_time_to_first_token_seconds": "frontend_ttft_s",
    "dynamo_frontend_inter_token_latency_seconds": "frontend_itl_s",
    "dynamo_frontend_request_duration_seconds": "frontend_request_duration_s",
    "dynamo_frontend_input_sequence_tokens": "frontend_input_seq_tokens",
    "dynamo_frontend_output_sequence_tokens": "frontend_output_seq_tokens",
    "dynamo_frontend_cached_tokens": "frontend_cached_tokens",
    "dynamo_frontend_tokenizer_latency_ms": "frontend_tokenizer_latency_ms",
    "dynamo_frontend_inflight_requests": "frontend_inflight_requests",
    "dynamo_frontend_queued_requests": "frontend_queued_requests",
    "dynamo_frontend_output_tokens": "frontend_output_tokens",
    "dynamo_frontend_requests": "frontend_requests",
}


def parse_server_metrics_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())

    summary = data.get("summary", {})
    metrics = data.get("metrics", {})

    out: dict[str, Any] = {
        "server_metrics_enabled": bool(summary.get("endpoints_configured")),
        "server_metrics_endpoint_count": len(summary.get("endpoints_configured", []) or []),
        "server_metrics_success_count": len(summary.get("endpoints_successful", []) or []),
    }

    # Endpoint scrape quality.
    endpoint_info = summary.get("endpoint_info", {})
    if isinstance(endpoint_info, dict) and endpoint_info:
        first = next(iter(endpoint_info.values()))
        if isinstance(first, dict):
            out["server_metrics_total_fetches"] = safe_float(first.get("total_fetches"))
            out["server_metrics_unique_updates"] = safe_float(first.get("unique_updates"))
            out["server_metrics_duration_seconds"] = safe_float(first.get("duration_seconds"))
            out["server_metrics_avg_update_interval_ms"] = safe_float(first.get("avg_update_interval_ms"))

    for metric_name, prefix in SERVER_METRIC_MAP.items():
        metric_obj = metrics.get(metric_name)
        if not isinstance(metric_obj, dict):
            continue

        metric_type = metric_obj.get("type")
        series = metric_obj.get("series", [])
        if not series:
            continue

        # Some metrics have multiple series, e.g. tokenizer tokenize/detokenize.
        # For main plots, we take the first meaningful series with stats.
        for s in series:
            if not isinstance(s, dict):
                continue
            stats = s.get("stats", {})
            if not isinstance(stats, dict):
                continue

            labels = s.get("labels", {})
            label_suffix = ""

            # Preserve tokenizer operation distinction if present.
            if metric_name == "dynamo_frontend_tokenizer_latency_ms":
                op = labels.get("operation") if isinstance(labels, dict) else None
                if op:
                    label_suffix = f"_{op}"

            pfx = f"{prefix}{label_suffix}"

            if metric_type == "counter":
                out[f"{pfx}_total"] = safe_float(stats.get("total"))
                out[f"{pfx}_rate"] = safe_float(stats.get("rate"))

            elif metric_type in ("gauge", "histogram"):
                out[f"{pfx}_avg"] = safe_float(stats.get("avg"))
                out[f"{pfx}_p50"] = safe_float(stats.get("p50"))
                out[f"{pfx}_p95"] = safe_float(stats.get("p95"))
                out[f"{pfx}_p99"] = safe_float(stats.get("p99"))

                # Histograms use estimated percentile names.
                out[f"{pfx}_p50"] = out[f"{pfx}_p50"] or safe_float(stats.get("p50_estimate"))
                out[f"{pfx}_p95"] = out[f"{pfx}_p95"] or safe_float(stats.get("p95_estimate"))
                out[f"{pfx}_p99"] = out[f"{pfx}_p99"] or safe_float(stats.get("p99_estimate"))

                out[f"{pfx}_count"] = safe_float(stats.get("count"))
                out[f"{pfx}_count_rate"] = safe_float(stats.get("count_rate"))
                out[f"{pfx}_sum"] = safe_float(stats.get("sum"))
                out[f"{pfx}_sum_rate"] = safe_float(stats.get("sum_rate"))
                out[f"{pfx}_min"] = safe_float(stats.get("min"))
                out[f"{pfx}_max"] = safe_float(stats.get("max"))
                out[f"{pfx}_std"] = safe_float(stats.get("std"))

            # For non-tokenizer metrics, first usable series is enough.
            if metric_name != "dynamo_frontend_tokenizer_latency_ms":
                break

    return out


# ---------------------------------------------------------------------------
# Merge behavior
# ---------------------------------------------------------------------------

def merge_prefer_existing(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for k, v in new.items():
        if k not in base or base[k] in (None, ""):
            base[k] = v
    return base


def merge_override(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for k, v in new.items():
        base[k] = v
    return base


# ---------------------------------------------------------------------------
# Main artifact parser
# ---------------------------------------------------------------------------

def parse_artifact_dir(artifact_dir: Path) -> dict[str, Any]:
    mode = infer_mode_from_name(artifact_dir.name)
    generation = infer_generation_from_name(artifact_dir.name)
    concurrency = infer_concurrency_from_name(artifact_dir.name)
    gpu_budget = gpu_budget_for_mode(mode)

    summary: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "artifact_name": artifact_dir.name,
        "mode": mode,
        "generation": generation,
        "concurrency": concurrency,
        "gpu_budget": gpu_budget,
        "comparison_framing": "stage_disaggregated_scaleout",
        "sources": [],
    }

    aiperf_json = artifact_dir / "profile_export_aiperf.json"
    profile_jsonl = artifact_dir / "profile_export.jsonl"
    gpu_jsonl = artifact_dir / "gpu_telemetry_export.jsonl"
    server_json = artifact_dir / "server_metrics_export.json"
    server_csv = artifact_dir / "server_metrics_export.csv"
    aiperf_csv = artifact_dir / "profile_export_aiperf.csv"

    # 1. Request-side metrics from AIPerf summary.
    if aiperf_json.exists():
        summary = merge_override(summary, parse_profile_export_aiperf_json(aiperf_json))
        summary["sources"].append(aiperf_json.name)

        # Prefer summarized telemetry from profile_export_aiperf.json when present.
        gpu_from_json = parse_gpu_telemetry_from_aiperf_json(aiperf_json, mode)
        if gpu_from_json.get("gpu_count_observed"):
            summary = merge_override(summary, gpu_from_json)

    # 2. JSONL request-level fallback.
    if profile_jsonl.exists():
        summary = merge_prefer_existing(summary, parse_profile_export_jsonl(profile_jsonl))
        summary["sources"].append(profile_jsonl.name)

    # 3. Raw telemetry fallback if summarized telemetry was absent.
    if gpu_jsonl.exists():
        if not summary.get("gpu_count_observed"):
            summary = merge_override(summary, parse_gpu_telemetry_jsonl(gpu_jsonl, mode))
        summary["sources"].append(gpu_jsonl.name)

    # 4. Server/Dynamo metrics when available.
    if server_json.exists():
        summary = merge_override(summary, parse_server_metrics_json(server_json))
        summary["sources"].append(server_json.name)

    if server_csv.exists():
        summary["sources"].append(server_csv.name)

    if aiperf_csv.exists():
        summary["sources"].append(aiperf_csv.name)

    # Derived normalized metrics.
    throughput = safe_float(summary.get("throughput"))
    output_tps = safe_float(summary.get("output_token_throughput"))
    gpu_budget_val = safe_float(summary.get("gpu_budget"))
    power_avg = safe_float(summary.get("gpu_power_w_avg"))
    energy_mj = safe_float(summary.get("gpu_energy_mj_sum"))
    request_count = safe_float(summary.get("request_count"))
    total_output_tokens = safe_float(summary.get("total_output_tokens"))

    if throughput is not None and gpu_budget_val:
        summary["throughput_per_gpu"] = throughput / gpu_budget_val

    if output_tps is not None and gpu_budget_val:
        summary["output_tps_per_gpu"] = output_tps / gpu_budget_val

    if throughput is not None and power_avg and power_avg > 0:
        summary["requests_per_watt"] = throughput / power_avg

    if output_tps is not None and power_avg and power_avg > 0:
        summary["output_tps_per_watt"] = output_tps / power_avg

    if energy_mj is not None and request_count and request_count > 0:
        summary["energy_mj_per_request"] = energy_mj / request_count

    if energy_mj is not None and total_output_tokens and total_output_tokens > 0:
        summary["energy_mj_per_output_token"] = energy_mj / total_output_tokens

    # Explicit nulls for key timing metrics when unavailable.
    for k in [
        "ttft_ms", "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
        "tbt_ms", "tbt_ms_p50", "tbt_ms_p95", "tbt_ms_p99",
    ]:
        summary.setdefault(k, None)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()

    print(json.dumps(parse_artifact_dir(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
