#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

        s = str(x).strip()

        for token in [
            " MiB", " GiB", " MB", " GB", " W", " %", " C",
            " ms", " s", " tokens/sec", " requests/sec",
        ]:
            s = s.replace(token, "")

        return float(s)
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


def merge_prefer_existing(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for k, v in new.items():
        if k not in base or base[k] in (None, ""):
            base[k] = v
    return base


def merge_override(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for k, v in new.items():
        base[k] = v
    return base


def normalize_metric_name(s: str) -> str:
    return (
        s.strip()
        .lower()
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_per_")
        .replace("%", "percent")
        .replace("°", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .replace("__", "_")
    )


def read_csv_sections(path: Path) -> list[list[list[str]]]:
    rows = list(csv.reader(path.open(newline="")))
    sections: list[list[list[str]]] = []
    cur: list[list[str]] = []

    for row in rows:
        if not any(cell.strip() for cell in row):
            if cur:
                sections.append(cur)
                cur = []
            continue
        cur.append(row)

    if cur:
        sections.append(cur)

    return sections


# ---------------------------------------------------------------------------
# Artifact metadata
# ---------------------------------------------------------------------------

def infer_concurrency_from_name(name: str) -> int | None:
    m = re.search(r"concurrency(\d+)", name)
    return int(m.group(1)) if m else None


def infer_mode_from_name(name: str) -> str | None:
    n = name.lower()

    if "aggregated-native-vllm" in n or "aggregated_native_vllm" in n:
        return "aggregated_native_vllm"

    if "aggregated" in n:
        return "aggregated_native_vllm"

    if "encoder-only" in n or "encoder_only" in n or "e_pd" in n:
        return "e_pd"

    if "full-disagg" in n or "full_disagg" in n or "e_p_d" in n:
        return "e_p_d"

    return None


def gpu_budget_for_mode(mode: str | None) -> int | None:
    if mode in {"aggregated_native_vllm", "e_pd", "e_p_d"}:
        return 8
    return None


def parse_workload_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return {"workload_json_error": str(e)}

    return {
        "workload_name": data.get("name"),
        "workload_type": data.get("workload_type"),
        "image_source": data.get("image_source"),
        "image_path": data.get("image_path"),
        "configured_image_width_mean": data.get("image_width_mean"),
        "configured_image_height_mean": data.get("image_height_mean"),
        "configured_concurrency_values": data.get("concurrency_values"),
        "configured_request_count": data.get("request_count"),
        "configured_output_tokens_mean": data.get("output_tokens_mean"),
        "configured_output_tokens_stddev": data.get("output_tokens_stddev"),
        "configured_endpoint_type": data.get("endpoint_type"),
        "configured_use_server_token_count": data.get("use_server_token_count"),
        "configured_num_images": data.get("num_images"),
        "configured_num_videos": data.get("num_videos"),
        "configured_prompt": data.get("prompt"),
    }


# ---------------------------------------------------------------------------
# AIPerf request metrics: CSV / JSON / JSONL
# ---------------------------------------------------------------------------

def parse_profile_export_aiperf_csv(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"profile_csv_source": path.name}
    sections = read_csv_sections(path)

    for sec in sections:
        header = [h.strip() for h in sec[0]]
        if not header:
            continue

        # Distribution table:
        # Metric, avg, min, max, p1, p5, p10, p25, p50, p75, p90, p95, p99, std...
        if header[0] == "Metric" and any(h in header for h in ["avg", "p50", "p95", "p99"]):
            idx = {h: i for i, h in enumerate(header)}

            for row in sec[1:]:
                if not row:
                    continue

                metric = row[0].strip()
                if not metric:
                    continue

                nm = normalize_metric_name(metric)

                prefix_map = {
                    "request_latency_ms": "e2e_ms",
                    "time_to_first_token_ms": "ttft_ms",
                    "time_to_first_output_token_ms": "ttft_ms",
                    "inter_token_latency_ms": "itl_ms",
                    "inter_chunk_latency_ms": "inter_chunk_latency_ms",
                    "output_token_throughput_per_user_tokens_per_sec_per_user": "output_tps_per_user",
                    "prefill_throughput_per_user_tokens_per_sec_per_user": "prefill_tps_per_user",
                    "input_sequence_length_tokens": "input_sequence_length",
                    "output_sequence_length_tokens": "output_sequence_length",
                    "output_token_count_tokens": "output_token_count",
                    "number_of_images_images": "num_images",
                    "image_latency_ms_per_image": "image_latency_ms",
                    "image_throughput_images_per_sec": "image_throughput",
                    "http_total_time_ms": "http_total_time_ms",
                    "http_waiting_ttfb_ms": "http_waiting_ttfb_ms",
                }

                prefix = prefix_map.get(nm)
                if not prefix:
                    continue

                for stat in ["avg", "std", "p50", "p95", "p99", "min", "max"]:
                    if stat in idx and idx[stat] < len(row):
                        out[f"{prefix}_{stat}"] = safe_float(row[idx[stat]])

                out[f"{prefix}_peak"] = out.get(f"{prefix}_max")

            out["e2e_ms"] = out.get("e2e_ms_avg")
            out["ttft_ms"] = out.get("ttft_ms_avg")
            out["itl_ms"] = out.get("itl_ms_avg")

        # Summary table:
        # Metric, Value
        elif header[:2] == ["Metric", "Value"]:
            for row in sec[1:]:
                if len(row) < 2:
                    continue

                metric = normalize_metric_name(row[0])
                val = safe_float(row[1])

                summary_map = {
                    "benchmark_duration_sec": "benchmark_duration_sec",
                    "request_count": "request_count",
                    "request_throughput_requests_per_sec": "throughput",
                    "output_token_throughput_tokens_per_sec": "output_token_throughput",
                    "total_token_throughput_tokens_per_sec": "total_token_throughput",
                    "total_input_sequence_length_tokens": "total_input_sequence_length",
                    "total_output_sequence_length_tokens": "total_output_sequence_length",
                    "total_output_tokens_tokens": "total_output_tokens",
                    "total_usage_prompt_tokens_tokens": "total_usage_prompt_tokens",
                    "total_usage_completion_tokens_tokens": "total_usage_completion_tokens",
                    "osl_mismatch_count": "osl_mismatch_count",
                }

                key = summary_map.get(metric)
                if key:
                    out[key] = val

            out["request_throughput"] = out.get("throughput")

    return out


def parse_profile_export_aiperf_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())

    out: dict[str, Any] = {
        "aiperf_version": data.get("aiperf_version"),
        "benchmark_id": data.get("benchmark_id"),
        "was_cancelled": data.get("was_cancelled"),
        "start_time": data.get("start_time"),
        "end_time": data.get("end_time"),
    }

    out["throughput"] = get_stat(data.get("request_throughput"))
    out["request_throughput"] = out["throughput"]

    latency = data.get("request_latency", {})
    out["e2e_ms"] = get_stat(latency, "avg")

    for s in ["p50", "p95", "p99", "min", "max", "std"]:
        out[f"e2e_ms_{s}"] = get_stat(latency, s)

    metric_map = [
        ("time_to_first_token", "ttft_ms"),
        ("time_to_first_output_token", "ttft_ms"),
        ("inter_token_latency", "itl_ms"),
        ("inter_chunk_latency", "inter_chunk_latency_ms"),
        ("input_sequence_length", "input_sequence_length"),
        ("output_sequence_length", "output_sequence_length"),
        ("output_token_count", "output_token_count"),
        ("num_images", "num_images"),
        ("image_latency", "image_latency_ms"),
        ("image_throughput", "image_throughput"),
    ]

    for raw, prefix in metric_map:
        obj = data.get(raw)
        if isinstance(obj, dict):
            for s in ["avg", "p50", "p95", "p99", "min", "max", "std"]:
                out[f"{prefix}_{s}"] = get_stat(obj, s)

    out["ttft_ms"] = out.get("ttft_ms_avg")
    out["itl_ms"] = out.get("itl_ms_avg")

    out["output_token_throughput"] = get_stat(data.get("output_token_throughput"))
    out["total_token_throughput"] = get_stat(data.get("total_token_throughput"))
    out["request_count"] = get_stat(data.get("request_count"))
    out["benchmark_duration_sec"] = get_stat(data.get("benchmark_duration"))
    out["total_output_tokens"] = get_stat(data.get("total_output_tokens"))
    out["total_input_sequence_length"] = get_stat(data.get("total_isl"))
    out["total_output_sequence_length"] = get_stat(data.get("total_osl"))

    return out


def parse_profile_export_jsonl(path: Path) -> dict[str, Any]:
    vals: dict[str, list[float]] = defaultdict(list)
    total_count = 0
    cancelled_count = 0

    metric_map = {
        "request_latency": "jsonl_e2e_ms",
        "time_to_first_token": "ttft_ms",
        "ttft": "ttft_ms",
        "inter_token_latency": "itl_ms",
        "time_between_tokens": "itl_ms",
        "tbt": "itl_ms",
        "itl": "itl_ms",
        "input_sequence_length": "input_sequence_length",
        "output_sequence_length": "output_sequence_length",
        "output_token_count": "output_token_count",
        "num_images": "num_images",
    }

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except Exception:
            continue

        total_count += 1

        if row.get("metadata", {}).get("was_cancelled") is True:
            cancelled_count += 1

        metrics = row.get("metrics", {})

        for raw, prefix in metric_map.items():
            obj = metrics.get(raw)
            if isinstance(obj, dict):
                v = safe_float(obj.get("value"))
                if v is not None:
                    vals[prefix].append(v)

    out: dict[str, Any] = {
        "profile_jsonl_request_count": total_count,
        "profile_jsonl_cancelled_count": cancelled_count,
    }

    for prefix, arr in vals.items():
        out.update(stat_summary(arr, prefix))

        if prefix in ["ttft_ms", "itl_ms"]:
            out[prefix] = out.get(f"{prefix}_avg")

    return out


# ---------------------------------------------------------------------------
# Phase 4 cluster GPU telemetry: gpu_telemetry_all_nodes.csv
# ---------------------------------------------------------------------------

def parse_gpu_telemetry_all_nodes_csv(path: Path) -> dict[str, Any]:
    per_gpu: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    hosts: set[str] = set()

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            host = (row.get("host") or "").strip()
            idx = safe_float(row.get("gpu_index"))

            if not host or idx is None:
                continue

            gpu_key = (host, int(idx))
            hosts.add(host)

            mem_mb = safe_float(row.get("memory_used_mb"))
            total_mb = safe_float(row.get("memory_total_mb"))
            util = safe_float(row.get("gpu_util_percent"))
            power = safe_float(row.get("power_watts"))
            temp = safe_float(row.get("temp_c"))

            if mem_mb is not None:
                per_gpu[gpu_key]["gpu_mem_used_gb"].append(mem_mb / 1024.0)
                per_gpu[gpu_key]["gpu_mem_used_mb"].append(mem_mb)

            if total_mb is not None:
                per_gpu[gpu_key]["gpu_mem_total_gb"].append(total_mb / 1024.0)

            if util is not None:
                per_gpu[gpu_key]["gpu_util"].append(util)

            if power is not None:
                per_gpu[gpu_key]["gpu_power_w"].append(power)

            if temp is not None:
                per_gpu[gpu_key]["gpu_temp_c"].append(temp)

    out: dict[str, Any] = {
        "gpu_telemetry_source": path.name,
        "cluster_num_hosts": len(hosts),
        "cluster_hosts": sorted(hosts),
        "cluster_num_gpus": len(per_gpu),
        "gpu_count_observed": len(per_gpu),
        "gpu_count_used": len(per_gpu),
    }

    for metric in [
        "gpu_mem_used_gb",
        "gpu_mem_used_mb",
        "gpu_util",
        "gpu_power_w",
        "gpu_temp_c",
    ]:
        all_vals: list[float] = []

        for g in per_gpu.values():
            all_vals.extend(g.get(metric, []))

        out.update(stat_summary(all_vals, metric))

    # Compatibility aliases.
    out["gpu_mem_used_avg_gb"] = out.get("gpu_mem_used_gb_avg")
    out["gpu_mem_used_peak_gb"] = out.get("gpu_mem_used_gb_peak")
    out["gpu_power_avg_w"] = out.get("gpu_power_w_avg")
    out["gpu_power_peak_w"] = out.get("gpu_power_w_peak")
    out["gpu_temp_peak_c"] = out.get("gpu_temp_c_peak")

    # Per-host summaries.
    host_metric_vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for (host, _idx), metrics in per_gpu.items():
        for metric, vals in metrics.items():
            host_metric_vals[host][metric].extend(vals)

    for host, metrics in host_metric_vals.items():
        clean = re.sub(r"[^A-Za-z0-9]+", "_", host).strip("_")

        for metric, vals in metrics.items():
            out.update(stat_summary(vals, f"host_{clean}_{metric}"))

    # Skew across GPUs using per-GPU peaks.
    mem_peaks = []
    util_peaks = []
    power_avgs = []

    for metrics in per_gpu.values():
        if metrics.get("gpu_mem_used_gb"):
            mem_peaks.append(max(metrics["gpu_mem_used_gb"]))

        if metrics.get("gpu_util"):
            util_peaks.append(max(metrics["gpu_util"]))

        if metrics.get("gpu_power_w"):
            power_avgs.append(mean(metrics["gpu_power_w"]))

    out["gpu_mem_peak_skew_gb"] = (
        max(mem_peaks) - min(mem_peaks)
        if len(mem_peaks) >= 2 else 0.0 if mem_peaks else None
    )

    out["gpu_util_peak_skew"] = (
        max(util_peaks) - min(util_peaks)
        if len(util_peaks) >= 2 else 0.0 if util_peaks else None
    )

    out["gpu_power_avg_skew_w"] = (
        max(power_avgs) - min(power_avgs)
        if len(power_avgs) >= 2 else 0.0 if power_avgs else None
    )

    # Approximate cluster total power from per-GPU aggregate.
    if out.get("gpu_power_w_avg") is not None and per_gpu:
        out["cluster_total_power_avg_w"] = out["gpu_power_w_avg"] * len(per_gpu)
    else:
        out["cluster_total_power_avg_w"] = None

    if out.get("gpu_power_w_peak") is not None and per_gpu:
        out["cluster_total_power_peak_w"] = out["gpu_power_w_peak"] * len(per_gpu)
    else:
        out["cluster_total_power_peak_w"] = None

    active = 0

    for metrics in per_gpu.values():
        util_avg = mean(metrics.get("gpu_util", [0.0]))
        mem_peak = max(metrics.get("gpu_mem_used_gb", [0.0]))

        if util_avg > 1.0 or mem_peak > 1.0:
            active += 1

    out["gpu_active_count_avg"] = active if per_gpu else None

    return out


# ---------------------------------------------------------------------------
# Server metrics: vLLM CSV / JSON
# ---------------------------------------------------------------------------

def parse_server_metrics_export_csv(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"server_metrics_source": path.name}
    sections = read_csv_sections(path)

    for sec in sections:
        header = [h.strip() for h in sec[0]]

        if not header or "Metric" not in header:
            continue

        idx = {h: i for i, h in enumerate(header)}

        for row in sec[1:]:
            if len(row) <= idx.get("Metric", 0):
                continue

            metric = row[idx["Metric"]].strip()

            if not metric:
                continue

            def get_col(*names: str) -> float | None:
                for n in names:
                    if n in idx and idx[n] < len(row):
                        v = safe_float(row[idx[n]])
                        if v is not None:
                            return v
                return None

            if metric == "vllm:kv_cache_usage_perc":
                out["kv_cache_usage_avg"] = get_col("avg", "Average")
                out["kv_cache_usage_max"] = get_col("max", "Max")
                out["kv_cache_usage_p95"] = get_col("p95", "p95_estimate")

            elif metric == "vllm:num_requests_running":
                out["num_requests_running_avg"] = get_col("avg", "Average")
                out["num_requests_running_max"] = get_col("max", "Max")

            elif metric == "vllm:num_requests_waiting":
                out["num_requests_waiting_avg"] = get_col("avg", "Average")
                out["num_requests_waiting_max"] = get_col("max", "Max")

            histogram_map = {
                "vllm:e2e_request_latency_seconds": "server_e2e_s",
                "vllm:time_to_first_token_seconds": "server_ttft_s",
                "vllm:inter_token_latency_seconds": "server_itl_s",
                "vllm:request_prefill_time_seconds": "prefill_s",
                "vllm:request_decode_time_seconds": "decode_s",
                "vllm:request_inference_time_seconds": "inference_s",
                "vllm:request_queue_time_seconds": "queue_s",
                "vllm:request_prompt_tokens": "server_prompt_tokens",
                "vllm:request_generation_tokens": "server_generation_tokens",
                "vllm:request_time_per_output_token_seconds": "server_time_per_output_token_s",
                "vllm:request_prefill_kv_computed_tokens": "server_prefill_kv_computed_tokens",
            }

            if metric in histogram_map:
                pfx = histogram_map[metric]

                for stat in ["avg", "p50", "p95", "p99", "min", "max"]:
                    val = get_col(stat, f"{stat}_estimate")

                    if val is not None:
                        out[f"{pfx}_{stat}"] = val

                ms_alias = {
                    "server_e2e_s": "server_e2e_ms",
                    "server_ttft_s": "server_ttft_ms",
                    "server_itl_s": "server_itl_ms",
                    "prefill_s": "prefill_ms",
                    "decode_s": "decode_ms",
                    "inference_s": "inference_ms",
                    "queue_s": "queue_ms",
                    "server_time_per_output_token_s": "server_time_per_output_token_ms",
                }.get(pfx)

                if ms_alias:
                    for stat in ["avg", "p50", "p95", "p99", "min", "max"]:
                        v = out.get(f"{pfx}_{stat}")

                        if v is not None:
                            out[f"{ms_alias}_{stat}"] = v * 1000.0

            counter_map = {
                "vllm:generation_tokens": "server_generation_tokens",
                "vllm:prompt_tokens": "server_prompt_tokens",
                "vllm:num_preemptions": "num_preemptions",
                "vllm:prefix_cache_hits": "prefix_cache_hits",
                "vllm:prefix_cache_queries": "prefix_cache_queries",
                "vllm:mm_cache_hits": "mm_cache_hits",
                "vllm:mm_cache_queries": "mm_cache_queries",
                "vllm:prompt_tokens_cached": "prompt_tokens_cached",
                "vllm:prompt_tokens_recomputed": "prompt_tokens_recomputed",
                "vllm:request_success": "request_success",
                "vllm:estimated_flops_per_gpu": "estimated_flops_per_gpu",
                "vllm:estimated_read_bytes_per_gpu": "estimated_read_bytes_per_gpu",
                "vllm:estimated_write_bytes_per_gpu": "estimated_write_bytes_per_gpu",
            }

            if metric in counter_map:
                pfx = counter_map[metric]
                total = get_col("total", "Total", "sum", "Sum", "value", "Value")
                rate = get_col("rate", "Rate", "sum_rate")

                if total is not None:
                    out[f"{pfx}_total"] = total

                if rate is not None:
                    out[f"{pfx}_rate"] = rate

    hits = safe_float(out.get("mm_cache_hits_total"))
    queries = safe_float(out.get("mm_cache_queries_total"))

    out["mm_cache_hit_rate"] = (
        hits / queries
        if hits is not None and queries and queries > 0 else None
    )

    hits = safe_float(out.get("prefix_cache_hits_total"))
    queries = safe_float(out.get("prefix_cache_queries_total"))

    out["prefix_cache_hit_rate"] = (
        hits / queries
        if hits is not None and queries and queries > 0 else None
    )

    return out


def parse_server_metrics_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    metrics = data.get("metrics", {})

    out: dict[str, Any] = {
        "server_metrics_enabled": bool(summary.get("endpoints_configured")),
        "server_metrics_endpoint_count": len(summary.get("endpoints_configured", []) or []),
        "server_metrics_success_count": len(summary.get("endpoints_successful", []) or []),
    }

    metric_map = {
        "vllm:kv_cache_usage_perc": "kv_cache_usage",
        "vllm:num_requests_running": "num_requests_running",
        "vllm:num_requests_waiting": "num_requests_waiting",
        "vllm:e2e_request_latency_seconds": "server_e2e_s",
        "vllm:time_to_first_token_seconds": "server_ttft_s",
        "vllm:inter_token_latency_seconds": "server_itl_s",
        "vllm:request_prefill_time_seconds": "prefill_s",
        "vllm:request_decode_time_seconds": "decode_s",
        "vllm:request_queue_time_seconds": "queue_s",
    }

    for metric_name, prefix in metric_map.items():
        obj = metrics.get(metric_name)

        if not isinstance(obj, dict):
            continue

        for s in obj.get("series", []):
            stats = s.get("stats", {}) if isinstance(s, dict) else {}

            if not isinstance(stats, dict):
                continue

            for stat in ["avg", "p50", "p95", "p99", "min", "max"]:
                v = safe_float(stats.get(stat))

                if v is None:
                    v = safe_float(stats.get(f"{stat}_estimate"))

                if v is not None:
                    out[f"{prefix}_{stat}"] = v

            break

    for s_prefix, ms_prefix in [
        ("server_e2e_s", "server_e2e_ms"),
        ("server_ttft_s", "server_ttft_ms"),
        ("server_itl_s", "server_itl_ms"),
        ("prefill_s", "prefill_ms"),
        ("decode_s", "decode_ms"),
        ("queue_s", "queue_ms"),
    ]:
        for stat in ["avg", "p50", "p95", "p99", "min", "max"]:
            v = out.get(f"{s_prefix}_{stat}")

            if v is not None:
                out[f"{ms_prefix}_{stat}"] = v * 1000.0

    return out


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_artifact_dir(artifact_dir: Path) -> dict[str, Any]:
    mode = infer_mode_from_name(artifact_dir.name)
    concurrency = infer_concurrency_from_name(artifact_dir.name)
    gpu_budget = gpu_budget_for_mode(mode)

    summary: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "artifact_name": artifact_dir.name,
        "mode": mode,
        "generation": "phase4",
        "concurrency": concurrency,
        "gpu_budget": gpu_budget,
        "comparison_framing": "phase4_aggregated_native_vllm_scaleout",
        "sources": [],
    }

    workload_json = artifact_dir / "workload.json"
    if workload_json.exists():
        summary = merge_override(summary, parse_workload_json(workload_json))
        summary["sources"].append(workload_json.name)

    aiperf_json = artifact_dir / "profile_export_aiperf.json"
    aiperf_csv = artifact_dir / "profile_export_aiperf.csv"
    profile_jsonl = artifact_dir / "profile_export.jsonl"
    gpu_all_nodes_csv = artifact_dir / "gpu_telemetry_all_nodes.csv"
    server_json = artifact_dir / "server_metrics_export.json"
    server_csv = artifact_dir / "server_metrics_export.csv"

    if aiperf_json.exists():
        summary = merge_override(summary, parse_profile_export_aiperf_json(aiperf_json))
        summary["sources"].append(aiperf_json.name)

    if aiperf_csv.exists():
        summary = merge_override(summary, parse_profile_export_aiperf_csv(aiperf_csv))
        summary["sources"].append(aiperf_csv.name)

    if profile_jsonl.exists():
        summary = merge_prefer_existing(summary, parse_profile_export_jsonl(profile_jsonl))
        summary["sources"].append(profile_jsonl.name)

    if server_json.exists():
        summary = merge_override(summary, parse_server_metrics_json(server_json))
        summary["sources"].append(server_json.name)

    if server_csv.exists():
        summary = merge_override(summary, parse_server_metrics_export_csv(server_csv))
        summary["sources"].append(server_csv.name)

    if gpu_all_nodes_csv.exists():
        summary = merge_override(summary, parse_gpu_telemetry_all_nodes_csv(gpu_all_nodes_csv))
        summary["sources"].append(gpu_all_nodes_csv.name)

    # Prefer measured cluster GPU count for per-GPU metrics.
    measured_gpus = safe_float(summary.get("cluster_num_gpus") or summary.get("gpu_count_observed"))

    if measured_gpus and measured_gpus > 0:
        summary["gpu_budget"] = int(measured_gpus)

    throughput = safe_float(summary.get("throughput"))
    output_tps = safe_float(summary.get("output_token_throughput"))
    total_tps = safe_float(summary.get("total_token_throughput"))
    gpu_budget_val = safe_float(summary.get("gpu_budget"))
    power_avg = safe_float(summary.get("cluster_total_power_avg_w") or summary.get("gpu_power_w_avg"))

    if throughput is not None and gpu_budget_val:
        summary["throughput_per_gpu"] = throughput / gpu_budget_val

    if output_tps is not None and gpu_budget_val:
        summary["output_tps_per_gpu"] = output_tps / gpu_budget_val

    if total_tps is not None and gpu_budget_val:
        summary["total_tps_per_gpu"] = total_tps / gpu_budget_val

    if throughput is not None and power_avg and power_avg > 0:
        summary["requests_per_watt"] = throughput / power_avg

    if output_tps is not None and power_avg and power_avg > 0:
        summary["output_tps_per_watt"] = output_tps / power_avg

    # Preferred aliases for plotting.
    summary.setdefault("ttft_ms", summary.get("ttft_ms_avg"))
    summary.setdefault("itl_ms", summary.get("itl_ms_avg"))

    if summary.get("request_count") is None and summary.get("profile_jsonl_request_count") is not None:
        summary["request_count"] = summary["profile_jsonl_request_count"]

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()

    print(json.dumps(parse_artifact_dir(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
