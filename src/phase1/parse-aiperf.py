#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def parse_profile_export_aiperf_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())

    return {
        "throughput": safe_float(data.get("request_throughput", {}).get("avg")),
        "ttft_ms": None,  # fill later from JSONL if available
        "ttft_ms_p50": None,
        "ttft_ms_p95": None,
        "tbt_ms": None,   # fill later from JSONL if available
        "tbt_ms_p50": None,
        "tbt_ms_p95": None,
        "e2e_ms": safe_float(data.get("request_latency", {}).get("avg")),
        "e2e_ms_p50": safe_float(data.get("request_latency", {}).get("p50")),
        "e2e_ms_p95": safe_float(data.get("request_latency", {}).get("p95")),
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
        "tbt_ms": mean(tbt_vals) if tbt_vals else None,
        "tbt_ms_p50": percentile(tbt_vals, 50),
        "tbt_ms_p95": percentile(tbt_vals, 95),
        # only use these as fallbacks if aiperf.json doesn't already have them
        "e2e_ms": mean(e2e_vals) if e2e_vals else None,
        "e2e_ms_p50": percentile(e2e_vals, 50),
        "e2e_ms_p95": percentile(e2e_vals, 95),
    }


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
        "concurrency": infer_concurrency_from_name(artifact_dir.name),
        "throughput": None,
        "ttft_ms": None,
        "ttft_ms_p50": None,
        "ttft_ms_p95": None,
        "tbt_ms": None,
        "tbt_ms_p50": None,
        "tbt_ms_p95": None,
        "e2e_ms": None,
        "e2e_ms_p50": None,
        "e2e_ms_p95": None,
        "output_token_throughput": None,
        "request_count": None,
        "output_token_count_avg": None,
        "input_sequence_length_avg": None,
        "output_sequence_length_avg": None,
        "benchmark_duration_sec": None,
        "total_output_tokens": None,
        "sources": [],
    }

    aiperf_json = artifact_dir / "profile_export_aiperf.json"
    jsonl = artifact_dir / "profile_export.jsonl"

    if aiperf_json.exists():
        summary = merge_fill_none(summary, parse_profile_export_aiperf_json(aiperf_json))
        summary["sources"].append(aiperf_json.name)

    if jsonl.exists():
        summary = merge_fill_none(summary, parse_profile_export_jsonl(jsonl))
        summary["sources"].append(jsonl.name)

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
