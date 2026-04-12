#!/usr/bin/env python3
"""Best-effort parser for AIPerf output bundles.

It scans the emitted JSON/CSV files inside one artifact directory and extracts a
compact metric summary. It is intentionally defensive because field names may
vary slightly across AIPerf versions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

KEYWORDS = {
    "throughput": ["throughput", "req_per_s", "requests_per_second", "tokens_per_second"],
    "ttft_ms": ["ttft", "time_to_first_token", "ttft_ms", "time_to_first_token_ms"],
    "tbt_ms": ["tbt", "itl", "inter_token_latency", "time_between_tokens", "tbt_ms"],
    "e2e_ms": ["e2e", "end_to_end", "latency", "latency_ms", "e2e_ms"],
    "output_token_throughput": ["output_token_throughput", "tokens_per_second"],
}


def normalize_key(k: str) -> str:
    return k.strip().lower().replace(" ", "_")


def search_obj(obj: Any, found: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            nk = normalize_key(k)
            for target, variants in KEYWORDS.items():
                if target in found and found[target] is not None:
                    continue
                if nk in variants or any(variant in nk for variant in variants):
                    if isinstance(v, (int, float, str)):
                        found[target] = v
            search_obj(v, found)
    elif isinstance(obj, list):
        for item in obj:
            search_obj(item, found)

def parse_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {
            "throughput": None,
            "ttft_ms": None,
            "tbt_ms": None,
            "e2e_ms": None,
            "output_token_throughput": None,
        }

    if path.name == "profile_export_aiperf.json":
        return {
            "throughput": data.get("request_throughput", {}).get("avg"),
            "ttft_ms": None,
            "tbt_ms": None,
            "e2e_ms": data.get("request_latency", {}).get("avg"),
            "output_token_throughput": data.get("output_token_throughput", {}).get("avg"),
        }

    if path.name == "server_metrics_export.json":
        return {
            "throughput": None,
            "ttft_ms": None,
            "tbt_ms": None,
            "e2e_ms": None,
            "output_token_throughput": None,
        }

    found = {
        "throughput": None,
        "ttft_ms": None,
        "tbt_ms": None,
        "e2e_ms": None,
        "output_token_throughput": None,
    }
    search_obj(data, found)
    return found

def parse_jsonl_file(path: Path) -> dict[str, Any]:
    found = {k: None for k in KEYWORDS}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            search_obj(data, found)
    except Exception:
        pass
    return found


def parse_csv_file(path: Path) -> dict[str, Any]:
    found = {k: None for k in KEYWORDS}
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return found
        # prefer first row; most aiperf exports are aggregate-ish
        for row in rows:
            norm = {normalize_key(k): v for k, v in row.items() if k is not None}
            for target, variants in KEYWORDS.items():
                if found[target] is not None:
                    continue
                for key, value in norm.items():
                    if key in variants or any(variant in key for variant in variants):
                        found[target] = value
                        break
    except Exception:
        pass
    return found


def merge(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for k, v in new.items():
        if base.get(k) is None and v not in (None, ""):
            base[k] = v
    return base


def parse_artifact_dir(artifact_dir: Path) -> dict[str, Any]:
    summary = {
        "artifact_dir": str(artifact_dir),
        "throughput": None,
        "ttft_ms": None,
        "tbt_ms": None,
        "e2e_ms": None,
        "output_token_throughput": None,
        "sources": [],
    }

    candidates = [
        artifact_dir / "profile_export_aiperf.json",
        artifact_dir / "server_metrics_export.json",
        artifact_dir / "profile_export.jsonl",
        artifact_dir / "profile_export_aiperf.csv",
        artifact_dir / "server_metrics_export.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".json":
            parsed = parse_json_file(path)
        elif path.suffix == ".jsonl":
            parsed = parse_jsonl_file(path)
        elif path.suffix == ".csv":
            parsed = parse_csv_file(path)
        else:
            continue
        summary = merge(summary, parsed)
        summary["sources"].append(path.name)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(parse_artifact_dir(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
