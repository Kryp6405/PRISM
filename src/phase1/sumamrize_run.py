#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_one(parser_script: Path, artifact_dir: Path) -> dict:
    result = subprocess.run(
        ["python", str(parser_script), str(artifact_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "artifact_dir": str(artifact_dir),
            "error": result.stderr.strip() or f"parser exited with {result.returncode}",
        }
    try:
        return json.loads(result.stdout)
    except Exception:
        return {
            "artifact_dir": str(artifact_dir),
            "error": "invalid parser output",
            "raw": result.stdout,
        }


def is_aiperf_artifact_dir(path: Path) -> bool:
    return (
        (path / "profile_export_aiperf.json").exists()
        or (path / "profile_export.jsonl").exists()
        or (path / "profile_export_aiperf.csv").exists()
    )


def sort_key(d: dict) -> tuple:
    c = d.get("concurrency")
    if c is None:
        return (999999, d.get("artifact_name", d.get("artifact_dir", "")))
    return (int(c), d.get("artifact_name", d.get("artifact_dir", "")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--parser-script", type=Path, default=Path("src/phase0/parse_aiperf.py"))
    args = parser.parse_args()

    if not args.artifact_root.exists():
        artifact_dirs = []
    else:
        artifact_dirs = sorted(
            p for p in args.artifact_root.iterdir()
            if p.is_dir()
            and p.name.startswith(args.run_prefix)
            and is_aiperf_artifact_dir(p)
        )

    summaries = [parse_one(args.parser_script, d) for d in artifact_dirs]
    summaries = sorted(summaries, key=sort_key)

    print(json.dumps({
        "artifact_root": str(args.artifact_root),
        "run_prefix": args.run_prefix,
        "run_count": len(summaries),
        "runs": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
