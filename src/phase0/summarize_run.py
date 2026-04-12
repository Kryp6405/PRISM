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
        return {"artifact_dir": str(artifact_dir), "error": result.stderr.strip()}
    try:
        return json.loads(result.stdout)
    except Exception:
        return {"artifact_dir": str(artifact_dir), "error": "invalid parser output", "raw": result.stdout}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--parser-script", type=Path, default=Path("src/phase0/parse_aiperf.py"))
    args = parser.parse_args()

    artifact_dirs = sorted(
        p for p in args.artifact_root.iterdir()
        if p.is_dir() and p.name.startswith(args.run_prefix)
    ) if args.artifact_root.exists() else []

    summaries = [parse_one(args.parser_script, d) for d in artifact_dirs]
    print(json.dumps({
        "artifact_root": str(args.artifact_root),
        "run_prefix": args.run_prefix,
        "runs": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
