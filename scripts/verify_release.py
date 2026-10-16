#!/usr/bin/env python3
"""Verify the compact distribution and, optionally, every prepared numerical file."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {
    ".git",
    ".venv",
    ".venv-figures311",
    "runs",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Check inventory and sizes without content hashes")
    p.add_argument(
        "--tables", action="store_true", help="Also require every prepared file and check all Parquet schemas"
    )
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    inventory = json.loads((ROOT / "manifest.json").read_text())
    recipes = json.loads((ROOT / "data_sources/recipes.json").read_text())
    original = json.loads((ROOT / "reference/expected_outputs.json").read_text())
    expanded = {r["path"]: r for r in original["files"]}
    generated = set(recipes["files"])
    listed = {r["path"] for r in inventory["files"]}
    actual = set()
    errors = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in IGNORED or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_symlink():
            errors.append(f"Symlink in distribution: {rel}")
        if path.is_file() and str(rel) not in generated | {"manifest.json", "manifest.sha256"}:
            actual.add(str(rel))
    errors.extend(f"Missing distributed file: {x}" for x in sorted(listed - actual))
    errors.extend(f"Unlisted distributed file: {x}" for x in sorted(actual - listed))
    for row in inventory["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            continue
        if path.stat().st_size != row["bytes"]:
            errors.append(f"Size mismatch: {row['path']}")
        if row["bytes"] > 25 * 1024**2:
            errors.append(f"Distributed file exceeds 25 MiB: {row['path']}")
        if not a.quick and sha256(path) != row["sha256"]:
            errors.append(f"Hash mismatch: {row['path']}")
    prepared = 0
    for name, row in recipes["files"].items():
        path = ROOT / name
        if not path.is_file():
            if a.tables:
                errors.append(f"Prepare this file first: {name}")
            continue
        prepared += 1
        if path.stat().st_size != row["bytes"] or (not a.quick and sha256(path) != row["sha256"]):
            errors.append(f"Prepared file differs from original: {name}")
    if a.tables:
        import pyarrow.parquet as pq

        for name, row in expanded.items():
            if name.endswith(".parquet") and (ROOT / name).is_file():
                meta = pq.read_metadata(ROOT / name)
                if meta.num_rows != row["rows"] or pq.read_schema(ROOT / name).names != row["columns"]:
                    errors.append(f"Table schema mismatch: {name}")
    if sha256(ROOT / "manifest.json") != (ROOT / "manifest.sha256").read_text().split()[0]:
        errors.append("Distribution manifest checksum mismatch")
    if (
        sha256(ROOT / "data_sources/recipes.json")
        != (ROOT / "data_sources/recipes.sha256").read_text().split()[0]
    ):
        errors.append("Recipe manifest checksum mismatch")
    report = {
        "status": "FAIL" if errors else "PASS",
        "distributed_files": len(listed),
        "distributed_bytes": sum(r["bytes"] for r in inventory["files"]),
        "prepared_files": prepared,
        "preparable_files": len(generated),
        "content_hashes_checked": not a.quick,
        "table_metadata_checked": a.tables,
        "errors": errors,
    }
    if a.output:
        output = a.output.resolve()
        if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "runs"):
            p.error("Put reports under runs/ or outside the distribution")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
