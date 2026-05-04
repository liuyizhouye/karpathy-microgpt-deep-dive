#!/usr/bin/env python3
"""Strip outputs from Jupyter notebooks before commit.

This script intentionally uses only the Python standard library so the Git hook
does not depend on nbstripout, pre-commit, or Jupyter being installed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def repo_root() -> Path:
    result = run_git(["rev-parse", "--show-toplevel"])
    return Path(result.stdout.strip())


def staged_notebooks() -> list[Path]:
    result = run_git(
        ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "--", "*.ipynb"]
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def all_notebooks(root: Path) -> list[Path]:
    ignored_dirs = {".git", ".ipynb_checkpoints", "__pycache__"}
    notebooks: list[Path] = []
    for path in root.rglob("*.ipynb"):
        if any(part in ignored_dirs for part in path.parts):
            continue
        notebooks.append(path.relative_to(root))
    return sorted(notebooks)


def has_unstaged_changes(path: Path) -> bool:
    result = run_git(["diff", "--quiet", "--", str(path)], check=False)
    return result.returncode != 0


def strip_outputs(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        if cell.get("outputs") != []:
            cell["outputs"] = []
            changed = True

        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

        metadata = cell.get("metadata")
        if isinstance(metadata, dict):
            for key in ("execution", "ExecuteTime"):
                if key in metadata:
                    del metadata[key]
                    changed = True

    metadata = notebook.get("metadata")
    if isinstance(metadata, dict) and "widgets" in metadata:
        del metadata["widgets"]
        changed = True

    if changed:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove outputs and execution counts from Jupyter notebooks."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--staged",
        action="store_true",
        help="clean staged notebooks and re-stage them for the current commit",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="clean every notebook in the repository",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report notebooks that would change; do not write files",
    )
    args = parser.parse_args()

    root = repo_root()
    targets = staged_notebooks() if args.staged else all_notebooks(root)
    if not targets:
        return 0

    changed: list[Path] = []

    for rel_path in targets:
        abs_path = root / rel_path
        if not abs_path.exists():
            continue

        if args.staged and has_unstaged_changes(rel_path):
            print(
                f"Notebook has both staged and unstaged changes: {rel_path}",
                file=sys.stderr,
            )
            print(
                "Stage or stash the unstaged changes, then commit again.",
                file=sys.stderr,
            )
            return 1

        if args.check:
            before = abs_path.read_text(encoding="utf-8")
            strip_outputs(abs_path)
            after = abs_path.read_text(encoding="utf-8")
            if before != after:
                changed.append(rel_path)
                abs_path.write_text(before, encoding="utf-8")
        elif strip_outputs(abs_path):
            changed.append(rel_path)
            if args.staged:
                run_git(["add", "--", str(rel_path)])

    if args.check and changed:
        print("Notebook outputs would be stripped from:")
        for path in changed:
            print(f"  {path}")
        return 1

    if changed:
        print("Stripped Jupyter outputs from:")
        for path in changed:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
