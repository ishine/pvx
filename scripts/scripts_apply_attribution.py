#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
"""Apply centralized attribution references to Python and Markdown files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from pvx.core.attribution import (  # noqa: E402
    ATTRIBUTION_DOC_PATH,
    COPYRIGHT_NOTICE,
    markdown_notice,
    python_header_reference,
)

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}
PYTHON_HEADER = python_header_reference()


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def _relative_attribution_path(path: Path) -> str:
    relative = os.path.relpath(ROOT / ATTRIBUTION_DOC_PATH, start=path.parent)
    return str(relative).replace("\\", "/")


def _insert_python_header(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if COPYRIGHT_NOTICE in text:
        return False
    lines = text.splitlines()
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if insert_at < len(lines) and re.match(r"^#.*coding[:=]", lines[insert_at]):
        insert_at += 1

    injected = [PYTHON_HEADER, ""]
    new_lines = lines[:insert_at] + injected + lines[insert_at:]
    path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    return True


def _insert_markdown_notice(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if COPYRIGHT_NOTICE in text:
        return False
    lines = text.splitlines()
    notice = markdown_notice(_relative_attribution_path(path))

    insert_at = 0
    if lines and lines[0].startswith("#"):
        insert_at = 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
        if insert_at < len(lines) and lines[insert_at].lstrip().startswith("!["):
            insert_at += 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1

    new_lines = lines[:insert_at] + ["", notice, ""] + lines[insert_at:]
    path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    return True


def apply_python_headers() -> int:
    count = 0
    for path in sorted(ROOT.rglob("*.py")):
        if _is_excluded(path):
            continue
        if _insert_python_header(path):
            count += 1
    return count


def apply_markdown_notices() -> int:
    count = 0
    for path in sorted(ROOT.rglob("*.md")):
        if _is_excluded(path):
            continue
        if path.name == ATTRIBUTION_DOC_PATH:
            continue
        if _insert_markdown_notice(path):
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply attribution references to code and docs")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python header updates")
    parser.add_argument("--skip-markdown", action="store_true", help="Skip Markdown notice updates")
    args = parser.parse_args()

    touched_py = 0
    touched_md = 0
    if not args.skip_python:
        touched_py = apply_python_headers()
    if not args.skip_markdown:
        touched_md = apply_markdown_notices()

    print(
        f"[done] attribution updates: python_files={touched_py}, markdown_files={touched_md}, notice='{COPYRIGHT_NOTICE}'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
