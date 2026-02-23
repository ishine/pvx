#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Compatibility entrypoint for the unified pvx CLI."""

from __future__ import annotations

from pvx.cli.pvx import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
