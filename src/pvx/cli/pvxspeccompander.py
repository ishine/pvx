#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Response-referenced spectral compander wrapper."""

from __future__ import annotations

from pvx.cli.pvxfilter import run_filter_cli


def main(argv: list[str] | None = None) -> int:
    return run_filter_cli(argv, default_operator="spec-compander", prog="pvx spec-compander")


if __name__ == "__main__":
    raise SystemExit(main())
