#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Time-varying ring + resonator filter wrapper."""

from __future__ import annotations

from pvx.cli.pvxring import run_ring_cli


def main(argv: list[str] | None = None) -> int:
    return run_ring_cli(argv, default_operator="ringtvfilter", prog="pvx ringtvfilter")


if __name__ == "__main__":
    raise SystemExit(main())
