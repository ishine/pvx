#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Compatibility wrapper for `pvx.cli.pvxspeccompander`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pvx.cli.pvxspeccompander import *  # noqa: F401,F403
from pvx.cli.pvxspeccompander import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
