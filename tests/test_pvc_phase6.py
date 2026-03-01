# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unit tests for PVC-inspired Phase 6 function-stream utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from pvx.cli.pvxenvelope import main as envelope_main
from pvx.cli.pvxreshape import main as reshape_main
from pvx.core.pvc_functions import (
    dump_control_points_csv,
    generate_envelope_points,
    parse_control_points_payload,
    reshape_control_points,
)


class TestPVCPhase6Utilities(unittest.TestCase):
    def test_generate_adsr_points_have_expected_bounds(self) -> None:
        times, values = generate_envelope_points(
            duration_sec=2.0,
            rate_hz=10.0,
            mode="adsr",
            start=0.0,
            peak=1.0,
            sustain=0.6,
            end=0.0,
            attack_sec=0.2,
            decay_sec=0.4,
            release_sec=0.4,
        )
        self.assertGreaterEqual(times.size, 2)
        self.assertAlmostEqual(float(times[0]), 0.0, places=8)
        self.assertAlmostEqual(float(times[-1]), 2.0, places=8)
        self.assertAlmostEqual(float(values[0]), 0.0, places=6)
        self.assertAlmostEqual(float(values[-1]), 0.0, places=5)
        self.assertGreater(float(np.max(values)), 0.95)

    def test_reshape_resample_then_scale(self) -> None:
        t = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
        v = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)
        rt, rv = reshape_control_points(
            t,
            v,
            operation="resample",
            resample_rate_hz=20.0,
            interp="linear",
        )
        self.assertGreater(rt.size, t.size)
        st, sv = reshape_control_points(rt, rv, operation="scale", factor=0.5)
        self.assertEqual(st.size, rt.size)
        self.assertAlmostEqual(float(np.max(sv)), 1.0, places=5)

    def test_csv_roundtrip_parser(self) -> None:
        t = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        v = np.asarray([1.2, 0.8, 1.0], dtype=np.float64)
        payload = dump_control_points_csv(t, v, key="stretch")
        out_t, out_v = parse_control_points_payload(
            payload,
            key="stretch",
            source_label="test",
            fmt="csv",
        )
        self.assertEqual(out_t.size, 3)
        self.assertTrue(np.allclose(out_v, v, atol=1e-9))


class TestPVCPhase6Cli(unittest.TestCase):
    def test_envelope_cli_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-p6-env-") as tmp:
            out = Path(tmp) / "env.csv"
            code = envelope_main(
                [
                    "--mode",
                    "ramp",
                    "--duration",
                    "1.0",
                    "--rate",
                    "10",
                    "--start",
                    "1.0",
                    "--end",
                    "2.0",
                    "--key",
                    "stretch",
                    "--output",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("time_sec,stretch", text)

    def test_reshape_cli_scales_csv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-p6-shape-") as tmp:
            src = Path(tmp) / "src.csv"
            dst = Path(tmp) / "dst.csv"
            src.write_text("time_sec,stretch\n0.0,1.0\n1.0,2.0\n", encoding="utf-8")
            code = reshape_main(
                [
                    str(src),
                    "--key",
                    "stretch",
                    "--operation",
                    "scale",
                    "--factor",
                    "0.5",
                    "--output",
                    str(dst),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            text = dst.read_text(encoding="utf-8")
            self.assertIn("0.500000000", text)
            self.assertIn("1.000000000", text)


if __name__ == "__main__":
    unittest.main()
