# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unit tests for stereo M/S conversion and channel validation utilities."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Add src to sys.path to allow imports from pvx
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.core.stereo import lr_to_ms, ms_to_lr, validate_ref_channel


class TestStereoUtils(unittest.TestCase):
    def test_validate_ref_channel_success(self) -> None:
        self.assertEqual(validate_ref_channel(0, 2), 0)
        self.assertEqual(validate_ref_channel(1, 2), 1)
        self.assertEqual(validate_ref_channel("0", 1), 0)

    def test_validate_ref_channel_failures(self) -> None:
        with self.assertRaises(ValueError):
            validate_ref_channel(0, 0)
        with self.assertRaises(ValueError):
            validate_ref_channel(0, -1)
        with self.assertRaises(ValueError):
            validate_ref_channel(-1, 2)
        with self.assertRaises(ValueError):
            validate_ref_channel(2, 2)
        with self.assertRaises(ValueError):
            validate_ref_channel(5, 2)

    def test_lr_to_ms_mathematical_correctness(self) -> None:
        # L=1, R=1 -> M=sqrt(2), S=0
        audio = np.array([[1.0, 1.0]], dtype=np.float64)
        expected_m = 2.0 / math.sqrt(2.0)  # (1+1)/sqrt(2) = sqrt(2)
        expected_s = 0.0
        out = lr_to_ms(audio)
        self.assertAlmostEqual(out[0, 0], expected_m)
        self.assertAlmostEqual(out[0, 1], expected_s)

        # L=1, R=-1 -> M=0, S=sqrt(2)
        audio = np.array([[1.0, -1.0]], dtype=np.float64)
        expected_m = 0.0
        expected_s = 2.0 / math.sqrt(2.0)
        out = lr_to_ms(audio)
        self.assertAlmostEqual(out[0, 0], expected_m)
        self.assertAlmostEqual(out[0, 1], expected_s)

    def test_lr_to_ms_shape_validation(self) -> None:
        # Mono
        with self.assertRaises(ValueError):
            lr_to_ms(np.array([1.0, 2.0, 3.0]))
        # 1D with 2 elements is still not 2D
        with self.assertRaises(ValueError):
            lr_to_ms(np.array([1.0, 1.0]))
        # 2D but 1 channel
        with self.assertRaises(ValueError):
            lr_to_ms(np.array([[1.0], [2.0]]))
        # 2D but 3 channels
        with self.assertRaises(ValueError):
            lr_to_ms(np.array([[1.0, 2.0, 3.0]]))
        # 3D
        with self.assertRaises(ValueError):
            lr_to_ms(np.array([[[1.0, 2.0]]]))

    def test_ms_to_lr_mathematical_correctness(self) -> None:
        # M=sqrt(2), S=0 -> L=1, R=1
        sqrt2 = math.sqrt(2.0)
        audio_ms = np.array([[sqrt2, 0.0]], dtype=np.float64)
        out = ms_to_lr(audio_ms)
        self.assertAlmostEqual(out[0, 0], 1.0)
        self.assertAlmostEqual(out[0, 1], 1.0)

        # M=0, S=sqrt(2) -> L=1, R=-1
        audio_ms = np.array([[0.0, sqrt2]], dtype=np.float64)
        out = ms_to_lr(audio_ms)
        self.assertAlmostEqual(out[0, 0], 1.0)
        self.assertAlmostEqual(out[0, 1], -1.0)

    def test_ms_to_lr_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            ms_to_lr(np.array([[1.0, 2.0, 3.0]]))

    def test_stereo_round_trip(self) -> None:
        # Random stereo data
        np.random.seed(42)
        audio = np.random.randn(100, 2)
        ms = lr_to_ms(audio)
        lr = ms_to_lr(ms)
        np.testing.assert_allclose(audio, lr, atol=1e-12)

    def test_empty_input(self) -> None:
        audio = np.empty((0, 2))
        ms = lr_to_ms(audio)
        self.assertEqual(ms.shape, (0, 2))
        lr = ms_to_lr(ms)
        self.assertEqual(lr.shape, (0, 2))

    def test_different_dtypes(self) -> None:
        audio = np.array([[1, 2]], dtype=np.int32)
        ms = lr_to_ms(audio)
        self.assertEqual(ms.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
