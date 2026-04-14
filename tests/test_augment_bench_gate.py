"""Regression tests for directional augment benchmark gates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"

if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from run_augment_bench import _relative_metric_gate


class TestAugmentBenchGate(unittest.TestCase):
    def test_lower_is_better_metrics_allow_improvement(self) -> None:
        ok, failures = _relative_metric_gate(
            {"clip_pct_max": 0.0, "required_field_errors": 0, "split_balance_l1": 0.2},
            {"clip_pct_max": 0.04, "required_field_errors": 1, "split_balance_l1": 0.4},
            tolerance=0.10,
        )
        self.assertTrue(ok)
        self.assertEqual(failures, [])

    def test_higher_is_better_metrics_allow_improvement(self) -> None:
        ok, failures = _relative_metric_gate(
            {"pair_coverage": 1.0},
            {"pair_coverage": 0.5},
            tolerance=0.10,
        )
        self.assertTrue(ok)
        self.assertEqual(failures, [])

    def test_directional_metrics_still_fail_on_regression(self) -> None:
        ok, failures = _relative_metric_gate(
            {"clip_pct_max": 0.2, "pair_coverage": 0.5},
            {"clip_pct_max": 0.05, "pair_coverage": 1.0},
            tolerance=0.10,
        )
        self.assertFalse(ok)
        self.assertEqual(len(failures), 2)


if __name__ == "__main__":
    unittest.main()
