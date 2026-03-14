# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unit tests for pvx augment helper mode."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.cli.pvx import _augment_group_key, _parse_split_ratios, _stable_seed_from_text, run_augment_mode


class TestAugmentMode(unittest.TestCase):
    def test_parse_split_ratios_normalizes(self) -> None:
        train, val, test = _parse_split_ratios("8,1,1")
        self.assertAlmostEqual(train, 0.8, places=8)
        self.assertAlmostEqual(val, 0.1, places=8)
        self.assertAlmostEqual(test, 0.1, places=8)

    def test_group_key_and_seed_are_deterministic(self) -> None:
        p = Path("/tmp/speaker42__takeA.wav")
        key = _augment_group_key(p, "stem-prefix", "__")
        self.assertEqual(key, "speaker42")
        s1 = _stable_seed_from_text(1337, key)
        s2 = _stable_seed_from_text(1337, key)
        self.assertEqual(s1, s2)

    def test_augment_dry_run_writes_manifest_with_grouping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-augment-test-") as tmp:
            root = Path(tmp)
            sr = 16000
            t = np.arange(0, int(0.10 * sr), dtype=np.float64) / float(sr)
            tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
            audio = tone[:, None]

            a = root / "speaker1__a.wav"
            b = root / "speaker1__b.wav"
            sf.write(str(a), audio, sr)
            sf.write(str(b), audio, sr)

            out_dir = root / "aug_out"
            code = run_augment_mode(
                [
                    str(root / "*.wav"),
                    "--output-dir",
                    str(out_dir),
                    "--variants-per-input",
                    "1",
                    "--intent",
                    "asr_robust",
                    "--seed",
                    "99",
                    "--dry-run",
                    "--silent",
                ]
            )
            self.assertEqual(code, 0)

            manifest = out_dir / "augment_manifest.jsonl"
            self.assertTrue(manifest.exists())
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["group_key"], "speaker1")
            self.assertEqual(rows[1]["group_key"], "speaker1")
            self.assertEqual(rows[0]["split"], rows[1]["split"])
            self.assertEqual(rows[0]["status"], "planned")
            self.assertEqual(rows[1]["status"], "planned")


if __name__ == "__main__":
    unittest.main()
