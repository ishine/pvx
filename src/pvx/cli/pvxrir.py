#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""pvxrir — simulate room acoustics via room impulse response convolution.

Convolves audio with a synthetic or user-provided room impulse response
to simulate the acoustic characteristics of different room environments.
Useful for building reverb-robust ASR, speaker verification, and source
separation datasets.

Examples
--------
# Synthetic room: RT60 between 0.3 and 1.2 s
pvxrir speech.wav --rt60 0.3,1.2 --wet 0.4,0.7 --output speech_reverb.wav

# Fixed reverb parameters:
pvxrir speech.wav --rt60 0.5 --wet 0.6 --output speech_reverb.wav

# Convolve with a real IR file:
pvxrir speech.wav --ir-file concert_hall.wav --wet 0.5 --output out.wav

# Convolve with a random IR from a directory:
pvxrir speech.wav --ir-dir irs/ --wet 0.3,0.8 --output out.wav
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pvxrir",
        description="Simulate room acoustics via impulse response convolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Input audio file")
    p.add_argument("--output", "-o", required=True, help="Output audio file")
    p.add_argument(
        "--rt60",
        default="0.2,1.5",
        help=(
            "Reverberation time in seconds. "
            "Single value or 'min,max' range (default: 0.2,1.5)"
        ),
    )
    p.add_argument(
        "--wet",
        default="0.3,0.7",
        help="Wet/dry mix 0–1. Single value or 'min,max' range (default: 0.3,0.7)",
    )
    p.add_argument(
        "--drr",
        default="3,12",
        help="Direct-to-reverb ratio in dB. Single value or 'min,max' range (default: 3,12)",
    )
    p.add_argument(
        "--pre-delay-ms",
        type=float,
        default=5.0,
        help="Pre-delay in ms before the reverberant tail (default: 5.0)",
    )
    p.add_argument(
        "--ir-file",
        default=None,
        help="Path to an impulse response WAV file (overrides synthetic generation)",
    )
    p.add_argument(
        "--ir-dir",
        default=None,
        help="Directory of impulse response files (one is chosen randomly)",
    )
    p.add_argument(
        "--no-trim",
        action="store_true",
        help="Do not trim output to input length (include reverb tail)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument(
        "--bit-depth",
        choices=["16", "24", "32", "float"],
        default="24",
        help="Output bit depth (default: 24)",
    )
    return p


def _parse_range(s: str) -> tuple[float, float]:
    parts = [float(x) for x in str(s).split(",")]
    return (parts[0], parts[-1])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from pvx.augment.core import load_audio, save_audio
    from pvx.augment.room import RoomSimulator, ImpulseResponseConvolver

    audio, sr = load_audio(args.input)

    if args.ir_file or args.ir_dir:
        src = args.ir_file or args.ir_dir
        wet_range = _parse_range(args.wet)
        aug = ImpulseResponseConvolver(
            src,
            wet_range=wet_range,
            preserve_length=not args.no_trim,
        )
    else:
        aug = RoomSimulator(
            rt60_range=_parse_range(args.rt60),
            drr_db_range=_parse_range(args.drr),
            wet_range=_parse_range(args.wet),
            pre_delay_ms=args.pre_delay_ms,
            preserve_length=not args.no_trim,
        )

    audio_out, sr_out = aug(audio, sr, seed=args.seed)

    subtype_map = {"16": "PCM_16", "24": "PCM_24", "32": "PCM_32", "float": "FLOAT"}
    save_audio(
        args.output,
        audio_out,
        sr_out,
        subtype=subtype_map.get(args.bit_depth, "PCM_24"),
    )
    print(f"[pvxrir] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
