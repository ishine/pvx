#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""PVC-style envelope function-stream generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pvx.core.common import (
    add_console_args,
    build_examples_epilog,
    build_status_bar,
    log_message,
)
from pvx.core.pvc_functions import (
    ENVELOPE_MODES,
    dump_control_points_csv,
    dump_control_points_json,
    generate_envelope_points,
)


def build_parser(prog: str = "pvx envelope") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate deterministic control-rate envelope maps for pvx CSV/JSON workflows.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx envelope --mode adsr --duration 8 --rate 20 --attack-sec 0.2 --decay-sec 0.6 --sustain 1.1 --release-sec 1.0 --key stretch --output stretch_env.csv",
                "pvx envelope --mode ramp --duration 6 --rate 10 --start 1.0 --end 0.5 --key pitch_ratio --stdout | pvx voc input.wav --stretch 1.0 --pitch-map-stdin --route pitch_ratio=pitch_ratio --output out.wav",
                "pvx envelope --mode sine --duration 12 --rate 30 --start 1.0 --peak 0.25 --sine-cycles 6 --min 0.75 --max 1.25 --key stretch --output stretch_lfo.json",
            ],
            notes=[
                "Generated maps are control-rate signals (not sample-rate audio).",
                "Use --key to emit directly as stretch or pitch_ratio for pvx voc map workflows.",
            ],
        ),
    )
    parser.add_argument("--mode", choices=list(ENVELOPE_MODES), default="adsr", help="Envelope mode")
    parser.add_argument("--duration", type=float, required=True, help="Envelope duration in seconds")
    parser.add_argument("--rate", type=float, default=20.0, help="Control points per second (default: 20)")
    parser.add_argument("--start", type=float, default=0.0, help="Start value (mode-dependent)")
    parser.add_argument("--peak", type=float, default=1.0, help="Peak or depth value (mode-dependent)")
    parser.add_argument("--sustain", type=float, default=0.7, help="Sustain value for ADSR mode")
    parser.add_argument("--end", type=float, default=0.0, help="End value for ADSR/ramp/exp modes")
    parser.add_argument("--attack-sec", type=float, default=0.1, help="ADSR attack in seconds")
    parser.add_argument("--decay-sec", type=float, default=0.2, help="ADSR decay in seconds")
    parser.add_argument("--release-sec", type=float, default=0.2, help="ADSR release in seconds")
    parser.add_argument("--exp-curve", type=float, default=4.0, help="Exponential curve steepness for exp mode")
    parser.add_argument("--sine-cycles", type=float, default=1.0, help="Number of cycles for sine mode")
    parser.add_argument("--sine-phase-rad", type=float, default=0.0, help="Initial sine phase in radians")
    parser.add_argument("--min", dest="min_value", type=float, default=None, help="Optional value clamp minimum")
    parser.add_argument("--max", dest="max_value", type=float, default=None, help="Optional value clamp maximum")
    parser.add_argument("--key", default="value", help="Output control column/key name (default: value)")
    parser.add_argument(
        "--format",
        choices=["csv", "json", "auto"],
        default="auto",
        help="Output format (default: auto from extension or csv for stdout)",
    )
    parser.add_argument("--output", "--out", type=Path, default=None, help="Output path. Default: stdout.")
    parser.add_argument("--stdout", action="store_true", help="Write map to stdout")
    add_console_args(parser)
    return parser


def _resolve_output_format(args: argparse.Namespace) -> str:
    fmt = str(getattr(args, "format", "auto")).strip().lower()
    if fmt in {"csv", "json"}:
        return fmt
    out = getattr(args, "output", None)
    if out is None:
        return "csv"
    if Path(out).suffix.lower() == ".json":
        return "json"
    return "csv"


def _render_payload(
    times,
    values,
    *,
    key: str,
    output_format: str,
) -> str:
    if output_format == "json":
        return dump_control_points_json(times, values, key=key)
    return dump_control_points_csv(times, values, key=key)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if float(args.duration) <= 0.0:
        parser.error("--duration must be > 0")
    if float(args.rate) <= 0.0:
        parser.error("--rate must be > 0")
    if str(args.key).strip() == "":
        parser.error("--key must not be empty")

    output_format = _resolve_output_format(args)
    write_stdout = bool(args.stdout or args.output is None)
    if not write_stdout and args.output is None:
        parser.error("--output is required unless --stdout is used")

    status = build_status_bar(args, "pvxenvelope", 1)
    times, values = generate_envelope_points(
        duration_sec=float(args.duration),
        rate_hz=float(args.rate),
        mode=str(args.mode),  # type: ignore[arg-type]
        start=float(args.start),
        peak=float(args.peak),
        sustain=float(args.sustain),
        end=float(args.end),
        attack_sec=float(args.attack_sec),
        decay_sec=float(args.decay_sec),
        release_sec=float(args.release_sec),
        exp_curve=float(args.exp_curve),
        sine_cycles=float(args.sine_cycles),
        sine_phase_rad=float(args.sine_phase_rad),
        min_value=args.min_value,
        max_value=args.max_value,
    )
    payload = _render_payload(times, values, key=str(args.key), output_format=output_format)
    status.step(1, "rendered")
    status.finish("done")

    if write_stdout:
        sys.stdout.write(payload)
        sys.stdout.flush()
        log_message(
            args,
            f"[done] pvxenvelope points={int(times.size)} key={args.key} format={output_format} output=stdout",
            min_level="verbose",
        )
        return 0

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    log_message(
        args,
        f"[done] pvxenvelope points={int(times.size)} key={args.key} format={output_format} output={out_path}",
        min_level="normal",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
