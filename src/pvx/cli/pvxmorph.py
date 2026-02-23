#!/usr/bin/env python3
"""Spectral morphing between two input files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvx.core.audio_metrics import (
    render_audio_comparison_table,
    render_audio_metrics_table,
    summarize_audio_metrics,
)
from pvx.core.common import (
    add_console_args,
    add_output_policy_args,
    add_vocoder_args,
    build_examples_epilog,
    build_status_bar,
    build_vocoder_config,
    ensure_runtime,
    log_error,
    log_message,
    read_audio,
    validate_vocoder_args,
    write_output,
)
from pvx.core.voc import add_mastering_args, apply_mastering_chain, cepstral_envelope, istft, resample_1d, stft

BLEND_MODE_CHOICES: tuple[str, ...] = (
    "linear",
    "geometric",
    "magnitude_b_phase_a",
    "magnitude_a_phase_b",
    "carrier_a_envelope_b",
    "carrier_b_envelope_a",
    "carrier_a_mask_b",
    "carrier_b_mask_a",
    "product",
    "max_mag",
    "min_mag",
)

EPS = 1e-12


def match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.shape[1] == channels:
        return audio
    if audio.shape[1] > channels:
        return audio[:, :channels]
    reps = channels - audio.shape[1]
    extra = np.repeat(audio[:, -1:], reps, axis=1)
    return np.hstack([audio, extra])


def _phase_blend(phase_a: np.ndarray, phase_b: np.ndarray, mix: float) -> np.ndarray:
    m = float(np.clip(mix, 0.0, 1.0))
    z = (1.0 - m) * np.exp(1j * phase_a) + m * np.exp(1j * phase_b)
    out = np.angle(z)
    out[np.abs(z) <= 1e-8] = phase_a[np.abs(z) <= 1e-8]
    return out


def _resolve_phase_mix(mode: str, alpha: float, user_mix: float | None) -> float:
    if user_mix is not None:
        return float(np.clip(user_mix, 0.0, 1.0))
    if mode in {"magnitude_b_phase_a", "carrier_a_envelope_b", "carrier_a_mask_b"}:
        return 0.0
    if mode in {"magnitude_a_phase_b", "carrier_b_envelope_a", "carrier_b_mask_a"}:
        return 1.0
    return float(np.clip(alpha, 0.0, 1.0))


def _safe_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _framewise_envelope(mag: np.ndarray, lifter: int) -> np.ndarray:
    env = np.zeros_like(mag, dtype=np.float64)
    for i in range(mag.shape[1]):
        env[:, i] = np.asarray(cepstral_envelope(mag[:, i], lifter), dtype=np.float64)
    return np.maximum(env, EPS)


def _mask_from_modulator(mag_mod: np.ndarray, exponent: float) -> np.ndarray:
    frame_mean = np.mean(mag_mod, axis=0, keepdims=True)
    base = mag_mod / np.maximum(frame_mean, EPS)
    return np.power(np.maximum(base, EPS), float(exponent))


def morph_pair(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    config,
    alpha: float,
    *,
    blend_mode: str,
    phase_mix: float | None,
    mask_exponent: float,
    envelope_lifter: int,
    normalize_energy: bool,
) -> np.ndarray:
    max_len = max(a.shape[0], b.shape[0])
    channels = max(a.shape[1], b.shape[1])
    a2 = np.zeros((max_len, channels), dtype=np.float64)
    b2 = np.zeros((max_len, channels), dtype=np.float64)
    a_adj = match_channels(a, channels)
    b_adj = match_channels(b, channels)
    a2[: a_adj.shape[0], :] = a_adj
    b2[: b_adj.shape[0], :] = b_adj

    out = np.zeros_like(a2)
    mode = str(blend_mode).strip().lower()
    pmix = _resolve_phase_mix(mode, alpha, phase_mix)
    for ch in range(channels):
        sa = stft(a2[:, ch], config)
        sb = stft(b2[:, ch], config)
        n_bins = max(sa.shape[0], sb.shape[0])
        n_frames = max(sa.shape[1], sb.shape[1])
        sa2 = np.zeros((n_bins, n_frames), dtype=np.complex128)
        sb2 = np.zeros((n_bins, n_frames), dtype=np.complex128)
        sa2[: sa.shape[0], : sa.shape[1]] = sa
        sb2[: sb.shape[0], : sb.shape[1]] = sb

        mag_a = np.abs(sa2)
        mag_b = np.abs(sb2)
        phase_a = np.angle(sa2)
        phase_b = np.angle(sb2)
        linear_mag = (1.0 - alpha) * mag_a + alpha * mag_b

        if mode == "linear":
            out_mag = linear_mag
        elif mode == "geometric":
            out_mag = np.exp((1.0 - alpha) * np.log(mag_a + EPS) + alpha * np.log(mag_b + EPS))
        elif mode == "magnitude_b_phase_a":
            out_mag = linear_mag
        elif mode == "magnitude_a_phase_b":
            out_mag = linear_mag
        elif mode == "carrier_a_envelope_b":
            env_a = _framewise_envelope(mag_a, envelope_lifter)
            env_b = _framewise_envelope(mag_b, envelope_lifter)
            target = (mag_a / env_a) * env_b
            out_mag = (1.0 - alpha) * mag_a + alpha * target
        elif mode == "carrier_b_envelope_a":
            env_a = _framewise_envelope(mag_a, envelope_lifter)
            env_b = _framewise_envelope(mag_b, envelope_lifter)
            target = (mag_b / env_b) * env_a
            out_mag = (1.0 - alpha) * mag_b + alpha * target
        elif mode == "carrier_a_mask_b":
            target = mag_a * _mask_from_modulator(mag_b, mask_exponent)
            out_mag = (1.0 - alpha) * mag_a + alpha * target
        elif mode == "carrier_b_mask_a":
            target = mag_b * _mask_from_modulator(mag_a, mask_exponent)
            out_mag = (1.0 - alpha) * mag_b + alpha * target
        elif mode == "product":
            target = np.sqrt(np.maximum(mag_a * mag_b, 0.0))
            out_mag = (1.0 - alpha) * mag_a + alpha * target
        elif mode == "max_mag":
            target = np.maximum(mag_a, mag_b)
            out_mag = (1.0 - alpha) * linear_mag + alpha * target
        elif mode == "min_mag":
            target = np.minimum(mag_a, mag_b)
            out_mag = (1.0 - alpha) * linear_mag + alpha * target
        else:
            raise ValueError(f"Unsupported blend mode: {blend_mode}")

        out_phase = _phase_blend(phase_a, phase_b, pmix)
        sm = np.maximum(out_mag, 0.0) * np.exp(1j * out_phase)
        rendered = istft(sm, config, expected_length=max_len)

        if normalize_energy:
            target_rms = (1.0 - alpha) * _safe_rms(a2[:, ch]) + alpha * _safe_rms(b2[:, ch])
            cur_rms = _safe_rms(rendered)
            if target_rms > 0.0 and cur_rms > EPS:
                rendered = rendered * (target_rms / cur_rms)
        out[:, ch] = rendered
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Morph two audio files in the STFT domain",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx morph source_a.wav source_b.wav --alpha 0.35 --output morph_35.wav",
                "pvx morph vocal.wav strings.wav --alpha 0.6 --n-fft 4096 --hop-size 512 --output vocal_strings.wav",
                "pvx morph voice.wav synth.wav --blend-mode carrier_a_envelope_b --alpha 0.8 --envelope-lifter 40 --output voice_synth_env.wav",
                "pvx morph drums.wav guitar.wav --blend-mode carrier_a_mask_b --alpha 0.7 --mask-exponent 1.2 --output drums_guitar_mask.wav",
                "pvx morph a.wav b.wav --alpha 0.5 --stdout | pvx freeze - --freeze-time 0.8 --duration 12 --output morph_freeze.wav",
            ],
            notes=[
                "--alpha 0.0 keeps input A, --alpha 1.0 keeps input B.",
                "Try --blend-mode carrier_a_envelope_b or carrier_a_mask_b for stronger cross-synthesis character.",
                "Only one morph input can be '-' (stdin).",
            ],
        ),
    )
    parser.add_argument("input_a", type=Path, help="Input A path or '-' for stdin")
    parser.add_argument("input_b", type=Path, help="Input B path or '-' for stdin")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file path")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write processed audio to stdout stream (for piping); equivalent to -o -",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        help="Output extension/format; for --stdout defaults to wav",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Morph amount 0..1 (0=A, 1=B)")
    parser.add_argument(
        "--blend-mode",
        choices=list(BLEND_MODE_CHOICES),
        default="linear",
        help=(
            "Cross-synthesis blend style. "
            "linear/geometric are symmetric blends; carrier_* modes transfer envelope/mask from modulator to carrier."
        ),
    )
    parser.add_argument(
        "--phase-mix",
        type=float,
        default=None,
        help=(
            "Phase blend in [0,1]. If omitted, mode-specific defaults apply "
            "(A-phase for *_phase_a/carrier_a_*, B-phase for *_phase_b/carrier_b_*, alpha for symmetric modes)."
        ),
    )
    parser.add_argument(
        "--mask-exponent",
        type=float,
        default=1.0,
        help="Exponent used by carrier_*_mask_* modes (default: 1.0).",
    )
    parser.add_argument(
        "--envelope-lifter",
        type=int,
        default=32,
        help="Cepstral lifter cutoff for carrier_*_envelope_* modes (default: 32).",
    )
    parser.add_argument(
        "--normalize-energy",
        action="store_true",
        help="Normalize each output channel RMS toward alpha-blended input RMS.",
    )
    add_mastering_args(parser)
    add_output_policy_args(parser)
    parser.add_argument("--overwrite", action="store_true")
    add_console_args(parser)
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if not (0.0 <= args.alpha <= 1.0):
        parser.error("--alpha must be between 0 and 1")
    if args.phase_mix is not None and not (0.0 <= args.phase_mix <= 1.0):
        parser.error("--phase-mix must be between 0 and 1")
    if args.mask_exponent <= 0.0:
        parser.error("--mask-exponent must be > 0")
    if args.envelope_lifter < 0:
        parser.error("--envelope-lifter must be >= 0")
    if str(args.input_a) == "-" and str(args.input_b) == "-":
        parser.error("At most one morph input may be '-' (stdin)")
    if args.output is not None and str(args.output) == "-":
        args.stdout = True
    if args.stdout and args.output is not None and str(args.output) != "-":
        parser.error("Use either --stdout (or -o -) or an explicit output file path, not both")

    config = build_vocoder_config(args, phase_locking="off", transient_preserve=False)
    status = build_status_bar(args, "pvxmorph", 1)
    try:
        a, sr_a = read_audio(args.input_a)
        b, sr_b = read_audio(args.input_b)
        if sr_b != sr_a:
            target = max(1, int(round(b.shape[0] * sr_a / sr_b)))
            rb = np.zeros((target, b.shape[1]), dtype=np.float64)
            for ch in range(b.shape[1]):
                rb[:, ch] = resample_1d(b[:, ch], target, "auto")
            b = rb

        out = morph_pair(
            a,
            b,
            sr_a,
            config,
            args.alpha,
            blend_mode=args.blend_mode,
            phase_mix=args.phase_mix,
            mask_exponent=args.mask_exponent,
            envelope_lifter=int(args.envelope_lifter),
            normalize_energy=bool(args.normalize_energy),
        )
        out = apply_mastering_chain(out, sr_a, args)

        if args.stdout:
            out_path = Path("-")
        elif args.output is not None:
            out_path = args.output
        else:
            base = Path("stdin.wav") if str(args.input_a) == "-" else args.input_a
            out_path = base.with_name(f"{base.stem}_morph.wav")
        metrics_table = render_audio_metrics_table(
            [
                (f"inA:{args.input_a}", summarize_audio_metrics(a, int(sr_a))),
                (f"inB:{args.input_b}", summarize_audio_metrics(b, int(sr_a))),
                (f"out:{out_path}", summarize_audio_metrics(out, int(sr_a))),
            ],
            title="Audio Metrics",
            include_delta_from_first=True,
        )
        compare_a = render_audio_comparison_table(
            reference_label=f"inA:{args.input_a}",
            reference_audio=a,
            reference_sr=int(sr_a),
            candidate_label=f"out:{out_path}",
            candidate_audio=out,
            candidate_sr=int(sr_a),
            title="Audio Compare Metrics (out vs inA)",
        )
        compare_b = render_audio_comparison_table(
            reference_label=f"inB:{args.input_b}",
            reference_audio=b,
            reference_sr=int(sr_a),
            candidate_label=f"out:{out_path}",
            candidate_audio=out,
            candidate_sr=int(sr_a),
            title="Audio Compare Metrics (out vs inB)",
        )
        log_message(args, f"{metrics_table}\n{compare_a}\n{compare_b}", min_level="quiet")
        write_output(
            out_path,
            out,
            sr_a,
            args,
            metadata_extra={
                "morph": {
                    "input_a": str(args.input_a),
                    "input_b": str(args.input_b),
                    "alpha": float(args.alpha),
                    "blend_mode": str(args.blend_mode),
                    "phase_mix": None if args.phase_mix is None else float(args.phase_mix),
                    "mask_exponent": float(args.mask_exponent),
                    "envelope_lifter": int(args.envelope_lifter),
                    "normalize_energy": bool(args.normalize_energy),
                }
            },
        )
        log_message(args, f"[ok] {args.input_a} + {args.input_b} -> {out_path}", min_level="verbose")
        status.step(1, out_path.name)
        status.finish("done")
        log_message(args, "[done] pvxmorph processed=1 failed=0", min_level="normal")
        return 0
    except Exception as exc:
        log_error(args, f"[error] {args.input_a} + {args.input_b}: {exc}")
        status.step(1, "error")
        status.finish("errors=1")
        log_message(args, "[done] pvxmorph processed=1 failed=1", min_level="normal")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
