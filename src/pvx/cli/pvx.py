#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unified top-level CLI for the pvx command suite."""

from __future__ import annotations

import argparse
import contextlib
import difflib
import importlib
import io
import json
import os
import random
import re
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from pvx.core.streaming import run_stateful_stream


@dataclass(frozen=True)
class ToolSpec:
    name: str
    entrypoint: str
    summary: str
    aliases: tuple[str, ...] = ()


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="voc",
        entrypoint="pvx.core.voc:main",
        summary="General-purpose phase-vocoder time/pitch processing",
        aliases=("pvxvoc", "vocoder", "timepitch"),
    ),
    ToolSpec(
        name="freeze",
        entrypoint="pvx.cli.pvxfreeze:main",
        summary="Freeze a spectral frame into a sustained texture",
        aliases=("pvxfreeze",),
    ),
    ToolSpec(
        name="harmonize",
        entrypoint="pvx.cli.pvxharmonize:main",
        summary="Generate harmony voices from one source",
        aliases=("pvxharmonize", "harm"),
    ),
    ToolSpec(
        name="conform",
        entrypoint="pvx.cli.pvxconform:main",
        summary="Apply CSV segment map for time/pitch conformity",
        aliases=("pvxconform",),
    ),
    ToolSpec(
        name="morph",
        entrypoint="pvx.cli.pvxmorph:main",
        summary="Morph two sources in the STFT domain",
        aliases=("pvxmorph",),
    ),
    ToolSpec(
        name="warp",
        entrypoint="pvx.cli.pvxwarp:main",
        summary="Apply variable stretch map from CSV",
        aliases=("pvxwarp",),
    ),
    ToolSpec(
        name="formant",
        entrypoint="pvx.cli.pvxformant:main",
        summary="Formant shift/preserve processing",
        aliases=("pvxformant",),
    ),
    ToolSpec(
        name="transient",
        entrypoint="pvx.cli.pvxtransient:main",
        summary="Transient-aware time/pitch processing",
        aliases=("pvxtransient",),
    ),
    ToolSpec(
        name="unison",
        entrypoint="pvx.cli.pvxunison:main",
        summary="Unison thickening and width enhancement",
        aliases=("pvxunison",),
    ),
    ToolSpec(
        name="denoise",
        entrypoint="pvx.cli.pvxdenoise:main",
        summary="Spectral denoise",
        aliases=("pvxdenoise",),
    ),
    ToolSpec(
        name="deverb",
        entrypoint="pvx.cli.pvxdeverb:main",
        summary="Dereverb spectral tail reduction",
        aliases=("pvxdeverb", "dereverb"),
    ),
    ToolSpec(
        name="retune",
        entrypoint="pvx.cli.pvxretune:main",
        summary="Monophonic pitch retune to scale/root",
        aliases=("pvxretune",),
    ),
    ToolSpec(
        name="layer",
        entrypoint="pvx.cli.pvxlayer:main",
        summary="Split/process harmonic and percussive layers",
        aliases=("pvxlayer",),
    ),
    ToolSpec(
        name="pitch-track",
        entrypoint="pvx.cli.hps_pitch_track:main",
        summary="Track f0 and emit control-map CSV",
        aliases=("hps-pitch-track", "hps", "track"),
    ),
    ToolSpec(
        name="analysis",
        entrypoint="pvx.cli.pvxanalysis:main",
        summary="Create/inspect reusable PVXAN analysis artifacts",
        aliases=("pvxanalysis",),
    ),
    ToolSpec(
        name="response",
        entrypoint="pvx.cli.pvxresponse:main",
        summary="Create/inspect reusable PVXRF response artifacts",
        aliases=("pvxresponse",),
    ),
    ToolSpec(
        name="envelope",
        entrypoint="pvx.cli.pvxenvelope:main",
        summary="Generate control-rate envelope maps (CSV/JSON)",
        aliases=("pvxenvelope", "lfo"),
    ),
    ToolSpec(
        name="reshape",
        entrypoint="pvx.cli.pvxreshape:main",
        summary="Reshape control-rate maps for pvx routing",
        aliases=("pvxreshape",),
    ),
    ToolSpec(
        name="filter",
        entrypoint="pvx.cli.pvxfilter:main",
        summary="Response-driven spectral filtering (PVC-inspired)",
        aliases=("pvxfilter",),
    ),
    ToolSpec(
        name="tvfilter",
        entrypoint="pvx.cli.pvxtvfilter:main",
        summary="Time-varying response filter with control maps",
        aliases=("pvxtvfilter",),
    ),
    ToolSpec(
        name="noisefilter",
        entrypoint="pvx.cli.pvxnoisefilter:main",
        summary="Response-referenced noise filtering",
        aliases=("pvxnoisefilter",),
    ),
    ToolSpec(
        name="bandamp",
        entrypoint="pvx.cli.pvxbandamp:main",
        summary="Response-peak band amplification",
        aliases=("pvxbandamp",),
    ),
    ToolSpec(
        name="spec-compander",
        entrypoint="pvx.cli.pvxspeccompander:main",
        summary="Response-referenced spectral compander",
        aliases=("pvxspeccompander", "speccompander"),
    ),
    ToolSpec(
        name="ring",
        entrypoint="pvx.cli.pvxring:main",
        summary="Ring modulation operator",
        aliases=("pvxring",),
    ),
    ToolSpec(
        name="ringfilter",
        entrypoint="pvx.cli.pvxringfilter:main",
        summary="Ring modulation plus resonator filtering",
        aliases=("pvxringfilter",),
    ),
    ToolSpec(
        name="ringtvfilter",
        entrypoint="pvx.cli.pvxringtvfilter:main",
        summary="Time-varying ring modulation plus resonator filtering",
        aliases=("pvxringtvfilter",),
    ),
    ToolSpec(
        name="chordmapper",
        entrypoint="pvx.cli.pvxchordmapper:main",
        summary="Chord-aware spectral mapping",
        aliases=("pvxchordmapper",),
    ),
    ToolSpec(
        name="inharmonator",
        entrypoint="pvx.cli.pvxinharmonator:main",
        summary="Inharmonic spectral warping",
        aliases=("pvxinharmonator",),
    ),
    ToolSpec(
        name="trajectory-reverb",
        entrypoint="pvx.cli.pvxtrajectoryreverb:main",
        summary="Mono-to-multichannel trajectory convolution reverb",
        aliases=("pvxtrajectoryreverb", "trajreverb", "spatial-reverb"),
    ),
)


EXAMPLE_COMMANDS: dict[str, tuple[str, str]] = {
    "doctor": ("Environment diagnostics", "pvx doctor"),
    "quickstart": (
        "Minimal launch/demo sequence",
        "pvx quickstart input.wav --output output.wav",
    ),
    "safe": (
        "Quality-first conservative voc wrapper",
        "pvx safe input.wav --material mix --output output_safe.wav",
    ),
    "transforms": (
        "Transform availability and recommendation guide",
        "pvx transforms",
    ),
    "smoke": (
        "Synthetic end-to-end smoke render",
        "pvx smoke --output smoke_out.wav",
    ),
    "basic": ("Basic stretch", "pvx voc input.wav --stretch 1.20 --output output.wav"),
    "speech": ("Slow speech for review", "pvx voc speech.wav --preset vocal_studio --stretch 1.30 --output speech_slow.wav"),
    "vocal": (
        "Vocal pitch/formant correction",
        "pvx voc vocal.wav --preset vocal_studio --pitch -2 --output vocal_fixed.wav",
    ),
    "retune": (
        "Scale retune",
        "pvx retune vocal.wav --root C --scale major --strength 0.85 --output vocal_retuned.wav",
    ),
    "freeze": (
        "Freeze a spectral moment",
        "pvx freeze hit.wav --freeze-time 0.2 --duration 10 --output hit_freeze.wav",
    ),
    "ambient": (
        "Extreme ambient stretch",
        "pvx voc one_shot.wav --preset extreme_ambient --target-duration 600 --output one_shot_ambient.wav",
    ),
    "drums": (
        "Transient-safe drums",
        "pvx voc drums.wav --preset drums_safe --stretch 1.25 --output drums_safe.wav",
    ),
    "morph": (
        "Source morph",
        "pvx morph source_a.wav source_b.wav --alpha controls/alpha_curve.csv --interp linear --output morph_traj.wav",
    ),
    "map": (
        "Time/pitch map conform",
        "pvx conform source.wav --map map_conform.csv --output source_conformed.wav",
    ),
    "microtonal": (
        "Microtonal pitch ratio",
        "pvx voc input.wav --stretch 1.0 --ratio 3/2 --output input_perfect_fifth.wav",
    ),
    "pipe": (
        "Short one-line pipe",
        "pvx voc input.wav --stretch 1.2 --stdout | pvx deverb - --strength 0.3 --output output.wav",
    ),
    "pipeline": (
        "Pitch-follow pipeline",
        "pvx pitch-track guide.wav --emit pitch_to_stretch --output - | pvx voc target.wav --control-stdin --output followed.wav",
    ),
    "follow": (
        "Single-command sidechain follow",
        "pvx follow guide.wav target.wav --output followed.wav --emit pitch_to_stretch --pitch-conf-min 0.75",
    ),
    "follow-feature": (
        "Feature-driven follow (MFCC + MPEG-7 spectral flux)",
        "pvx follow guide.wav target.wav --feature-set all --mfcc-count 13 --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(mfcc_01,0.002,1.0) --route pitch_ratio=clip(pitch_ratio,0.5,2.0) --route stretch=affine(mpeg7_spectral_flux,0.05,1.0) --route stretch=clip(stretch,0.85,1.6) --output followed_feature.wav",
    ),
    "follow-formant": (
        "Feature-driven follow (formant and onset)",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(formant_f1_hz,0.0016,0.2) --route pitch_ratio=clip(pitch_ratio,0.7,1.5) --route stretch=affine(onset_norm,-0.35,1.2) --route stretch=clip(stretch,0.8,1.3) --output followed_formant_onset.wav",
    ),
    "follow-noise-aware": (
        "Feature-driven follow (noise-aware hiss/hum control)",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route stretch=affine(hiss_ratio,-0.6,1.2) --route stretch=clip(stretch,0.8,1.2) --route pitch_ratio=affine(hum_60_ratio,-0.4,1.15) --route pitch_ratio=clip(pitch_ratio,0.9,1.2) --output followed_noise_aware.wav",
    ),
    "analysis": (
        "Create reusable analysis artifact",
        "pvx analysis create input.wav --output input.pvxan.npz --n-fft 4096 --hop-size 256",
    ),
    "response": (
        "Derive reusable response artifact",
        "pvx response create input.pvxan.npz --output input.pvxrf.npz --method median --normalize peak",
    ),
    "envelope": (
        "Generate a stretch envelope/LFO control map",
        "pvx lfo --wave triangle --duration 8 --frequency-hz 0.5 --center 1.0 --amplitude 0.2 --key stretch --output stretch_lfo.csv",
    ),
    "reshape": (
        "Reshape and resample a control map",
        "pvx reshape stretch_env.csv --key stretch --operation resample --rate 50 --interp polynomial --order 5 --output stretch_env_dense.csv",
    ),
    "filter": (
        "Response-driven static filter",
        "pvx filter input.wav --response input.pvxrf.npz --response-mix 1.0 --output filtered.wav",
    ),
    "tvfilter": (
        "Time-varying response filter",
        "pvx tvfilter input.wav --response input.pvxrf.npz --tv-map mix_map.csv --tv-interp linear --output tvfiltered.wav",
    ),
    "ringfilter": (
        "Ring + resonator filter",
        "pvx ringfilter input.wav --frequency-hz 55 --resonance-hz 1200 --resonance-q 9 --output ringfilter.wav",
    ),
    "chordmapper": (
        "Chord-aware harmonic mapping",
        "pvx chordmapper input.wav --root-hz 220 --chord minor --strength 0.75 --output chordmapped.wav",
    ),
    "inharmonator": (
        "Inharmonic spectral warping",
        "pvx inharmonator input.wav --inharmonic-f0-hz 220 --inharmonicity 0.0002 --inharmonic-mix 1.0 --output inharm.wav",
    ),
    "trajectory-reverb": (
        "Mono through multichannel room IR with A->B movement",
        "pvx trajectory-reverb source.wav --ir room_4ch.wav --coord-system cartesian --start -1,0,1 --end 1,0,1 --output flythrough.wav",
    ),
    "chain": (
        "Managed multi-stage chain",
        "pvx chain input.wav --pipeline \"voc --stretch 1.2 | formant --mode preserve\" --output output_chain.wav",
    ),
    "stream": (
        "Chunked stream wrapper over pvx voc",
        "pvx stream input.wav --output output_stream.wav --chunk-seconds 0.2 --time-stretch 2.0 --preset extreme_ambient",
    ),
    "stretch-budget": (
        "Estimate max safe stretch from a file and disk budget",
        "pvx stretch-budget input.wav --disk-budget 20GB --bit-depth 16 --requested-stretch 1000000",
    ),
}


FOLLOW_EXAMPLE_COMMANDS: dict[str, tuple[str, str]] = {
    "basic": (
        "Pitch-to-stretch sidechain",
        "pvx follow guide.wav target.wav --emit pitch_to_stretch --pitch-conf-min 0.75 --output followed.wav",
    ),
    "pitch": (
        "Pitch-map follow with fixed stretch",
        "pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 --output followed_pitch.wav",
    ),
    "mfcc_flux": (
        "MFCC + MPEG-7 flux dual control",
        "pvx follow guide.wav target.wav --feature-set all --mfcc-count 13 --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(mfcc_01,0.002,1.0) --route pitch_ratio=clip(pitch_ratio,0.5,2.0) --route stretch=affine(mpeg7_spectral_flux,0.05,1.0) --route stretch=clip(stretch,0.85,1.6) --output followed_mfcc_flux.wav",
    ),
    "formant_onset": (
        "Formant + onset dual control",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(formant_f1_hz,0.0016,0.2) --route pitch_ratio=clip(pitch_ratio,0.7,1.5) --route stretch=affine(onset_norm,-0.35,1.2) --route stretch=clip(stretch,0.8,1.3) --output followed_formant_onset.wav",
    ),
    "noise_aware": (
        "Noise-aware hiss/hum routing",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route stretch=affine(hiss_ratio,-0.6,1.2) --route stretch=clip(stretch,0.8,1.2) --route pitch_ratio=affine(hum_60_ratio,-0.4,1.15) --route pitch_ratio=clip(pitch_ratio,0.9,1.2) --output followed_noise_aware.wav",
    ),
}

FOLLOW_EXAMPLE_CHOICES: tuple[str, ...] = ("all", *tuple(FOLLOW_EXAMPLE_COMMANDS.keys()))

_AUDIO_EXTENSIONS: set[str] = {
    ".wav",
    ".flac",
    ".aiff",
    ".aif",
    ".ogg",
    ".oga",
    ".caf",
    ".mp3",
    ".m4a",
    ".aac",
    ".wma",
}

_CHAIN_TOOL_ALLOWLIST: set[str] = {
    "voc",
    "freeze",
    "harmonize",
    "conform",
    "warp",
    "formant",
    "transient",
    "unison",
    "denoise",
    "deverb",
    "retune",
    "layer",
    "filter",
    "tvfilter",
    "noisefilter",
    "bandamp",
    "spec-compander",
    "ring",
    "ringfilter",
    "ringtvfilter",
    "chordmapper",
    "inharmonator",
    "trajectory-reverb",
}
_CHAIN_STAGE_FORBIDDEN_FLAGS: set[str] = {
    "-o",
    "--out",
    "--output",
    "--output-dir",
    "--stdout",
}

_LUCKY_SUPPORTED_TOOLS: set[str] = set(_CHAIN_TOOL_ALLOWLIST) | {"morph"}
_LUCKY_PRESETS: tuple[str, ...] = ("default", "vocal_studio", "drums_safe", "extreme_ambient", "stereo_coherent")
_LUCKY_WINDOWS: tuple[str, ...] = ("hann", "hamming", "blackmanharris", "kaiser", "tukey")


def _tool_index() -> dict[str, ToolSpec]:
    out: dict[str, ToolSpec] = {}
    for spec in TOOL_SPECS:
        out[spec.name] = spec
        for alias in spec.aliases:
            out[alias] = spec
    return out


TOOL_INDEX = _tool_index()


def _load_entrypoint(entrypoint: str) -> Callable[[list[str] | None], int]:
    module_name, func_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    return fn


def _looks_like_audio_input(token: str) -> bool:
    if token == "-":
        return True
    if any(ch in token for ch in "*?["):
        return True
    path = Path(token)
    if path.suffix.lower() in _AUDIO_EXTENSIONS:
        return True
    return path.exists()


def _tool_names_csv() -> str:
    return ", ".join(spec.name for spec in TOOL_SPECS)


def print_tools() -> None:
    print("pvx command list")
    print("")
    print("Primary subcommands:")
    for spec in TOOL_SPECS:
        aliases = ""
        if spec.aliases:
            aliases = f" [aliases: {', '.join(spec.aliases)}]"
        print(f"  {spec.name:<12} {spec.summary}{aliases}")
    print("")
    print("Helper commands:")
    print("  list         Show this command table")
    print("  examples     Show copy-paste examples (use `pvx examples <name>`)")
    print("  quickstart   Print a minimal launch sequence")
    print("  doctor       Run environment diagnostics and suggested fixes")
    print("  transforms   Show transform choices and recommendations")
    print("  safe         Run `pvx voc` with conservative quality-first defaults")
    print("  smoke        Fast synthetic end-to-end smoke render")
    print("  guided       Interactive command builder")
    print("  follow       Track one file and control another in one command")
    print("  chain        Run a managed multi-stage one-line tool chain")
    print("  stream       Chunked stream wrapper around `pvx voc`")
    print("  stretch-budget  Estimate max safe stretch from file size/budget assumptions")
    print("  help <tool>  Show subcommand help")
    print("")
    print("Global randomizer:")
    print("  --lucky N [--lucky-seed S]  Run selected workflow N times with randomized DSP settings")
    print("")
    print("Use installed commands (`pvx`, `pvxvoc`, `pvxfreeze`, ...) or `python -m pvx...` module entry points.")


def print_examples(which: str = "all") -> None:
    key = str(which).strip().lower()
    if key == "all":
        print("pvx example commands")
        print("")
        for name, (title, command) in EXAMPLE_COMMANDS.items():
            print(f"[{name}] {title}")
            print(command)
            print("")
        return
    if key not in EXAMPLE_COMMANDS:
        raise ValueError(
            f"Unknown example '{which}'. Use one of: {', '.join(sorted(EXAMPLE_COMMANDS))}, all"
        )
    title, command = EXAMPLE_COMMANDS[key]
    print(f"[{key}] {title}")
    print(command)


def _prompt_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def _prompt_choice(prompt: str, choices: tuple[str, ...], default: str) -> str:
    value = _prompt_text(prompt, default).strip().lower()
    if value not in choices:
        raise ValueError(f"Expected one of: {', '.join(choices)}")
    return value


def _print_command_preview(command: str, forwarded_args: list[str]) -> None:
    cmd = " ".join([shlex.quote("pvx"), shlex.quote(command)] + [shlex.quote(a) for a in forwarded_args])
    print("")
    print("Generated command:")
    print(cmd)
    print("")


def print_follow_examples(which: str = "basic") -> None:
    key = str(which).strip().lower()
    if key == "all":
        print("pvx follow example commands")
        print("")
        for name, (title, command) in FOLLOW_EXAMPLE_COMMANDS.items():
            print(f"[{name}] {title}")
            print(command)
            print("")
        return
    if key not in FOLLOW_EXAMPLE_COMMANDS:
        raise ValueError(
            f"Unknown follow example '{which}'. Use one of: {', '.join(FOLLOW_EXAMPLE_CHOICES)}"
        )
    title, command = FOLLOW_EXAMPLE_COMMANDS[key]
    print(f"[{key}] {title}")
    print(command)


def run_doctor_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx doctor",
        description="Environment and launch-readiness diagnostics for pvx.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code if warnings are found",
    )
    args = parser.parse_args(forwarded_args)

    cwd = Path.cwd().resolve()
    venv_active = bool(getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    python_exe = str(Path(sys.executable).resolve())
    pvx_on_path = shutil.which("pvx")
    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    venv_bin = str((cwd / ".venv" / "bin").resolve())

    try:
        importlib.import_module("scipy")
        scipy_ok = True
    except Exception:
        scipy_ok = False
    try:
        importlib.import_module("cupy")
        cupy_ok = True
    except Exception:
        cupy_ok = False

    warnings: list[str] = []
    if not venv_active:
        warnings.append("Python virtual environment is not active.")
    if pvx_on_path is None:
        warnings.append("`pvx` executable is not on PATH.")
    if (cwd / ".venv").exists() and venv_bin not in path_entries:
        warnings.append("Project virtualenv bin directory is not on PATH.")
    if not scipy_ok:
        warnings.append("SciPy not installed: czt/dct/dst/hartley transforms may be unavailable.")

    report = {
        "python_executable": python_exe,
        "python_version": sys.version.split()[0],
        "cwd": str(cwd),
        "venv_active": venv_active,
        "pvx_on_path": pvx_on_path,
        "venv_bin_expected": venv_bin,
        "venv_bin_on_path": venv_bin in path_entries,
        "scipy_installed": scipy_ok,
        "cupy_installed": cupy_ok,
        "warnings": list(warnings),
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("pvx doctor")
        print(f"- python: {report['python_executable']} (v{report['python_version']})")
        print(f"- working directory: {report['cwd']}")
        print(f"- virtual environment active: {'yes' if venv_active else 'no'}")
        print(f"- pvx on PATH: {pvx_on_path if pvx_on_path is not None else 'no'}")
        print(f"- scipy installed: {'yes' if scipy_ok else 'no'}")
        print(f"- cupy installed: {'yes' if cupy_ok else 'no'}")
        if warnings:
            print("")
            print("Warnings:")
            for item in warnings:
                print(f"- {item}")
            print("")
            print("Suggested fixes:")
            print("- Activate virtual environment: source .venv/bin/activate")
            print("- Add pvx to PATH (zsh):")
            print('  printf \'export PATH="%s/.venv/bin:$PATH"\\n\' "$(pwd)" >> ~/.zshrc && source ~/.zshrc')
            print("- Install optional transforms/GPU dependencies:")
            print("  python3 -m pip install scipy")
            print("  python3 -m pip install cupy-cuda12x  # optional, NVIDIA CUDA only")
        else:
            print("")
            print("No launch-blocking warnings found.")

    if args.strict and warnings:
        return 1
    return 0


def run_quickstart_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx quickstart",
        description="Print a minimal copy-paste launch sequence for announcement demos.",
    )
    parser.add_argument("input", nargs="?", default="input.wav", help="Input audio path (default: input.wav)")
    parser.add_argument("--output", default="output.wav", help="Output audio path (default: output.wav)")
    parser.add_argument(
        "--material",
        choices=["mix", "speech", "vocal", "drums", "ambient"],
        default="mix",
        help="Material profile for `pvx safe` command generation",
    )
    args = parser.parse_args(forwarded_args)

    print("pvx quickstart")
    print("")
    print("1) Diagnose environment")
    print("pvx doctor")
    print("")
    print("2) Run quality-safe first render")
    print(
        f"pvx safe {shlex.quote(str(args.input))} --material {args.material} "
        f"--output {shlex.quote(str(args.output))}"
    )
    print("")
    print("3) Inspect transform options")
    print("pvx transforms")
    print("")
    print("4) Print curated examples")
    print("pvx examples basic")
    print("")
    print("5) Run a synthetic smoke test")
    print("pvx smoke --output smoke_out.wav")
    return 0


def run_safe_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx safe",
        description="Run `pvx voc` with conservative, quality-first defaults for first-pass renders.",
    )
    parser.add_argument("input", help="Input audio path")
    parser.add_argument("--output", "--out", dest="output", required=True, help="Output audio path")
    parser.add_argument(
        "--material",
        choices=["mix", "speech", "vocal", "drums", "ambient"],
        default="mix",
        help="Material profile (default: mix)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs")
    parser.add_argument("--silent", action="store_true", help="Suppress logs")
    args, passthrough = parser.parse_known_args(forwarded_args)

    passthrough_flags = {_token_flag(token) for token in passthrough if token.startswith("-")}
    forbidden_passthrough = {"--output", "--out", "-o", "--stdout"}
    bad_flags = sorted(passthrough_flags & forbidden_passthrough)
    if bad_flags:
        parser.error(
            f"Do not pass {bad_flags} via passthrough in `pvx safe`; safe mode manages output routing."
        )

    preset_by_material = {
        "mix": "stereo_coherent",
        "speech": "vocal_studio",
        "vocal": "vocal_studio",
        "drums": "drums_safe",
        "ambient": "extreme_ambient",
    }

    voc_args: list[str] = [
        str(args.input),
        "--preset",
        preset_by_material[str(args.material)],
    ]

    if "--phase-locking" not in passthrough_flags:
        voc_args.extend(["--phase-locking", "identity"])
    if "--transient-mode" not in passthrough_flags:
        voc_args.extend(["--transient-mode", "hybrid"])
    if "--transient-sensitivity" not in passthrough_flags:
        voc_args.extend(["--transient-sensitivity", "0.60"])
    if "--stereo-mode" not in passthrough_flags:
        voc_args.extend(["--stereo-mode", "mid_side_lock"])
    if "--coherence-strength" not in passthrough_flags:
        voc_args.extend(["--coherence-strength", "0.85"])

    voc_args.extend(["--output", str(args.output)])
    if args.overwrite:
        voc_args.append("--overwrite")
    if args.quiet:
        voc_args.append("--quiet")
    if args.silent:
        voc_args.append("--silent")
    voc_args.extend(passthrough)
    return dispatch_tool("voc", voc_args)


def run_transforms_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx transforms",
        description="Show available per-frame transform backends and practical recommendations.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(forwarded_args)

    try:
        importlib.import_module("scipy.fft")
        scipy_fft_ok = True
    except Exception:
        scipy_fft_ok = False
    try:
        from scipy.signal import czt as _scipy_czt  # noqa: F401
        scipy_czt_ok = True
    except Exception:
        scipy_czt_ok = False

    transforms = [
        {"name": "fft", "available": True, "recommended_for": "default production use", "notes": "Best overall speed/quality baseline"},
        {"name": "dft", "available": True, "recommended_for": "reference and non-power-of-two research", "notes": "Slowest but direct reference path"},
        {"name": "czt", "available": scipy_czt_ok, "recommended_for": "zoomed/custom spectral focus", "notes": "Requires scipy.signal.czt"},
        {"name": "dct", "available": scipy_fft_ok, "recommended_for": "real-transform experiments", "notes": "Requires scipy.fft"},
        {"name": "dst", "available": scipy_fft_ok, "recommended_for": "real-transform experiments", "notes": "Requires scipy.fft"},
        {"name": "hartley", "available": scipy_fft_ok, "recommended_for": "real-transform experiments", "notes": "Requires scipy.fft"},
    ]

    if args.json:
        print(json.dumps({"transforms": transforms}, indent=2, sort_keys=True))
        return 0

    print("pvx transform guide")
    print("")
    for item in transforms:
        state = "yes" if bool(item["available"]) else "no"
        print(f"- {item['name']}: available={state}")
        print(f"  use: {item['recommended_for']}")
        print(f"  note: {item['notes']}")
    print("")
    print("Rule of thumb: start with --transform fft, then A/B against alternatives only if you need a specific behavior.")
    return 0


def _build_smoke_signal(sample_rate: int, duration: float) -> np.ndarray:
    frames = max(1024, int(round(float(sample_rate) * float(duration))))
    t = np.arange(frames, dtype=np.float64) / float(sample_rate)
    tone = 0.18 * np.sin(2.0 * np.pi * 220.0 * t) + 0.08 * np.sin(2.0 * np.pi * 440.0 * t)
    fade = min(256, max(8, frames // 20))
    ramp = np.linspace(0.0, 1.0, num=fade, endpoint=True, dtype=np.float64)
    tone[:fade] *= ramp
    tone[-fade:] *= ramp[::-1]
    return np.stack([tone, tone], axis=1)


def run_smoke_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx smoke",
        description="Run a fast synthetic end-to-end smoke render for launch confidence.",
    )
    parser.add_argument("--output", default="smoke_out.wav", help="Output path for smoke render")
    parser.add_argument("--duration", type=float, default=0.30, help="Synthetic input duration seconds (default: 0.30)")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Synthetic input sample rate (default: 24000)")
    parser.add_argument("--stretch", type=float, default=1.25, help="Smoke render stretch factor (default: 1.25)")
    parser.add_argument("--pitch", type=float, default=0.0, help="Smoke render pitch semitones (default: 0.0)")
    args = parser.parse_args(forwarded_args)

    if float(args.duration) <= 0.0:
        parser.error("--duration must be > 0")
    if int(args.sample_rate) <= 1000:
        parser.error("--sample-rate must be > 1000")
    if float(args.stretch) <= 0.0:
        parser.error("--stretch must be > 0")

    with tempfile.TemporaryDirectory(prefix="pvx-smoke-") as tmp:
        tmp_dir = Path(tmp)
        in_path = tmp_dir / "smoke_in.wav"
        out_path = Path(args.output).expanduser().resolve()
        signal = _build_smoke_signal(int(args.sample_rate), float(args.duration))
        sf.write(str(in_path), signal, int(args.sample_rate))
        code = dispatch_tool(
            "voc",
            [
                str(in_path),
                "--stretch",
                f"{float(args.stretch):.8g}",
                "--pitch",
                f"{float(args.pitch):.8g}",
                "--preset",
                "stereo_coherent",
                "--phase-locking",
                "identity",
                "--transient-mode",
                "hybrid",
                "--output",
                str(out_path),
                "--overwrite",
                "--silent",
            ],
        )
        if int(code) != 0:
            print(f"[smoke] failed: voc exited with code {code}", file=sys.stderr)
            return int(code)
        if not out_path.exists():
            print("[smoke] failed: output file was not created", file=sys.stderr)
            return 1
        info = sf.info(str(out_path))
        print(
            f"[smoke] ok -> {out_path} | frames={int(info.frames)} sr={int(info.samplerate)} channels={int(info.channels)}"
        )
    return 0


def _extract_follow_example_request(args: list[str]) -> str | None:
    tokens = [str(token).strip() for token in list(args or [])]
    for idx, token in enumerate(tokens):
        if token == "--example":
            if idx + 1 < len(tokens):
                candidate = tokens[idx + 1]
                if candidate and not candidate.startswith("-"):
                    return candidate
            return "basic"
        if token.startswith("--example="):
            candidate = token.split("=", 1)[1].strip()
            return candidate or "basic"
    return None


def run_guided_mode() -> int:
    if not sys.stdin.isatty():
        raise ValueError("`pvx guided` requires an interactive terminal (TTY stdin)")

    print("pvx guided mode")
    print("Press Enter to accept defaults.\n")

    mode = _prompt_choice(
        "Workflow (voc/freeze/harmonize/retune/morph)",
        ("voc", "freeze", "harmonize", "retune", "morph"),
        "voc",
    )

    if mode == "voc":
        input_path = _prompt_text("Input path", "input.wav")
        output_path = _prompt_text("Output path", "output.wav")
        stretch = _prompt_text("Stretch factor", "1.20")
        semitones = _prompt_text("Pitch shift semitones", "0")
        preset = _prompt_text("Preset", "default")
        forwarded = [
            input_path,
            "--stretch",
            stretch,
            "--pitch",
            semitones,
            "--preset",
            preset,
            "--output",
            output_path,
        ]
    elif mode == "freeze":
        input_path = _prompt_text("Input path", "input.wav")
        output_path = _prompt_text("Output path", "output_freeze.wav")
        freeze_time = _prompt_text("Freeze time (seconds)", "0.25")
        duration = _prompt_text("Output duration (seconds)", "8.0")
        forwarded = [
            input_path,
            "--freeze-time",
            freeze_time,
            "--duration",
            duration,
            "--output",
            output_path,
        ]
    elif mode == "harmonize":
        input_path = _prompt_text("Input path", "input.wav")
        output_path = _prompt_text("Output path", "output_harm.wav")
        intervals = _prompt_text("Intervals (semitones CSV)", "0,4,7")
        forwarded = [
            input_path,
            "--intervals",
            intervals,
            "--output",
            output_path,
        ]
    elif mode == "retune":
        input_path = _prompt_text("Input path", "input.wav")
        output_path = _prompt_text("Output path", "output_retune.wav")
        root = _prompt_text("Root note", "C")
        scale = _prompt_text("Scale", "major")
        strength = _prompt_text("Correction strength", "0.85")
        forwarded = [
            input_path,
            "--root",
            root,
            "--scale",
            scale,
            "--strength",
            strength,
            "--output",
            output_path,
        ]
    else:
        input_a = _prompt_text("Input A path", "a.wav")
        input_b = _prompt_text("Input B path", "b.wav")
        output_path = _prompt_text("Output path", "morph.wav")
        alpha = _prompt_text("Morph alpha (0..1 or CSV/JSON control file)", "0.50")
        forwarded = [
            input_a,
            input_b,
            "--alpha",
            alpha,
            "--output",
            output_path,
        ]

    _print_command_preview(mode, forwarded)
    run_now = _prompt_choice("Run now? (yes/no)", ("yes", "no"), "yes")
    if run_now == "no":
        print("Command preview only; no processing executed.")
        return 0
    return dispatch_tool(mode, forwarded)


def _split_pipeline_stages(pipeline: str) -> list[str]:
    return [stage.strip() for stage in str(pipeline).split("|") if stage.strip()]


def _token_flag(token: str) -> str:
    return token.split("=", 1)[0]


def _extract_flag_value(args: list[str], flags: tuple[str, ...]) -> str | None:
    value: str | None = None
    i = 0
    while i < len(args):
        token = str(args[i])
        flag = _token_flag(token)
        if flag in flags:
            if "=" in token:
                value = token.split("=", 1)[1]
                i += 1
                continue
            if i + 1 < len(args):
                value = str(args[i + 1])
                i += 2
                continue
            value = ""
            i += 1
            continue
        i += 1
    return value


def _strip_flags(args: list[str], flag_has_value: dict[str, bool]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        token = str(args[i])
        flag = _token_flag(token)
        has_value = flag_has_value.get(flag)
        if has_value is None:
            out.append(token)
            i += 1
            continue
        if "=" in token:
            i += 1
            continue
        if has_value:
            i += 2
            continue
        i += 1
    return out


def _replace_flag_value(args: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    replaced = False
    i = 0
    while i < len(args):
        token = str(args[i])
        tok_flag = _token_flag(token)
        if tok_flag == flag:
            if not replaced:
                out.extend([flag, value])
                replaced = True
            if "=" in token:
                i += 1
            else:
                i += 2
            continue
        out.append(token)
        i += 1
    if not replaced:
        out.extend([flag, value])
    return out


def _consume_lucky_options(args: list[str]) -> tuple[list[str], int | None, int | None]:
    clean: list[str] = []
    lucky_count: int | None = None
    lucky_seed: int | None = None
    i = 0
    while i < len(args):
        token = str(args[i])
        if token == "--lucky":
            if i + 1 >= len(args):
                raise ValueError("--lucky requires an integer value")
            lucky_count = int(str(args[i + 1]))
            i += 2
            continue
        if token.startswith("--lucky="):
            lucky_count = int(token.split("=", 1)[1])
            i += 1
            continue
        if token == "--lucky-seed":
            if i + 1 >= len(args):
                raise ValueError("--lucky-seed requires an integer value")
            lucky_seed = int(str(args[i + 1]))
            i += 2
            continue
        if token.startswith("--lucky-seed="):
            lucky_seed = int(token.split("=", 1)[1])
            i += 1
            continue
        clean.append(token)
        i += 1

    if lucky_count is not None and lucky_count <= 0:
        raise ValueError("--lucky must be a positive integer")
    return clean, lucky_count, lucky_seed


def _lucky_output_variant(base: Path, idx: int) -> Path:
    stem = base.stem if base.stem else "output"
    suffix = base.suffix if base.suffix else ".wav"
    return base.with_name(f"{stem}_lucky_{idx:03d}{suffix}")


def _lucky_mastering_overrides(rng: random.Random) -> list[str]:
    out: list[str] = []
    if rng.random() < 0.85:
        out.extend(["--target-lufs", f"{rng.uniform(-18.0, -11.0):.3f}"])
    if rng.random() < 0.70:
        out.extend(
            [
                "--compressor-threshold-db",
                f"{rng.uniform(-30.0, -12.0):.3f}",
                "--compressor-ratio",
                f"{rng.uniform(1.5, 4.8):.3f}",
                "--compressor-attack-ms",
                f"{rng.uniform(2.0, 40.0):.3f}",
                "--compressor-release-ms",
                f"{rng.uniform(60.0, 280.0):.3f}",
                "--compressor-makeup-db",
                f"{rng.uniform(0.0, 6.0):.3f}",
            ]
        )
    if rng.random() < 0.90:
        out.extend(["--limiter-threshold", f"{rng.uniform(0.88, 0.995):.4f}"])
    if rng.random() < 0.65:
        out.extend(
            [
                "--soft-clip-level",
                f"{rng.uniform(0.90, 0.995):.4f}",
                "--soft-clip-type",
                rng.choice(["tanh", "arctan", "cubic"]),
                "--soft-clip-drive",
                f"{rng.uniform(0.8, 2.4):.4f}",
            ]
        )
    if rng.random() < 0.30:
        out.extend(["--hard-clip-level", f"{rng.uniform(0.95, 0.999):.4f}"])
    return out


def _lucky_tool_overrides(tool: str, rng: random.Random) -> list[str]:
    if tool == "voc":
        window = rng.choice(_LUCKY_WINDOWS)
        out = [
            "--preset",
            rng.choice(_LUCKY_PRESETS),
            "--stretch",
            f"{rng.uniform(0.4, 3.4):.4f}",
            "--pitch",
            f"{rng.uniform(-12.0, 12.0):.4f}",
            "--window",
            window,
        ]
        if window == "kaiser":
            out.extend(["--kaiser-beta", f"{rng.uniform(7.0, 22.0):.4f}"])
        return out
    if tool == "freeze":
        out = [
            "--freeze-time",
            f"{rng.uniform(0.02, 0.92):.4f}",
            "--duration",
            f"{rng.uniform(5.0, 90.0):.4f}",
            "--phase-mode",
            rng.choice(["instantaneous", "bin", "hold"]),
        ]
        if rng.random() < 0.55:
            out.append("--random-phase")
        return out
    if tool == "harmonize":
        return [
            "--intervals",
            rng.choice(["0,7,12", "0,4,7,11", "0,3,7,10", "0,7,14,19"]),
            "--force-stereo",
        ]
    if tool in {"conform", "warp"}:
        return ["--crossfade-ms", f"{rng.uniform(2.0, 35.0):.4f}"]
    if tool == "formant":
        return [
            "--mode",
            rng.choice(["shift", "preserve"]),
            "--formant-shift-ratio",
            f"{rng.uniform(0.72, 1.42):.4f}",
            "--pitch-shift-semitones",
            f"{rng.uniform(-5.0, 5.0):.4f}",
        ]
    if tool == "transient":
        return [
            "--time-stretch",
            f"{rng.uniform(0.55, 2.4):.4f}",
            "--pitch-shift-semitones",
            f"{rng.uniform(-8.0, 8.0):.4f}",
            "--transient-threshold",
            f"{rng.uniform(1.1, 2.4):.4f}",
        ]
    if tool == "unison":
        return [
            "--voices",
            str(rng.randint(3, 9)),
            "--detune-cents",
            f"{rng.uniform(4.0, 32.0):.4f}",
            "--width",
            f"{rng.uniform(0.2, 1.0):.4f}",
            "--dry-mix",
            f"{rng.uniform(0.05, 0.45):.4f}",
        ]
    if tool == "denoise":
        return [
            "--reduction-db",
            f"{rng.uniform(4.0, 18.0):.4f}",
            "--floor",
            f"{rng.uniform(0.05, 0.25):.4f}",
            "--smooth",
            str(rng.randint(3, 12)),
        ]
    if tool == "deverb":
        return [
            "--strength",
            f"{rng.uniform(0.15, 0.75):.4f}",
            "--decay",
            f"{rng.uniform(0.75, 0.97):.4f}",
            "--floor",
            f"{rng.uniform(0.05, 0.30):.4f}",
        ]
    if tool == "retune":
        return [
            "--root",
            rng.choice(["C", "D", "E", "F", "G", "A", "B"]),
            "--scale",
            rng.choice(["major", "minor", "pentatonic", "chromatic"]),
            "--strength",
            f"{rng.uniform(0.45, 1.0):.4f}",
        ]
    if tool == "layer":
        return [
            "--harmonic-stretch",
            f"{rng.uniform(0.6, 2.2):.4f}",
            "--percussive-stretch",
            f"{rng.uniform(0.7, 1.8):.4f}",
            "--harmonic-pitch-semitones",
            f"{rng.uniform(-6.0, 6.0):.4f}",
            "--percussive-pitch-semitones",
            f"{rng.uniform(-2.0, 2.0):.4f}",
            "--harmonic-gain",
            f"{rng.uniform(0.6, 1.4):.4f}",
            "--percussive-gain",
            f"{rng.uniform(0.6, 1.4):.4f}",
        ]
    if tool in {"filter", "tvfilter", "noisefilter", "bandamp", "spec-compander"}:
        return [
            "--response-mix",
            f"{rng.uniform(0.3, 1.0):.4f}",
            "--dry-mix",
            f"{rng.uniform(0.0, 0.35):.4f}",
            "--response-gain-db",
            f"{rng.uniform(-6.0, 8.0):.4f}",
            "--noise-floor",
            f"{rng.uniform(0.6, 2.0):.4f}",
            "--band-gain-db",
            f"{rng.uniform(2.0, 14.0):.4f}",
            "--peak-count",
            str(rng.randint(4, 16)),
            "--comp-ratio",
            f"{rng.uniform(1.1, 3.6):.4f}",
            "--expand-ratio",
            f"{rng.uniform(1.0, 2.5):.4f}",
        ]
    if tool in {"ring", "ringfilter", "ringtvfilter"}:
        return [
            "--frequency-hz",
            f"{rng.uniform(12.0, 1800.0):.4f}",
            "--depth",
            f"{rng.uniform(0.2, 1.0):.4f}",
            "--mix",
            f"{rng.uniform(0.25, 1.0):.4f}",
            "--feedback",
            f"{rng.uniform(0.0, 0.45):.4f}",
            "--resonance-hz",
            f"{rng.uniform(120.0, 4800.0):.4f}",
            "--resonance-q",
            f"{rng.uniform(1.0, 18.0):.4f}",
            "--resonance-mix",
            f"{rng.uniform(0.15, 0.95):.4f}",
        ]
    if tool == "chordmapper":
        return [
            "--root-hz",
            f"{rng.uniform(80.0, 440.0):.4f}",
            "--chord",
            rng.choice(["major", "minor", "sus4"]),
            "--strength",
            f"{rng.uniform(0.3, 1.0):.4f}",
            "--boost-db",
            f"{rng.uniform(2.0, 12.0):.4f}",
            "--attenuation",
            f"{rng.uniform(0.15, 0.85):.4f}",
        ]
    if tool == "inharmonator":
        return [
            "--inharmonic-f0-hz",
            f"{rng.uniform(60.0, 440.0):.4f}",
            "--inharmonicity",
            f"{rng.uniform(1e-6, 6e-4):.8f}",
            "--inharmonic-mix",
            f"{rng.uniform(0.25, 1.0):.4f}",
            "--dry-mix",
            f"{rng.uniform(0.0, 0.35):.4f}",
        ]
    if tool == "morph":
        return [
            "--alpha",
            f"{rng.uniform(0.15, 0.92):.4f}",
            "--blend-mode",
            rng.choice(["linear", "geometric", "carrier_a_envelope_b", "carrier_a_mask_b"]),
            "--phase-mix",
            f"{rng.uniform(0.0, 1.0):.4f}",
            "--mask-exponent",
            f"{rng.uniform(0.7, 2.2):.4f}",
            "--envelope-lifter",
            str(rng.randint(16, 72)),
        ]
    return []


def _run_lucky_tool_mode(tool: str, forwarded_args: list[str], lucky_count: int, lucky_seed: int | None) -> int:
    if tool not in _LUCKY_SUPPORTED_TOOLS:
        raise ValueError(
            f"`--lucky` is not supported for `{tool}`. "
            f"Supported tools: {', '.join(sorted(_LUCKY_SUPPORTED_TOOLS))}"
        )
    if lucky_count <= 0:
        raise ValueError("--lucky must be > 0")
    seed = int(lucky_seed) if lucky_seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(seed)
    print(f"[lucky] seed={seed} tool={tool} runs={lucky_count}")

    if tool == "morph":
        output_value = _extract_flag_value(forwarded_args, ("--output", "--out", "-o"))
        output_base = Path(output_value if output_value not in {None, "", "-"} else "morph_lucky.wav")
        if output_base.suffix == "":
            output_base = output_base.with_suffix(".wav")
        output_dir = output_base.parent if str(output_base.parent) else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        base = _strip_flags(
            forwarded_args,
            {
                "--output": True,
                "--out": True,
                "-o": True,
                "--stdout": False,
            },
        )
        for run_idx in range(1, lucky_count + 1):
            out_path = _lucky_output_variant(output_dir / output_base.name, run_idx)
            run_args = list(base)
            run_args.extend(_lucky_tool_overrides(tool, rng))
            run_args.extend(_lucky_mastering_overrides(rng))
            run_args.extend(["--output", str(out_path), "--overwrite"])
            print(f"[lucky] {tool} run {run_idx}/{lucky_count} -> {out_path}")
            code = dispatch_tool(tool, run_args)
            if code != 0:
                return int(code)
        return 0

    output_dir_value = _extract_flag_value(forwarded_args, ("--output-dir", "-o"))
    if output_dir_value in {None, ""}:
        explicit_out = _extract_flag_value(forwarded_args, ("--output", "--out"))
        if explicit_out not in {None, "", "-"}:
            output_dir = Path(str(explicit_out)).parent
        else:
            output_dir = Path("lucky_out")
    else:
        output_dir = Path(str(output_dir_value))
    output_dir.mkdir(parents=True, exist_ok=True)

    base = _strip_flags(
        forwarded_args,
        {
            "--output": True,
            "--out": True,
            "--output-dir": True,
            "-o": True,
            "--stdout": False,
            "--suffix": True,
        },
    )
    for run_idx in range(1, lucky_count + 1):
        run_args = list(base)
        run_args.extend(_lucky_tool_overrides(tool, rng))
        run_args.extend(_lucky_mastering_overrides(rng))
        run_args.extend(
            [
                "--output-dir",
                str(output_dir),
                "--suffix",
                f"_lucky_{run_idx:03d}",
                "--overwrite",
            ]
        )
        print(f"[lucky] {tool} run {run_idx}/{lucky_count} -> {output_dir}/*_lucky_{run_idx:03d}.*")
        code = dispatch_tool(tool, run_args)
        if code != 0:
            return int(code)
    return 0


def _run_lucky_helper_mode(
    helper: str,
    forwarded_args: list[str],
    lucky_count: int,
    lucky_seed: int | None,
) -> int:
    seed = int(lucky_seed) if lucky_seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(seed)
    print(f"[lucky] seed={seed} helper={helper} runs={lucky_count}")

    output_value = _extract_flag_value(forwarded_args, ("--output", "--out"))
    output_base = Path(output_value if output_value not in {None, "", "-"} else f"{helper}_lucky.wav")
    if output_base.suffix == "":
        output_base = output_base.with_suffix(".wav")
    output_dir = output_base.parent if str(output_base.parent) else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _strip_flags(
        forwarded_args,
        {
            "--output": True,
            "--out": True,
        },
    )

    for run_idx in range(1, lucky_count + 1):
        out_path = _lucky_output_variant(output_dir / output_base.name, run_idx)
        run_args = list(base)
        if helper == "chain":
            current_pipeline = _extract_flag_value(run_args, ("--pipeline",))
            if current_pipeline not in {None, ""}:
                random_stage = (
                    "voc "
                    f"--stretch {rng.uniform(0.55, 2.4):.4f} "
                    f"--pitch {rng.uniform(-7.0, 7.0):.4f} "
                    f"--preset {rng.choice(_LUCKY_PRESETS)}"
                )
                run_args = _replace_flag_value(run_args, "--pipeline", f"{current_pipeline} | {random_stage}")
            run_args.extend(["--output", str(out_path)])
            print(f"[lucky] chain run {run_idx}/{lucky_count} -> {out_path}")
            code = run_chain_mode(run_args)
        elif helper == "follow":
            run_args.extend(
                [
                    "--stretch",
                    f"{rng.uniform(0.7, 1.5):.4f}",
                    "--pitch-conf-min",
                    f"{rng.uniform(0.45, 0.9):.4f}",
                    "--pitch-map-smooth-ms",
                    f"{rng.uniform(0.0, 40.0):.4f}",
                    "--pitch-map-crossfade-ms",
                    f"{rng.uniform(8.0, 40.0):.4f}",
                    "--output",
                    str(out_path),
                    "--overwrite",
                ]
            )
            print(f"[lucky] follow run {run_idx}/{lucky_count} -> {out_path}")
            code = run_follow_mode(run_args)
        elif helper == "stream":
            run_args.extend(
                [
                    "--output",
                    str(out_path),
                    "--chunk-seconds",
                    f"{rng.uniform(0.06, 0.35):.4f}",
                    "--time-stretch",
                    f"{rng.uniform(0.6, 3.0):.4f}",
                    "--pitch",
                    f"{rng.uniform(-9.0, 9.0):.4f}",
                    "--preset",
                    rng.choice(_LUCKY_PRESETS),
                ]
            )
            print(f"[lucky] stream run {run_idx}/{lucky_count} -> {out_path}")
            code = run_stream_mode(run_args)
        else:
            raise ValueError(f"Unsupported helper for --lucky: {helper}")
        if code != 0:
            return int(code)
    return 0


def _run_stage_command(stage_name: str, stage_args: list[str]) -> int:
    try:
        return int(dispatch_tool(stage_name, stage_args))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return int(code)


def _run_stage_capture_stdout(stage_name: str, stage_args: list[str]) -> tuple[int, str]:
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            code = int(dispatch_tool(stage_name, stage_args))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"[error] {stage_name}: {exc}", file=sys.stderr)
        code = 1
    return int(code), capture.getvalue()


class _BytesStdin:
    def __init__(self, payload: bytes) -> None:
        self.buffer = io.BytesIO(payload)

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def _patched_stdin_bytes(payload: bytes):
    original_stdin = sys.stdin
    sys.stdin = _BytesStdin(payload)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdin = original_stdin


def run_follow_mode(forwarded_args: list[str]) -> int:
    example_request = _extract_follow_example_request(forwarded_args)
    if example_request is not None:
        try:
            print_follow_examples(example_request)
            return 0
        except ValueError as exc:
            print(f"pvx follow: error: {exc}", file=sys.stderr)
            return 2

    parser = argparse.ArgumentParser(
        prog="pvx follow",
        description=(
            "Single-command sidechain helper: track guide pitch/f0 and apply the resulting control map "
            "to a target via `pvx voc --control-stdin`."
        ),
    )
    parser.add_argument("guide", help="Guide/input A used for pitch tracking")
    parser.add_argument("target", help="Target/input B to be processed by pvx voc")
    parser.add_argument("--output", "--out", dest="output", required=True, help="Output audio path")
    parser.add_argument(
        "--emit",
        choices=["pitch_map", "stretch_map", "pitch_to_stretch"],
        default="pitch_to_stretch",
        help="Control map emit mode for the guide track (default: pitch_to_stretch)",
    )
    parser.add_argument("--backend", choices=["auto", "pyin", "acf"], default="auto", help="Pitch tracker backend")
    parser.add_argument("--fmin", type=float, default=50.0, help="Minimum tracked f0 in Hz")
    parser.add_argument("--fmax", type=float, default=1200.0, help="Maximum tracked f0 in Hz")
    parser.add_argument("--frame-length", type=int, default=2048, help="Tracker frame length in samples")
    parser.add_argument("--hop-size", type=int, default=256, help="Tracker hop size in samples")
    parser.add_argument(
        "--ratio-reference",
        choices=["median", "mean", "first", "hz"],
        default="median",
        help="Reference mode for pitch_ratio derivation in tracking",
    )
    parser.add_argument("--reference-hz", type=float, default=None, help="Reference Hz when --ratio-reference hz")
    parser.add_argument("--ratio-min", type=float, default=0.25, help="Minimum pitch_ratio clamp")
    parser.add_argument("--ratio-max", type=float, default=4.0, help="Maximum pitch_ratio clamp")
    parser.add_argument("--smooth-frames", type=int, default=5, help="Smoothing window in frames")
    parser.add_argument("--confidence-floor", type=float, default=0.0, help="Minimum tracker confidence")
    parser.add_argument(
        "--feature-set",
        choices=["none", "basic", "advanced", "all"],
        default="all",
        help="Feature columns emitted by pitch tracker (default: all)",
    )
    parser.add_argument(
        "--mfcc-count",
        type=int,
        default=13,
        help="MFCC column count emitted by pitch tracker (default: 13)",
    )
    parser.add_argument(
        "--stretch-from",
        choices=["pitch_ratio", "inv_pitch_ratio", "f0_hz"],
        default="pitch_ratio",
        help="Source for deriving stretch in stretch-oriented emit modes",
    )
    parser.add_argument("--stretch-scale", type=float, default=1.0, help="Scale factor for derived stretch track")
    parser.add_argument("--stretch-min", type=float, default=0.25, help="Lower clamp for derived stretch")
    parser.add_argument("--stretch-max", type=float, default=4.0, help="Upper clamp for derived stretch")
    parser.add_argument("--stretch", type=float, default=1.0, help="Constant stretch value when --emit pitch_map")
    parser.add_argument(
        "--pitch-conf-min",
        type=float,
        default=0.75,
        help="Minimum accepted map confidence for pvx voc (default: 0.75)",
    )
    parser.add_argument(
        "--pitch-lowconf-mode",
        choices=["hold", "unity", "interp"],
        default="hold",
        help="Low-confidence handling mode in pvx voc (default: hold)",
    )
    parser.add_argument(
        "--pitch-map-smooth-ms",
        type=float,
        default=0.0,
        help="Additional map smoothing in pvx voc (milliseconds)",
    )
    parser.add_argument(
        "--pitch-map-crossfade-ms",
        type=float,
        default=20.0,
        help="Map segment crossfade in pvx voc (milliseconds, default: 20)",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="EXPR",
        help=(
            "Optional pvx voc control route expression. Repeat to chain. "
            "Example: --route stretch=pitch_ratio --route pitch_ratio=const(1.0)"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--quiet", action="store_true", help="Reduce helper logs and hide progress bars")
    parser.add_argument("--silent", action="store_true", help="Suppress helper logs")
    parser.add_argument(
        "--example",
        nargs="?",
        const="basic",
        default=None,
        choices=list(FOLLOW_EXAMPLE_CHOICES),
        metavar="NAME",
        help=(
            "Print follow example command(s) and exit. "
            "Use `--example` for basic or `--example all` for the full set."
        ),
    )
    args, passthrough = parser.parse_known_args(forwarded_args)
    if args.example is not None:
        print_follow_examples(str(args.example))
        return 0

    passthrough_flags = {_token_flag(token) for token in passthrough if token.startswith("-")}
    forbidden_passthrough = {"--output", "--out", "--stdout", "--pitch-map", "--pitch-map-stdin", "--control-stdin"}
    bad_flags = sorted(passthrough_flags & forbidden_passthrough)
    if bad_flags:
        parser.error(
            f"Do not pass {bad_flags} via passthrough in `pvx follow`; "
            "follow mode manages control-map and output routing."
        )
    if int(args.mfcc_count) < 0 or int(args.mfcc_count) > 40:
        parser.error("--mfcc-count must be in [0, 40]")

    track_args: list[str] = [
        str(args.guide),
        "--output",
        "-",
        "--emit",
        str(args.emit),
        "--backend",
        str(args.backend),
        "--fmin",
        f"{float(args.fmin):.12g}",
        "--fmax",
        f"{float(args.fmax):.12g}",
        "--frame-length",
        str(int(args.frame_length)),
        "--hop-size",
        str(int(args.hop_size)),
        "--ratio-reference",
        str(args.ratio_reference),
        "--ratio-min",
        f"{float(args.ratio_min):.12g}",
        "--ratio-max",
        f"{float(args.ratio_max):.12g}",
        "--smooth-frames",
        str(int(args.smooth_frames)),
        "--confidence-floor",
        f"{float(args.confidence_floor):.12g}",
        "--feature-set",
        str(args.feature_set),
        "--mfcc-count",
        str(int(args.mfcc_count)),
        "--stretch-from",
        str(args.stretch_from),
        "--stretch-scale",
        f"{float(args.stretch_scale):.12g}",
        "--stretch-min",
        f"{float(args.stretch_min):.12g}",
        "--stretch-max",
        f"{float(args.stretch_max):.12g}",
        "--stretch",
        f"{float(args.stretch):.12g}",
    ]
    if args.reference_hz is not None:
        track_args.extend(["--reference-hz", f"{float(args.reference_hz):.12g}"])
    if args.quiet:
        track_args.append("--quiet")
    if args.silent:
        track_args.append("--silent")

    code, control_csv = _run_stage_capture_stdout("pitch-track", track_args)
    if code != 0:
        return int(code)
    if not control_csv.strip():
        print("[error] follow: pitch tracker emitted an empty control map", file=sys.stderr)
        return 1

    voc_args: list[str] = [
        str(args.target),
        "--control-stdin",
        "--pitch-conf-min",
        f"{float(args.pitch_conf_min):.12g}",
        "--pitch-lowconf-mode",
        str(args.pitch_lowconf_mode),
        "--pitch-map-smooth-ms",
        f"{float(args.pitch_map_smooth_ms):.12g}",
        "--pitch-map-crossfade-ms",
        f"{float(args.pitch_map_crossfade_ms):.12g}",
        "--output",
        str(args.output),
    ]
    for route in list(args.route or []):
        voc_args.extend(["--route", str(route)])
    if args.overwrite:
        voc_args.append("--overwrite")
    if args.quiet:
        voc_args.append("--quiet")
    if args.silent:
        voc_args.append("--silent")
    voc_args.extend(passthrough)

    payload = control_csv.encode("utf-8")
    with _patched_stdin_bytes(payload):
        return _run_stage_command("voc", voc_args)


def run_chain_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx chain",
        description=(
            "Managed one-line chain runner for serial pvx audio tools. "
            "Each stage receives the previous stage output as input."
        ),
    )
    parser.add_argument("input", help="Initial input audio path or '-' for stdin")
    parser.add_argument(
        "--pipeline",
        required=True,
        help=(
            "Pipeline string with stages separated by '|'. "
            "Example: \"voc --stretch 1.2 | formant --mode preserve\""
        ),
    )
    parser.add_argument("--output", "--out", dest="output", required=True, help="Final output path (or '-')")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional directory for intermediate stage files",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate stage files after successful completion",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Print a copy-paste chain example and exit",
    )
    args = parser.parse_args(forwarded_args)

    if args.example:
        print(EXAMPLE_COMMANDS["chain"][1])
        return 0

    raw_stages = _split_pipeline_stages(args.pipeline)
    if not raw_stages:
        parser.error("--pipeline produced no stages")

    stages: list[tuple[str, list[str]]] = []
    for stage_idx, stage_text in enumerate(raw_stages, start=1):
        try:
            tokens = shlex.split(stage_text)
        except ValueError as exc:
            parser.error(f"Invalid stage {stage_idx} syntax: {exc}")
        if not tokens:
            parser.error(f"Stage {stage_idx} is empty")

        stage_cmd = tokens[0].strip().lower()
        if stage_cmd not in TOOL_INDEX:
            parser.error(f"Unknown chain stage command '{tokens[0]}' in stage {stage_idx}")
        stage_tool = TOOL_INDEX[stage_cmd].name
        if stage_tool not in _CHAIN_TOOL_ALLOWLIST:
            parser.error(
                f"Chain stage '{stage_tool}' is not supported in managed chain mode. "
                f"Supported: {', '.join(sorted(_CHAIN_TOOL_ALLOWLIST))}"
            )

        stage_flags = {_token_flag(token) for token in tokens[1:] if token.startswith("-")}
        bad_flags = sorted(stage_flags & _CHAIN_STAGE_FORBIDDEN_FLAGS)
        if bad_flags:
            parser.error(
                f"Stage {stage_idx} ({stage_tool}) contains output-routing flags {bad_flags}. "
                "Managed chain mode controls stage outputs automatically."
            )
        stages.append((stage_tool, tokens[1:]))

    stretch_product = 1.0
    stretch_terms = 0
    for stage_tool, stage_args in stages:
        if stage_tool not in {"voc", "transient"}:
            continue
        stretch_text = _extract_flag_value(stage_args, ("--stretch", "--time-stretch", "--time-stretch-factor"))
        if stretch_text in {None, ""}:
            continue
        try:
            stretch_val = float(str(stretch_text))
        except ValueError:
            continue
        if stretch_val <= 0.0:
            continue
        stretch_product *= stretch_val
        stretch_terms += 1
    if stretch_terms >= 2 and abs(stretch_product - 1.0) <= 0.06:
        print(
            (
                "[chain] note: cumulative stretch across pipeline is near unity "
                f"({stretch_product:.6f}x). Perceptual change may be subtle."
            ),
            file=sys.stderr,
        )

    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        if args.keep_intermediate:
            work_dir = Path(tempfile.mkdtemp(prefix="pvx-chain-"))
        else:
            temp_ctx = tempfile.TemporaryDirectory(prefix="pvx-chain-")
            work_dir = Path(temp_ctx.name)
    else:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

    current_input = str(args.input)
    for stage_idx, (stage_tool, stage_args) in enumerate(stages, start=1):
        is_last = stage_idx == len(stages)
        if is_last:
            stage_out = Path(str(args.output))
        else:
            stage_out = work_dir / f"stage_{stage_idx:02d}_{stage_tool}.wav"

        command_args = [
            current_input,
            *stage_args,
            "--output",
            str(stage_out),
            "--overwrite",
            "--quiet",
        ]
        print(f"[chain] stage {stage_idx}/{len(stages)}: {stage_tool}")
        code = _run_stage_command(stage_tool, command_args)
        if code != 0:
            print(f"[chain] stage {stage_idx} failed with exit code {code}", file=sys.stderr)
            return int(code)

        current_input = str(stage_out)

    if temp_ctx is not None:
        temp_ctx.cleanup()
    elif args.keep_intermediate:
        print(f"[chain] intermediates kept in {work_dir}")

    print(f"[chain] done -> {args.output}")
    return 0


def run_stream_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx stream",
        description=(
            "Chunked streaming wrapper over `pvx voc` for long renders and pipe-friendly one-liners."
        ),
    )
    parser.add_argument("input", help="Input audio path or '-' for stdin")
    parser.add_argument("--output", "--out", dest="output", required=True, help="Output path (or '-')")
    parser.add_argument(
        "--mode",
        choices=["stateful", "wrapper"],
        default="stateful",
        help="Stream engine: stateful chunk processor (default) or wrapper compatibility mode",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.25,
        help="Chunk/segment duration for `--auto-segment-seconds` (default: 0.25)",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=float,
        default=0.0,
        help="Crossfade used for segment assembly in milliseconds (default: 0.0)",
    )
    parser.add_argument(
        "--context-ms",
        type=float,
        default=None,
        help="Optional stateful context window in milliseconds (default: auto from window/hop)",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Print a copy-paste stream example and exit",
    )
    args, passthrough = parser.parse_known_args(forwarded_args)

    if args.example:
        print(EXAMPLE_COMMANDS["stream"][1])
        return 0

    if args.chunk_seconds <= 0.0:
        parser.error("--chunk-seconds must be > 0")
    if args.crossfade_ms < 0.0:
        parser.error("--crossfade-ms must be >= 0")
    if args.context_ms is not None and float(args.context_ms) < 0.0:
        parser.error("--context-ms must be >= 0")

    passthrough_flags = {_token_flag(token) for token in passthrough if token.startswith("-")}
    if passthrough_flags & {"--output", "--out", "--stdout"}:
        parser.error("Do not pass --output/--stdout in passthrough args; use `pvx stream --output ...`")

    if args.mode == "stateful":
        return run_stateful_stream(
            input_token=str(args.input),
            output_token=str(args.output),
            passthrough=list(passthrough),
            chunk_seconds=float(args.chunk_seconds),
            context_ms=None if args.context_ms is None else float(args.context_ms),
            crossfade_ms=float(args.crossfade_ms),
        )

    voc_args: list[str] = [str(args.input)]
    if "--auto-segment-seconds" not in passthrough_flags:
        voc_args.extend(["--auto-segment-seconds", f"{float(args.chunk_seconds):.6g}"])
    if "--pitch-map-crossfade-ms" not in passthrough_flags:
        voc_args.extend(["--pitch-map-crossfade-ms", f"{float(args.crossfade_ms):.6g}"])

    if str(args.output) == "-":
        voc_args.append("--stdout")
    else:
        voc_args.extend(["--output", str(args.output)])

    voc_args.extend(passthrough)
    return _run_stage_command("voc", voc_args)


_SIZE_UNITS_BYTES: dict[str, float] = {
    "b": 1.0,
    "kb": 1_000.0,
    "mb": 1_000_000.0,
    "gb": 1_000_000_000.0,
    "tb": 1_000_000_000_000.0,
    "kib": 1024.0,
    "mib": 1024.0 * 1024.0,
    "gib": 1024.0 * 1024.0 * 1024.0,
    "tib": 1024.0 * 1024.0 * 1024.0 * 1024.0,
}

_SUBTYPE_BYTES_PER_SAMPLE: dict[str, int] = {
    "PCM_S8": 1,
    "PCM_U8": 1,
    "PCM_16": 2,
    "PCM_24": 3,
    "PCM_32": 4,
    "FLOAT": 4,
    "DOUBLE": 8,
}


def _parse_size_bytes(text: str) -> float:
    raw = str(text).strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-z]*)", raw)
    if match is None:
        raise ValueError(
            f"Invalid size '{text}'. Use forms like 500MB, 20GB, 2.5TB, or 10GiB."
        )
    value = float(match.group(1))
    unit = str(match.group(2) or "b")
    if unit not in _SIZE_UNITS_BYTES:
        choices = ", ".join(sorted(_SIZE_UNITS_BYTES))
        raise ValueError(f"Unsupported size unit '{unit}'. Supported units: {choices}")
    out = value * _SIZE_UNITS_BYTES[unit]
    if out <= 0.0:
        raise ValueError("Size must be > 0")
    return out


def _format_bytes_human(value: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    x = float(value)
    idx = 0
    while idx < len(units) - 1 and abs(x) >= 1000.0:
        x /= 1000.0
        idx += 1
    return f"{x:.3f} {units[idx]}"


def _infer_output_format(input_path: Path, requested: str) -> str:
    token = str(requested).strip().lower()
    if token == "auto":
        token = input_path.suffix.lower().lstrip(".")
    token = token.lstrip(".")
    if token == "aif":
        token = "aiff"
    if token == "oga":
        token = "ogg"
    allowed = {"wav", "flac", "aiff", "ogg", "caf"}
    if token not in allowed:
        raise ValueError(
            f"Unsupported output format '{requested}'. Choose one of: {', '.join(sorted(allowed))}, auto"
        )
    return token


def _bytes_per_sample_from_subtype(subtype: str) -> int | None:
    key = str(subtype).strip().upper()
    if not key:
        return None
    return _SUBTYPE_BYTES_PER_SAMPLE.get(key)


def run_stretch_budget_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx stretch-budget",
        description=(
            "Estimate maximum safe time-stretch for an input file under a disk budget.\n"
            "This is a storage-budget estimate, not a quality guarantee."
        ),
    )
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument(
        "--disk-budget",
        type=str,
        default=None,
        help="Total budget size (e.g., 500MB, 20GB, 2TiB). If omitted, use free space at --budget-path.",
    )
    parser.add_argument(
        "--budget-path",
        type=Path,
        default=Path("."),
        help="Path used to query free space when --disk-budget is omitted (default: current directory).",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.90,
        help="Usable fraction of budget in (0,1]; default: 0.90 (10%% headroom).",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="auto",
        help="Output format assumption: auto, wav, flac, aiff, ogg, caf (default: auto from input extension).",
    )
    parser.add_argument(
        "--bit-depth",
        choices=["inherit", "16", "24", "32f"],
        default="inherit",
        help="Bit-depth assumption when --subtype is not set (default: inherit from input subtype).",
    )
    parser.add_argument(
        "--subtype",
        type=str,
        default=None,
        help="Explicit libsndfile subtype assumption (e.g., PCM_16, PCM_24, FLOAT).",
    )
    parser.add_argument(
        "--requested-stretch",
        type=float,
        default=None,
        help="Optional stretch ratio to evaluate against the computed budget.",
    )
    parser.add_argument(
        "--fail-if-exceeds",
        action="store_true",
        help="Return non-zero when --requested-stretch does not fit the usable budget.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    args = parser.parse_args(forwarded_args)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input not found: {input_path}")
    if args.safety_margin <= 0.0 or args.safety_margin > 1.0:
        parser.error("--safety-margin must be in (0, 1]")
    if args.requested_stretch is not None and float(args.requested_stretch) <= 0.0:
        parser.error("--requested-stretch must be > 0")

    if args.disk_budget is not None:
        try:
            budget_bytes = _parse_size_bytes(str(args.disk_budget))
        except ValueError as exc:
            parser.error(str(exc))
        budget_source = f"explicit:{str(args.disk_budget).strip()}"
    else:
        budget_root = Path(args.budget_path).expanduser().resolve()
        try:
            budget_bytes = float(shutil.disk_usage(str(budget_root)).free)
        except Exception as exc:
            parser.error(f"Failed to query free disk at {budget_root}: {exc}")
        budget_source = f"free-space:{budget_root}"

    if budget_bytes <= 0.0:
        parser.error("Resolved budget must be > 0 bytes")

    try:
        info = sf.info(str(input_path))
    except Exception as exc:
        parser.error(f"Failed to read input metadata: {exc}")
    frames = int(getattr(info, "frames", 0) or 0)
    channels = int(getattr(info, "channels", 0) or 0)
    sample_rate = int(getattr(info, "samplerate", 0) or 0)
    if frames <= 0:
        parser.error("Input has no audio frames")
    if channels <= 0:
        parser.error("Input has invalid channel count")
    if sample_rate <= 0:
        parser.error("Input has invalid sample rate")

    try:
        output_format = _infer_output_format(input_path, str(args.output_format))
    except ValueError as exc:
        parser.error(str(exc))

    bytes_per_sample: int | None = None
    bytes_source = ""
    subtype_assumed = ""
    if args.subtype is not None:
        bytes_per_sample = _bytes_per_sample_from_subtype(str(args.subtype))
        if bytes_per_sample is None:
            supported = ", ".join(sorted(_SUBTYPE_BYTES_PER_SAMPLE))
            parser.error(f"Unsupported --subtype '{args.subtype}' for estimator. Supported: {supported}")
        subtype_assumed = str(args.subtype).strip().upper()
        bytes_source = "explicit --subtype"
    elif str(args.bit_depth) != "inherit":
        if str(args.bit_depth) == "16":
            bytes_per_sample = 2
            subtype_assumed = "PCM_16"
        elif str(args.bit_depth) == "24":
            bytes_per_sample = 3
            subtype_assumed = "PCM_24"
        else:
            bytes_per_sample = 4
            subtype_assumed = "FLOAT"
        bytes_source = "explicit --bit-depth"
    else:
        inferred_subtype = str(getattr(info, "subtype", "") or "")
        inferred_bps = _bytes_per_sample_from_subtype(inferred_subtype)
        if inferred_bps is None:
            bytes_per_sample = 4
            subtype_assumed = "FLOAT"
            bytes_source = f"fallback from input subtype '{inferred_subtype or 'unknown'}'"
        else:
            bytes_per_sample = inferred_bps
            subtype_assumed = inferred_subtype.strip().upper()
            bytes_source = "inherited input subtype"

    base_bytes = float(frames) * float(channels) * float(bytes_per_sample)
    usable_budget_bytes = float(budget_bytes) * float(args.safety_margin)
    max_safe_stretch = usable_budget_bytes / max(base_bytes, 1.0)
    input_duration_sec = float(frames) / float(sample_rate)
    max_duration_sec = input_duration_sec * max_safe_stretch
    max_frames = max(1, int(round(float(frames) * max_safe_stretch)))
    conservative_for_compressed = output_format in {"flac", "ogg"}

    requested_bytes: float | None = None
    requested_ok: bool | None = None
    requested_duration_sec: float | None = None
    if args.requested_stretch is not None:
        requested = float(args.requested_stretch)
        requested_bytes = base_bytes * requested
        requested_ok = bool(requested_bytes <= usable_budget_bytes)
        requested_duration_sec = input_duration_sec * requested

    payload = {
        "input_path": str(input_path),
        "input_frames": int(frames),
        "input_channels": int(channels),
        "input_sample_rate": int(sample_rate),
        "input_duration_sec": float(input_duration_sec),
        "input_subtype": str(getattr(info, "subtype", "") or ""),
        "output_format_assumed": str(output_format),
        "subtype_assumed": str(subtype_assumed),
        "bytes_per_sample_assumed": int(bytes_per_sample),
        "bytes_assumption_source": str(bytes_source),
        "estimate_mode": "conservative_pcm_equivalent" if conservative_for_compressed else "pcm_equivalent",
        "budget_source": str(budget_source),
        "budget_bytes": float(budget_bytes),
        "safety_margin": float(args.safety_margin),
        "usable_budget_bytes": float(usable_budget_bytes),
        "estimated_bytes_at_1x": float(base_bytes),
        "max_safe_stretch": float(max_safe_stretch),
        "max_safe_duration_sec": float(max_duration_sec),
        "max_safe_output_frames": int(max_frames),
        "requested_stretch": None if args.requested_stretch is None else float(args.requested_stretch),
        "requested_estimated_bytes": None if requested_bytes is None else float(requested_bytes),
        "requested_duration_sec": None if requested_duration_sec is None else float(requested_duration_sec),
        "requested_fits_budget": requested_ok,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("pvx stretch budget estimate")
        print(f"- input: {payload['input_path']}")
        print(
            f"- input shape: {payload['input_channels']} ch, {payload['input_sample_rate']} Hz, "
            f"{payload['input_frames']} frames ({payload['input_duration_sec']:.3f} s)"
        )
        print(
            f"- output assumption: format={payload['output_format_assumed']}, "
            f"subtype={payload['subtype_assumed']}, bytes/sample={payload['bytes_per_sample_assumed']} "
            f"({payload['bytes_assumption_source']})"
        )
        print(
            f"- budget: {_format_bytes_human(payload['budget_bytes'])} "
            f"(usable {_format_bytes_human(payload['usable_budget_bytes'])} at safety-margin={payload['safety_margin']:.3f})"
        )
        print(f"- estimated size at 1.0x: {_format_bytes_human(payload['estimated_bytes_at_1x'])}")
        print(f"- max safe stretch: {payload['max_safe_stretch']:.6f}x")
        print(
            f"- max safe output: {_format_bytes_human(payload['usable_budget_bytes'])}, "
            f"{payload['max_safe_output_frames']} frames, {payload['max_safe_duration_sec']:.3f} s"
        )
        if conservative_for_compressed:
            print("- note: compressed format selected; estimate is conservative PCM-equivalent.")
        if args.requested_stretch is not None:
            fits_text = "yes" if bool(payload["requested_fits_budget"]) else "no"
            print(
                f"- requested stretch: {payload['requested_stretch']:.6f}x "
                f"(est. {_format_bytes_human(float(payload['requested_estimated_bytes'] or 0.0))}, "
                f"duration {float(payload['requested_duration_sec'] or 0.0):.3f} s) -> fits budget: {fits_text}"
            )

    if bool(args.fail_if_exceeds) and args.requested_stretch is not None and not bool(requested_ok):
        print("[stretch-budget] requested stretch exceeds usable budget", file=sys.stderr)
        return 1
    return 0


def dispatch_tool(command: str, forwarded_args: list[str]) -> int:
    spec = TOOL_INDEX.get(command)
    if spec is None:
        raise ValueError(f"Unknown tool command: {command}")
    main_fn = _load_entrypoint(spec.entrypoint)
    return int(main_fn(forwarded_args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pvx",
        description=(
            "Unified CLI for pvx (audio quality first, speed second).\n"
            "Use subcommands to access all existing pvx tools from one entrypoint."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  pvx voc input.wav --stretch 1.2 --output output.wav\n"
            "  pvx input.wav --stretch 1.2 --output output.wav   # defaults to `voc`\n"
            "  pvx follow guide.wav target.wav --output followed.wav --emit pitch_to_stretch\n"
            "  pvx chain input.wav --pipeline \"voc --stretch 1.2 | formant --mode preserve\" --output out.wav\n"
            "  pvx stream input.wav --output out.wav --chunk-seconds 0.2 --time-stretch 2.0\n"
            "  pvx stretch-budget input.wav --disk-budget 20GB --bit-depth 16 --requested-stretch 1000000\n"
            "  pvx doctor\n"
            "  pvx quickstart input.wav --output output.wav\n"
            "  pvx safe input.wav --material mix --output output.wav\n"
            "  pvx transforms\n"
            "  pvx smoke --output smoke_out.wav\n"
            "  pvx voc input.wav --output-dir out --lucky 8\n"
            "  pvx list\n"
            "  pvx examples basic\n"
            "  pvx help voc\n"
            "\n"
            f"Available tool commands: {_tool_names_csv()}"
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Subcommand name, helper command, or input path (defaults to `voc` when an input path is provided)",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded directly to the selected subcommand",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    command_raw = args.command
    forwarded_raw = list(args.args or [])

    if command_raw is None:
        parser.print_help()
        print("")
        print("Tip: run `pvx list` for tool descriptions.")
        return 0

    try:
        forwarded, lucky_count, lucky_seed = _consume_lucky_options(forwarded_raw)
    except ValueError as exc:
        parser.error(str(exc))

    command = str(command_raw).strip().lower()
    helper_commands = {
        "list",
        "ls",
        "tools",
        "examples",
        "example",
        "quickstart",
        "doctor",
        "transforms",
        "safe",
        "smoke",
        "guided",
        "guide",
        "follow",
        "chain",
        "stream",
        "stretch-budget",
        "stretchbudget",
        "budget",
        "help",
    }

    if command in {"list", "ls", "tools"}:
        print_tools()
        return 0
    if command in {"examples", "example"}:
        which = forwarded[0] if forwarded else "all"
        try:
            print_examples(which)
        except ValueError as exc:
            parser.error(str(exc))
        return 0
    if command == "quickstart":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx quickstart`")
        return run_quickstart_mode(forwarded)
    if command == "doctor":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx doctor`")
        return run_doctor_mode(forwarded)
    if command == "transforms":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx transforms`")
        return run_transforms_mode(forwarded)
    if command == "safe":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx safe`")
        return run_safe_mode(forwarded)
    if command == "smoke":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx smoke`")
        return run_smoke_mode(forwarded)
    if command in {"guided", "guide"}:
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx guided`")
        try:
            return run_guided_mode()
        except ValueError as exc:
            parser.error(str(exc))
    if command == "follow":
        if lucky_count is not None:
            try:
                return _run_lucky_helper_mode("follow", forwarded, lucky_count, lucky_seed)
            except ValueError as exc:
                parser.error(str(exc))
        try:
            return run_follow_mode(forwarded)
        except ValueError as exc:
            parser.error(str(exc))
    if command == "chain":
        if lucky_count is not None:
            try:
                return _run_lucky_helper_mode("chain", forwarded, lucky_count, lucky_seed)
            except ValueError as exc:
                parser.error(str(exc))
        try:
            return run_chain_mode(forwarded)
        except ValueError as exc:
            parser.error(str(exc))
    if command == "stream":
        if lucky_count is not None:
            try:
                return _run_lucky_helper_mode("stream", forwarded, lucky_count, lucky_seed)
            except ValueError as exc:
                parser.error(str(exc))
        try:
            return run_stream_mode(forwarded)
        except ValueError as exc:
            parser.error(str(exc))
    if command in {"stretch-budget", "stretchbudget", "budget"}:
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx stretch-budget`")
        try:
            return run_stretch_budget_mode(forwarded)
        except ValueError as exc:
            parser.error(str(exc))
    if command == "help":
        if not forwarded:
            parser.print_help()
            return 0
        target = str(forwarded[0]).strip().lower()
        if target in helper_commands:
            if target in {"examples", "example"}:
                print_examples("all")
                return 0
            if target == "quickstart":
                print("Run `pvx quickstart` for a minimal launch/demo sequence.")
                return 0
            if target == "doctor":
                print("Run `pvx doctor` for environment diagnostics and actionable fixes.")
                return 0
            if target == "transforms":
                print("Run `pvx transforms` for transform choices and backend availability.")
                return 0
            if target == "safe":
                print("Run `pvx safe --help` for conservative quality-first voc defaults.")
                return 0
            if target == "smoke":
                print("Run `pvx smoke --help` for a fast synthetic end-to-end verification render.")
                return 0
            if target in {"list", "ls", "tools"}:
                print_tools()
                return 0
            if target in {"guided", "guide"}:
                print("Run `pvx guided` from an interactive terminal.")
                return 0
            if target == "follow":
                print("Run `pvx follow --help` for one-command sidechain control mapping.")
                return 0
            if target == "chain":
                print("Run `pvx chain --help` for managed one-line tool chaining.")
                return 0
            if target == "stream":
                print("Run `pvx stream --help` for chunked streaming wrapper options.")
                return 0
            if target in {"stretch-budget", "stretchbudget", "budget"}:
                print("Run `pvx stretch-budget --help` for file-based stretch-budget estimates.")
                return 0
            parser.print_help()
            return 0
        if target in TOOL_INDEX:
            return dispatch_tool(target, ["--help"])
        parser.error(f"Unknown help target '{forwarded[0]}'. Use `pvx list`.")

    if command in TOOL_INDEX:
        spec = TOOL_INDEX[command]
        if lucky_count is not None:
            try:
                return _run_lucky_tool_mode(spec.name, forwarded, lucky_count, lucky_seed)
            except ValueError as exc:
                parser.error(str(exc))
        return dispatch_tool(command, forwarded)

    # Beginner shortcut: if first token looks like an input path or glob, treat as `pvx voc ...`.
    if _looks_like_audio_input(command_raw):
        shortcut_args = [command_raw] + forwarded
        if lucky_count is not None:
            try:
                return _run_lucky_tool_mode("voc", shortcut_args, lucky_count, lucky_seed)
            except ValueError as exc:
                parser.error(str(exc))
        return dispatch_tool("voc", shortcut_args)

    candidates = sorted(
        set(
            [spec.name for spec in TOOL_SPECS]
            + [alias for spec in TOOL_SPECS for alias in spec.aliases]
            + list(helper_commands)
        )
    )
    suggestions = difflib.get_close_matches(command, candidates, n=3, cutoff=0.45)
    detail = ""
    if suggestions:
        detail = f" Did you mean: {', '.join(suggestions)}?"
    parser.error(f"Unknown command '{command_raw}'.{detail} Run `pvx list` to inspect commands.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
