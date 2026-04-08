#!/usr/bin/env python3

"""Benchmark runner for pvx augmentation workflows.

Supports profile-driven benchmark presets (speech/music/noisy/stereo) with
both relative (baseline drift) and absolute metric gates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_PROFILES_PATH = ROOT / "benchmarks" / "augment_profiles.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


PROFILE_FLAG_MAP: dict[str, tuple[str, ...]] = {
    "input_glob": ("--input-glob",),
    "intent": ("--intent",),
    "pair_mode": ("--pair-mode",),
    "seed": ("--seed",),
    "split": ("--split",),
    "workers": ("--workers",),
    "variants_per_input": ("--variants-per-input",),
    "baseline": ("--baseline",),
    "gate_tolerance": ("--gate-tolerance",),
}


def _load_manifest_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _required_error_count(rows: list[dict[str, Any]]) -> int:
    required = ("source_path", "output_path", "intent", "seed", "split", "group_key", "status")
    errors = 0
    for row in rows:
        for key in required:
            if row.get(key) in {None, ""}:
                errors += 1
    return errors


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _split_balance_l1(rows: list[dict[str, Any]], target: tuple[float, float, float]) -> float:
    if not rows:
        return float("nan")
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        split = str(row.get("split", "")).strip().lower()
        if split in counts:
            counts[split] += 1
    total = float(sum(counts.values()))
    if total <= 0.0:
        return float("nan")
    observed = (counts["train"] / total, counts["val"] / total, counts["test"] / total)
    return float(sum(abs(observed[i] - target[i]) for i in range(3)))


def _summarize(
    rows: list[dict[str, Any]], *, target_split: tuple[float, float, float]
) -> dict[str, Any]:
    rendered = [row for row in rows if str(row.get("status", "")).startswith("rendered")]
    splits = {"train": 0, "val": 0, "test": 0}
    intents: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", "")).strip().lower()
        if split in splits:
            splits[split] += 1
        intent = str(row.get("intent", "")).strip().lower() or "unknown"
        intents[intent] = intents.get(intent, 0) + 1

    stretch_values: list[float] = []
    pitch_values: list[float] = []
    clip_values: list[float] = []
    peak_values: list[float] = []
    pair_to_views: dict[str, set[str]] = {}
    for row in rows:
        params = row.get("params", {})
        if isinstance(params, dict):
            s = _safe_float(params.get("stretch"))
            p = _safe_float(params.get("pitch"))
            if s is not None:
                stretch_values.append(s)
            if p is not None:
                pitch_values.append(p)
        audit = row.get("audit", {})
        if isinstance(audit, dict):
            clip = _safe_float(audit.get("clip_pct"))
            peak = _safe_float(audit.get("peak_dbfs"))
            if clip is not None:
                clip_values.append(clip)
            if peak is not None:
                peak_values.append(peak)
        pair_id = str(row.get("pair_id", "")).strip()
        view_id = str(row.get("view_id", "")).strip()
        if pair_id:
            pair_to_views.setdefault(pair_id, set()).add(view_id)

    pair_complete = 0
    if pair_to_views:
        for views in pair_to_views.values():
            if {"a", "b"}.issubset(views):
                pair_complete += 1
    pair_coverage = float(pair_complete / len(pair_to_views)) if pair_to_views else float("nan")

    peak_p95 = float("nan")
    if peak_values:
        ordered = sorted(peak_values)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        peak_p95 = float(ordered[idx])

    return {
        "records_total": len(rows),
        "rendered_total": len(rendered),
        "required_field_errors": _required_error_count(rows),
        "unique_sources": len(
            {str(row.get("source_path", "")) for row in rows if row.get("source_path")}
        ),
        "split_counts": splits,
        "intent_counts": intents,
        "split_balance_l1": _split_balance_l1(rows, target_split),
        "stretch_mean": float(statistics.mean(stretch_values)) if stretch_values else float("nan"),
        "stretch_std": float(statistics.pstdev(stretch_values)) if len(stretch_values) > 1 else 0.0,
        "pitch_mean": float(statistics.mean(pitch_values)) if pitch_values else float("nan"),
        "pitch_std": float(statistics.pstdev(pitch_values)) if len(pitch_values) > 1 else 0.0,
        "pitch_abs_mean": float(statistics.mean(abs(v) for v in pitch_values))
        if pitch_values
        else float("nan"),
        "clip_pct_max": float(max(clip_values)) if clip_values else float("nan"),
        "peak_dbfs_p95": peak_p95,
        "pair_coverage": pair_coverage,
    }


def _relative_metric_gate(
    cur: dict[str, Any], base: dict[str, Any], tolerance: float
) -> tuple[bool, list[str]]:
    keys = (
        "records_total",
        "rendered_total",
        "required_field_errors",
        "split_balance_l1",
        "stretch_std",
        "pitch_std",
        "clip_pct_max",
        "pair_coverage",
    )
    lower_is_better = {"required_field_errors", "split_balance_l1", "clip_pct_max"}
    higher_is_better = {"pair_coverage"}
    failures: list[str] = []
    for key in keys:
        if key not in cur or key not in base:
            continue
        a = _safe_float(cur[key])
        b = _safe_float(base[key])
        if a is None or b is None:
            continue
        if key in lower_is_better and a <= b:
            continue
        if key in higher_is_better and a >= b:
            continue
        denom = max(abs(b), 1e-9)
        rel = abs(a - b) / denom
        if rel > tolerance:
            failures.append(
                f"relative:{key}: current={a:.6g} baseline={b:.6g} rel={rel:.4f} tol={tolerance:.4f}"
            )
    return len(failures) == 0, failures


def _absolute_metric_gate(cur: dict[str, Any], rules: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for metric, rule_obj in rules.items():
        if not isinstance(rule_obj, dict):
            continue
        value = _safe_float(cur.get(metric))
        if value is None:
            continue
        min_v = _safe_float(rule_obj.get("min"))
        max_v = _safe_float(rule_obj.get("max"))
        if min_v is not None and value < min_v:
            failures.append(f"absolute:{metric}: {value:.6g} < min {min_v:.6g}")
        if max_v is not None and value > max_v:
            failures.append(f"absolute:{metric}: {value:.6g} > max {max_v:.6g}")
    return len(failures) == 0, failures


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _flag_present(argv: list[str], flags: tuple[str, ...]) -> bool:
    for arg in argv:
        for flag in flags:
            if arg == flag or arg.startswith(f"{flag}="):
                return True
    return False


def _load_profiles(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"profiles": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"profiles": {}}
    if not isinstance(payload.get("profiles"), dict):
        payload["profiles"] = {}
    return payload


def _profile_names(payload: dict[str, Any]) -> list[str]:
    profiles = payload.get("profiles", {})
    if not isinstance(profiles, dict):
        return []
    return sorted(str(k) for k in profiles)


def _resolve_profile_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _apply_profile_overrides(
    args: argparse.Namespace,
    *,
    argv: list[str],
    profile_name: str,
    profiles_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    profiles = profiles_payload.get("profiles", {})
    if not isinstance(profiles, dict) or profile_name not in profiles:
        available = ", ".join(_profile_names(profiles_payload)) or "(none)"
        raise SystemExit(f"Unknown --profile {profile_name!r}. Available profiles: {available}")

    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise SystemExit(f"Profile {profile_name!r} is malformed (expected object)")

    for key, flags in PROFILE_FLAG_MAP.items():
        if key not in profile:
            continue
        if _flag_present(argv, flags):
            continue
        value = profile[key]
        if key in {"seed", "workers", "variants_per_input"}:
            value = int(value)
        elif key == "gate_tolerance":
            value = float(value)
        elif key == "baseline":
            rp = _resolve_profile_path(str(value))
            value = None if rp is None else rp
        else:
            value = str(value)
        setattr(args, key, value)

    if bool(args.quick) and (not _flag_present(argv, ("--variants-per-input",))):
        quick_var = profile.get("quick_variants_per_input")
        if quick_var is not None:
            args.variants_per_input = int(quick_var)

    absolute_rules = profile.get("absolute_gates", {})
    if not isinstance(absolute_rules, dict):
        absolute_rules = {}

    return profile, absolute_rules


def main(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        description="Run pvx augmentation benchmark and optional regression gate."
    )
    parser.add_argument(
        "--input-glob", default="test.wav", help="Input glob for augmentation benchmark"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("benchmarks/out_augment"), help="Output directory"
    )
    parser.add_argument("--variants-per-input", type=int, default=2, help="Variants per input")
    parser.add_argument(
        "--intent", choices=["asr_robust", "mir_music", "ssl_contrastive"], default="asr_robust"
    )
    parser.add_argument("--pair-mode", choices=["off", "contrastive2"], default="off")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split", default="0.8,0.1,0.1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline JSON path")
    parser.add_argument(
        "--gate", action="store_true", help="Enable regression gate against baseline/absolute rules"
    )
    parser.add_argument(
        "--gate-tolerance", type=float, default=0.20, help="Relative tolerance for baseline drift"
    )
    parser.add_argument(
        "--refresh-baseline", action="store_true", help="Write current metrics to baseline path"
    )
    parser.add_argument("--quick", action="store_true", help="Run smaller/faster benchmark")
    parser.add_argument("--profile", default=None, help="Named profile from --profiles-file")
    parser.add_argument(
        "--profiles-file", type=Path, default=DEFAULT_PROFILES_PATH, help="Profile JSON file path"
    )
    parser.add_argument(
        "--list-profiles", action="store_true", help="List available profiles and exit"
    )
    parser.add_argument(
        "--absolute-gates-json", default=None, help="Override absolute gate rules with JSON object"
    )
    args = parser.parse_args(cli_argv)

    profiles_payload = _load_profiles(Path(args.profiles_file).expanduser().resolve())

    if bool(args.list_profiles):
        names = _profile_names(profiles_payload)
        if names:
            print("Available profiles:")
            for name in names:
                desc = ""
                profile = profiles_payload.get("profiles", {}).get(name, {})
                if isinstance(profile, dict):
                    desc = str(profile.get("description", "")).strip()
                if desc:
                    print(f"- {name}: {desc}")
                else:
                    print(f"- {name}")
        else:
            print("No profiles available.")
        return 0

    profile_name = str(args.profile).strip() if args.profile else ""
    profile_data: dict[str, Any] = {}
    profile_abs_rules: dict[str, Any] = {}
    if profile_name:
        profile_data, profile_abs_rules = _apply_profile_overrides(
            args,
            argv=cli_argv,
            profile_name=profile_name,
            profiles_payload=profiles_payload,
        )

    if args.absolute_gates_json:
        try:
            payload = json.loads(args.absolute_gates_json)
            if isinstance(payload, dict):
                profile_abs_rules = payload
            else:
                raise ValueError("must decode to object")
        except Exception as exc:
            raise SystemExit(f"Invalid --absolute-gates-json: {exc}") from exc

    out_dir = Path(args.out_dir).expanduser().resolve()
    aug_dir = out_dir / "aug"
    out_dir.mkdir(parents=True, exist_ok=True)
    aug_dir.mkdir(parents=True, exist_ok=True)
    manifest_jsonl = out_dir / "augment_manifest.jsonl"
    manifest_csv = out_dir / "augment_manifest.csv"

    variants = max(1, int(args.variants_per_input))
    if bool(args.quick):
        variants = max(1, min(variants, 1))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "pvx.cli.pvx",
        "augment",
        str(args.input_glob),
        "--output-dir",
        str(aug_dir),
        "--variants-per-input",
        str(variants),
        "--intent",
        str(args.intent),
        "--pair-mode",
        str(args.pair_mode),
        "--seed",
        str(int(args.seed)),
        "--split",
        str(args.split),
        "--workers",
        str(int(args.workers)),
        "--manifest-jsonl",
        str(manifest_jsonl),
        "--manifest-csv",
        str(manifest_csv),
        "--overwrite",
        "--quiet",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed_sec = float(time.perf_counter() - t0)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return int(proc.returncode or 1)

    rows = _load_manifest_jsonl(manifest_jsonl)
    split_parts = [float(x.strip()) for x in str(args.split).split(",")]
    total = max(1e-12, sum(split_parts))
    target = (split_parts[0] / total, split_parts[1] / total, split_parts[2] / total)
    metrics = _summarize(rows, target_split=target)
    report = {
        "meta": {
            "profile": profile_name or None,
            "profile_description": str(profile_data.get("description", "")).strip()
            if profile_data
            else "",
            "profiles_file": str(Path(args.profiles_file).expanduser().resolve()),
            "input_glob": str(args.input_glob),
            "out_dir": str(out_dir),
            "intent": str(args.intent),
            "pair_mode": str(args.pair_mode),
            "seed": int(args.seed),
            "split": str(args.split),
            "variants_per_input": int(variants),
            "elapsed_sec": elapsed_sec,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "command": cmd,
        },
        "metrics": metrics,
    }

    baseline_payload: dict[str, Any] | None = None
    baseline_path = (
        Path(args.baseline).expanduser().resolve() if args.baseline is not None else None
    )
    relative_failures: list[str] = []
    absolute_failures: list[str] = []
    relative_ok = True
    absolute_ok = True

    if baseline_path is not None and baseline_path.exists():
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        base_metrics = dict(baseline_payload.get("metrics", {}))
        relative_ok, relative_failures = _relative_metric_gate(
            metrics, base_metrics, float(args.gate_tolerance)
        )

    if profile_abs_rules:
        absolute_ok, absolute_failures = _absolute_metric_gate(metrics, profile_abs_rules)

    if bool(args.refresh_baseline) and baseline_path is not None:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    gate_ok = bool(relative_ok and absolute_ok)
    report["gate"] = {
        "enabled": bool(args.gate),
        "ok": bool(gate_ok),
        "relative": {
            "enabled": baseline_path is not None,
            "ok": bool(relative_ok),
            "tolerance": float(args.gate_tolerance),
            "failures": relative_failures,
            "baseline_path": None if baseline_path is None else str(baseline_path),
            "baseline_loaded": baseline_payload is not None,
        },
        "absolute": {
            "enabled": bool(profile_abs_rules),
            "ok": bool(absolute_ok),
            "rules": profile_abs_rules,
            "failures": absolute_failures,
        },
    }

    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    safe_report = _json_safe(report)
    report_json.write_text(
        json.dumps(safe_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines: list[str] = []
    lines.append("# pvx Augmentation Benchmark")
    lines.append("")
    if profile_name:
        lines.append(f"- Profile: `{profile_name}`")
        profile_desc = str(profile_data.get("description", "")).strip()
        if profile_desc:
            lines.append(f"- Profile description: {profile_desc}")
    lines.append(f"- Input glob: `{args.input_glob}`")
    lines.append(f"- Intent: `{args.intent}`")
    lines.append(f"- Pair mode: `{args.pair_mode}`")
    lines.append(f"- Variants/input: `{variants}`")
    lines.append(f"- Elapsed: `{elapsed_sec:.3f}` seconds")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            if math.isnan(value):
                lines.append(f"| `{key}` | `nan` |")
            else:
                lines.append(f"| `{key}` | `{value:.6f}` |")
        else:
            lines.append(f"| `{key}` | `{value}` |")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"- Enabled: `{bool(args.gate)}`")
    lines.append(f"- Pass: `{bool(gate_ok)}`")
    lines.append(f"- Relative gate pass: `{bool(relative_ok)}`")
    lines.append(f"- Absolute gate pass: `{bool(absolute_ok)}`")
    if relative_failures:
        lines.append("- Relative failures:")
        for item in relative_failures:
            lines.append(f"  - {item}")
    if absolute_failures:
        lines.append("- Absolute failures:")
        for item in absolute_failures:
            lines.append(f"  - {item}")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[augment-bench] report json -> {report_json}")
    print(f"[augment-bench] report md   -> {report_md}")
    if bool(args.gate) and not bool(gate_ok):
        for item in relative_failures + absolute_failures:
            print(f"[augment-bench] gate failure: {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
