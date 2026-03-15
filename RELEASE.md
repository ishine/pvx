# Release Process

`pvx` uses tag-based releases.

## Prerequisites

- `main` is green on required checks.
- `CHANGELOG.md` is updated.
- `pyproject.toml` version is updated.

## Pre-Tag Validation

Run release validation locally before creating the tag:

```bash
uv run pytest -q
python benchmarks/run_augment_profile_suite.py --quick --gate --out-dir benchmarks/out_augment_profiles_release
```

Optional baseline refresh (only when benchmark behavior intentionally changed):

```bash
python benchmarks/run_augment_profile_suite.py --quick --refresh-baselines --out-dir benchmarks/out_augment_profiles_refresh
```

## Create a Release Tag

```bash
git checkout main
git pull --rebase
git tag -a v0.1.0a1 -m "pvx v0.1.0a1 (alpha)"
git push origin v0.1.0a1
```

Pushing a `v*` tag triggers `.github/workflows/release.yml` to build artifacts
and publish a GitHub release.

## Attribution

Copyright (c) 2026 Colby Leider and contributors. See [ATTRIBUTION.md](ATTRIBUTION.md).
