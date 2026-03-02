# Contributing to pvx

Thanks for contributing to `pvx`.

This project is research- and production-oriented. Changes should prioritize
audio quality first, speed second, and preserve command-line interface (CLI)
backward compatibility unless explicitly discussed.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e ".[dev]"
```

Optional:

```bash
pre-commit install
```

## Required Checks Before Opening a Pull Request

```bash
make docs
make test
make lint
make typecheck
```

If you do not use `make`, run the underlying commands directly.

## Pull Request Scope

- Keep changes focused and reviewable.
- Include tests for behavioral changes.
- Update documentation in the same pull request when behavior, parameters,
  outputs, or defaults change.
- Preserve existing command-line interfaces (`pvx`, `pvxvoc`, `pvxfreeze`, etc.).

## Commit Messages

Use clear imperative messages, for example:

- `Add transient region merge guard`
- `Fix CSV interpolation edge case at file end`
- `Regenerate docs after CLI flag update`

## DSP and Audio Quality Expectations

- Document algorithm assumptions and failure modes.
- Provide deterministic mode behavior for reproducibility.
- Avoid regressions in transient handling and stereo coherence.
- Validate objective metrics when changing core transforms or phase logic.

## Documentation Contract

When touching documentation generators, regenerate and commit generated outputs:

```bash
python3 scripts/scripts_generate_python_docs.py
python3 scripts/scripts_generate_theory_docs.py
python3 scripts/scripts_generate_docs_extras.py
python3 scripts/scripts_generate_html_docs.py
```

## Attribution

Copyright (c) 2026 Colby Leider and contributors. See [ATTRIBUTION.md](ATTRIBUTION.md).
