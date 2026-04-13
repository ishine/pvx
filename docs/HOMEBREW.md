# Homebrew Install

`pvx` ships with a Homebrew formula in `Formula/pvx.rb`.

## Install From This Repo

Current development / pre-release flow:

From a local checkout:

```bash
brew install --HEAD ./Formula/pvx.rb
```

From GitHub once the formula is on `main`:

```bash
brew install --HEAD https://raw.githubusercontent.com/TheColby/pvx/main/Formula/pvx.rb
```

That formula creates an isolated Python virtualenv, installs the runtime Python dependencies, and links the `pvx*` CLI entry points.

## Stable Tagged Release Flow

After a tag is published, stamp the formula with the tagged source tarball checksum:

```bash
./scripts/refresh_homebrew_formula.sh v0.1.0a1
```

That updates `Formula/pvx.rb` with:
- the tagged GitHub source tarball URL
- the matching `sha256`
- the explicit formula version

Once committed, users can install the tagged formula without `--HEAD`:

```bash
brew install https://raw.githubusercontent.com/TheColby/pvx/main/Formula/pvx.rb
```

## Notes

- The formula currently targets `python@3.12`.
- Runtime audio I/O depends on Homebrew `libsndfile`.
- This formula is aimed at the project tap/raw-formula workflow, not Homebrew core submission.
