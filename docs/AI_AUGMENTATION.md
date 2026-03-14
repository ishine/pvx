<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# AI Research and Data Augmentation with pvx

`pvx augment` provides deterministic, manifest-driven audio augmentation for machine-learning workflows.

Primary use cases:
- automatic speech recognition (ASR) robustness experiments
- music information retrieval (MIR) augmentation studies
- self-supervised learning (SSL) contrastive view generation
- ablation studies where exact augmentation parameters must be replayed

## Core Command

```bash
pvx augment data/*.wav --output-dir aug_out --variants-per-input 4 --intent asr_robust --seed 1337
```

What this does:
- resolves input files/globs/directories
- renders `N` deterministic variants per input (`--variants-per-input`)
- writes per-variant metadata to:
  - `augment_manifest.jsonl`
  - `augment_manifest.csv`

## Intent Profiles

| Intent | Typical target | Behavior |
| --- | --- | --- |
| `asr_robust` | Speech pipelines | Mild time/pitch perturbation with conservative vocal/formant defaults |
| `mir_music` | Music analytics | Moderate timing/pitch/spectral variation suited for musical content |
| `ssl_contrastive` | Contrastive representation learning | Wider perturbation envelope for view diversity while remaining reproducible |

## Reproducibility Controls

| Option | Meaning |
| --- | --- |
| `--seed` | Global deterministic seed for all sampled parameters |
| `--split train,val,test` | Deterministic split assignment ratios written into manifest |
| `--grouping` | Split assignment grouping strategy (`stem-prefix` or `none`) |
| `--group-separator` | Prefix separator for grouping (default: `__`) |
| `--dry-run` | Plan outputs and write manifests without rendering audio |
| `--manifest-jsonl` / `--manifest-csv` | Explicit manifest output paths |

## Manifest Fields

Each JSON Lines (JSONL) row contains:

- `source_path`
- `output_path`
- `intent`
- `seed`
- `split` (`train`, `val`, `test`)
- `group_key` (split-group identifier)
- `status` (`planned`, `rendered`, or `error:<code>`)
- `params` object, including:
  - `stretch`
  - `pitch`
  - `preset`
  - `window`
  - `transform`
  - `formant_strength`
  - `transient_sensitivity`
  - `target_lufs`

CSV manifest includes the same essential fields in tabular form.

## Example Workflows

### 1) Speech Robustness Set

```bash
pvx augment corpus/speech/*.wav \
  --output-dir data_aug/speech \
  --variants-per-input 6 \
  --intent asr_robust \
  --grouping stem-prefix \
  --group-separator "__" \
  --split 0.8,0.1,0.1 \
  --seed 1337
```

### 2) Music Information Retrieval Set

```bash
pvx augment corpus/music/**/*.wav \
  --output-dir data_aug/music \
  --variants-per-input 4 \
  --intent mir_music \
  --split 0.7,0.2,0.1 \
  --seed 2026
```

### 3) Contrastive Planning-Only Pass

```bash
pvx augment corpus/*.wav \
  --output-dir data_aug/plan_only \
  --variants-per-input 3 \
  --intent ssl_contrastive \
  --dry-run \
  --seed 42
```

### 4) With explicit manifest paths

```bash
pvx augment corpus/*.wav \
  --output-dir data_aug/run_a \
  --variants-per-input 5 \
  --intent asr_robust \
  --manifest-jsonl reports/run_a_manifest.jsonl \
  --manifest-csv reports/run_a_manifest.csv \
  --seed 9001
```

## Running with uv

```bash
uv run pvx augment data/*.wav --output-dir aug_out --variants-per-input 4 --intent asr_robust --seed 1337
```

## Research Notes

- Keep original clean file IDs in your training metadata so you can group augmented siblings.
- Avoid split leakage: use `--grouping stem-prefix` with a stable naming convention (for example `speaker42__take3.wav`).
- Prefer fixed seeds for published experiments; change seeds only for explicit variance studies.
- Use `--dry-run` before large jobs to validate output counts and manifest content.

## Attribution

Copyright (c) 2026 Colby Leider and contributors. See [ATTRIBUTION.md](../ATTRIBUTION.md).
