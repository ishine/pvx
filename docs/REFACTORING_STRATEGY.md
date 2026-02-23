# pvx Refactoring Strategy

This document outlines a phased implementation plan to address architectural deficiencies in the `pvx` repository, specifically targeting the "God Object" anti-pattern, monolithic core files, and root directory clutter.

## Overview of Deficiencies

1.  **"God Object" Anti-pattern**: `src/pvx/algorithms/base.py` centralizes implementation logic for almost all algorithms, despite a directory structure that suggests modularity.
2.  **Monolithic Core**: `src/pvx/core/voc.py` mixes low-level DSP logic (STFT, phase locking) with high-level CLI argument parsing and execution flow.
3.  **Root Clutter**: Numerous legacy wrapper scripts (`pvxvoc.py`, etc.) pollute the root directory.
4.  **Fragile Dispatch**: Algorithms are invoked via string-based `if/elif` dispatch in `base.py` rather than a robust registry.

---

## Phase 1: Project Organization & Cleanup

**Goal**: Reduce root directory clutter and standardize entry points.

1.  **Create Compatibility Directory**:
    *   Create `scripts/compat/`.
2.  **Move Legacy Wrappers**:
    *   Move `pvxvoc.py`, `pvxfreeze.py`, `pvxformant.py`, etc., from the root to `scripts/compat/`.
    *   **Action**: Update the `sys.path` insertion logic in these scripts to account for the new depth (from `parent` to `parent.parent`).
3.  **Update Documentation**:
    *   Update `README.md` to reflect that legacy wrappers are now in `scripts/compat/` (or strictly recommend the unified `pvx` CLI).

## Phase 2: Core Separation (`voc.py`)

**Goal**: Decouple DSP logic from CLI logic to improve testability and reusability.

1.  **Create DSP Module**:
    *   Create `src/pvx/core/dsp.py` (or `dsp_voc.py`).
2.  **Migrate Pure Functions**:
    *   Move purely functional DSP code from `src/pvx/core/voc.py` to the new module.
    *   Candidates: `stft`, `istft`, `phase_vocoder_time_stretch`, `wsola_time_stretch`, window generation functions.
3.  **Refactor CLI**:
    *   Update `src/pvx/core/voc.py` to import these functions from the new module.
    *   Ensure `voc.py` focuses solely on `argparse`, configuration building, and I/O.

## Phase 3: Algorithm Decoupling ("God Object" Dismantling)

**Goal**: Move algorithm implementation logic from `base.py` into their respective dedicated modules.

*This phase should be executed iteratively, one "theme" at a time.*

1.  **Define Standard Interface**:
    *   Formalize the `process(audio, sample_rate, **params)` contract for algorithm modules.
2.  **Pilot Migration**:
    *   Target: `time_scale_and_pitch_core`.
    *   Move logic from `_dispatch_time_scale` in `base.py` to:
        *   `src/pvx/algorithms/time_scale_and_pitch_core/wsola_waveform_similarity_overlap_add.py`
        *   `src/pvx/algorithms/time_scale_and_pitch_core/td_psola.py`
        *   etc.
3.  **Iterate**:
    *   Repeat for `pitch_detection_and_tracking`.
    *   Repeat for `retune_and_intonation`.
    *   (Continue until `base.py` contains only shared utilities).

## Phase 4: Registry & Dispatch Modernization

**Goal**: Replace string-based `if/elif` dispatch with a dynamic registry.

1.  **Enhance Registry**:
    *   Update `src/pvx/algorithms/registry.py` to support lazy loading of modules and direct function resolution.
2.  **Update Base Dispatch**:
    *   Rewrite `run_algorithm` in `base.py` to:
        1.  Look up the module/function in the registry using `algorithm_id`.
        2.  Dynamically import the module.
        3.  Invoke the standard `process()` entry point.
    *   Remove the massive `if/elif` blocks (`_dispatch_time_scale`, `_dispatch_denoise`, etc.).

## Phase 5: Standardization

**Goal**: Eliminate hardcoded values and standardize parameter handling.

1.  **Externalize Configuration**:
    *   Move magic numbers (default FFT sizes, overlaps, constants) to a `config.py` or constants file.
2.  **Schema Validation**:
    *   (Optional) Add schema validation for algorithm parameters using `pydantic` or similar, ensuring robust error handling before DSP execution.
