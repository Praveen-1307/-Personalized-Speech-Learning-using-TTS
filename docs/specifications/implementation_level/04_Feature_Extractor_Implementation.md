# Implementation Specification: Feature Extractor (`personalization_engine/feature_extractor.py`)

## Overview
This component measures the physical and acoustic traits of a recorded voice. It translates raw audio data into mathematical statistics representing pitch, loudness, speed, and timbre.

## Input Data and Structure
| Input | Type | Structure |
| :--- | :--- | :--- |
| `audio` | `np.ndarray` | Sanitized 1D audio array. |\n| `sr` | `int` | Confirmed sampling rate. |

## Sequence of Steps
1. Perform F0 tracking algorithms (YIN/PYIN) to deduce fundamental frequency ranges.\n2. Calculate RMS matrix to map continuous volume energy levels.\n3. Estimate physical syllable counts analyzing energy oscillation peaks measuring speaking limits.\n4. Calculate spectral centroids dictating audio brightness vectors.\n5. Synthesize time-series arrays into flat absolute float variables.

## Data Transformations
- `1D Array -> 2D Matrix`: Extracts time-frequency frames via Fast Fourier Transform analysis.\n- `2D Matrix -> float`: Reduces dynamic multi-frame matrices into flat mean averages.

## Output Data and Structure
| Output | Type | Structure |
| :--- | :--- | :--- |
| `profile_stats` | `dict` | Statistical dataset containing keys like 'f0_mean' and 'energy_std'. |

## Notes
- Pitch extraction via YIN/PYIN calculations are heavily resource intensive on extended clips.\n- Struggles structurally if polyphonic background noises conflict with harmonic measurements.
