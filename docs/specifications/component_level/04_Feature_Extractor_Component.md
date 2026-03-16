# Component Specification: Feature Extractor (`feature_extractor.py`)

## 1. Overview
The Feature Extractor is part of the Analysis Engine. Unlike the Emotion Detector (which predicts a psychological state), the Feature Extractor measures the strict *physical* properties of the voice. It computes pitch (how high/low), energy (how loud), and speaking rate (how fast) to create a mechanical "Voice Profile".

## 2. Input Data & Structure
| Input Name | Data Type | Description |
| :--- | :--- | :--- |
| `audio` | `np.ndarray` | A clean, 1-dimensional NumPy float array representing the audio wave. |

## 3. High-Level Sequence of Execution
1.  **Pitch Tracking (F0)**: Uses the `pyin` algorithm to track the fundamental pitch (Hz) of the speaker through time.
2.  **Energy Profiling (RMS)**: Calculates the physical loudness (Root Mean Square energy) of every millisecond.
3.  **Speaking Rate Estimation**: Slices the sound into non-silent segments to guess how many syllables are being spoken per second.
4.  **Spectral Analysis**: Computes the "Spectral Centroid" and "Bandwidth", showing where the bulk of the audio frequencies live (e.g., bass-heavy vs. treble-heavy).
5.  **Aggregation**: Takes the average (mean) and variance of all these tracks over time.

## 4. Internal Data Transformations
*   **Time-Domain Array $\rightarrow$ Frequency-Domain Arrays**: The raw sound wave is translated into a set of frequency arrays using Fast Fourier Transforms (FFT).
*   **Continuous Tracks $\rightarrow$ Statistical Summaries**: A continuous track of 1000 pitch values gets mathematically transformed into simple statistics like `mean_pitch`, `max_pitch`, and `min_pitch`.

## 5. Output Data & Structure
| Output Name | Data Type | Description |
| :--- | :--- | :--- |
| `features` | `Dictionary` | A structured dictionary containing physical metrics. |

Example structure of `features`:
```json
{
  "prosodic": {"f0_mean": 120.5, "f0_range": 45.0, "energy_mean": 0.05},
  "spectral": {"centroid_mean": 1500.2},
  "speaking_pattern": {"articulation_rate": 4.5}
}
```
