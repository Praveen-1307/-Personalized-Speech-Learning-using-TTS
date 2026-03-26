# Component Specification: Emotion Detector (`personalization_engine/emotion_detector.py`)

## Overview
This component acts as an acoustic classifier. It extracts high-level spectral and prosodic features from audio to predict the speaker's emotional state, outputting categorical labels such as 'happy' or 'angry'.

## Input Data and Structure
| Input | Type | Structure |
| :--- | :--- | :--- |
| `audio` | `np.ndarray` | 1D numpy array of the audio waveform. |\n| `sr` | `int` | The audio sampling rate in Hz. |

## Sequence of Steps
1. Check if an active SVM or RandomForest model is trained and loaded.\n2. Extract 5 core acoustic features (Pitch, RMS Energy, Spectral Centroid, MFCCs, ZCR) if the model is active.\n3. Normalize the extracted features using a pre-fitted StandardScaler.\n4. Feed the scaled features into the machine learning classifier.\n5. Map the output predictions back to string labels and return a confidence dictionary.

## Data Transformations
- `np.ndarray -> np.ndarray`: Audio sequence is abstracted into a condensed 1D array of statistical feature means.\n- `np.ndarray -> scaled matrix`: The feature array normalizes against global dataset distributions using Z-scores.\n- `scaled matrix -> dict`: Class probabilities are formatted into an easily digestible dictionary.

## Output Data and Structure
| Output | Type | Structure |
| :--- | :--- | :--- |
| `emotion` | `str` | The core predicted emotion label (e.g., 'neutral'). |\n| `confidence` | `float` | Score representing the model's certainty (0.0 to 1.0). |\n| `probabilities` | `dict` | Key-value pairs mapping all possible emotions to their respective probabilities. |

## Notes
- By default, the detector acts as a passive mock, defaulting to 'neutral' with a 100% confidence fallback unless explicitly trained.\n- Relies heavily on acoustic data rather than semantic text meaning.
