# Implementation Specification: Emotion Detector (`emotion_detector.py`)

## 1. Overview
The `emotion_detector.py` module houses the `EmotionDetector` class. It utilizes `scikit-learn` to load an SVM and `librosa` to compute low-level speech acoustics from non-silent audio frames. It bridges the gap between raw signal processing and classical machine learning emotion classification.

## 2. Input Data & Structure (Code Level)
*   **Audio Data:** A `numpy.ndarray` containing float amplitudes.
*   **Sample Rate:** An integer `sr` used as a multiplier for time-domain conversions.
*   **Pre-trained Model:** Expected at `models/emotion/svm_model.pkl` (pickled sklearn pipelines).

## 3. Detailed Execution Flow (Step-by-Step)
1.  **Class Initialization:** The `__init__` constructor sets up mapping dictionaries translating numerical classes (`0`, `1`, `2`) to human-readable strings (`'angry'`, `'calm'`, `'happy'`).
2.  **Audio Ingestion & Silence Trimming:** To prevent the model from analyzing dead air, the detector executes `librosa.effects.trim(audio, top_db=30)`.
3.  **Fundamental Frequency Tracking:** The `librosa.pyin(audio, fmin=50, fmax=500)` function calculates the average Pitch (`F0`). `pyin` is used globally for high-accuracy pitch tracking.
4.  **Energy Profiling:** The `librosa.feature.rms(y=audio)` module calculates the Root Mean Square of the signal's energy (loudness and stress).
5.  **Timber & Texture Analysis:** `librosa.feature.mfcc(..., n_mfcc=13)` extracts the primary 13 Mel-Frequency Cepstral Coefficients, capturing the shape of the vocal tract.
6.  **Feature Flattening:** The mean values of all features (F0, RMS, MFCC, Spectral Centroid) are concatenated into a single `1D` NumPy array representing one unified "Feature Vector."
7.  **Inference:** Using `self.model.predict(features)` on the SVM.
8.  **Probability Mapping:** `self.model.predict_proba(features)` is called to determine the certainty of the prediction as a percentage (`confidence = np.max(proba)`).

## 4. Internal Data Transformations
1.  `audio (np.ndarray)` + `sr` $\rightarrow$ `MFCC Array (13xT)` and `F0 Array (1xT)`.
2.  `np.mean()` operations on the arrays across the time axis (T) collapse the vectors into 1D float averages.
3.  Feature Vectors $\rightarrow$ `scikit-learn` standard scalar normalization (mean=0, variance=1).
4.  `model.predict` vector $\rightarrow$ Integer (`[0..7]`).
5.  Dictionary Mapping $\rightarrow$ String (`'Sad'`).

## 5. Output Data & Error Handling
*   **Success Output:** Returns a robust Python `dict` with fields:
    *   `'emotion'`: The mapped string label.
    *   `'confidence'`: A float percentage.
    *   `'raw_scores'`: The probability distributions across all 7 emotions for deeper logging.
*   **Fallback Handling:** If `models/emotion/svm_model.pkl` is missing or corrupted, the `detect()` wrapper catches the exception and returns a dummy output: `{'emotion': 'neutral', 'confidence': 0.5}`. This ensures the rest of the TTS pipeline (Qwen) doesn't crash if the analysis layer fails.
