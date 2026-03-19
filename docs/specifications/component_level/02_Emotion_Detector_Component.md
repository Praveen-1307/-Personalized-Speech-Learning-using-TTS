# Component Specification: Emotion Detector (`emotion_detector.py`)

## 1. Overview
The Emotion Detector acts as the cognitive layer of the Analysis Engine. Its purpose is to predict the emotional state of a user's voice based on a short audio recording. This is a crucial component for ensuring the synthesized speech matches the natural "vibe" of the original speaker, rather than sounding robotic.

## 2. Input Data & Structure
| Input Name | Data Type | Description |
| :--- | :--- | :--- |
| `audio` | `np.ndarray` | A 1-dimensional NumPy array of the loaded `.wav` audio signal. |
| `sr` | `Integer` | The original Sample Rate of the audio in Hz (e.g., `16000` or `22050`). |

## 3. High-Level Sequence of Execution
1.  **Model Loading**: The component attempts to load a pre-trained `scikit-learn` Support Vector Machine (SVM) from the local disk (`models/emotion/svm_model.pkl`).
2.  **Audio Ingestion**: The component receives the audio waveform data (loaded using the `librosa` library).
3.  **Feature Extraction**: The detector runs intense mathematical operations to extract **5 Core Features** (Pitch, Loudness, Spectral Shape, MFCC Texture, Zero-Crossing Rate).
4.  **Prediction**: The combined feature logic is fed through the SVM model to retrieve the most likely emotional classification.
5.  **Output Generation**: The component returns an easy-to-read dictionary containing the predicted emotion string and its percentage confidence.

## 4. Internal Data Transformations
*   **Audio $\rightarrow$ Features**: The physical `.wav` file is transformed into mathematical arrays.
*   **Features $\rightarrow$ Vector**: Those raw numbers are normalized and combined into a flattened 1D feature vector array that the machine learning model can understand.
*   **Vector $\rightarrow$ String**: The ML model predicts an integer associated with a string label (e.g., `0 = Angry`, `3 = Happy`).

## 5. Output Data & Structure
| Output Name | Data Type | Description |
| :--- | :--- | :--- |
| `emotion` | `String` | The predicted core emotion (e.g., `'neutral', 'happy', 'sad', 'angry'`). |
| `confidence` | `Float` | The probability score between 0.0 and 1.0 that the prediction is correct. |
| `(Internal)` | `Dictionary` | The returned data contains raw technical numbers alongside the prediction for use in logging. |
