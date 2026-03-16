# Component Specification: Audio Preprocessor (`preprocessor.py`)

## 1. Overview
The Audio Preprocessor is the first line of defense in the Analysis Engine. Its job is to take raw, potentially messy microphone recordings and clean them up so the rest of the system (like the Emotion Detector and Qwen Adapter) receives high-quality, standardized audio data.

## 2. Input Data & Structure
| Input Name | Data Type | Description |
| :--- | :--- | :--- |
| `audio_path` | `String` | Absolute path to the raw `.wav` file recorded by the user. |

## 3. High-Level Sequence of Execution
1.  **Loading**: Reads the raw audio file from the disk.
2.  **Resampling**: Forces the audio to a standard Sample Rate (e.g., 16,000 Hz or 24,000 Hz) so all downstream models know what to expect.
3.  **Volume Normalization**: Adjusts the overall volume so it's not too quiet and not clipping (too loud).
4.  **Silence Trimming**: Detects and chops off any "dead air" or silence at the beginning and end of the recording.
5.  **Noise Reduction**: Applies a pre-emphasis filter to balance the frequencies and reduce background hum.

## 4. Internal Data Transformations
*   **File $\rightarrow$ Audio Tensor**: The physical `.wav` file is read and transformed into a 1-dimensional float array (NumPy) representing the audio waveform.
*   **Messy Array $\rightarrow$ Clean Array**: The array's values are mathematically scaled (normalized) and sliced (trimmed of zeros/quiet parts).

## 5. Output Data & Structure
| Output Name | Data Type | Description |
| :--- | :--- | :--- |
| `clean_audio` | `np.ndarray` | A clean, 1-dimensional NumPy float array containing the processed waveform. |
| `sr` | `Integer` | The standardized Sample Rate of the clean audio. |
