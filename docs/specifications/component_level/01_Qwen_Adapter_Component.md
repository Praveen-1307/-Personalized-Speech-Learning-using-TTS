# Component Specification: Qwen Adapter (`qwen_adapter.py`)

## 1. Overview
The Qwen Adapter is the core synthesis wrapper component. It is responsible for bridging the gap between the application's data layer (text and user profiles) and the heavy deep learning model (`Qwen3-TTS`). Its primary responsibility is to take in target text and reference audio and return a seamless, voice-consistent audio waveform.

## 2. Input Data & Structure
This component accepts three primary inputs:
| Input Name | Data Type | Description |
| :--- | :--- | :--- |
| `model` | `Qwen3TTSModel` | The pre-loaded PyTorch neural network model object. |
| `text` | `String` | The raw UTF-8 string that the user wants the system to say. |
| `ref_audio_path` | `String` | The absolute file path to the user's reference `.wav` file on disk. |
| `speed` (Optional) | `Float` | A multiplier for playback speed (e.g., `1.0` is normal, `1.2` is faster). |

## 3. High-Level Sequence of Execution
1.  **Validation**: Verify the reference audio file actually exists on the disk.
2.  **Text Chunking**: The input string is broken into an array of smaller string sentences (max 1200 chars).
3.  **Prompt Extraction**: The component calls the model to analyze the reference `.wav` and extract a singular voice embedding (`x-vector`).
4.  **Batched Inference**: The component feeds the entire array of text chunks AND the singular voice embedding to the model simultaneously.
5.  **Output Reconstruction**: The component stitches together the returned audio pieces into a single file and applies optional speed adjustments.

## 4. Internal Data Transformations
*   **String $\rightarrow$ List[String]**: The raw text string is transformed into a list of strings via regex (`split_text_into_chunks`).
*   **Audio $\rightarrow$ Vector**: The physical `.wav` file is transformed into a dense mathematical tensor (`x-vector`) representing the speaker's vocal characteristics.

## 5. Output Data & Structure
| Output Name | Data Type | Description |
| :--- | :--- | :--- |
| `output_path` | `String` | The absolute file path to the newly generated `.wav` file (e.g., `output/cloned_12345.wav`). |
| `(Internal)` | `np.ndarray` | Before saving, the audio exists as a 1D NumPy float array (Concatenated PCM). |
