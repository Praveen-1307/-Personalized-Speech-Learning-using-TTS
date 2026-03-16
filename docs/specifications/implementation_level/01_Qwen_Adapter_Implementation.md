# Implementation Specification: Qwen Adapter (`qwen_adapter.py`)

## 1. Overview
The Qwen Adapter is a Python module implementing the `QwenAdapter` class and the core `generate_audio` function. It acts as a Windows-optimized compatibility layer for the `Qwen3-TTS` architecture, specifically designed to solve the problem of voice drift during long-form generation using Batched Inference.

## 2. Input Data & Structure (Code Level)
*   **The Model:** `Qwen3TTSModel.from_pretrained(model_id, trust_remote_code=True)` loaded into memory, explicitly placed on `device='cuda'` (GPU) or `device='cpu'` depending on system availability.
*   **The Text:** A Python `str` (the input text).
*   **The Audio:** A Python `str` representing the file path to a 16kHz or 24kHz `.wav` reference file.

## 3. Detailed Execution Flow (Step-by-Step)
1.  **Dependency Handling & Initialization:** The module attempts to import `Qwen3TTSModel` and provides graceful fallbacks if it fails. The `load_model()` function handles PyTorch device placement (`model.to(device)`).
2.  **Text Pre-processing (`split_text_into_chunks`):**
    *   **Logic:** Uses Regular Expressions `re.split(r'(?<=[.!?])\s+', text)` to split the text into an array based on sentence-ending punctuation (., !, ?).
    *   **Constraint:** It groups sentences back together until a chunk reaches 1200 characters to maximize GPU throughput without exceeding VRAM bounds.
3.  **Prompt Extraction (The `x-vector`):**
    *   **Logic:** Calls `model.create_voice_clone_prompt(..., x_vector_only_mode=True)`. This strips context and returns only the mathematical speaker embedding.
4.  **Batched Inference (`generate_audio`):**
    *   **Broadcasting:** The single `prompt` is duplicated to match the number of text chunks: `voice_clone_prompt=[prompt] * len(chunks)`.
    *   **Inference Call:** `model.generate_voice_clone()` is called with the list of text strings. Key parameters:
        *   `temperature=0.1`: Forces the model to be highly deterministic, minimizing variation in voice tone across chunks.
        *   `repetition_penalty=1.1`: Decreases the likelihood of the model stuttering or repeating syllables.
5.  **Concatenation & Speed Adjustment:**
    *   **Reconstruction:** The model returns a list of NumPy arrays (`wavs_list`). The system uses `np.concatenate(wavs_list)` to continuously stitch the PCM audio together.
    *   **Speed Modification:** If `speed != 1.0`, it uses the `librosa.effects.time_stretch()` phase vocoder to speed up/slow down the audio without altering pitch.

## 4. Internal Data Transformations
1.  Text string $\rightarrow$ `List[str]` (size N).
2.  File Path $\rightarrow$ PyTorch Tensor (size: `[1, embedding_dim]`).
3.  Model Inference $\rightarrow$ `List[np.ndarray]` representing discrete audio segments.
4.  Concatenation $\rightarrow$ Single `np.ndarray` float32 array.
5.  Disk Write $\rightarrow$ `.wav` file with RIFF headers via `soundfile.write`.

## 5. Output Data & Error Handling
*   **Success Output:** Returns a Python `str` containing the absolute output path of the `.wav` file.
*   **Fallback:** If Batched Inference fails (e.g., CUDA Out Of Memory), a `try/except` block catches the error and falls back to a sequentially executed `for` loop, parsing one chunk at a time.
