# System Architecture

The TTS Personalization Engine is a high-performance voice cloning pipeline. Originally based on Piper TTS, it has been evolved into a sophisticated **Zero-Shot Voice Cloning** system powered by the **Qwen3-TTS** model. It enables real-time extraction of user-specific speech patterns and emotions, coupling them with large-scale pre-trained models for state-of-the-art synthesis.

## High-Level Architecture

The system follows a modular adapter-based architecture:

1.  **Input Layer**: Accepts **Raw Audio (Microphone/WAV)** and **Input Text (String)**.
2.  **Analysis Layer**: Processes **Raw Audio** to output **Acoustic Features** (F0, Energy) and **Emotion Labels**.
3.  **Core Intelligence**: The **Qwen3-TTS (0.6B/1.7B)** transformer model, which takes **Reference Style** and **Target Text** to output **Mel-Spectrograms/Waveforms**.
4.  **Adapter Layer**: A Windows-optimized PyTorch wrapper that converts **Model Tensors** into **WAV signals**.
5.  **Data Layer**: Dual output system saving synthesized **Cloned Audio (.wav)** and its technical **Metadata Fingerprint (.json)**.

```mermaid
graph TD
    User[User] -->|Text String & Voice Sample| InputLayer[Input Layer]
    
    subgraph Analysis [Analysis Layer]
        InputLayer -->|Raw WAV| FE[Feature Extractor]
        InputLayer -->|Raw WAV| ED[Emotion Detector]
        FE -->|Pitch/Stress Tensors| Profile[Voice Metadata Profile]
        ED -->|Emotion Label & Confidence| Profile
    end
    
    subgraph Core [Core Intelligence & Adapter]
        Profile -->|Acoustic Context Vector| Qwen[Qwen3-TTS Model]
        InputLayer -->|Reference Audio WAV| Qwen
        InputLayer -->|Target Text String| Qwen
        Qwen -->|Generated Logit/Waveform Tensors| Adapter[Audio Adapter]
    end
    
    subgraph Data [Data Layer]
        Adapter -->|PCM Audio Data| WavOut[Cloned Audio .wav File]
        Profile -->|JSON Serialized Specs| JsonOut[Metadata Fingerprint .json]
        WavOut -->|Synthesized Audio| Val[Validator]
        InputLayer -->|Original Text| Val
        Val -->|Validation Status & Metrics| ReportOut[Validation Report .json]
    end
```

## Component Details

### 1. Analysis Engine (`personalization_engine.feature_extractor`)
Derived from signal processing research, this component computes the "Voice DNA".
*   **Input**: Raw audio waveform (`np.ndarray` or WAV file).
*   **Output**: Multi-dimensional feature set including F0 Mean, Pitch Range, and RMS Energy.
*   **Pitch Analysis**: Fundamental Frequency (F0) tracking to capture user register.
*   **Energy/Stress**: RMS energy mapping to understand emphasis patterns.
*   **Speaking Rate**: Syllables-per-second estimation.

### 2. Emotion Intelligence (`personalization_engine.emotion_detector`)
A machine learning module that classifies the atmospheric "vibe" of the input.
*   **Input**: Mel-Frequency Cepstral Coefficients (MFCCs) extracted from input audio.
*   **Output**: Emotion Label (e.g., 'happy', 'neutral') and Confidence Score (0.0-1.0).
*   **Model**: SVM Classifier (`svm_model.pkl`).
*   **Dynamic Response**: Detects 7 core emotions including Happy, Sad, Angry, and Calm.

### 3. Qwen Adapter (`personalization_engine.qwen_adapter`)
The heart of the new system, providing a high-level interface for complex model interactions.
*   **Input**: Target Text (UTF-8), Reference Audio (WAV), and Model Configuration.
*   **Output**: High-fidelity synthesized waveform (WAV).
*   **Zero-Shot Cloning**: Unlike legacy systems, Qwen does not require fine-tuning; it uses **In-Context Learning (ICL)** to mimic a voice from just 5 seconds of audio.
*   **Windows Optimization**: Handles device placement (CPU/CUDA) gracefully on Windows machines.
*   **Tokenization**: Integrated `Qwen3TTSTokenizer` for advanced linguistic processing.

### 4. Interactive Interface (`run_qwen_interactive.py`)
A comprehensive command-line UI built with `Rich`.
*   **Iterative Workflow**: Maintains the model in memory to allow rapid text-to-speech loops without reloading weights.
*   **Auto-Play**: Integrated sounddevice playback for immediate feedback.

## Data Flow

### The "Clone & Fingerprint" Loop
1.  **Capture**: User records 5 seconds of reference audio.
2.  **Extract**: Pitch, Energy, and Emotion are computed and cached in RAM.
3.  **Load**: Qwen3-TTS weights are loaded into GPU/RAM.
4.  **Synthesize**: The text is combined with the reference audio using **x\_vector\_only\_mode** for rapid zero-shot transfer.
5.  **Persist**: The system writes a synchronized pair:
    *   `cloned_TIMESTAMP.wav`: The actual voice.
    *   `cloned_TIMESTAMP.json`: The technical specs (Pitch, Emotion, Stress).

## Scaling & Implementation
*   **Model Switching**: Supports 0.6B Base for speed/low-RAM and 1.7B VoiceDesign for high-fidelity clones.
*   **Platform Independence**: The code is designed to run on standard Windows hardware without specialized macOS chips.
