# Logging & Metadata Specification

## Overview
The personalization engine implements a multi-tier logging system:
1.  **Rich Console Stream**: Real-time, color-coded diagnostic logs for immediate feedback.
2.  **Persistent File Log**: Holistic record of all session activity saved to `logs/app.log`.
3.  **Product Metadata (JSON)**: Analytical record of specific voice characteristics for every clone.

## 1. Centralized Logging System
The system uses a custom logging module (`personalization_engine/logger.py`) to ensure consistency across all components.

### Console Output (Rich)
*   **Levels**: INFO for standard operations, DEBUG for technical details, WARNING/ERROR for issues.
*   **Features**: Includes timestamps, stylized levels, and automatic traceback formatting for crashes.

### File Logging (Rotating)
*   **Path**: `logs/app.log`
*   **Retention**: Keeps up to 5 historical copies, each capped at 10MB.
*   **Verbosity**: Always captures DEBUG level information for comprehensive troubleshooting.

**Example Layout (`cloned_1770739260.json`):**
```json
{
    "text": "The quick brown fox...",
    "timestamp": 1770739260,
    "reference_voice_analysis": {
        "pitch": 185.4,
        "pitch_range": [120.5, 250.1],
        "stress_energy": 0.045,
        "emotion": "happy",
        "emotion_confidence": 0.89,
        "speaking_pattern": {
            "wpm": 145,
            "pause_entropy": 0.76
        }
    },
    "model_used": "Qwen3-TTS-0.6B-Base"
}
```

## 2. System Execution Logs

### Model Initialization
Logged during `load_model()` in `qwen_adapter.py`:
```log
INFO: Loading Qwen TTS model: Qwen/Qwen3-TTS-12Hz-0.6B-Base on cuda
INFO: Transformers version: 4.57.3
INFO: Qwen TTS model loaded successfully via adapter.
```

### Synthesis Workflow
Logged during `generate_audio()`:
```log
INFO: Generating audio for text: Hello World
INFO: Analysis: Extracted F0 Pitch (Mean: 122Hz)
INFO: Analysis: Emotion Detected: Neutral (92% Confidence)
INFO: Saved audio to output/cloned_123.wav
INFO: Saved metadata to output/cloned_123.json
```

## 3. Monitoring Metrics
The system now implements automated execution tracking for all high-computation functions.

### Detailed Execution Logs (`[ExecutionStart]` / `[ExecutionComplete]`)
Every major function in the `personalization_engine` is decorated to log:
*   **Latency**: Total time taken for the function to complete.
*   **Memory Delta**: RSS memory change during function execution.
*   **Object Metadata**: Sizes, shapes, and lengths of returned objects.

### Object Data & Metadata
*   **Feature Vectors**: Dimensions of extracted F0 and Energy contours are logged.
*   **Model Parameters**: Transformer parameter counts are logged during initialization (e.g., `Loaded Qwen/Qwen3-TTS ... | Parameters: 620,000,000`).
*   **Input Context**: Input audio duration, sample rate, and text length are tracked for every request.

### System Health & Hardware
*   **Real-Time Hardware Tracking**: `[Metrics] [MemoryFootprint]` entries show CPU% and RAM usage (MB) at high-frequency checkpoints.
*   **VRAM**: GPU utilization is monitored via `torch.cuda` when active.

### Complexity Observations (`[Complexity]`)
The system logs theoretical time and space complexity markers for transparency:
*   **Feature Extraction**: `O(N log N)` (via FFT/Spectral analysis).
*   **Qwen Inference**: `O(Tokens * Model_Size)`.
*   **Profile Learning**: `O(Iterations * K * N * D)` for GMM-based pattern learning.

## 4. Error Logging Table

| Logger Tag | Message | Severity | Meaning |
| :--- | :--- | :--- | :--- |
| `qwen_adapter` | `qwen-tts not installed` | ERROR | environment/dependency failure. |
| `qwen_adapter` | `Ref audio not found` | WARNING | Reference file missing; using fallback. |
| `emotion_detector` | `index out of range` | ERROR | Model label mismatch (Fixed in v2). |
| `feature_extractor`| `No voiced segments` | WARNING | Input audio is silent or too noisy. |

## 5. Log Analysis
To check the average pitch of all generated clones:
```powershell
# Extract pitch from all JSON fingerprints
Get-Content output/*.json | ConvertFrom-Json | Select-Object -ExpandProperty reference_voice_analysis | Measure-Object -Property pitch -Average
```
