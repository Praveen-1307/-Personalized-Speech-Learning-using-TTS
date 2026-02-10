# Operational Guide: Qwen3-TTS Voice Cloning

This guide details how to operate, monitor, and troubleshoot the Qwen3-TTS enabled personalized voice cloning system.

## 1. Running the System

### Primary Entry Point
The main way to interact with the engine is via the interactive script:
```powershell
python run_qwen_interactive.py
```

### Operational Workflow
1.  **Reference Capture**: You will be prompted to record 5 seconds of audio. This audio is saved to `profiles/current_session_user.wav`.
2.  **Model Loading**: The script initializes the Qwen3-TTS engine. (Note: The first run requires a ~1.2GB download from HuggingFace).
3.  **Synthesis Loop**: You can type text iteratively. Each text input generates a new audio file using the *same* recorded voice from step 1.
4.  **Re-recording**: Type `r` during the loop to update the reference voice without restarting the program.

## 2. Output Management

The system generates paired data files for every synthesis in the `output/` directory:

| Extension | Content | Purpose |
| :--- | :--- | :--- |
| **.wav** | Synthesized Audio | The actual cloned speech. |
| **.json** | Voice Fingerprint | Metadata including extracted Pitch, Emotion, and Stress Energy. |

### Monitoring Metadata
Open the `.json` files to verify system accuracy:
*   `pitch`: Average frequency in Hz.
*   `emotion`: Predicted emotional state of the speaker.
*   `stress_energy`: Volume/Emphasis profile.

## 3. Configuration

Key variables in `config.yaml` or `run_qwen_interactive.py`:
*   `model_id`: Defaults to `Qwen/Qwen3-TTS-12Hz-0.6B-Base` for Windows stability.
*   `use_gpu`: Uses CUDA if available, falls back to CPU automatically.
*   `duration`: Recording length for the reference voice (default: 5s).

## 4. Troubleshooting Qwen Integration

| Issue | Potential Cause | Resolution |
| :--- | :--- | :--- |
| **"System Crashed during loading"** | RAM/VRAM Exhaustion | Ensure you are using the **0.6B** model instead of 1.7B. |
| **"JSON file missing data"** | Emotion Model Error | Ensure `models/emotion/svm_model.pkl` exists; restart the script to re-init analyzers. |
| **"Audio contains noise"** | Mic Quality | Record in a quiet room; use a higher quality mic for better zero-shot cloning. |
| **"ImportError: transformers"** | Version Mismatch | Run `pip install transformers>=4.57.3`. |

## 5. Performance Metrics

The interactive console provides real-time feedback:
*   **Loading Status**: Spinner indicating model weights transfer.
*   **Synthesis Time**: Displayed after each generation.
*   **JSON Link**: Direct path to the metadata fingerprint.

## 6. Cleanup
To clear old results without breaking the system:
```powershell
# Safe to delete cloned files
del output/cloned_*.*
```
*Note: Do not delete `profiles/current_session_user.wav` if you wish to reuse your previous voice session.*
