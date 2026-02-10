# Personalized TTS Engine (Piper Integration)

A comprehensive Python-based personalization engine that extends Piper TTS. It analyzes user audio to extract speaking patterns, pitch, and emotion, creating a personalized voice profile for synthesis.

##  Key Features

1.  **Qwen3-TTS Voice Cloning**: Integrated the Qwen3-TTS 0.6B engine for zero-shot voice cloning.
2.  **Iterative Synthesis**: Record your voice once and generate multiple speech segments without re-recording.
3.  **Real-Time Voice Profiling**: Automatically extracts Pitch (F0), Stress (RMS Energy), and Emotion from recorded samples.
4.  **JSON Metadata Generation**: Every generated audio file is accompanied by a matching `.json` file containing the source voice's characteristics.
5.  **Windows Optimized**: Custom PyTorch-based adapter for Windows environments (alternative to MLX).

##  Quick Start

### 1. Installation
```bash
# Install core dependencies
pip install -r requirements.txt
# Install TTS-specific libraries
pip install transformers torch torchaudio sounddevice soundfile qwen-tts
```

### 2. Interactive Voice Cloning (Recommended)
This is the main entry point for the modern cloning system.
```bash
python run_qwen_interactive.py
```
**Process:**
1.  **Record**: Capture 5 seconds of your voice naturally.
2.  **Analyze**: The system extracts your profile (Pitch, Stress, Emotion).
3.  **Clone**: Enter text repeatedly to hear it in your voice.
4.  **Save**: Outputs and matching metadata are saved to the `output/` folder.

##  Legacy Piper Personalization Workflow
*(Original Piper-based engine for pattern learning)*

#### **Step A: Train a Voice Profile**
Analyze a folder of `.wav` files to create a profile.
```bash
# Example: Training on LJSpeech subset
python -m personalization_engine.cli train --audio-dir samples/ljspeech_subset --user-id my_voice
```

**Training Details:**
*   **Input**: A folder containing clean `.wav` audio files (mono, 22050Hz recommended).
*   **Command**: `python -m personalization_engine.cli train ...`
*   **Output**:
    1.  **Console**: A Rich-formatted summary of extracted features (Pitch, Speed, Emotion).
    2.  **File Storage**:
        *   **JSON Profiles**: Saved to `./profiles/{user_id}_profile.json` (Portable, human-readable).
        *   **YAML Profiles**: Saved to `./profiles/{user_id}_profile.yaml` (Alternative format).

#### **Step B: Inspect the Profile (CLI & JSON)**
View the extracted features (Pitch, Tone, Speed) in the terminal.
```bash
# View details for a specific profile (e.g., ljspeech_with_emotion)
python -m personalization_engine.cli list-profiles --user-id ljspeech_with_emotion
```

**Sample Output (JSON Structure):**
```json
{
  "user_id": "ljspeech_with_emotion",
  "features": {
    "speaking_pattern": { "words_per_minute": 150.0, "rhythm_entropy": 0.91 },
    "prosodic": { "f0_mean": 203.26, "energy_mean": 0.52 },
    "emotion_detected": { "emotion": "happy", "confidence": 0.85 },
    "spectral": { "spectral_centroid_mean": 2634.83 }
  }
}
```

#### **Step C: Synthesize Speech**
Generate audio using the profile.

```bash
python -m personalization_engine.cli synthesize --text "This is my personalized voice with emotion detection." --user-id ljspeech_with_emotion --output final_result.wav
```

#### **Step D: Live Microphone Analysis**
Analyze your own voice in real-time using the microphone tool. This displays your Pitch (F0), Speaking Rate (WPM), and Emotion instantly.

```bash
# Analyze for 5 seconds (default)
python analyze_mic.py

# Analyze for 10 seconds
python analyze_mic.py --duration 10
```

##  Emotion Detection
The system includes an `EmotionDetector` module.
*   **Training**: Requires an emotion-labeled dataset (like EMO-DB).
*   **Inference**: If a model is present (`models/emotion/svm_model.pkl`), the system automatically detects the emotion of the input samples and tags the profile.
*   **Synthesis**: synthesis parameters (pitch shift, speed) are subtly adjusted based on the detected emotion tag.

##  Project Structure

*   `personalization_engine/`: Core Python source code.
    *   `feature_extractor.py`: Librosa-based audio analysis.
    *   `pattern_learner.py`: Statistical profiling (GMM).
    *   `emotion_detector.py`: SVM Classifier for native emotion recognition.
*   `profiles/`: Stored user profiles (`.json`, `.pkl`).
*   `synthesize_voice.py`: Main synthesis entry point.

##  Documentation
*   [Dataset Analysis](DATASET_ANALYSIS.md)
*   [System Architecture](ARCHITECTURE.md)
*   [Operational & Logging Guide](OPERATIONS.md)
*   [Logging Specification](LOGS.md)
*   [Changelog](CHANGELOG.md)

