# Personalized TTS Engine 

A comprehensive Python-based personalization engine that extends Qwen3-TTS. It analyzes user audio to extract speaking patterns, pitch, and emotion, creating a personalized voice profile for synthesis.

##  Key Features

1.  **Qwen3-TTS Voice Cloning**: Integrated the Qwen3-TTS 0.6B engine for zero-shot voice cloning.
2.  **Iterative Synthesis**: Record your voice once and generate multiple speech segments without re-recording.
3.  **Real-Time Voice Profiling**: Automatically extracts Pitch (F0), Stress (RMS Energy), and Emotion from recorded samples.
4.  **JSON Metadata Generation**: Every generated audio file is accompanied by a matching `.json` file containing the source voice's characteristics.
5.  **Real-time Monitoring**: Centralized logging system with beautiful console output (via Rich) and persistent file logging.
6.  **Windows Optimized**: Custom PyTorch-based adapter for Windows environments (alternative to MLX).

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

### 3. Monitoring & Logs
The system now features real-time logging. To view the persistent history:
```bash
# View the application log
cat logs/app.log
```
Logs are also displayed in real-time in the terminal with color coding.


##  Project Structure

*   `personalization_engine/`: Core Python source code.
    *   `feature_extractor.py`: Librosa-based audio analysis.
    *   `pattern_learner.py`: Statistical profiling (GMM).
    *   `emotion_detector.py`: SVM Classifier for native emotion recognition.
*   `profiles/`: Stored user profiles (`.json`, `.pkl`).
*   `synthesize_voice.py`: Main synthesis entry point.

##  Documentation
*   [Dataset Analysis](docs/DATASET_ANALYSIS.md)
*   [System Architecture](docs/ARCHITECTURE.md)
*   [Operational & Logging Guide](docs/OPERATIONS.md)
*   [Logging Specification](docs/LOGS.md)
*   [Changelog](docs/CHANGELOG.md)


