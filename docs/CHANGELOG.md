# Changelog

All notable changes to the TTS Personalization Engine project are documented here.

## [2.0.0] - 2026-02-10 (Current)

### Added
- **Qwen3-TTS Core Integration**: Transitioned from legacy Piper to state-of-the-art Qwen3-TTS Large Language Model for voice cloning.
- **Zero-Shot Voice Cloning**: Enabled cloning of any voice from a 5-second reference without fine-tuning.
- **Windows Adaptive Wrapper**: Implemented `qwen_adapter.py` to provide MLX-style functions (`load_model`, `generate_audio`) on Windows/PyTorch environments.
- **JSON Voice Fingerprinting**: Added automatic generation of `.json` files for every synthesized audio, containing pitch, emotion, and stress metadata.
- **Iterative Live Interface**: Created `run_qwen_interactive.py` for seamless voice capture and repeat synthesis loops.
- **Model Stability Fallback**: Support for 0.6B Base model for stable performance on standard hardware/RAM.

### Modified
- **Architecture**: Redesigned as a Zero-Shot Transformer pipeline instead of a statistical pattern learner.
- **Dependencies**: Upgraded `transformers` to `4.57.3` and added `qwen-tts`.
- **Documentation**: Overhauled `README.md`, `ARCHITECTURE.md`, `OPERATIONS.md`, and `LOGS.md` to reflect the Qwen-first focus.

### Fixed
- **Emotion Index Error**: Resolved "index out of range" crash in the emotion detector when model labels returned string formats.
- **Parameter Conflict**: Fixed `generate_voice_clone` keyword argument errors.
- **Windows Stability**: Fixed system crashes associated with the 1.7B model by optimizing 0.6B as the default stable choice.

## [1.0.0] - 2026-02-03 (Legacy)

### Added
- Initial Piper TTS integration.
- SVM-based Emotion Classification.
- Feature extraction for pitch and speaking rate.
- Profile management system (JSON/YAML).
- CLI for profile training.
