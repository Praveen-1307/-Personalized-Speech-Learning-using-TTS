
import os
import sys
import json
import logging
import subprocess
import requests
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class SynthesisAdapter:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.piper_config = self.config.get('piper_tts', {})
        self.model_dir = Path(self.piper_config.get('model_path', './models/piper'))
        self.base_model_name = "en_US-lessac-medium"
        
        # Base stats for the default Lessac model (approximate)
        self.base_stats = {
            'f0_mean': 210.0, # Female voice avg
            'wpm': 155.0      # Approx speaking rate
        }
        
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        """Download base Piper model if missing."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = self.model_dir / f"{self.base_model_name}.onnx"
        json_path = self.model_dir / f"{self.base_model_name}.onnx.json"
        
        if not onnx_path.exists() or not json_path.exists():
            logger.info(f"Downloading base model: {self.base_model_name}...")
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"
            
            try:
                self._download_file(f"{base_url}/{self.base_model_name}.onnx", onnx_path)
                self._download_file(f"{base_url}/{self.base_model_name}.onnx.json", json_path)
                logger.info("Base model downloaded.")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")

    def _download_file(self, url: str, path: Path):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise

    def adapt(self, text: str, profile: Dict) -> Dict:
        """
        Synthesize text and adapt it to the profile.
        Returns dictionary with 'audio_path' and 'metadata'.
        """
        # 1. Determine Piper Parameters (Speed/Length Scale)
        target_wpm = profile.get('features', {}).get('speaking_pattern', {}).get('words_per_minute', self.base_stats['wpm'])
        # Piper Uses length_scale: higher = slower. 
        # If target is faster (higher WPM), scale should be < 1.
        length_scale = self.base_stats['wpm'] / max(target_wpm, 50.0) 
        
        # 2. Run Piper Synthesis
        temp_output = self.model_dir / "temp_synthesis.wav"
        model_path = self.model_dir / f"{self.base_model_name}.onnx"
        
        # Simple text input mode (piper reads from stdin)
        try:
            process = subprocess.Popen(
                ['piper', '--model', str(model_path), '--output_file', str(temp_output), '--length_scale', str(length_scale)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(input=text.encode('utf-8'))
            
            if process.returncode != 0:
                logger.error(f"Piper error: {stderr.decode()}")
                raise RuntimeError(f"Piper synthesis failed: {stderr.decode()}")
                
        except FileNotFoundError:
            raise RuntimeError("Piper executable not found in PATH")

        if not temp_output.exists():
            raise RuntimeError("Output file not created by Piper")

        # 3. Post-Processing Adaptation (Pitch Shift)
        final_output = temp_output # Default if no processing
        
        target_f0 = profile.get('features', {}).get('prosodic', {}).get('f0_mean')
        if target_f0:
            final_output = self._apply_dsp_adaptation(temp_output, target_f0)

        return {
            'audio_path': str(final_output),
            'metadata': {
                'base_model': self.base_model_name,
                'length_scale_applied': length_scale,
                'target_wpm': target_wpm,
                'target_f0': target_f0
            }
        }

    def _apply_dsp_adaptation(self, audio_path: Path, target_f0: float) -> Path:
        """Apply pitch shifting using librosa."""
        try:
            y, sr = librosa.load(str(audio_path), sr=None)
            
            # Simple shift: semitones = 12 * log2(target / base)
            # Avoid divide by zero
            base_f0 = max(self.base_stats['f0_mean'], 1.0)
            target_f0 = max(target_f0, 1.0)
            
            n_steps = 12 * np.log2(target_f0 / base_f0)
            
            # Limit shift to avoid extreme artifacts
            n_steps = max(-12, min(12, n_steps))
            
            if abs(n_steps) > 0.5:
                logger.info(f"Shifting pitch by {n_steps:.2f} semitones")
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                
                output_path = audio_path.parent / f"adapted_{audio_path.name}"
                sf.write(str(output_path), y_shifted, sr)
                return output_path
                
        except Exception as e:
            logger.warning(f"DSP adaptation failed: {e}. Returning original.")
        
        return audio_path
