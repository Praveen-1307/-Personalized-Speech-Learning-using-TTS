import librosa
import numpy as np
import soundfile as sf
from typing import Optional, Tuple
import logging
import webrtcvad
from pydub import AudioSegment, effects
import noisereduce as nr
import time

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(self, target_sr: int = 22050, min_duration: float = 5.0):
        self.target_sr = target_sr
        self.min_duration = min_duration
        self.vad = webrtcvad.Vad(2)  # Aggressiveness mode 2
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and validate audio file."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            if duration < self.min_duration:
                logger.warning(f"Audio too short: {duration:.2f}s (min: {self.min_duration}s)")
                
            logger.info(f"Loaded audio: {duration:.2f}s, {sr}Hz, {audio.shape}")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise
            
    def remove_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction."""
        try:
            # Use spectral gating for noise reduction
            audio_clean = nr.reduce_noise(y=audio, sr=sr)
            logger.info("Noise reduction applied")
            return audio_clean
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
            
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target dB level."""
        # Convert to AudioSegment for normalization
        audio_normalized = librosa.util.normalize(audio)
        # Target -3 dB
        rms = np.sqrt(np.mean(audio_normalized**2))
        target_rms = 10**(-3/20)  # -3 dB
        audio_normalized = audio_normalized * (target_rms / rms)
        
        logger.info(f"Audio normalized to {-3}dB")
        return audio_normalized
        
    def remove_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove leading/trailing silence."""
        # Use voice activity detection
        audio_no_silence = librosa.effects.trim(audio, top_db=30)
        logger.info(f"Silence removed: {len(audio)} -> {len(audio_no_silence[0])} samples")
        return audio_no_silence[0]
        
    def calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        signal_power = np.mean(audio**2)
        noise = audio - librosa.effects.preemphasis(audio)
        noise_power = np.mean(noise**2)
        
        if noise_power == 0:
            return float('inf')
            
        snr = 10 * np.log10(signal_power / noise_power)
        logger.info(f"SNR calculated: {snr:.2f} dB")
        return snr
        
    def segment_audio(self, audio: np.ndarray, sr: int, 
                     segment_duration: float = 3.0) -> list:
        """Segment audio into chunks for processing."""
        segment_samples = int(segment_duration * sr)
        segments = []
        
        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i + segment_samples]
            if len(segment) > sr * 0.5:  # At least 0.5 second
                segments.append(segment)
                
        logger.info(f"Audio segmented into {len(segments)} parts")
        return segments
        
    def preprocess(self, audio_path: str) -> dict:
        """Complete preprocessing pipeline."""
        start_time = time.time()
        
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Calculate original metrics
        original_snr = self.calculate_snr(audio)
        original_duration = len(audio) / sr
        
        # Apply preprocessing steps
        audio = self.remove_noise(audio, sr)
        audio = self.normalize_audio(audio)
        audio = self.remove_silence(audio, sr)
        
        # Calculate final metrics
        final_snr = self.calculate_snr(audio)
        final_duration = len(audio) / sr
        
        processing_time = time.time() - start_time
        
        # Log results
        logger.info({
            'original_snr': original_snr,
            'final_snr': final_snr,
            'original_duration': original_duration,
            'final_duration': final_duration,
            'processing_time': processing_time,
            'samples_removed': (original_duration - final_duration)
        })
        
        return {
            'audio': audio,
            'sample_rate': sr,
            'snr': final_snr,
            'duration': final_duration,
            'processing_time': processing_time
        }
