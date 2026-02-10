import numpy as np
import librosa
import pyworld as pw
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json
from scipy import stats
import matplotlib.pyplot as plt
import time

logger = logging.getLogger(__name__)

@dataclass
class SpeakingPattern:
    words_per_minute: float
    syllables_per_second: float
    pause_durations: List[float]
    inter_word_delays: List[float]
    rhythm_entropy: float
    
@dataclass
class ProsodicFeatures:
    pitch_contours: np.ndarray
    pitch_range: Tuple[float, float]
    mean_pitch: float
    energy_contours: np.ndarray
    stress_patterns: List[float]
    intonation_patterns: List[str]

class FeatureExtractor:
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
        
    def extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract fundamental frequency using WORLD."""
        try:
            # Use WORLD for accurate pitch extraction
            f0, timeaxis = pw.dio(audio.astype(np.float64), self.sr)
            f0 = pw.stonemask(audio.astype(np.float64), f0, timeaxis, self.sr)
            
            # Remove unvoiced frames
            f0 = f0[f0 > 0]
            
            logger.info(f"F0 extracted: {len(f0)} frames, range: {f0.min():.1f}-{f0.max():.1f} Hz")
            return f0
            
        except Exception as e:
            logger.error(f"F0 extraction failed: {e}")
            # Fallback to librosa
            f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'),
                                   fmax=librosa.note_to_hz('C7'), sr=self.sr)
            f0 = f0[~np.isnan(f0)]
            return f0
            
    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract energy contours."""
        energy = librosa.feature.rms(y=audio, frame_length=self.frame_length,
                                    hop_length=self.hop_length)[0]
        logger.info(f"Energy extracted: {len(energy)} frames")
        return energy
        
    def detect_silences(self, audio: np.ndarray, threshold: float = 0.01) -> List[Tuple[int, int]]:
        """Detect silent regions in audio."""
        non_silent = librosa.effects.split(audio, top_db=30)
        silences = []
        
        for i in range(1, len(non_silent)):
            silence_start = non_silent[i-1][1]
            silence_end = non_silent[i][0]
            duration = (silence_end - silence_start) / self.sr
            if duration > 0.05:  # Only consider silences > 50ms
                silences.append((silence_start, silence_end, duration))
                
        logger.info(f"Detected {len(silences)} silence regions")
        return silences
        
    def analyze_speaking_pattern(self, audio: np.ndarray, 
                                transcript: Optional[str] = None) -> SpeakingPattern:
        """Analyze speaking patterns and rhythm."""
        # Detect pauses
        silences = self.detect_silences(audio)
        pause_durations = [s[2] for s in silences]
        
        # Calculate speaking rate (approximate)
        # This would be more accurate with proper speech recognition
        words_per_minute = 150  # Default, should be calculated from transcript
        
        # Calculate rhythm entropy
        if len(pause_durations) > 1:
            rhythm_entropy = stats.entropy(np.histogram(pause_durations, bins=10)[0])
        else:
            rhythm_entropy = 0
            
        # Estimate inter-word delays (simplified)
        inter_word_delays = pause_durations if pause_durations else [0.3]
        
        pattern = SpeakingPattern(
            words_per_minute=words_per_minute,
            syllables_per_second=words_per_minute / 60 * 1.5,  # Approximate
            pause_durations=pause_durations,
            inter_word_delays=inter_word_delays,
            rhythm_entropy=rhythm_entropy
        )
        
        logger.info(f"Speaking pattern analyzed: {pattern}")
        return pattern
        
    def extract_emotion_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract features related to emotion."""
        f0 = self.extract_f0(audio)
        energy = self.extract_energy(audio)
        
        # Pitch statistics
        pitch_mean = np.mean(f0) if len(f0) > 0 else 0
        pitch_std = np.std(f0) if len(f0) > 0 else 0
        pitch_range = np.ptp(f0) if len(f0) > 0 else 0
        
        # Energy statistics
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        
        # Speaking rate approximation
        speaking_rate = self.estimate_speaking_rate(audio)
        
        # MFCCs for spectral features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        emotion_features = {
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'pitch_range': float(pitch_range),
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            'speaking_rate': float(speaking_rate),
            'mfccs': mfcc_mean.tolist()[:5]  # First 5 coefficients
        }
        
        logger.info(f"Emotion features extracted: {list(emotion_features.keys())}")
        return emotion_features
        
    def estimate_speaking_rate(self, audio: np.ndarray) -> float:
        """Estimate speaking rate in syllables per second."""
        # Simplified approach - in production, use ASR
        # Calculate based on voiced segments
        voiced_segments = self.detect_voiced_segments(audio)
        total_voiced_time = sum([v[1] - v[0] for v in voiced_segments]) / self.sr
        
        if total_voiced_time > 0:
            # Approximate: 5 syllables per second of voiced speech
            return len(voiced_segments) * 5 / total_voiced_time
        return 4.0  # Default
        
    def detect_voiced_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Detect voiced segments in audio."""
        f0 = self.extract_f0(audio)
        # This is simplified - actual implementation would need frame alignment
        return [(0, len(audio))]  # Placeholder
        
    def extract_all_features(self, audio: np.ndarray) -> Dict:
        """Extract all features for voice profile."""
        start_time = time.time()
        
        # Extract prosodic features
        f0 = self.extract_f0(audio)
        energy = self.extract_energy(audio)
        
        # Analyze speaking patterns
        speaking_pattern = self.analyze_speaking_pattern(audio)
        
        # Extract emotion features
        emotion_features = self.extract_emotion_features(audio)
        
        # Extract spectral features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=20)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        
        processing_time = time.time() - start_time
        
        features = {
            'prosodic': {
                'f0_mean': float(np.mean(f0)) if len(f0) > 0 else 0,
                'f0_std': float(np.std(f0)) if len(f0) > 0 else 0,
                'f0_range': (float(f0.min()), float(f0.max())) if len(f0) > 0 else (0, 0),
                'energy_mean': float(np.mean(energy)),
                'energy_std': float(np.std(energy))
            },
            'speaking_pattern': {
                'words_per_minute': speaking_pattern.words_per_minute,
                'syllables_per_second': speaking_pattern.syllables_per_second,
                'pause_durations': speaking_pattern.pause_durations,
                'inter_word_delays': speaking_pattern.inter_word_delays,
                'rhythm_entropy': speaking_pattern.rhythm_entropy
            },
            'emotion': emotion_features,
            'spectral': {
                'mfccs_mean': np.mean(mfccs, axis=1).tolist(),
                'spectral_centroid_mean': float(np.mean(spectral_centroid))
            },
            'metadata': {
                'processing_time': processing_time,
                'audio_duration': len(audio) / self.sr,
                'sample_rate': self.sr
            }
        }
        
        logger.info(f"All features extracted in {processing_time:.2f}s")
        return features
