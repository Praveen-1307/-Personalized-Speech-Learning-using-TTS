import pytest
import numpy as np
from personalization_engine.feature_extractor import FeatureExtractor, SpeakingPattern

class TestFeatureExtractor:
    
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor(sr=22050)
        
    @pytest.fixture
    def sample_audio(self):
        # Generate synthetic speech-like audio
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Simulate pitch variation (like speech)
        f0 = 100 + 50 * np.sin(2 * np.pi * 2 * t)  # Varying pitch
        audio = 0.3 * np.sin(2 * np.pi * f0 * t)
        
        # Add some "pauses"
        audio[int(1.0*sr):int(1.2*sr)] = 0
        audio[int(2.0*sr):int(2.1*sr)] = 0
        
        return audio
        
    def test_extract_f0(self, extractor, sample_audio):
        f0 = extractor.extract_f0(sample_audio)
        
        assert len(f0) > 0
        assert np.all(f0 > 0)  # All pitches should be positive
        
        # Mean pitch should be around 100 Hz
        mean_pitch = np.mean(f0)
        assert 80 < mean_pitch < 120
        
    def test_extract_energy(self, extractor, sample_audio):
        energy = extractor.extract_energy(sample_audio)
        
        assert len(energy) > 0
        assert energy.shape[0] > 10  # Should have multiple frames
        
    def test_detect_silences(self, extractor, sample_audio):
        silences = extractor.detect_silences(sample_audio)
        
        # Should detect the two pauses we added
        assert len(silences) >= 2
        
        # Check silence durations
        for _, _, duration in silences:
            assert duration > 0.05  # Should be at least 50ms
            
    def test_analyze_speaking_pattern(self, extractor, sample_audio):
        pattern = extractor.analyze_speaking_pattern(sample_audio)
        
        assert isinstance(pattern, SpeakingPattern)
        assert pattern.words_per_minute > 0
        assert pattern.syllables_per_second > 0
        assert len(pattern.pause_durations) >= 2
        
    def test_extract_emotion_features(self, extractor, sample_audio):
        features = extractor.extract_emotion_features(sample_audio)
        
        assert 'pitch_mean' in features
        assert 'pitch_std' in features
        assert 'energy_mean' in features
        assert 'speaking_rate' in features
        assert 'mfccs' in features
        
        assert isinstance(features['mfccs'], list)
        assert len(features['mfccs']) == 5
        
    def test_extract_all_features(self, extractor, sample_audio):
        features = extractor.extract_all_features(sample_audio)
        
        assert 'prosodic' in features
        assert 'speaking_pattern' in features
        assert 'emotion' in features
        assert 'spectral' in features
        assert 'metadata' in features
        
        # Check prosodic features
        prosodic = features['prosodic']
        assert 'f0_mean' in prosodic
        assert 'f0_std' in prosodic
        assert 'energy_mean' in prosodic
        
        # Check metadata
        metadata = features['metadata']
        assert 'processing_time' in metadata
        assert metadata['audio_duration'] == pytest.approx(3.0, rel=0.1)
