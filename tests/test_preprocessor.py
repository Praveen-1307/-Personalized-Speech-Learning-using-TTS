import pytest
import numpy as np
import tempfile
import soundfile as sf
from personalization_engine.preprocessor import AudioPreprocessor

class TestAudioPreprocessor:
    
    @pytest.fixture
    def preprocessor(self):
        return AudioPreprocessor(target_sr=22050, min_duration=1.0)
        
    @pytest.fixture
    def sample_audio(self):
        # Generate 2 seconds of synthetic audio
        sr = 22050
        t = np.linspace(0, 2, sr * 2)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        return audio, sr
        
    def test_load_audio(self, preprocessor, sample_audio):
        audio, sr = sample_audio
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            
            # Test loading
            loaded_audio, loaded_sr = preprocessor.load_audio(f.name)
            
            assert loaded_sr == preprocessor.target_sr
            assert len(loaded_audio) > 0
            assert abs(len(loaded_audio)/loaded_sr - 2.0) < 0.1
            
    def test_normalize_audio(self, preprocessor, sample_audio):
        audio, _ = sample_audio
        
        # Create audio with low volume
        quiet_audio = audio * 0.1
        
        normalized = preprocessor.normalize_audio(quiet_audio)
        
        # Check that volume increased
        assert np.max(np.abs(normalized)) > np.max(np.abs(quiet_audio))
        # Check that it's not clipped
        assert np.max(np.abs(normalized)) <= 1.0
        
    def test_calculate_snr(self, preprocessor, sample_audio):
        audio, _ = sample_audio
        
        # Clean audio should have high SNR
        snr_clean = preprocessor.calculate_snr(audio)
        assert snr_clean > 20  # dB
        
        # Add noise
        noisy_audio = audio + 0.1 * np.random.randn(len(audio))
        snr_noisy = preprocessor.calculate_snr(noisy_audio)
        
        assert snr_noisy < snr_clean
        
    def test_segment_audio(self, preprocessor, sample_audio):
        audio, sr = sample_audio
        
        segments = preprocessor.segment_audio(audio, sr, segment_duration=1.0)
        
        assert len(segments) == 2  # 2-second audio into 1-second segments
        assert len(segments[0]) == sr  # 1 second of samples
        
    def test_full_preprocessing(self, preprocessor):
        # Create test audio with silence at beginning and end
        sr = 22050
        t = np.linspace(0, 3, sr * 3)
        audio = np.concatenate([
            np.zeros(int(0.5 * sr)),  # 0.5s silence
            0.5 * np.sin(2 * np.pi * 440 * t[:int(2 * sr)]),  # 2s tone
            np.zeros(int(0.5 * sr))   # 0.5s silence
        ])
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            
            result = preprocessor.preprocess(f.name)
            
            assert 'audio' in result
            assert 'sample_rate' in result
            assert 'snr' in result
            assert 'duration' in result
            
            # Check that silence was removed (approximately)
            assert result['duration'] < 3.0
