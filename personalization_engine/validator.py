
import os
import re
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SynthesisValidator:
    """
    Validates input text and generated audio for TTS synthesis.
    """
    
    def __init__(self, target_sr=22050):
        self.target_sr = target_sr
        
    def validate_text(self, text: str) -> Dict[str, Any]:
        """
        Validates the input text.
        """
        errors = []
        warnings = []
        
        if not text or not text.strip():
            errors.append("Input text is empty.")
            return {"valid": False, "errors": errors, "metrics": {}}
            
        # Basic metrics
        word_count = len(text.split())
        char_count = len(text)
        
        if char_count < 2:
            warnings.append("Input text is extremely short.")
            
        if char_count > 1000:
            warnings.append("Input text is very long; may lead to synthesis artifacts.")
            
        # Check for invalid characters (simplified)
        # Assuming only standard punctuation and alphanumeric are allowed for base models
        invalid_chars = re.findall(r'[^\w\s\d.,!?\'"-]', text)
        if invalid_chars:
            warnings.append(f"Unexpected characters detected: {set(invalid_chars)}")
            
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "word_count": word_count,
                "character_count": char_count
            }
        }
        
    def validate_audio(self, audio_path: str, original_text: str = None) -> Dict[str, Any]:
        """
        Validates the generated audio file.
        """
        errors = []
        warnings = []
        
        if not os.path.exists(audio_path):
            errors.append(f"Audio file not found at: {audio_path}")
            return {"valid": False, "errors": errors, "metrics": {}}
            
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            # 1. Basic Audio Checks
            if duration <= 0:
                errors.append("Audio duration is zero.")
            
            max_amp = np.max(np.abs(audio))
            if max_amp < 0.01:
                errors.append("Audio is nearly silent.")
            elif max_amp > 0.99:
                warnings.append("Audio is clipping (peak at 1.0).")
                
            # 2. Heuristic "Word Match" Metric
            # Without STT, we compare expected duration vs actual duration
            # Average speaking rate is ~2.5 words per second
            word_match_metrics = {}
            if original_text:
                words = original_text.split()
                word_count = len(words)
                expected_duration = word_count / 2.5 # approximation
                
                # If duration is 2x off, something is wrong
                if duration < expected_duration * 0.4:
                    warnings.append("Audio is significantly shorter than expected for the given text.")
                elif duration > expected_duration * 2.5:
                    warnings.append("Audio is significantly longer than expected (potential halluncinations/repetition).")
                
                # Placeholder for actual word matching (e.g. via Cross-Correlation or STT)
                # For now, we simulate a 'match' score based on duration and energy peaks
                # In a real system, this would use whispered/ASR transcription
                peaks = librosa.effects.trim(audio, top_db=20)[0]
                peak_duration = librosa.get_duration(y=peaks, sr=sr)
                
                # Simulating word match: 
                # We'll assume a "match" if the audio has enough active segments for the words
                active_segments = librosa.effects.split(audio, top_db=30)
                num_segments = len(active_segments)
                
                # Rough heuristic: num segments usually correlates with words/phrases
                match_confidence = min(1.0, num_segments / max(1, word_count))
                
                word_match_metrics = {
                    "expected_words": word_count,
                    "estimated_words_detected": num_segments, # rough proxy
                    "match_score": match_confidence,
                    "words_matched_count": int(word_count * match_confidence) # Simulated
                }
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "metrics": {
                    "duration_seconds": duration,
                    "sample_rate": sr,
                    "max_amplitude": float(max_amp),
                    **word_match_metrics
                }
            }
            
        except Exception as e:
            errors.append(f"Failed to process audio: {str(e)}")
            return {"valid": False, "errors": errors, "metrics": {}}

def generate_report(text: str, audio_path: str, output_path: str = "validation_report.json"):
    """
    Generates a comprehensive validation report.
    """
    validator = SynthesisValidator()
    
    text_res = validator.validate_text(text)
    audio_res = validator.validate_audio(audio_path, text)
    
    report = {
        "summary": {
            "overall_status": "PASS" if text_res['valid'] and audio_res['valid'] else "FAIL",
            "timestamp": os.path.getmtime(audio_path) if os.path.exists(audio_path) else None
        },
        "input_validation": text_res,
        "output_validation": audio_res
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    return report
