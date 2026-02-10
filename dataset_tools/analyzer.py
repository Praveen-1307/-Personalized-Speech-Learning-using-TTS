import json
import yaml
from typing import Dict, List
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DatasetMetrics:
    name: str
    total_duration: float  # hours
    speaker_count: int
    sample_rate: int
    languages: List[str]
    emotion_labels: bool
    accent_diversity: str

class DatasetAnalyzer:
    DATASET_INFO = {
        "LJSpeech": {
            "total_duration": 24,
            "speaker_count": 1,
            "sample_rate": 22050,
            "languages": ["English"],
            "emotion_labels": False,
            "accent_diversity": "None (single speaker)",
            "url": "https://keithito.com/LJ-Speech-Dataset/",
            "voice_characteristics": {
                "clarity": "High",
                "naturalness": "High",
                "accent": "Neutral US English",
                "pitch_range": "Medium",
                "emotional_range": "Limited"
            }
        },
        "VCTK": {
            "total_duration": 44,
            "speaker_count": 109,
            "sample_rate": 48000,
            "languages": ["English"],
            "emotion_labels": False,
            "accent_diversity": "High (multiple UK accents)",
            "url": "https://datashare.ed.ac.uk/handle/10283/3443",
            "voice_characteristics": {
                "clarity": "Medium-High",
                "naturalness": "Medium",
                "accent": "Various UK accents",
                "pitch_range": "Wide",
                "emotional_range": "Limited"
            }
        },
        "LibriTTS": {
            "total_duration": 585,
            "speaker_count": 2456,
            "sample_rate": 24000,
            "languages": ["English"],
            "emotion_labels": False,
            "accent_diversity": "Medium",
            "url": "http://www.openslr.org/60/",
            "voice_characteristics": {
                "clarity": "Variable",
                "naturalness": "High (audiobook style)",
                "accent": "Mostly US English",
                "pitch_range": "Wide",
                "emotional_range": "Limited to narrative"
            }
        },
        "HiFi": {
            "total_duration": 292,
            "speaker_count": 10,
            "sample_rate": 44100,
            "languages": ["English"],
            "emotion_labels": False,
            "accent_diversity": "Low",
            "url": "https://github.com/coqui-ai/TTS/blob/master/TTS/tts/datasets/hifi_tts.py",
            "voice_characteristics": {
                "clarity": "Very High",
                "naturalness": "High",
                "accent": "Neutral US English",
                "pitch_range": "Medium",
                "emotional_range": "Limited"
            }
        },
        "HUI": {
            "total_duration": 100,
            "speaker_count": 5,
            "sample_rate": 48000,
            "languages": ["German"],
            "emotion_labels": True,
            "accent_diversity": "Low (German)",
            "url": "https://github.com/DigitalPhonetics/hui-audio-corpus",
            "voice_characteristics": {
                "clarity": "High",
                "naturalness": "High",
                "accent": "German",
                "pitch_range": "Wide",
                "emotional_range": "Good (emotionally labeled)"
            }
        }
    }
    
    def analyze(self, dataset_name: str) -> Dict:
        """Analyze a specific dataset."""
        if dataset_name not in self.DATASET_INFO:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        info = self.DATASET_INFO[dataset_name]
        
        # Calculate additional metrics
        avg_duration_per_speaker = info["total_duration"] * 3600 / info["speaker_count"]
        data_density = info["total_duration"] / max(1, info["speaker_count"])
        
        analysis = {
            "basic_info": info,
            "metrics": {
                "total_duration_hours": info["total_duration"],
                "speaker_count": info["speaker_count"],
                "avg_recording_per_speaker_seconds": avg_duration_per_speaker,
                "data_density_hours_per_speaker": data_density,
                "sample_rate_hz": info["sample_rate"],
                "has_emotion_labels": info["emotion_labels"]
            },
            "suitability": self._calculate_suitability(info),
            "recommended_use_cases": self._recommend_use_cases(info),
            "limitations": self._identify_limitations(info),
            "preprocessing_requirements": self._get_preprocessing_needs(info)
        }
        
        logger.info(f"Analyzed dataset: {dataset_name}")
        return analysis
        
    def _calculate_suitability(self, info: Dict) -> Dict:
        """Calculate suitability scores for different use cases."""
        scores = {}
        
        # Single speaker TTS
        scores["single_speaker"] = 100 if info["speaker_count"] == 1 else 20
        
        # Multi-speaker TTS
        scores["multi_speaker"] = min(100, info["speaker_count"] * 5)
        
        # Emotional TTS
        scores["emotional_tts"] = 90 if info["emotion_labels"] else 30
        
        # Accent diversity
        if "High" in info["accent_diversity"]:
            scores["accent_modeling"] = 85
        elif "Medium" in info["accent_diversity"]:
            scores["accent_modeling"] = 60
        else:
            scores["accent_modeling"] = 20
            
        # Overall quality
        scores["overall_quality"] = (
            scores["single_speaker"] * 0.2 +
            scores["multi_speaker"] * 0.2 +
            scores["emotional_tts"] * 0.3 +
            scores["accent_modeling"] * 0.3
        )
        
        return scores
        
    def _recommend_use_cases(self, info: Dict) -> List[str]:
        """Recommend use cases for the dataset."""
        recommendations = []
        
        if info["speaker_count"] == 1:
            recommendations.append("Single-speaker voice cloning")
            
        if info["speaker_count"] > 10:
            recommendations.append("Multi-speaker TTS training")
            
        if info["emotion_labels"]:
            recommendations.append("Emotional speech synthesis")
            
        if "High" in info["accent_diversity"]:
            recommendations.append("Accent modeling")
            
        if info["sample_rate"] >= 44100:
            recommendations.append("High-fidelity synthesis")
            
        return recommendations
        
    def _identify_limitations(self, info: Dict) -> List[str]:
        """Identify limitations of the dataset."""
        limitations = []
        
        if info["speaker_count"] == 1:
            limitations.append("Limited to single speaker")
            
        if not info["emotion_labels"]:
            limitations.append("No emotion labels")
            
        if info["total_duration"] < 10:
            limitations.append("Limited total duration")
            
        if info["sample_rate"] < 22050:
            limitations.append("Low sample rate")
            
        return limitations
        
    def _get_preprocessing_needs(self, info: Dict) -> List[str]:
        """Get preprocessing requirements."""
        needs = []
        
        needs.append("Resample to target rate (if needed)")
        needs.append("Audio normalization")
        
        if info["sample_rate"] > 22050:
            needs.append("Downsampling may be required")
            
        if not info["emotion_labels"]:
            needs.append("Emotion labeling required for emotional TTS")
            
        return needs
        
    def compare_datasets(self, dataset_names: List[str]) -> Dict:
        """Compare multiple datasets side by side."""
        comparison = {}
        
        for name in dataset_names:
            if name in self.DATASET_INFO:
                analysis = self.analyze(name)
                comparison[name] = {
                    "metrics": analysis["metrics"],
                    "suitability": analysis["suitability"],
                    "voice_characteristics": self.DATASET_INFO[name]["voice_characteristics"]
                }
                
        # Add comparative analysis
        comparison["summary"] = self._create_comparative_summary(comparison)
        
        return comparison
        
    def _create_comparative_summary(self, comparison: Dict) -> Dict:
        """Create summary comparing datasets."""
        if not comparison:
            return {}
            
        best_for = {}
        metrics = ["total_duration_hours", "speaker_count", "avg_recording_per_speaker_seconds"]
        
        for metric in metrics:
            datasets = []
            for name, data in comparison.items():
                if name != "summary":
                    datasets.append((name, data["metrics"].get(metric, 0)))
                    
            if datasets:
                best_dataset = max(datasets, key=lambda x: x[1])
                worst_dataset = min(datasets, key=lambda x: x[1])
                best_for[metric] = {
                    "best": best_dataset[0],
                    "worst": worst_dataset[0],
                    "range": f"{worst_dataset[1]:.1f} - {best_dataset[1]:.1f}"
                }
                
        return best_for
