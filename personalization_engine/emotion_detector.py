import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import librosa

logger = logging.getLogger(__name__)

class EmotionDetector:
    # Emotion categories based on research
    EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'calm']
    
    def __init__(self, model_type: str = 'svm'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
    def extract_emotion_features(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Extract features relevant for emotion detection."""
        features = []
        
        # Pitch features
        f0 = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)[0]
        f0 = f0[~np.isnan(f0)]
        
        if len(f0) > 0:
            features.append(np.mean(f0))
            features.append(np.std(f0))
            features.append(np.ptp(f0))
            features.append(np.median(f0))
        else:
            features.extend([0, 0, 0, 0])
            
        # Energy features
        energy = librosa.feature.rms(y=audio)[0]
        features.append(np.mean(energy))
        features.append(np.std(energy))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for i in range(5):  # First 5 coefficients
            features.append(np.mean(mfccs[i]))
            features.append(np.std(mfccs[i]))
            
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.append(np.mean(zcr))
        
        # Harmonics-to-noise ratio
        try:
            harmonic = librosa.effects.harmonic(audio)
            percussive = librosa.effects.percussive(audio)
            hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-10)
            features.append(hnr)
        except:
            features.append(0)
            
        # Speaking rate approximation
        voiced_frames = np.sum(f0 > 0)
        total_frames = len(f0)
        speaking_rate = voiced_frames / total_frames if total_frames > 0 else 0
        features.append(speaking_rate)
        
        return np.array(features)
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train emotion classifier."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, class_weight='balanced')
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        accuracy = self.model.score(X_scaled, y)
        logger.info(f"Emotion detector trained with {accuracy:.2%} accuracy")
        
    def detect(self, audio: np.ndarray, sr: int = 22050) -> Dict:
        """Detect emotion from audio."""
        if not self.is_trained:
            # logger.warning("Emotion detector not trained, using default neutral")
            return {
                'emotion': 'neutral', 
                'confidence': 1.0, 
                'probabilities': {'neutral': 1.0}
            }
            
        # Extract features
        features = self.extract_emotion_features(audio, sr)
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Handle case where model returns strings (e.g. 'happy') instead of indices (0, 1...)
        if isinstance(prediction, (str, np.str_)):
            emotion = str(prediction)
            try:
                # Get index of the class for confidence mapping
                emotion_idx = list(self.model.classes_).index(prediction)
            except:
                emotion_idx = 0 # Fallback
        else:
            emotion_idx = int(prediction)
            emotion = self.EMOTIONS[emotion_idx] if emotion_idx < len(self.EMOTIONS) else 'neutral'
            
        confidence = float(probabilities[emotion_idx])
        
        # Create probability dictionary
        if hasattr(self.model, 'classes_'):
            labels = [str(c) for c in self.model.classes_]
            prob_dict = {labels[i]: float(prob) for i, prob in enumerate(probabilities)}
        else:
            prob_dict = {self.EMOTIONS[i]: float(prob) for i, prob in enumerate(probabilities) if i < len(self.EMOTIONS)}
        
        result = {
            'emotion': emotion,
            'confidence': float(confidence),
            'probabilities': prob_dict,
            'features': features.tolist()
        }
        
        logger.info(f"Detected emotion: {emotion} (confidence: {confidence:.2%})")
        return result
        
    def map_emotion_to_synthesis_params(self, emotion: str) -> Dict:
        """Map emotion to TTS synthesis parameters."""
        # Based on research on emotional speech synthesis
        params = {
            'neutral': {
                'pitch_shift': 0,
                'speaking_rate': 1.0,
                'energy_scale': 1.0,
                'pause_duration': 1.0
            },
            'happy': {
                'pitch_shift': 2.0,  # Higher pitch
                'speaking_rate': 1.2,  # Faster
                'energy_scale': 1.3,  # More energy
                'pause_duration': 0.8  # Shorter pauses
            },
            'sad': {
                'pitch_shift': -1.5,  # Lower pitch
                'speaking_rate': 0.8,  # Slower
                'energy_scale': 0.7,  # Less energy
                'pause_duration': 1.5  # Longer pauses
            },
            'angry': {
                'pitch_shift': 1.0,
                'speaking_rate': 1.1,
                'energy_scale': 1.5,
                'pause_duration': 0.9
            },
            'calm': {
                'pitch_shift': -0.5,
                'speaking_rate': 0.9,
                'energy_scale': 0.9,
                'pause_duration': 1.2
            }
        }
        
        return params.get(emotion, params['neutral'])
        
    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return
            
        model_data = {
            'model_type': self.model_type,
            'scaler': self.scaler,
            'model': self.model,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Emotion model saved to {path}")
        
    def load_model(self, path: str):
        """Load trained model from disk."""
        model_data = joblib.load(path)
        
        self.model_type = model_data['model_type']
        self.scaler = model_data['scaler']
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Emotion model loaded from {path}")
