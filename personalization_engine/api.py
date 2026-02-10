from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
import tempfile
import json
from pathlib import Path

from .preprocessor import AudioPreprocessor
from .feature_extractor import FeatureExtractor
from .pattern_learner import PatternLearner
from .profile_manager import VoiceProfileManager
from .emotion_detector import EmotionDetector

app = FastAPI(title="Personalized TTS Engine API", version="1.0.0")
logger = logging.getLogger(__name__)

# Initialize components
preprocessor = AudioPreprocessor()
feature_extractor = FeatureExtractor()
profile_manager = VoiceProfileManager()
emotion_detector = EmotionDetector()

class TrainRequest(BaseModel):
    user_id: str
    audio_format: str = "wav"
    min_duration: float = 5.0

class SynthesizeRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    emotion: Optional[str] = None
    output_format: str = "wav"

class ProfileResponse(BaseModel):
    user_id: str
    profile_id: str
    created_at: str
    updated_at: str
    version: int
    feature_summary: dict

@app.get("/")
async def root():
    return {
        "service": "Personalized TTS Engine API",
        "version": "1.0.0",
        "endpoints": [
            "/train",
            "/synthesize",
            "/profiles",
            "/profiles/{user_id}",
            "/analyze/emotion"
        ]
    }

@app.post("/train")
async def train_model(
    user_id: str,
    files: List[UploadFile] = File(...),
    method: str = "gmm"
):
    """Train a personalized voice model from uploaded audio files"""
    
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="At least one audio file required")
        
    try:
        all_features = []
        
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
                
            # Process audio
            result = preprocessor.preprocess(tmp_path)
            features = feature_extractor.extract_all_features(result['audio'])
            
            # Detect emotion
            emotion_result = emotion_detector.detect(result['audio'])
            features['emotion_detected'] = emotion_result
            
            all_features.append(features)
            
            # Clean up
            Path(tmp_path).unlink()
            
        # Learn patterns
        pattern_learner = PatternLearner(method=method)
        features_matrix = pattern_learner.prepare_features(all_features)
        
        if method == "gmm":
            pattern_learner.learn_gmm(features_matrix)
        else:
            pattern_learner.learn_neural(features_matrix)
            
        # Create average features for profile
        avg_features = {}
        for key in all_features[0].keys():
            if isinstance(all_features[0][key], dict):
                avg_features[key] = {}
                for subkey in all_features[0][key].keys():
                    if isinstance(all_features[0][key][subkey], (int, float)):
                        values = [f[key][subkey] for f in all_features]
                        avg_features[key][subkey] = sum(values) / len(values)
                        
        # Create profile
        profile = profile_manager.create_profile(
            user_id=user_id,
            features=avg_features,
            model_info={
                "method": method,
                "n_samples": len(files),
                "features_extracted": list(avg_features.keys())
            }
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "profile_id": profile["profile_id"],
            "samples_processed": len(files),
            "profile_path": f"./profiles/{user_id}_profile.json"
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize")
async def synthesize_text(request: SynthesizeRequest):
    """Synthesize text with personalized voice"""
    
    try:
        # This would integrate with Piper TTS
        # For now, return parameters that would be used
        
        synthesis_params = {
            "text": request.text,
            "output_format": request.output_format,
            "personalization_applied": False
        }
        
        if request.user_id:
            profile = profile_manager.load_profile(request.user_id)
            if profile:
                synthesis_params["personalization_applied"] = True
                synthesis_params["user_id"] = request.user_id
                synthesis_params["pitch_adjustment"] = profile["features"].get(
                    "prosodic", {}).get("f0_mean", 0)
                
        if request.emotion:
            emotion_params = emotion_detector.map_emotion_to_synthesis_params(request.emotion)
            synthesis_params["emotion"] = request.emotion
            synthesis_params["emotion_params"] = emotion_params
            
        # In a real implementation, this would generate audio
        # For now, create a placeholder response
        synthesis_params["audio_generated"] = True
        synthesis_params["estimated_duration"] = len(request.text) * 0.05  # 50ms per char
        
        return synthesis_params
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profiles")
async def list_all_profiles():
    """List all available voice profiles"""
    
    profiles = profile_manager.list_profiles()
    return {"profiles": profiles, "count": len(profiles)}

@app.get("/profiles/{user_id}")
async def get_profile(user_id: str, format: str = "json"):
    """Get voice profile for specific user"""
    
    profile = profile_manager.load_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found for {user_id}")
        
    if format == "yaml":
        import yaml
        return JSONResponse(content=yaml.dump(profile, default_flow_style=False))
        
    return profile

@app.delete("/profiles/{user_id}")
async def delete_profile(user_id: str):
    """Delete a user's voice profile"""
    
    profile_manager.delete_profile(user_id)
    return {"status": "success", "message": f"Profile deleted for {user_id}"}

@app.post("/analyze/emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    """Analyze emotion from audio file"""
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        # Load and process audio
        audio, sr = preprocessor.load_audio(tmp_path)
        result = emotion_detector.detect(audio, sr)
        
        # Clean up
        Path(tmp_path).unlink()
        
        return result
        
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "preprocessor": "operational",
            "feature_extractor": "operational",
            "profile_manager": "operational",
            "emotion_detector": "operational" if emotion_detector.is_trained else "not_trained"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
