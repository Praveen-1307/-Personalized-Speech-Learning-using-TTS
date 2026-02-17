import json
import yaml
from typing import Dict, List, Optional
import logging
from pathlib import Path
import hashlib
from datetime import datetime
import pickle
import numpy as np

from personalization_engine.logger import get_logger
logger = get_logger(__name__)

class VoiceProfileManager:
    def __init__(self, storage_dir: str = "./profiles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.profiles = {}
        
    def create_profile(self, user_id: str, features: Dict, 
                      model_info: Optional[Dict] = None) -> Dict:
        """Create a voice profile for a user."""
        profile_id = self._generate_profile_id(user_id, features)
        
        profile = {
            'profile_id': profile_id,
            'user_id': user_id,
            'features': features,
            'model_info': model_info or {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': 1
        }
        
        # Store in memory
        self.profiles[user_id] = profile
        
        # Save to disk
        self.save_profile(user_id)
        
        logger.info(f"Created profile for user {user_id}: {profile_id}")
        return profile
        
    def save_profile(self, user_id: str, format: str = 'both'):
        """Save profile to disk in specified format."""
        if user_id not in self.profiles:
            logger.warning(f"No profile found for user {user_id}")
            return
            
        profile = self.profiles[user_id]
        
        # Save as JSON
        if format in ['json', 'both']:
            json_path = self.storage_dir / f"{user_id}_profile.json"
            with open(json_path, 'w') as f:
                json.dump(profile, f, indent=2, default=self._json_serializer)
            logger.info(f"Saved JSON profile to {json_path}")
            
        # Save as YAML
        if format in ['yaml', 'both']:
            yaml_path = self.storage_dir / f"{user_id}_profile.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(profile, f, default_flow_style=False)
            logger.info(f"Saved YAML profile to {yaml_path}")
            
        # Save binary version for fast loading
        pickle_path = self.storage_dir / f"{user_id}_profile.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(profile, f)
            
    def load_profile(self, user_id: str, format: str = 'auto') -> Optional[Dict]:
        """Load profile from disk."""
        # Try pickle first for speed
        pickle_path = self.storage_dir / f"{user_id}_profile.pkl"
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    profile = pickle.load(f)
                self.profiles[user_id] = profile
                logger.info(f"Loaded profile from pickle: {pickle_path}")
                return profile
            except Exception as e:
                logger.warning(f"Failed to load pickle: {e}")
                
        # Try JSON
        json_path = self.storage_dir / f"{user_id}_profile.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    profile = json.load(f)
                self.profiles[user_id] = profile
                logger.info(f"Loaded profile from JSON: {json_path}")
                return profile
            except Exception as e:
                logger.warning(f"Failed to load JSON: {e}")
                
        # Try YAML
        yaml_path = self.storage_dir / f"{user_id}_profile.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    profile = yaml.safe_load(f)
                self.profiles[user_id] = profile
                logger.info(f"Loaded profile from YAML: {yaml_path}")
                return profile
            except Exception as e:
                logger.warning(f"Failed to load YAML: {e}")
                
        logger.warning(f"No profile found for user {user_id}")
        return None
        
    def update_profile(self, user_id: str, new_features: Dict):
        """Update existing profile with new features."""
        if user_id not in self.profiles:
            logger.warning(f"No existing profile for user {user_id}, creating new")
            return self.create_profile(user_id, new_features)
            
        profile = self.profiles[user_id]
        
        # Merge new features (simple update, could be more sophisticated)
        for key, value in new_features.items():
            if key in profile['features']:
                # Update existing features (could use weighted average)
                if isinstance(value, dict) and isinstance(profile['features'][key], dict):
                    profile['features'][key].update(value)
                else:
                    profile['features'][key] = value
            else:
                profile['features'][key] = value
                
        profile['updated_at'] = datetime.now().isoformat()
        profile['version'] += 1
        
        self.save_profile(user_id)
        logger.info(f"Updated profile for user {user_id} to version {profile['version']}")
        return profile
        
    def list_profiles(self) -> List[Dict]:
        """List all available profiles."""
        profiles = []
        for file in self.storage_dir.glob("*_profile.json"):
            user_id = file.name.replace("_profile.json", "")
            try:
                profile = self.load_profile(user_id)
                if profile:
                    profiles.append({
                        'user_id': user_id,
                        'created_at': profile.get('created_at'),
                        'updated_at': profile.get('updated_at'),
                        'version': profile.get('version', 1)
                    })
            except Exception as e:
                logger.error(f"Failed to load profile {file}: {e}")
                
        logger.info(f"Found {len(profiles)} profiles")
        return profiles
        
    def delete_profile(self, user_id: str):
        """Delete a user's profile."""
        # Remove from memory
        if user_id in self.profiles:
            del self.profiles[user_id]
            
        # Remove from disk
        for ext in ['.json', '.yaml', '.pkl']:
            file_path = self.storage_dir / f"{user_id}_profile{ext}"
            if file_path.exists():
                file_path.unlink()
                
        logger.info(f"Deleted profile for user {user_id}")
        
    def _generate_profile_id(self, user_id: str, features: Dict) -> str:
        """Generate unique profile ID."""
        # Create hash from user_id and feature summary
        feature_str = json.dumps(features, sort_keys=True, default=str)
        hash_input = f"{user_id}_{feature_str}"
        
        profile_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        return profile_id
        
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default."""
        if isinstance(obj, (datetime, np.integer, np.floating)):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")
        
    def export_profile_summary(self, user_id: str) -> Dict:
        """Export a summary of the profile."""
        profile = self.load_profile(user_id)
        if not profile:
            return {}
            
        summary = {
            'user_id': profile['user_id'],
            'profile_id': profile['profile_id'],
            'version': profile['version'],
            'created': profile['created_at'],
            'updated': profile['updated_at'],
            'feature_summary': {}
        }
        
        # Add feature statistics
        features = profile['features']
        if 'prosodic' in features:
            summary['feature_summary']['pitch_mean'] = features['prosodic'].get('f0_mean', 0)
            summary['feature_summary']['speaking_rate'] = features['speaking_pattern'].get('words_per_minute', 0)
            
        return summary
