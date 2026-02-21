import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import joblib
import json

from personalization_engine.logger import get_logger, log_execution_details, log_complexity
logger = get_logger(__name__)

class PatternLearner:
    def __init__(self, method: str = 'gmm', n_components: int = 8):
        self.method = method
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = {}
        
    @log_execution_details
    def prepare_features(self, features_list: List[Dict]) -> np.ndarray:
        """Prepare features for learning."""
        # Extract key features
        feature_vectors = []
        
        for features in features_list:
            vector = []
            
            # Prosodic features
            if 'prosodic' in features:
                prosodic = features['prosodic']
                vector.extend([
                    prosodic.get('f0_mean', 0),
                    prosodic.get('f0_std', 0),
                    prosodic.get('energy_mean', 0),
                    prosodic.get('energy_std', 0)
                ])
                
            # Speaking pattern features
            if 'speaking_pattern' in features:
                pattern = features['speaking_pattern']
                vector.extend([
                    pattern.get('words_per_minute', 0),
                    pattern.get('syllables_per_second', 0),
                    pattern.get('rhythm_entropy', 0)
                ])
                
            # Emotion features
            if 'emotion' in features:
                emotion = features['emotion']
                vector.extend([
                    emotion.get('pitch_mean', 0),
                    emotion.get('pitch_std', 0),
                    emotion.get('speaking_rate', 0)
                ])
                
            feature_vectors.append(vector)
            
        feature_matrix = np.array(feature_vectors)
        
        # Scale features
        if len(feature_matrix) > 1:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
            
        logger.info(f"Prepared features: {feature_matrix.shape}")
        return feature_matrix
        
    @log_execution_details
    def learn_gmm(self, features_matrix: np.ndarray):
        """Learn patterns using Gaussian Mixture Model."""
        log_complexity("GMM_Trainer", f"O(Iterations * K * N * D)", "O(K * D)")
        logger.info(f"[ObjectData] Feature Matrix Shape: {features_matrix.shape} | Components: {self.n_components}")
        try:
            # Handle small datasets
            if features_matrix.shape[0] < self.n_components:
                logger.warning(f"Insufficient samples ({features_matrix.shape[0]}) for GMM ({self.n_components}). Augmenting data.")
                # Simple augmentation: ensure we have at least n_components + 1 samples
                target_samples = self.n_components + 2
                original_matrix = features_matrix.copy()
                
                while features_matrix.shape[0] < target_samples:
                    noise = np.random.normal(0, 0.01, original_matrix.shape)
                    features_matrix = np.vstack([features_matrix, original_matrix + noise])

            gmm = GaussianMixture(n_components=self.n_components, 
                                 covariance_type='diag',
                                 random_state=42)
            gmm.fit(features_matrix)
            
            self.model = gmm
            logger.info(f"GMM trained with {self.n_components} components")
            
            # Calculate feature importance
            self.calculate_feature_importance(features_matrix)
            
            return gmm
            
        except Exception as e:
            logger.error(f"GMM training failed: {e}")
            raise
            
    @log_execution_details
    def learn_neural(self, features_matrix: np.ndarray, epochs: int = 100):
        """Learn patterns using neural network."""
        log_complexity("NeuralTrainer", f"O(Epochs * N * D^2)", "O(Weights)")
        logger.info(f"[ObjectData] Training Vector Count: {len(features_matrix)} | Hidden Dim: 64")
        class PatternNet(nn.Module):
            def __init__(self, input_dim, hidden_dim=64):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim // 4, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
                
        # Convert to torch tensors
        features_tensor = torch.FloatTensor(features_matrix)
        
        # Initialize model
        model = PatternNet(features_matrix.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(features_tensor)
            loss = criterion(outputs, features_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")
                
        self.model = model
        logger.info(f"Neural network trained for {epochs} epochs")
        return model
        
    def calculate_feature_importance(self, features_matrix: np.ndarray):
        """Calculate importance of each feature."""
        if self.method == 'gmm' and self.model is not None:
            # For GMM, use variance explained
            variances = np.var(features_matrix, axis=0)
            total_variance = np.sum(variances)
            
            self.feature_importance = {
                f'feature_{i}': var / total_variance
                for i, var in enumerate(variances)
            }
            
        logger.info(f"Feature importance calculated: {self.feature_importance}")
        
    @log_execution_details
    def generate_profile(self, features: Dict) -> Dict:
        """Generate voice profile from learned patterns."""
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        # Prepare single feature vector
        feature_vector = self.prepare_features([features])[0]
        
        if self.method == 'gmm':
            # Calculate posterior probabilities
            probs = self.model.predict_proba([feature_vector])[0]
            
            profile = {
                'method': 'gmm',
                'n_components': self.n_components,
                'component_probs': probs.tolist(),
                'means': self.model.means_.tolist(),
                'covariances': self.model.covariances_.tolist(),
                'feature_vector': feature_vector.tolist(),
                'feature_importance': self.feature_importance
            }
            
        elif self.method == 'neural':
            # Get encoded representation
            feature_tensor = torch.FloatTensor([feature_vector])
            encoded = self.model.encoder(feature_tensor)
            
            profile = {
                'method': 'neural',
                'encoded_vector': encoded.detach().numpy().tolist()[0],
                'feature_vector': feature_vector.tolist()
            }
            
        logger.info(f"Voice profile generated using {self.method}")
        return profile
        
    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return
            
        model_data = {
            'method': self.method,
            'n_components': self.n_components,
            'scaler': self.scaler,
            'model': self.model,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load trained model from disk."""
        model_data = joblib.load(path)
        
        self.method = model_data['method']
        self.n_components = model_data['n_components']
        self.scaler = model_data['scaler']
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        
        logger.info(f"Model loaded from {path}")
