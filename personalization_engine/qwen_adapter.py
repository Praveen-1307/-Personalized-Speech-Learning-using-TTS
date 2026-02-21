
import os
import torch
import logging
import soundfile as sf
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

try:
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
except ImportError as e:
    import logging
    logging.getLogger(__name__).error(f"Failed to import qwen_tts: {e}")
    Qwen3TTSModel = None
    Qwen3TTSTokenizer = None

from personalization_engine.logger import get_logger, log_execution_details, log_complexity, log_system_metrics
logger = get_logger(__name__)

# Compatibility wrapper for the MLX-style API on Windows
@log_execution_details
def load_model(model_id: str, device: str = None):
    """Load Qwen3 TTS model for Windows."""
    if Qwen3TTSModel is None:
        raise ImportError("qwen-tts is not installed correctly.")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading Qwen TTS model: {model_id} on {device}")
    model = Qwen3TTSModel.from_pretrained(model_id, trust_remote_code=True)
    
    # Handle device placement for the wrapper
    if hasattr(model, 'to'):
        model.to(device)
    elif hasattr(model, 'model') and hasattr(model.model, 'to'):
        model.model.to(device)
    
    # Log model parameters
    num_params = sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else None
    params_str = f"{num_params:,}" if num_params is not None else "Unknown"
    logger.info(f"[ModelMetadata] Loaded {model_id} | Parameters: {params_str} | Device: {device}")
    log_complexity("QwenModel", "O(L * D^2)", "O(L * D)") # L=length, D=dim
    
    return model

import re

def split_text_into_chunks(text: str, max_chars: int = 400) -> List[str]:
    """Split text into manageable chunks for TTS based on sentence boundaries."""
    # Split by period, exclamation, or question mark followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += (sentence + " ")
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If a single sentence is longer than max_chars, split it by logical breaks
            if len(sentence) > max_chars:
                # Fallback: split by commas if sentence is too long
                sub_sentences = re.split(r'(?<=,)\s+', sentence)
                sub_chunk = ""
                for ss in sub_sentences:
                    if len(sub_chunk) + len(ss) <= max_chars:
                        sub_chunk += (ss + " ")
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = ss + " "
                current_chunk = sub_chunk
            else:
                current_chunk = sentence + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return [c for c in chunks if c.strip()]

@log_execution_details
def generate_audio(model, text: str, ref_audio: str, file_prefix: str = "output"):
    """Generate audio using Qwen TTS with automatic chunking for long texts."""
    log_system_metrics(logger)
    log_complexity("QwenInference", "O(Tokens * Model_Size)", "O(K_V_Cache)")
    
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
        
    # Split text into manageable chunks (e.g., 400 characters ~ 60-80 words)
    chunks = split_text_into_chunks(text)
    logger.info(f"Processing text in {len(chunks)} chunks")
    
    all_audio = []
    sample_rate = 24000
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Synthesizing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
        
        # model.generate_voice_clone returns (wavs, sample_rate)
        output = model.generate_voice_clone(
            text=chunk,
            ref_audio=ref_audio,
            x_vector_only_mode=True
        )
        
        if isinstance(output, tuple):
            wavs, sr = output[0], output[1]
            sample_rate = sr
            if isinstance(wavs, list) and len(wavs) > 0:
                chunk_audio = wavs[0]
            else:
                chunk_audio = wavs
        else:
            chunk_audio = output
            
        if chunk_audio is not None:
            all_audio.append(chunk_audio)
            log_system_metrics(logger)
            
    if not all_audio:
        raise RuntimeError("Synthesis failed to produce any audio chunks.")
        
    # Concatenate all chunks
    combined_audio = np.concatenate(all_audio)
    
    output_path = f"{file_prefix}.wav"
    sf.write(output_path, combined_audio, sample_rate)
    logger.info(f"Saved concatenated audio to {output_path}")
    log_system_metrics(logger)
    return output_path

class QwenAdapter:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.qwen_config = self.config.get('qwen_tts', {})
        # Use the 0.6B model for stability on Windows
        self.model_id = self.qwen_config.get('model_id', "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        self.device = "cuda" if (self.qwen_config.get('use_gpu', False) and torch.cuda.is_available()) else "cpu"
        
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = load_model(self.model_id, self.device)
            logger.info("Qwen TTS model loaded successfully via adapter.")
        except Exception as e:
            logger.error(f"Failed to load Qwen TTS model: {e}")
            raise

    def adapt(self, text: str, profile: Dict) -> Dict:
        user_id = profile.get('user_id', 'unknown')
        
        # Locate reference audio logic
        ref_audio_path = self._find_ref_audio(user_id)
        
        try:
            output_path = generate_audio(
                model=self.model,
                text=text,
                ref_audio=str(ref_audio_path),
                file_prefix=f"output/qwen_{user_id}_{int(torch.randint(0, 1000, (1,)))}"
            )
            
            return {
                'audio_path': output_path,
                'metadata': {
                    'engine': 'qwen',
                    'model': self.model_id,
                    'reference_audio': str(ref_audio_path)
                }
            }
        except Exception as e:
            logger.error(f"Qwen synthesis failed: {e}")
            raise

    def _find_ref_audio(self, user_id: str) -> Path:
        possible_paths = [
            Path(f"profiles/{user_id}.wav"),
            Path(f"samples/{user_id}.wav"),
            Path("profiles/current_session_user.wav"),
            Path("samples/ljspeech_subset/LJ002-0132.wav")
        ]
        for p in possible_paths:
            if p.exists():
                return p
        
        # Absolute fallback: find any wav in samples
        sample_dir = Path("samples")
        if sample_dir.exists():
            wavs = list(sample_dir.glob("**/*.wav"))
            if wavs:
                return wavs[0]
                
        raise FileNotFoundError(f"No reference audio found for user {user_id}")
