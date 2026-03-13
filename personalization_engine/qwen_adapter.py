
import os
import torch
import logging
import soundfile as sf
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

import librosa
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

def split_text_into_chunks(text: str, max_chars: int = 1200) -> List[str]:
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
def generate_audio(model, text: str, ref_audio: str, file_prefix: str = "output", **kwargs):
    """Generate audio using Qwen TTS with batching for voice consistency."""
    log_system_metrics(logger)
    log_complexity("QwenInference", "O(Batch * Tokens * Model_Size)", "O(Batch * K_V_Cache)")
    
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
        
    # Split text into larger chunks to minimize transitions (1200 chars ~ 200 words)
    chunks = split_text_into_chunks(text, max_chars=1200)
    logger.info(f"Processing text in {len(chunks)} chunks as a single batch")
    
    # Pre-extract speaker embedding once
    logger.info(f"Extracting voice clone prompt from {ref_audio}")
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        x_vector_only_mode=True
    )
    
    # Process ALL chunks in one batch for better consistency across segments
    try:
        # Lower temperature and higher top_p for more stable voice characteristics
        # Batch processing ensures the same speaker encoder state is used for all
        output = model.generate_voice_clone(
            text=chunks,
            voice_clone_prompt=[prompt] * len(chunks), # Manually broadcast prompt to all chunks for consistency
            x_vector_only_mode=True,
            temperature=0.1,  # Lower temperature for much higher voice stability
            top_p=0.95,       # Slightly more constrained
            repetition_penalty=1.1 # Prevent artifacts in long runs
        )
        
        if isinstance(output, tuple):
            wavs_list, sample_rate = output[0], output[1]
        else:
            # Handle unexpected return types
            logger.error(f"Unexpected output type from generate_voice_clone: {type(output)}")
            raise RuntimeError("Synthesis failed to produce expected output format.")

        if not wavs_list or len(wavs_list) == 0:
            raise RuntimeError("Synthesis failed to produce any audio chunks.")
            
        # Concatenate all chunks from the batch
        combined_audio = np.concatenate(wavs_list)
        
        # Apply Speed Adjustment (Time Stretching)
        speed_factor = kwargs.get('speed', 1.0)
        if speed_factor != 1.0:
            logger.info(f"Adjusting audio speed by factor: {speed_factor}")
            # rate < 1.0 makes it slower, > 1.0 makes it faster
            combined_audio = librosa.effects.time_stretch(y=combined_audio, rate=speed_factor)
            
        output_path = f"{file_prefix}.wav"
        sf.write(output_path, combined_audio, sample_rate)
        logger.info(f"Saved concatenated audio to {output_path} (Speed: {speed_factor}x)")
        log_system_metrics(logger)
        return output_path
        
    except Exception as e:
        logger.error(f"Batch synthesis failed: {e}")
        # Fallback to serial processing if batching fails for some reason
        logger.info("Retrying with serial processing...")
        all_audio = []
        sample_rate = 24000
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Fallback synthesizing chunk {i+1}/{len(chunks)}")
            out = model.generate_voice_clone(
                text=chunk,
                voice_clone_prompt=prompt,
                x_vector_only_mode=True,
                temperature=0.1
            )
            if isinstance(out, tuple):
                wavs, sr = out[0], out[1]
                sample_rate = sr
                all_audio.append(wavs[0] if isinstance(wavs, list) else wavs)
        
        if not all_audio:
            raise RuntimeError("Synthesis failed completely.")
            
        combined_audio = np.concatenate(all_audio)
        output_path = f"{file_prefix}.wav"
        sf.write(output_path, combined_audio, sample_rate)
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
        
        # Determine speed factor (1.0 is default, < 1.0 is slower, > 1.0 is faster)
        # Check global config first
        speed = self.config.get('audio', {}).get('speed_factor', 1.0)
        
        # Profile specific override if present
        speed = profile.get('features', {}).get('speaking_pattern', {}).get('speed_factor', speed)
        
        try:
            output_path = generate_audio(
                model=self.model,
                text=text,
                ref_audio=str(ref_audio_path),
                file_prefix=f"output/qwen_{user_id}_{int(torch.randint(0, 1000, (1,)))}",
                speed=speed
            )
            
            return {
                'audio_path': output_path,
                'metadata': {
                    'engine': 'qwen',
                    'model': self.model_id,
                    'reference_audio': str(ref_audio_path),
                    'speed_applied': speed
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
