
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from personalization_engine.synthesis_adapter import SynthesisAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Synthesize text using a personalized profile.")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--user-id", type=str, required=True, help="User ID to use for personalization")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file path")
    args = parser.parse_args()

    # 1. Load Profile
    profile_path = Path(f"profiles/{args.user_id}_profile.json")
    if not profile_path.exists():
        logger.error(f"Profile for user {args.user_id} not found at {profile_path}")
        # Try fallback to standard user profile
        profile_path = Path("user_profile.json")
        if profile_path.exists():
             logger.warning(f"Using fallback user_profile.json")
        else:
             sys.exit(1)
             
    with open(profile_path, 'r') as f:
        profile = json.load(f)
        
    logger.info(f"Loaded profile for {args.user_id}")
    
    # Load Config
    import yaml
    config = {}
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # 2. Initialize Adapter
    engine_type = config.get('tts_engine', 'piper')
    logger.info(f"Using TTS Engine: {engine_type}")
    
    if engine_type == 'qwen':
        try:
            from personalization_engine.qwen_adapter import QwenAdapter
            adapter = QwenAdapter(config)
        except ImportError as e:
            logger.error(f"Failed to import QwenAdapter: {e}")
            sys.exit(1)
    else:
        adapter = SynthesisAdapter(config)
    # 3. Synthesize
    logger.info(f"Synthesizing: '{args.text}'")
    result = adapter.adapt(args.text, profile)
    
    # 4. Save/Move Output
    output_path = Path(args.output)
    generated_path = Path(result['audio_path'])
    
    if generated_path.exists():
        import shutil
        shutil.move(str(generated_path), str(output_path))
        logger.info(f"Saved personalized audio to: {output_path}")
        logger.info(f"Metadata: {result['metadata']}")
    else:
        logger.error("Synthesis failed to produce file.")

if __name__ == "__main__":
    main()
