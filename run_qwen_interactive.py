
import os
import sys
import time
import logging
import torch
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import our custom wrapper that mimics the MLX API but works on Windows
from personalization_engine.qwen_adapter import load_model, generate_audio
from personalization_engine.validator import SynthesisValidator

# Configure logging
from personalization_engine.logger import setup_logger, log_system_metrics
logger = setup_logger("qwen_interactive")

console = Console()

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone."""
    console.print(Panel(f"[bold cyan]Recording for {duration} seconds...[/bold cyan]\n[yellow]Speak naturally to capture your voice style![/yellow]", title="Microphone Input"))
    
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[red]Recording...", total=duration)
            for _ in range(duration):
                time.sleep(1)
                progress.advance(task)
                
        sd.wait()
        console.print("[green]Recording complete![/green]")
        return recording.flatten(), sample_rate
    except Exception as e:
        console.print(f"[bold red]Microphone Error:[/bold red] {e}")
        return None, None

def play_audio(audio_path):
    """Play audio file."""
    try:
        data, fs = sf.read(audio_path)
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        console.print(f"[red]Playback failed: {e}[/red]")

def main():
    console.print(Panel.fit("[bold magenta]Qwen3-TTS Voice Cloning Interface (Windows Optimized)[/bold magenta]", border_style="magenta"))
    
    # 1. Setup User ID and Reference Audio
    user_id = "current_session_user"
    profile_dir = Path("profiles")
    profile_dir.mkdir(exist_ok=True)
    ref_audio_path = profile_dir / f"{user_id}.wav"
    logger.info(f"Initialized session for user: {user_id}")
    
    # 2. Record or Use Existing
    if not ref_audio_path.exists() or Confirm.ask("Do you want to record a new voice reference?", default=True):
        duration = int(Prompt.ask("Recording duration (seconds)", default="5"))
        audio_data, sr = record_audio(duration=duration)
        
        if audio_data is not None:
            sf.write(str(ref_audio_path), audio_data, sr)
            logger.info(f"Reference audio saved to {ref_audio_path}")
            console.print(f"[green]Reference audio saved to {ref_audio_path}[/green]")
        else:
            console.print("[red]Recording failed. Exiting.[/red]")
            return
    else:
        console.print(f"[green]Using existing reference: {ref_audio_path}[/green]")

    # 3. Initialize Model and Analysis Tools
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    console.print(f"\n[bold]Initializing Qwen TTS Model: {model_id}...[/bold]")
    
    # Initialize analyzers
    from personalization_engine.feature_extractor import FeatureExtractor
    from personalization_engine.emotion_detector import EmotionDetector
    extractor = FeatureExtractor(sr=16000)
    detector = EmotionDetector()
    
    # Try to load pre-trained emotion model
    emotion_model_path = Path("models/emotion/svm_model.pkl")
    if emotion_model_path.exists():
        try:
            detector.load_model(str(emotion_model_path))
            console.print("[dim]Loaded emotion detector model.[/dim]")
        except:
            console.print("[dim yellow]Note: Could not load emotion model, using default analysis.[/dim]")

    # Initialize Validator
    validator = SynthesisValidator()

    try:
        with console.status("[bold green]Loading model...[/bold green]"):
            logger.info(f"Loading Qwen TTS model: {model_id}")
            model = load_model(model_id)
        logger.info("Model loaded successfully")
        console.print("[bold green]Model loaded successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to load model:[/bold red] {e}")
        return

    # Helper function to analyze voice
    def get_voice_metadata(audio_path):
        try:
            audio, sr = sf.read(str(audio_path))
            # Extract features
            features = extractor.extract_all_features(audio)
            # Detect emotion
            emotion_res = detector.detect(audio, sr=sr)
            
            return {
                "pitch": features['prosodic']['f0_mean'],
                "pitch_range": features['prosodic']['f0_range'],
                "stress_energy": features['prosodic']['energy_mean'],
                "emotion": emotion_res['emotion'],
                "emotion_confidence": emotion_res['confidence'],
                "speaking_pattern": features['speaking_pattern']
            }
        except Exception as e:
            return {"error": str(e)}

    # Initial analysis of recorded voice
    with console.status("[bold blue]Analyzing recorded voice...[/bold blue]"):
        logger.info("Starting initial voice analysis")
        ref_metadata = get_voice_metadata(ref_audio_path)
        logger.info(f"Initial analysis complete. Detected emotion: {ref_metadata.get('emotion', 'unknown')}")

    # 4. Interactive Synthesis Loop
    console.print("\n[bold green]Ready for synthesis![/bold green]")
    console.print("The same recorded voice will be used for all generations.")
    console.print("Each output will include a .json file with voice analysis.")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    import json

    while True:
        try:
            text = Prompt.ask("\n[bold cyan]Text to synthesize[/bold cyan]")
            
            if text.strip().lower() in ['exit', 'quit', 'stop']:
                break
            if not text.strip():
                continue
                
            timestamp = int(time.time())
            file_prefix = str(output_dir / f"cloned_{timestamp}")
            
            with console.status("[bold yellow]Cloning voice and generating audio...[/bold yellow]"):
                output_path = generate_audio(
                    model=model,
                    text=text,
                    ref_audio=str(ref_audio_path),
                    file_prefix=file_prefix
                )
                log_system_metrics(logger)
                
                # Save Metadata JSON
                json_path = output_dir / f"cloned_{timestamp}.json"
                metadata = {
                    "text": text,
                    "timestamp": timestamp,
                    "reference_voice_analysis": ref_metadata,
                    "model_used": model_id
                }
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                logger.info(f"Metadata written to {json_path}")
                
                # Generate Validation Report
                report_path = output_dir / f"cloned_{timestamp}_report.json"
                text_res = validator.validate_text(text)
                audio_res = validator.validate_audio(output_path, text)
                
                report = {
                    "summary": {"status": "PASS" if text_res['valid'] and audio_res['valid'] else "FAIL"},
                    "text": text_res,
                    "audio": audio_res
                }
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=4)
                
                logger.info(f"Validation report saved to {report_path}")
                
            console.print(f"✅ [green]Generated:[/green] [white]{output_path}[/white]")
            console.print(f"📄 [dim]Metadata saved to: {json_path}[/dim]")
            console.print(f"📊 [dim]Validation report: {report_path}[/dim]")
            
            # Display brief match info
            match_score = audio_res['metrics'].get('match_score', 0)
            console.print(f"🎯 [bold cyan]Estimated Word Match:[/bold cyan] {match_score:.1%}")
            
            play_audio(output_path)
            
            choice = Prompt.ask("[dim]Next action? [Enter for more text, 'r' to re-record, 'e' to exit][/dim]", default="")
            
            if choice.lower() == 'r':
                duration = int(Prompt.ask("Recording duration (seconds)", default="5"))
                audio_data, sr = record_audio(duration=duration)
                if audio_data is not None:
                    sf.write(str(ref_audio_path), audio_data, sr)
                    console.print(f"[green]Voice updated![/green]")
                    with console.status("[bold blue]Analyzing new voice...[/bold blue]"):
                        ref_metadata = get_voice_metadata(ref_audio_path)
            elif choice.lower() in ['e', 'exit']:
                break
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error during synthesis:[/bold red] {e}")

    console.print(Panel("[bold blue]Session ended. All outputs and metadata are in the 'output' folder.[/bold blue]"))

if __name__ == "__main__":
    main()
